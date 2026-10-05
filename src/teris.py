"""可由外部智能体控制的俄罗斯方块。

启动游戏与本地控制服务：
    python src/teris.py run

另一个终端中的控制示例：
    python src/teris.py start
    python src/teris.py state
    python src/teris.py legal
    python src/teris.py fast-place R0_X2 --round-serial 1 --piece-serial 1
    python src/teris.py end

控制协议是 localhost TCP JSONL；默认端口 8765。

决策时序约束：
* 方块在外部规划和 Laya 复核期间始终自动下落。
* run 只准备窗口与接口；start_game 被调用前不会开始计时或自动下落。
* 回合开始后按游戏时间逐级加速；默认从 600ms 开始，每 30 秒减少 75ms，最低 150ms。
* place 必须携带生成候选时的 round_serial 与 piece_serial。
* 若该方块已经自然落地，迟到的决策会被拒绝，绝不会应用到下一块。
* 游戏窗口关闭后控制服务一并结束；调用端应把连接终止视为测试结束。
"""

import argparse
import json
import os
import random
import socket
import socketserver
import threading
import time

import pygame


SHAPES = {
    "I": [[1, 1, 1, 1]],
    "O": [[1, 1], [1, 1]],
    "T": [[0, 1, 0], [1, 1, 1]],
    "L": [[1, 0, 0], [1, 1, 1]],
    "J": [[0, 0, 1], [1, 1, 1]],
    "S": [[0, 1, 1], [1, 1, 0]],
    "Z": [[1, 1, 0], [0, 1, 1]],
}
LINE_CLEAR_POINTS = (0, 100, 300, 500, 800)

BLOCK_COLORS = [
    (80, 180, 255), (255, 100, 100), (100, 220, 120), (255, 200, 80),
    (200, 120, 255), (255, 150, 80), (100, 220, 220), (255, 120, 180),
]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = int(os.getenv("TETRIS_CONTROL_PORT", "8765"))
DEFAULT_FALL_MS = int(os.getenv("TETRIS_FALL_MS", "600"))
DEFAULT_MIN_FALL_MS = int(os.getenv("TETRIS_MIN_FALL_MS", "150"))
DEFAULT_SPEEDUP_EVERY_SECONDS = float(os.getenv("TETRIS_SPEEDUP_EVERY_SECONDS", "30"))
DEFAULT_SPEEDUP_STEP_MS = int(os.getenv("TETRIS_SPEEDUP_STEP_MS", "75"))


def game_rules():
    """供规划者读取的客观规则，不包含评分、推荐或候选落点。"""
    return {
        "board_width": 10,
        "board_height": 20,
        "coordinates": "x 从左到右为 0..9；y 从上到下为 0..19",
        "pieces": {name: copy_shape(shape) for name, shape in SHAPES.items()},
        "rotation": "rotation_cw 表示顺时针旋转次数，允许 0..3",
        "placement": (
            "target_x 是旋转后方块包围盒左边缘列；执行时保持该旋转和 x 并直接落底"
        ),
        "line_clear": (
            "横向 10 格全部占用时消行；单次消除 1/2/3/4 行分别计 "
            "100/300/500/800 分"
        ),
        "gravity": (
            "当前方块会自动下落，且下落间隔随游戏时间逐步缩短；"
            "左右移动和旋转不会重置、推迟或暂停自动下落计时"
        ),
        "late_decision": "若 round_serial 或 piece_serial 已变化，策略必须被拒绝",
        "ai_execution": (
            "Laya 选定策略后，必须按顺序执行旋转、逐格左右移动和 hard_drop；"
            "禁止直接改写方块形状或坐标"
        ),
    }


def copy_shape(shape):
    return [row.copy() for row in shape]


class TetrisEngine:
    W = 10
    H = 20

    def __init__(self, spawn_initial=True):
        self.grid = [[0] * self.W for _ in range(self.H)]
        self.score = 0
        self.lines_cleared = 0
        self.pieces_locked = 0
        self.piece_serial = 0
        self.game_over = False
        self._bag = []
        self.piece = []
        self.piece_name = ""
        self.piece_color = None
        self.px = 0
        self.py = 0
        self.next_piece_name = ""
        self.next_piece = []
        # UI 可选消费的消行动画事件。游戏逻辑仍会立即完成删行，
        # 因此动画不会阻塞自动下落、控制接口或智能体决策。
        self.clear_event_serial = 0
        self.last_clear_rows = ()
        self.last_clear_grid = None
        self.last_clear_started_at = 0.0
        if spawn_initial:
            self.next_piece_name, self.next_piece = self._draw_from_bag()
            self.spawn_new_piece()

    def _draw_from_bag(self):
        if not self._bag:
            self._bag = list(SHAPES)
            random.shuffle(self._bag)
        name = self._bag.pop()
        return name, copy_shape(SHAPES[name])

    def spawn_new_piece(self):
        shape = copy_shape(self.next_piece)
        name = self.next_piece_name
        x = self.W // 2 - len(shape[0]) // 2
        if self.check_collision(shape, x, 0):
            self.game_over = True
            self.piece = []
            self.piece_color = None
            return False
        self.piece = shape
        self.piece_name = name
        self.piece_color = random.choice(BLOCK_COLORS)
        self.px = x
        self.py = 0
        self.piece_serial += 1
        self.next_piece_name, self.next_piece = self._draw_from_bag()
        return True

    def check_collision(self, shape, x, y, grid=None):
        grid = self.grid if grid is None else grid
        for dy, row in enumerate(shape):
            for dx, cell in enumerate(row):
                if not cell:
                    continue
                nx = x + dx
                ny = y + dy
                if nx < 0 or nx >= self.W or ny >= self.H:
                    return True
                if ny >= 0 and grid[ny][nx]:
                    return True
        return False

    @staticmethod
    def rotate_shape(shape):
        return [list(row) for row in zip(*shape[::-1])]

    def move(self, dx):
        if self.game_over or self.check_collision(self.piece, self.px + dx, self.py):
            return False
        self.px += dx
        return True

    def rotate_cw(self):
        if self.game_over:
            return False
        rotated = self.rotate_shape(self.piece)
        for kick in (0, 1, -1, 2, -2):
            if not self.check_collision(rotated, self.px + kick, self.py):
                self.piece = rotated
                self.px += kick
                return True
        return False

    def soft_drop(self):
        if self.game_over:
            return {"moved": False, "locked": False}
        if not self.check_collision(self.piece, self.px, self.py + 1):
            self.py += 1
            return {"moved": True, "locked": False}
        self.lock_piece()
        return {"moved": False, "locked": True}

    def hard_drop(self):
        if self.game_over:
            return {"distance": 0, "locked": False}
        start_y = self.py
        while not self.check_collision(self.piece, self.px, self.py + 1):
            self.py += 1
        distance = self.py - start_y
        self.lock_piece()
        return {"distance": distance, "locked": True}

    def lock_piece(self):
        for dy, row in enumerate(self.piece):
            for dx, cell in enumerate(row):
                if cell:
                    ny = self.py + dy
                    nx = self.px + dx
                    if 0 <= ny < self.H and 0 <= nx < self.W:
                        self.grid[ny][nx] = self.piece_color
        self.pieces_locked += 1
        self.clear_lines()
        self.spawn_new_piece()

    def clear_lines(self):
        cleared_rows = tuple(y for y, row in enumerate(self.grid) if all(row))
        if cleared_rows:
            self.clear_event_serial += 1
            self.last_clear_rows = cleared_rows
            self.last_clear_grid = [row.copy() for row in self.grid]
            self.last_clear_started_at = time.monotonic()
        remaining = [row for row in self.grid if not all(row)]
        cleared = self.H - len(remaining)
        self.grid = [[0] * self.W for _ in range(cleared)] + remaining
        self.lines_cleared += cleared
        self.score += LINE_CLEAR_POINTS[cleared]
        return cleared

    def grid_ascii(self, include_active=False):
        grid = [[bool(cell) for cell in row] for row in self.grid]
        if include_active and self.piece:
            for dy, row in enumerate(self.piece):
                for dx, cell in enumerate(row):
                    y = self.py + dy
                    x = self.px + dx
                    if cell and 0 <= y < self.H and 0 <= x < self.W:
                        grid[y][x] = True
        return "\n".join("".join("#" if cell else "." for cell in row) for row in grid)

    def simulate_landing(self, shape, x, start_y=None):
        y = self.py if start_y is None else start_y
        if self.check_collision(shape, x, y):
            return None
        while not self.check_collision(shape, x, y + 1):
            y += 1
        grid = [row.copy() for row in self.grid]
        for dy, row in enumerate(shape):
            for dx, cell in enumerate(row):
                if cell:
                    grid[y + dy][x + dx] = True
        lines = sum(1 for row in grid if all(row))
        grid = [[False] * self.W for _ in range(lines)] + [row for row in grid if not all(row)]
        metrics = self.analyze_grid(grid)
        metrics["lines"] = lines
        metrics["score_delta"] = LINE_CLEAR_POINTS[lines]
        return y, grid, metrics

    def analyze_grid(self, grid):
        heights = []
        holes_by_column = []
        for x in range(self.W):
            height = 0
            first_block = None
            for y in range(self.H):
                if grid[y][x]:
                    first_block = y
                    height = self.H - y
                    break
            heights.append(height)
            if first_block is None:
                holes_by_column.append(0)
            else:
                holes_by_column.append(sum(not grid[y][x] for y in range(first_block, self.H)))
        return {
            "holes": sum(holes_by_column),
            "holes_by_column": holes_by_column,
            "heights": heights,
            "max_height": max(heights, default=0),
            "aggregate_height": sum(heights),
            "bumpiness": sum(abs(heights[i] - heights[i + 1]) for i in range(self.W - 1)),
        }

    def legal_landings(self):
        if self.game_over or not self.piece:
            return []
        results = []
        shape = copy_shape(self.piece)
        seen = set()
        rotation = 0
        while True:
            key = tuple(tuple(row) for row in shape)
            if key in seen:
                break
            seen.add(key)
            for x in range(self.W - len(shape[0]) + 1):
                simulated = self.simulate_landing(shape, x)
                if simulated is None:
                    continue
                final_y, grid, metrics = simulated
                results.append({
                    "id": f"R{rotation}_X{x}",
                    "rotation_cw": rotation,
                    "target_x": x,
                    "final_y": final_y,
                    "metrics": metrics,
                    "board_after": "\n".join(
                        "".join("#" if cell else "." for cell in row) for row in grid
                    ),
                })
            shape = self.rotate_shape(shape)
            rotation += 1
        return results

    def resolve_plan(self, rotation_cw, target_x):
        """只校验并解析一个外部策略，不枚举、不评价、不推荐其他策略。"""
        try:
            rotation_cw = int(rotation_cw)
            target_x = int(target_x)
        except (TypeError, ValueError):
            return None
        if rotation_cw < 0 or rotation_cw > 3 or self.game_over or not self.piece:
            return None
        shape = copy_shape(self.piece)
        for _ in range(rotation_cw):
            shape = self.rotate_shape(shape)
        if target_x < 0 or target_x + len(shape[0]) > self.W:
            return None
        simulated = self.simulate_landing(shape, target_x)
        if simulated is None:
            return None
        final_y, grid, metrics = simulated
        return {
            "rotation_cw": rotation_cw,
            "target_x": target_x,
            "final_y": final_y,
            "cleared_lines": metrics["lines"],
            "board_after": "\n".join(
                "".join("#" if cell else "." for cell in row) for row in grid
            ),
        }


class GameController:
    def __init__(self, fall_ms=DEFAULT_FALL_MS):
        self.lock = threading.RLock()
        # 服务启动时是真正的空闲态：不生成、抽取或显示任何方块。
        self.game = TetrisEngine(spawn_initial=False)
        self.initial_fall_ms = max(50, int(fall_ms))
        self.minimum_fall_ms = min(self.initial_fall_ms, max(50, DEFAULT_MIN_FALL_MS))
        self.speedup_every_seconds = max(1.0, DEFAULT_SPEEDUP_EVERY_SECONDS)
        self.speedup_step_ms = max(1, DEFAULT_SPEEDUP_STEP_MS)
        self.revision = 0
        self.round_serial = 0
        self.round_active = False
        self.round_end_reason = None
        self.started_at = None
        self.round_ended_at = None
        self.last_fall_at = time.perf_counter()
        self.last_command = "等待 start_game"
        self.stop_event = threading.Event()
        self.termination_reason = None

    def round_elapsed(self, now=None):
        if self.started_at is None:
            return 0.0
        now = time.perf_counter() if now is None else now
        end = now if self.round_active else (self.round_ended_at or now)
        return max(0.0, end - self.started_at)

    def speed_level(self, now=None):
        raw_level = int(self.round_elapsed(now) // self.speedup_every_seconds)
        max_level = (
            self.initial_fall_ms - self.minimum_fall_ms + self.speedup_step_ms - 1
        ) // self.speedup_step_ms
        return min(raw_level, max_level)

    def current_fall_ms(self, now=None):
        return max(
            self.minimum_fall_ms,
            self.initial_fall_ms - self.speed_level(now) * self.speedup_step_ms,
        )

    @property
    def fall_ms(self):
        """兼容旧调用方：返回当前速度对应的下落间隔。"""
        return self.current_fall_ms()

    def _finish_round(self, reason):
        self.round_active = False
        self.round_end_reason = reason
        self.round_ended_at = time.perf_counter()

    def snapshot(self):
        game = self.game
        now = time.perf_counter()
        fall_ms = self.current_fall_ms(now)
        level = self.speed_level(now)
        at_max_speed = fall_ms <= self.minimum_fall_ms
        elapsed = self.round_elapsed(now)
        next_speedup = None
        if self.round_active and not at_max_speed:
            next_speedup = max(
                0.0,
                self.speedup_every_seconds - (elapsed % self.speedup_every_seconds),
            )
        remaining = (
            max(0, fall_ms - int((now - self.last_fall_at) * 1000))
            if self.round_active else None
        )
        if self.stop_event.is_set():
            lifecycle = "terminated"
        elif self.round_active:
            lifecycle = "running"
        elif self.round_end_reason == "game_over":
            lifecycle = "game_over"
        elif self.round_end_reason:
            lifecycle = "ended"
        else:
            lifecycle = "waiting"
        return {
            "revision": self.revision,
            "paused": False,
            "lifecycle": lifecycle,
            "service_running": not self.stop_event.is_set(),
            "termination_reason": self.termination_reason,
            "round_serial": self.round_serial,
            "round_active": self.round_active,
            "round_end_reason": self.round_end_reason,
            "game_over": game.game_over,
            "score": game.score,
            "lines_cleared": game.lines_cleared,
            "pieces_locked": game.pieces_locked,
            "piece_serial": game.piece_serial,
            "current_piece": {
                "name": game.piece_name,
                "shape": game.piece,
                "x": game.px,
                "y": game.py,
            },
            "next_piece": {"name": game.next_piece_name, "shape": game.next_piece},
            "fall_ms": fall_ms,
            "initial_fall_ms": self.initial_fall_ms,
            "minimum_fall_ms": self.minimum_fall_ms,
            "speed_level": level,
            "speedup_every_seconds": self.speedup_every_seconds,
            "speedup_step_ms": self.speedup_step_ms,
            "seconds_until_speedup": (
                round(next_speedup, 3) if next_speedup is not None else None
            ),
            "ms_until_auto_fall": remaining,
            "elapsed_seconds": round(elapsed, 3),
            "board": game.grid_ascii(False),
            "board_with_active_piece": game.grid_ascii(True),
            "last_command": self.last_command,
        }

    def auto_tick(self):
        with self.lock:
            if not self.round_active or self.game.game_over:
                return
            now = time.perf_counter()
            if (now - self.last_fall_at) * 1000 >= self.current_fall_ms(now):
                result = self.game.soft_drop()
                self.last_fall_at = now
                self.revision += 1
                self.last_command = "自动下落并锁定" if result["locked"] else "自动下落"
                if self.game.game_over:
                    self._finish_round("game_over")
                    self.last_command = "游戏结束"

    def _apply_action(self, action):
        game = self.game
        if action == "left":
            return {"moved": game.move(-1)}
        if action == "right":
            return {"moved": game.move(1)}
        if action == "rotate_cw":
            return {"rotated": game.rotate_cw()}
        if action == "soft_drop":
            return game.soft_drop()
        if action == "hard_drop":
            return game.hard_drop()
        raise ValueError(f"未知动作：{action}")

    @staticmethod
    def _action_preserves_gravity_timer(action):
        """水平移动与旋转不得改变既有的自动下落时刻。"""
        return action in ("left", "right", "rotate_cw")

    def _execute_interaction_plan(self, rotation_cw, target_x):
        """用与玩家相同的动作原语执行策略，成功后以 hard_drop 落底。"""
        game = self.game
        original_piece = copy_shape(game.piece)
        original_x = game.px
        actions = []
        results = []

        def rollback(error):
            game.piece = original_piece
            game.px = original_x
            return {
                "ok": False,
                "error": error,
                "actions": actions,
                "results": results,
            }

        for _ in range(rotation_cw):
            result = self._apply_action("rotate_cw")
            actions.append("rotate_cw")
            results.append({"action": "rotate_cw", "result": result})
            if not result.get("rotated"):
                return rollback("rotation_blocked")

        movement = "right" if target_x > game.px else "left"
        for _ in range(abs(target_x - game.px)):
            result = self._apply_action(movement)
            actions.append(movement)
            results.append({"action": movement, "result": result})
            if not result.get("moved"):
                return rollback("horizontal_path_blocked")

        if game.px != target_x:
            return rollback("target_x_not_reached")

        drop_result = self._apply_action("hard_drop")
        actions.append("hard_drop")
        results.append({"action": "hard_drop", "result": drop_result})
        return {
            "ok": True,
            "actions": actions,
            "results": results,
            "drop": drop_result,
        }

    def handle(self, request):
        if not isinstance(request, dict):
            raise ValueError("请求必须是 JSON object")
        operation = request.get("op")
        with self.lock:
            if self.stop_event.is_set() and operation not in ("get_state", "get_rules"):
                return {
                    "ok": False,
                    "error": "game_terminated",
                    "terminal": True,
                    "reason": self.termination_reason,
                    "state": self.snapshot(),
                }

            if operation in ("start_game", "restart"):
                self.game = TetrisEngine()
                self.round_serial += 1
                self.round_active = True
                self.round_end_reason = None
                self.round_ended_at = None
                self.started_at = time.perf_counter()
                self.last_fall_at = self.started_at
                self.last_command = "新回合开始"
                self.revision += 1
                return {"ok": True, "event": "game_started", "state": self.snapshot()}
            if operation == "end_game":
                if self.round_active:
                    self._finish_round("external_end")
                    self.last_command = "回合已结束"
                    self.revision += 1
                return {"ok": True, "event": "game_ended", "state": self.snapshot()}
            if operation == "quit":
                if self.round_active:
                    self._finish_round("service_quit")
                else:
                    self.round_end_reason = self.round_end_reason or "service_quit"
                self.last_command = "退出"
                self.termination_reason = "remote_quit"
                self.stop_event.set()
                return {"ok": True, "event": "quitting"}
            if operation == "get_state":
                return {"ok": True, "state": self.snapshot()}
            if operation == "get_rules":
                return {"ok": True, "rules": game_rules()}
            if operation == "set_fall_ms":
                self.initial_fall_ms = max(50, int(request["fall_ms"]))
                self.minimum_fall_ms = min(
                    self.initial_fall_ms, max(50, DEFAULT_MIN_FALL_MS)
                )
                self.last_fall_at = time.perf_counter()
                self.last_command = f"初始下落间隔 {self.initial_fall_ms}ms"
                self.revision += 1
                return {"ok": True, "state": self.snapshot()}

            if not self.round_active:
                return {
                    "ok": False,
                    "error": "round_not_active",
                    "terminal": self.round_end_reason == "game_over",
                    "state": self.snapshot(),
                }

            if operation in ("place", "fast_place", "validate_plan", "execute_plan") and (
                "expected_round_serial" not in request
                or "expected_piece_serial" not in request
            ):
                return {
                    "ok": False,
                    "error": "missing_decision_identity",
                    "message": "策略操作需要 expected_round_serial 和 expected_piece_serial",
                    "state": self.snapshot(),
                }

            expected_round = request.get("expected_round_serial")
            if expected_round is not None and int(expected_round) != self.round_serial:
                return {
                    "ok": False,
                    "error": "round_changed",
                    "decision_expired": True,
                    "expected": int(expected_round),
                    "actual": self.round_serial,
                    "state": self.snapshot(),
                }
            expected = request.get("expected_revision")
            if expected is not None and int(expected) != self.revision:
                return {
                    "ok": False,
                    "error": "revision_conflict",
                    "expected": int(expected),
                    "actual": self.revision,
                    "state": self.snapshot(),
                }
            expected_piece = request.get("expected_piece_serial")
            if expected_piece is not None and int(expected_piece) != self.game.piece_serial:
                return {
                    "ok": False,
                    "error": "piece_changed",
                    "decision_expired": True,
                    "natural_landing_preserved": True,
                    "expected": int(expected_piece),
                    "actual": self.game.piece_serial,
                    "state": self.snapshot(),
                }

            if operation == "validate_plan":
                validation = self.game.resolve_plan(
                    request.get("rotation_cw"), request.get("target_x")
                )
                if validation is None:
                    return {
                        "ok": False,
                        "error": "invalid_or_unreachable_plan",
                        "state": self.snapshot(),
                    }
                return {"ok": True, "validation": validation, "state": self.snapshot()}
            if operation == "legal_landings":
                future_names = list(self.game._bag) if self.game._bag else list(SHAPES)
                future_probability = 1.0 / len(future_names)
                return {
                    "ok": True,
                    "revision": self.revision,
                    "round_serial": self.round_serial,
                    "piece_serial": self.game.piece_serial,
                    "current_piece": {
                        "name": self.game.piece_name,
                        "shape": self.game.piece,
                        "x": self.game.px,
                        "y": self.game.py,
                    },
                    "next_piece": {
                        "name": self.game.next_piece_name,
                        "shape": self.game.next_piece,
                    },
                    "board": self.game.grid_ascii(False),
                    "piece_after_next_probabilities": {
                        name: future_probability for name in future_names
                    },
                    "ms_until_auto_fall": self.snapshot()["ms_until_auto_fall"],
                    "landings": self.game.legal_landings(),
                }
            if operation == "action":
                action = request["action"]
                count = max(1, int(request.get("count", 1)))
                results = [self._apply_action(action) for _ in range(count)]
                if self.game.game_over:
                    self._finish_round("game_over")
                if not self._action_preserves_gravity_timer(action):
                    self.last_fall_at = time.perf_counter()
                self.last_command = f"动作 {action} x{count}"
                self.revision += 1
                return {"ok": True, "results": results, "state": self.snapshot()}
            elif operation == "sequence":
                actions = request.get("actions", [])
                if not isinstance(actions, list) or not actions:
                    raise ValueError("sequence 需要非空 actions 数组")
                results = []
                for action in actions:
                    results.append({"action": action, "result": self._apply_action(action)})
                if self.game.game_over:
                    self._finish_round("game_over")
                if any(
                    not self._action_preserves_gravity_timer(action)
                    for action in actions
                ):
                    self.last_fall_at = time.perf_counter()
                self.last_command = "序列 " + " → ".join(actions)
                self.revision += 1
                return {"ok": True, "results": results, "state": self.snapshot()}
            elif operation == "execute_plan":
                validation = self.game.resolve_plan(
                    request.get("rotation_cw"), request.get("target_x")
                )
                if validation is None:
                    return {
                        "ok": False,
                        "error": "invalid_or_unreachable_plan",
                        "state": self.snapshot(),
                    }
                execution = self._execute_interaction_plan(
                    validation["rotation_cw"], validation["target_x"]
                )
                if not execution["ok"]:
                    return {
                        "ok": False,
                        "error": execution["error"],
                        "interaction_actions": execution["actions"],
                        "interaction_results": execution["results"],
                        "state": self.snapshot(),
                    }
                if self.game.game_over:
                    self._finish_round("game_over")
                self.last_fall_at = time.perf_counter()
                self.last_command = (
                    "交互执行复核策略 "
                    f"R{validation['rotation_cw']}_X{validation['target_x']}"
                )
                self.revision += 1
                return {
                    "ok": True,
                    "validation": validation,
                    "interaction_actions": execution["actions"],
                    "interaction_results": execution["results"],
                    "drop": execution["drop"],
                    "state": self.snapshot(),
                }
            elif operation in ("place", "fast_place"):
                if self.game.game_over or not self.game.piece:
                    return {
                        "ok": False,
                        "error": "game_over",
                        "terminal": True,
                        "decision_expired": True,
                        "natural_landing_preserved": True,
                        "state": self.snapshot(),
                    }
                landing_id = str(request["landing_id"])
                landing = next(
                    (item for item in self.game.legal_landings() if item["id"] == landing_id),
                    None,
                )
                if landing is None:
                    return {
                        "ok": False,
                        "error": "landing_no_longer_reachable",
                        "landing_id": landing_id,
                        "state": self.snapshot(),
                    }
                execution = self._execute_interaction_plan(
                    landing["rotation_cw"], landing["target_x"]
                )
                if not execution["ok"]:
                    return {
                        "ok": False,
                        "error": execution["error"],
                        "landing_id": landing_id,
                        "interaction_actions": execution["actions"],
                        "interaction_results": execution["results"],
                        "state": self.snapshot(),
                    }
                if self.game.game_over:
                    self._finish_round("game_over")
                self.last_fall_at = time.perf_counter()
                self.last_command = f"Laya交互执行并落底 {landing_id}"
                self.revision += 1
                return {
                    "ok": True,
                    "landing": landing,
                    "interaction_actions": execution["actions"],
                    "interaction_results": execution["results"],
                    "drop": execution["drop"],
                    "state": self.snapshot(),
                }
            else:
                raise ValueError(f"未知操作：{operation}")
            return {"ok": True, "state": self.snapshot()}


class ControlHandler(socketserver.StreamRequestHandler):
    def handle(self):
        line = self.rfile.readline()
        try:
            request = json.loads(line.decode("utf-8"))
            response = self.server.controller.handle(request)
        except Exception as exc:
            response = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
        self.wfile.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))


class ControlServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, controller):
        self.controller = controller
        super().__init__(address, ControlHandler)


# ====================== Pygame 面板 ======================
BLOCK_SIZE = 30
MARGIN = 2
CELL_SIZE = BLOCK_SIZE + MARGIN
BOARD_W = CELL_SIZE * 10
PANEL_X = BOARD_W + 24
PANEL_W = 340
SCREEN_W = PANEL_X + PANEL_W + 20
SCREEN_H = CELL_SIZE * 20

COLOR_BG = (12, 16, 24)
COLOR_BOARD_BG = (18, 23, 33)
COLOR_PANEL = (24, 30, 43)
COLOR_CARD = (31, 39, 55)
COLOR_BORDER = (53, 66, 88)
COLOR_GRID = (55, 64, 78)
COLOR_TEXT = (229, 235, 244)
COLOR_MUTED = (146, 158, 178)
COLOR_ACCENT = (91, 192, 235)
COLOR_SUCCESS = (103, 211, 145)
COLOR_WARNING = (255, 191, 92)
_FONT_CACHE = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _FONT_CACHE:
        path = pygame.font.match_font("microsoftyahei,simhei,arial")
        _FONT_CACHE[key] = pygame.font.Font(path, size)
        _FONT_CACHE[key].set_bold(bold)
    return _FONT_CACHE[key]


def text(screen, value, x, y, size=18, color=COLOR_TEXT, bold=False):
    screen.blit(font(size, bold).render(str(value), True, color), (x, y))


def card(screen, rect, title):
    pygame.draw.rect(screen, COLOR_CARD, rect, border_radius=10)
    pygame.draw.rect(screen, COLOR_BORDER, rect, 1, border_radius=10)
    text(screen, title, rect.x + 16, rect.y + 11, 16, COLOR_MUTED, True)


def duration(seconds):
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def preview(screen, shape, rect, color):
    pygame.draw.rect(screen, COLOR_BOARD_BG, rect, border_radius=8)
    pygame.draw.rect(screen, COLOR_BORDER, rect, 1, border_radius=8)
    if not shape:
        return
    cell = 18
    start_x = rect.centerx - len(shape[0]) * cell // 2
    start_y = rect.centery - len(shape) * cell // 2
    for y, row in enumerate(shape):
        for x, occupied in enumerate(row):
            if occupied:
                pygame.draw.rect(
                    screen,
                    color,
                    (start_x + x * cell, start_y + y * cell, cell - 2, cell - 2),
                    border_radius=3,
                )


def draw(screen, controller, port):
    with controller.lock:
        game = controller.game
        state = controller.snapshot()
        screen.fill(COLOR_BG)
        pygame.draw.rect(screen, COLOR_BOARD_BG, (0, 0, BOARD_W, SCREEN_H))
        for y in range(game.H):
            for x in range(game.W):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, BLOCK_SIZE - MARGIN, BLOCK_SIZE - MARGIN)
                if game.grid[y][x]:
                    pygame.draw.rect(screen, game.grid[y][x], rect, border_radius=3)
                pygame.draw.rect(screen, COLOR_GRID, rect, 1, border_radius=3)
        if game.piece and game.piece_color:
            for dy, row in enumerate(game.piece):
                for dx, occupied in enumerate(row):
                    x = game.px + dx
                    y = game.py + dy
                    if occupied and 0 <= x < game.W and 0 <= y < game.H:
                        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, BLOCK_SIZE - MARGIN, BLOCK_SIZE - MARGIN)
                        pygame.draw.rect(screen, game.piece_color, rect, border_radius=3)
                        pygame.draw.rect(screen, COLOR_GRID, rect, 1, border_radius=3)

        pygame.draw.rect(screen, COLOR_PANEL, (PANEL_X, 0, PANEL_W, SCREEN_H), border_radius=12)
        text(screen, "CODEX TETRIS", PANEL_X + 18, 16, 24, COLOR_ACCENT, True)
        status = {
            "waiting": "等待开始",
            "running": "自动下落",
            "game_over": "游戏结束",
            "ended": "回合结束",
            "terminated": "已关闭",
        }[state["lifecycle"]]
        status_color = COLOR_SUCCESS if state["lifecycle"] == "running" else COLOR_WARNING
        text(screen, status, PANEL_X + 230, 21, 16, status_color, True)

        score_rect = pygame.Rect(PANEL_X + 12, 58, PANEL_W - 24, 100)
        card(screen, score_rect, "游戏统计")
        for label, value, x, color in (
            ("分数", game.score, 16, COLOR_TEXT),
            ("消行", game.lines_cleared, 118, COLOR_SUCCESS),
            ("落块", game.pieces_locked, 218, COLOR_TEXT),
        ):
            text(screen, label, score_rect.x + x, score_rect.y + 39, 15, COLOR_MUTED)
            text(screen, value, score_rect.x + x, score_rect.y + 57, 28, color, True)

        piece_rect = pygame.Rect(PANEL_X + 12, 170, PANEL_W - 24, 116)
        card(screen, piece_rect, "方块队列")
        current_name = game.piece_name or "--"
        next_name = game.next_piece_name or "--"
        text(screen, f"当前  {current_name}", piece_rect.x + 16, piece_rect.y + 43, 18, COLOR_TEXT, True)
        text(screen, f"下一块  {next_name}", piece_rect.x + 16, piece_rect.y + 73, 18, COLOR_ACCENT, True)
        preview(screen, game.next_piece, pygame.Rect(piece_rect.right - 104, piece_rect.y + 28, 86, 72), COLOR_ACCENT)

        run_rect = pygame.Rect(PANEL_X + 12, 298, PANEL_W - 24, 116)
        card(screen, run_rect, "运行状态")
        text(screen, "总耗时", run_rect.x + 16, run_rect.y + 43, 16, COLOR_MUTED)
        text(screen, duration(state["elapsed_seconds"]), run_rect.x + 104, run_rect.y + 41, 19, COLOR_TEXT, True)
        text(screen, "自动下落", run_rect.x + 16, run_rect.y + 72, 16, COLOR_MUTED)
        text(screen, f"{state['fall_ms']} ms", run_rect.x + 104, run_rect.y + 70, 18, COLOR_WARNING, True)
        if state["round_active"]:
            speedup = state["seconds_until_speedup"]
            speed_text = (
                f"Lv{state['speed_level']} · {speedup:.0f}s后加速"
                if speedup is not None else f"Lv{state['speed_level']} · 最高速"
            )
        else:
            speed_text = "尚未开始"
        text(screen, speed_text, run_rect.x + 198, run_rect.y + 72, 14, COLOR_ACCENT)

        control_rect = pygame.Rect(PANEL_X + 12, 426, PANEL_W - 24, 194)
        card(screen, control_rect, "外部控制接口")
        text(screen, f"127.0.0.1:{port}", control_rect.x + 16, control_rect.y + 42, 17, COLOR_SUCCESS, True)
        text(screen, f"Round  #{controller.round_serial}", control_rect.x + 16, control_rect.y + 75, 16, COLOR_MUTED)
        text(screen, f"Piece  #{game.piece_serial}", control_rect.x + 170, control_rect.y + 75, 16, COLOR_MUTED)
        text(screen, "最近动作", control_rect.x + 16, control_rect.y + 108, 15, COLOR_MUTED)
        command = controller.last_command
        if len(command) > 28:
            command = command[:27] + "…"
        text(screen, command, control_rect.x + 16, control_rect.y + 131, 17, COLOR_TEXT, True)
        text(screen, "start · end · state · legal · place", control_rect.x + 16, control_rect.y + 163, 14, COLOR_MUTED)


def run_game(host, port, fall_ms):
    controller = GameController(fall_ms)
    server = ControlServer((host, port), controller)
    server_thread = threading.Thread(target=server.serve_forever, name="tetris-control", daemon=True)
    server_thread.start()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Codex Tetris【自动下落 + 外部控制】")
    clock = pygame.time.Clock()
    print(json.dumps({
        "event": "ready",
        "host": host,
        "port": port,
        "fall_ms": fall_ms,
        "minimum_fall_ms": controller.minimum_fall_ms,
        "speedup_every_seconds": controller.speedup_every_seconds,
        "speedup_step_ms": controller.speedup_step_ms,
        "round_active": False,
        "instruction": "准备完成后调用 start_game",
    }, ensure_ascii=False), flush=True)

    try:
        while not controller.stop_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    controller.termination_reason = "window_closed"
                    controller.last_command = "窗口已关闭"
                    controller.stop_event.set()
                elif event.type == pygame.KEYDOWN:
                    with controller.lock:
                        if event.key == pygame.K_LEFT:
                            controller.handle({"op": "action", "action": "left"})
                        elif event.key == pygame.K_RIGHT:
                            controller.handle({"op": "action", "action": "right"})
                        elif event.key == pygame.K_UP:
                            controller.handle({"op": "action", "action": "rotate_cw"})
                        elif event.key == pygame.K_DOWN:
                            controller.handle({"op": "action", "action": "soft_drop"})
                        elif event.key == pygame.K_SPACE:
                            controller.handle({"op": "action", "action": "hard_drop"})
            controller.auto_tick()
            draw(screen, controller, port)
            pygame.display.flip()
            clock.tick(60)
    finally:
        if controller.termination_reason is None:
            controller.termination_reason = "run_loop_ended"
        server.shutdown()
        server.server_close()
        pygame.quit()
        print(json.dumps({
            "event": "stopped",
            "terminal": True,
            "reason": controller.termination_reason,
        }, ensure_ascii=False), flush=True)


def send_request(host, port, request):
    with socket.create_connection((host, port), timeout=5) as connection:
        connection.sendall((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
        chunks = []
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    return json.loads(b"".join(chunks).decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Externally controlled Tetris")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--fall-ms", type=int, default=DEFAULT_FALL_MS)
    sub.add_parser("start")
    sub.add_parser("end")
    sub.add_parser("state")
    sub.add_parser("rules")
    sub.add_parser("legal")
    sub.add_parser("restart")
    sub.add_parser("soft-drop")
    sub.add_parser("hard-drop")
    sub.add_parser("quit")
    move_parser = sub.add_parser("move")
    move_parser.add_argument("direction", choices=("left", "right"))
    move_parser.add_argument("--count", type=int, default=1)
    rotate_parser = sub.add_parser("rotate")
    rotate_parser.add_argument("--count", type=int, default=1)
    sequence_parser = sub.add_parser("sequence")
    sequence_parser.add_argument("actions", nargs="+", choices=("left", "right", "rotate_cw", "soft_drop", "hard_drop"))
    fall_parser = sub.add_parser("fall-ms")
    fall_parser.add_argument("milliseconds", type=int)
    place_parser = sub.add_parser("place")
    place_parser.add_argument("landing_id")
    place_parser.add_argument("--round-serial", type=int, required=True)
    place_parser.add_argument("--piece-serial", type=int, required=True)
    fast_place_parser = sub.add_parser("fast-place")
    fast_place_parser.add_argument("landing_id")
    fast_place_parser.add_argument("--round-serial", type=int, required=True)
    fast_place_parser.add_argument("--piece-serial", type=int, required=True)
    for command in ("validate-plan", "execute-plan"):
        plan_parser = sub.add_parser(command)
        plan_parser.add_argument("rotation_cw", type=int)
        plan_parser.add_argument("target_x", type=int)
        plan_parser.add_argument("--round-serial", type=int, required=True)
        plan_parser.add_argument("--piece-serial", type=int, required=True)
    request_parser = sub.add_parser("request")
    request_parser.add_argument("json_request")
    args = parser.parse_args()

    if args.command == "run":
        run_game(args.host, args.port, args.fall_ms)
        return

    request_map = {
        "state": {"op": "get_state"},
        "rules": {"op": "get_rules"},
        "legal": {"op": "legal_landings"},
        "start": {"op": "start_game"},
        "restart": {"op": "start_game"},
        "end": {"op": "end_game"},
        "soft-drop": {"op": "action", "action": "soft_drop"},
        "hard-drop": {"op": "action", "action": "hard_drop"},
        "quit": {"op": "quit"},
    }
    if args.command in request_map:
        request = request_map[args.command]
    elif args.command == "move":
        request = {"op": "action", "action": args.direction, "count": args.count}
    elif args.command == "rotate":
        request = {"op": "action", "action": "rotate_cw", "count": args.count}
    elif args.command == "sequence":
        request = {"op": "sequence", "actions": args.actions}
    elif args.command == "fall-ms":
        request = {"op": "set_fall_ms", "fall_ms": args.milliseconds}
    elif args.command in ("place", "fast-place"):
        request = {
            "op": "fast_place" if args.command == "fast-place" else "place",
            "landing_id": args.landing_id,
            "expected_round_serial": args.round_serial,
            "expected_piece_serial": args.piece_serial,
        }
    elif args.command in ("validate-plan", "execute-plan"):
        request = {
            "op": "validate_plan" if args.command == "validate-plan" else "execute_plan",
            "rotation_cw": args.rotation_cw,
            "target_x": args.target_x,
            "expected_round_serial": args.round_serial,
            "expected_piece_serial": args.piece_serial,
        }
    else:
        request = json.loads(args.json_request)

    try:
        response = send_request(args.host, args.port, request)
    except OSError as exc:
        response = {
            "ok": False,
            "error": "game_unavailable",
            "terminal": True,
            "message": str(exc),
            "instruction": "停止本次控制；不要重启游戏或继续调用 Laya。",
        }
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
