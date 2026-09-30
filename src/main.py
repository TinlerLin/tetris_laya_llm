"""第三方 LLM / 启发式规划 + 可选本地 Laya 复核的实时俄罗斯方块。

统一入口组合 teris.py 游戏主体、tetris_llm.py 规划模块与
可插拔的 tetris_laya.py 复核模块，并负责执行最终选定的策略。
玩家可在开始前选择第三方 LLM、三层 Expectimax 启发式规划或玩家自主模式。

从仓库根目录启动：python src/main.py

可选环境变量（仅作为面板初始值）：
THIRD_PARTY_LLM_BASE_URL、THIRD_PARTY_LLM_API_KEY、
THIRD_PARTY_LLM_MODEL、THIRD_PARTY_LLM_TIMEOUT。
"""

import argparse
import os
import random
import threading
import time

import pygame

import teris as game_api
import tetris_heuristic as heuristic_module
import tetris_laya as laya_module
import tetris_llm as llm_module
import tetris_player as player_module


DEFAULT_BASE_URL = os.getenv("THIRD_PARTY_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
DEFAULT_API_KEY = os.getenv("THIRD_PARTY_LLM_API_KEY", "")
DEFAULT_MODEL = os.getenv("THIRD_PARTY_LLM_MODEL", "")
DEFAULT_TIMEOUT = float(os.getenv("THIRD_PARTY_LLM_TIMEOUT", "45"))
DEFAULT_PLANNER_MODE = os.getenv("TETRIS_PLANNER_MODE", "llm").strip().lower()
DEFAULT_USE_LAYA = os.getenv("TETRIS_USE_LAYA", "1").strip().lower() not in (
    "0", "false", "no", "off",
)
MAX_LLM_CANDIDATES = 4

BLOCK_SIZE = 36
MARGIN = 2
CELL_SIZE = BLOCK_SIZE + MARGIN
BOARD_W = CELL_SIZE * 10
SCREEN_H = CELL_SIZE * 20
PANEL_X = BOARD_W + 20
PANEL_W = 600
SCREEN_W = PANEL_X + PANEL_W + 16
FONT_SCALE = 1.10

BG = (11, 15, 23)
BOARD_BG = (18, 23, 33)
PANEL_BG = (23, 29, 42)
CARD_BG = (31, 39, 55)
BORDER = (57, 70, 93)
TEXT = (231, 236, 245)
MUTED = (145, 158, 179)
ACCENT = (91, 192, 235)
SUCCESS = (103, 211, 145)
WARNING = (255, 191, 92)
ERROR = (255, 112, 112)
BUTTON = (49, 113, 150)
_FONTS = {}


def font(size, bold=False):
    scaled_size = max(1, round(size * FONT_SCALE))
    key = (scaled_size, bold)
    if key not in _FONTS:
        # 与 tetris_laya_expectimax.py 使用完全相同的字体解析方式。
        font_path = pygame.font.match_font("microsoftyahei,simhei,arial")
        _FONTS[key] = pygame.font.Font(font_path, scaled_size)
        _FONTS[key].set_bold(bold)
    return _FONTS[key]


def draw_text(screen, value, x, y, size=18, color=TEXT, bold=False):
    screen.blit(font(size, bold).render(str(value), True, color), (x, y))


def card(screen, rect, title):
    pygame.draw.rect(screen, CARD_BG, rect, border_radius=10)
    pygame.draw.rect(screen, BORDER, rect, 1, border_radius=10)
    draw_text(screen, title, rect.x + 14, rect.y + 9, 15, MUTED, True)


def checkbox(screen, rect, checked, label, enabled=True):
    fill = (32, 44, 58) if enabled else (42, 47, 57)
    border = ACCENT if checked and enabled else BORDER
    pygame.draw.rect(screen, fill, rect, border_radius=4)
    pygame.draw.rect(screen, border, rect, 1, border_radius=4)
    if checked:
        pygame.draw.line(screen, SUCCESS, (rect.x + 4, rect.y + 9),
                         (rect.x + 8, rect.y + 13), 2)
        pygame.draw.line(screen, SUCCESS, (rect.x + 8, rect.y + 13),
                         (rect.x + 15, rect.y + 5), 2)
    draw_text(screen, label, rect.right + 8, rect.y - 1, 14, TEXT if enabled else MUTED)


def shorten(value, limit):
    value = str(value)
    return value if len(value) <= limit else value[: limit - 1] + "…"


def duration(seconds):
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _decode_clipboard_data(raw):
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, (bytes, bytearray)):
        return ""
    raw = bytes(raw)
    if not raw:
        return ""
    # Windows 剪贴板经常提供 UTF-16LE；Pygame 在不同后端也可能返回 UTF-8。
    encodings = (
        ("utf-16-le", "utf-8")
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or raw.count(b"\x00") > len(raw) // 4
        else ("utf-8", "utf-16-le")
    )
    for encoding in encodings:
        try:
            return raw.decode(encoding).lstrip("\ufeff").rstrip("\x00")
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore").replace("\x00", "")


def clipboard_text():
    """读取文本并保留换行；优先读取系统剪贴板，Pygame 仅作回退。"""
    root = None
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        root.update()
        pasted = root.clipboard_get()
        if pasted:
            return str(pasted).replace("\r\n", "\n").replace("\r", "\n").strip()
    except Exception:
        pass
    finally:
        if root is not None:
            root.destroy()
    try:
        pasted = _decode_clipboard_data(pygame.scrap.get(pygame.SCRAP_TEXT))
        if pasted:
            return pasted.replace("\r\n", "\n").replace("\r", "\n").strip()
    except (pygame.error, TypeError):
        pass
    return ""


def set_clipboard_text(value):
    """将文本写入系统剪贴板；优先使用 Tk，失败时回退到 Pygame。"""
    value = str(value)
    if not value:
        return False
    root = None
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(value)
        # update() 让 Windows 在窗口销毁后仍保留剪贴板所有权。
        root.update()
        return True
    except Exception:
        try:
            pygame.scrap.put(pygame.SCRAP_TEXT, value.encode("utf-8"))
            return True
        except (pygame.error, TypeError):
            return False
    finally:
        if root is not None:
            root.destroy()


class InputField:
    REPEATABLE_KEYS = {
        pygame.K_BACKSPACE, pygame.K_DELETE, pygame.K_LEFT, pygame.K_RIGHT,
    }
    REPEAT_DELAY_SECONDS = 0.36
    REPEAT_INTERVAL_SECONDS = 0.045

    def __init__(self, name, label, value="", secret=False):
        self.name = name
        self.label = label
        self.value = value
        self.secret = secret
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.active = False
        self.cursor = len(value)
        self.selection_anchor = self.cursor
        self.dragging = False
        self.view_start = 0
        self.view_end = len(value)
        self.repeat_key = None
        self.repeat_mod = 0
        self.next_repeat_at = None

    def display_value(self):
        if self.secret and self.value:
            # 保持显示字符数与真实值索引一一对应，才能正确定位光标和选区；
            # 超出输入框的部分由横向视口裁切。
            return "•" * len(self.value)
        return self.value

    def selection_bounds(self):
        return tuple(sorted((self.cursor, self.selection_anchor)))

    def has_selection(self):
        start, end = self.selection_bounds()
        return start != end

    def set_value(self, value):
        self.value = str(value)
        self.cursor = len(self.value)
        self.selection_anchor = self.cursor
        self.view_start = 0

    def replace_selection(self, text):
        start, end = self.selection_bounds()
        self.value = self.value[:start] + text + self.value[end:]
        self.cursor = start + len(text)
        self.selection_anchor = self.cursor

    def copy_selection(self, cut=False):
        if not self.has_selection():
            return False
        start, end = self.selection_bounds()
        if not set_clipboard_text(self.value[start:end]):
            return False
        if cut:
            self.replace_selection("")
        return True

    def _set_cursor(self, position, extend=False):
        position = max(0, min(len(self.value), int(position)))
        if not extend:
            self.selection_anchor = position
        self.cursor = position

    def _index_at_x(self, mouse_x):
        shown = self.display_value()
        if mouse_x <= self.rect.x + 9:
            return self.view_start
        if mouse_x >= self.rect.right - 9:
            return self.view_end
        relative_x = mouse_x - (self.rect.x + 9)
        visible = shown[self.view_start:self.view_end]
        previous_width = 0
        for offset in range(1, len(visible) + 1):
            width = font(15).size(visible[:offset])[0]
            if relative_x < (previous_width + width) / 2:
                return self.view_start + offset - 1
            previous_width = width
        return self.view_end

    def begin_mouse_selection(self, mouse_x, extend=False):
        position = self._index_at_x(mouse_x)
        self._set_cursor(position, extend=extend)
        self.dragging = True

    def drag_selection(self, mouse_x):
        if not self.dragging:
            return
        if mouse_x <= self.rect.x:
            position = 0
        elif mouse_x >= self.rect.right:
            position = len(self.value)
        else:
            position = self._index_at_x(mouse_x)
        self._set_cursor(position, extend=True)

    def end_mouse_selection(self):
        self.dragging = False

    def begin_key_repeat(self, event, now=None):
        if event.key not in self.REPEATABLE_KEYS:
            return
        now = time.perf_counter() if now is None else now
        self.repeat_key = event.key
        self.repeat_mod = event.mod
        self.next_repeat_at = now + self.REPEAT_DELAY_SECONDS

    def end_key_repeat(self, key=None):
        if key is None or key == self.repeat_key:
            self.repeat_key = None
            self.repeat_mod = 0
            self.next_repeat_at = None

    def repeat_due(self, fields=None, now=None):
        if self.repeat_key is None or self.next_repeat_at is None:
            return False
        now = time.perf_counter() if now is None else now
        if now < self.next_repeat_at:
            return False
        event = pygame.event.Event(
            pygame.KEYDOWN,
            key=self.repeat_key,
            mod=self.repeat_mod,
            unicode="",
        )
        self.handle_key(event, fields)
        self.next_repeat_at = now + self.REPEAT_INTERVAL_SECONDS
        return True

    def draw(self, screen, label_x, box_x, y, box_w, enabled=True):
        self.rect = pygame.Rect(box_x, y, box_w, 34)
        draw_text(screen, self.label, label_x, y + 7, 14, MUTED)
        fill = (25, 33, 47) if enabled else (34, 39, 49)
        border = ACCENT if self.active and enabled else BORDER
        pygame.draw.rect(screen, fill, self.rect, border_radius=6)
        pygame.draw.rect(screen, border, self.rect, 1, border_radius=6)
        shown = self.display_value()
        available = box_w - 18
        self.cursor = min(self.cursor, len(self.value))
        self.selection_anchor = min(self.selection_anchor, len(self.value))
        if self.active:
            if self.cursor < self.view_start:
                self.view_start = self.cursor
            while (
                self.view_start < self.cursor
                and font(15).size(shown[self.view_start:self.cursor])[0] > available
            ):
                self.view_start += 1
        else:
            self.view_start = 0
            while (
                self.view_start < len(shown)
                and font(15).size(shown[self.view_start:])[0] > available
            ):
                self.view_start += 1
        self.view_end = self.view_start
        while self.view_end < len(shown):
            candidate = shown[self.view_start:self.view_end + 1]
            if font(15).size(candidate)[0] > available:
                break
            self.view_end += 1
        visible = shown[self.view_start:self.view_end]

        if self.active and self.has_selection():
            selected_start, selected_end = self.selection_bounds()
            selected_start = max(selected_start, self.view_start)
            selected_end = min(selected_end, self.view_end)
            if selected_start < selected_end:
                left = font(15).size(shown[self.view_start:selected_start])[0]
                width = font(15).size(shown[selected_start:selected_end])[0]
                pygame.draw.rect(
                    screen, (45, 91, 120),
                    pygame.Rect(box_x + 9 + left, y + 5, max(1, width), 24),
                    border_radius=2,
                )

        draw_text(screen, visible or ("可留空" if self.secret else ""),
                  box_x + 9, y + 7, 15, TEXT if visible else MUTED)
        if self.active and self.view_start <= self.cursor <= self.view_end:
            caret_x = box_x + 9 + font(15).size(
                shown[self.view_start:self.cursor]
            )[0]
            pygame.draw.line(screen, TEXT, (caret_x, y + 6), (caret_x, y + 28), 1)

    def handle_key(self, event, fields=None):
        ctrl = bool(event.mod & pygame.KMOD_CTRL)
        shift = bool(event.mod & pygame.KMOD_SHIFT)
        if ctrl and event.key == pygame.K_a:
            self.selection_anchor = 0
            self.cursor = len(self.value)
        elif ctrl and event.key == pygame.K_c:
            self.copy_selection()
        elif ctrl and event.key == pygame.K_x:
            self.copy_selection(cut=True)
        elif event.key == pygame.K_LEFT:
            if self.has_selection() and not shift:
                self._set_cursor(self.selection_bounds()[0])
            else:
                self._set_cursor(self.cursor - 1, extend=shift)
        elif event.key == pygame.K_RIGHT:
            if self.has_selection() and not shift:
                self._set_cursor(self.selection_bounds()[1])
            else:
                self._set_cursor(self.cursor + 1, extend=shift)
        elif event.key == pygame.K_HOME:
            self._set_cursor(0, extend=shift)
        elif event.key == pygame.K_END:
            self._set_cursor(len(self.value), extend=shift)
        elif event.key == pygame.K_BACKSPACE:
            if self.has_selection():
                self.replace_selection("")
            elif self.cursor > 0:
                self.selection_anchor = self.cursor - 1
                self.replace_selection("")
        elif event.key == pygame.K_DELETE:
            if self.has_selection():
                self.replace_selection("")
            elif self.cursor < len(self.value):
                self.selection_anchor = self.cursor + 1
                self.replace_selection("")
        elif (
            event.key == pygame.K_v and event.mod & pygame.KMOD_CTRL
        ) or (
            event.key == pygame.K_INSERT and event.mod & pygame.KMOD_SHIFT
        ):
            self.paste(fields)
        elif event.unicode and event.unicode.isprintable() and not ctrl:
            self.replace_selection(event.unicode)

    def paste(self, fields=None, pasted=None):
        pasted = clipboard_text() if pasted is None else str(pasted)
        if pasted:
            normalized = pasted.replace("\r\n", "\n").replace("\r", "\n").strip()
            lines = [line.strip() for line in normalized.split("\n")]
            if fields is not None and len(fields) >= 3 and len(lines) == 3:
                # 三行配置块无论粘贴到哪个输入框，都按固定顺序覆盖填充。
                for field, value in zip(fields[:3], lines):
                    field.set_value(value)
                return True
            # 非三行内容维持原来的单输入框行为，去除换行和制表符。
            self.replace_selection(normalized.replace("\n", "").replace("\t", ""))
            return True
        return False


class DecisionCoordinator:
    def __init__(self):
        self.lock = threading.RLock()
        self.laya_agent = None
        self.laya_status = "未加载"
        self.laya_error = ""
        self.pending_identity = None
        self.decision_started_at = None
        self.processed_identity = None
        self.generation = 0
        self.phase = "等待开始"
        self.last_error = ""
        self.last_summary = ""
        self.last_decision = "尚无决策"
        self.llm_ms = 0.0
        self.laya_ms = 0.0
        self.execute_ms = 0.0
        self.llm_calls = 0
        self.laya_calls = 0
        self.discarded_decisions = 0
        self.last_discarded_identity = None

    def ensure_laya_loading(self):
        """仅在用户启用可选复核时加载 Laya。"""
        with self.lock:
            if self.laya_agent is not None or self.laya_status != "未加载":
                return
            self.laya_status = "加载中"
            self.laya_error = ""
        threading.Thread(target=self._load_laya, name="load-laya", daemon=True).start()

    def _load_laya(self):
        try:
            agent = laya_module.load_agent()
            with self.lock:
                self.laya_agent = agent
                self.laya_status = "已就绪"
                self.laya_error = ""
        except Exception as exc:
            with self.lock:
                self.laya_status = "加载失败"
                self.laya_error = str(exc)

    def snapshot(self):
        with self.lock:
            planning_elapsed_ms = (
                (time.perf_counter() - self.decision_started_at) * 1000
                if self.pending_identity is not None and self.decision_started_at is not None
                else 0.0
            )
            return {
                "laya_status": self.laya_status,
                "laya_error": self.laya_error,
                "phase": self.phase,
                "last_error": self.last_error,
                "last_summary": self.last_summary,
                "last_decision": self.last_decision,
                "llm_ms": self.llm_ms,
                "laya_ms": self.laya_ms,
                "execute_ms": self.execute_ms,
                "llm_calls": self.llm_calls,
                "laya_calls": self.laya_calls,
                "discarded_decisions": self.discarded_decisions,
                "last_discarded_identity": self.last_discarded_identity,
                "pending": self.pending_identity is not None,
                "planning_elapsed_ms": planning_elapsed_ms,
            }

    def reset_round(self, mode="llm"):
        with self.lock:
            # 使上一回合尚未返回的 LLM/Laya 后台任务失效。
            self.generation += 1
            self.pending_identity = None
            self.decision_started_at = None
            self.processed_identity = None
            self.phase = "玩家控制中" if mode == "player" else "等待首个方块"
            self.last_error = ""
            self.last_summary = ""
            self.last_decision = "尚无决策"
            self.discarded_decisions = 0
            self.last_discarded_identity = None

    def end_round(self):
        with self.lock:
            self.generation += 1
            self.pending_identity = None
            self.decision_started_at = None
            self.processed_identity = None
            self.phase = "回合已手动结束"

    def maybe_start(self, controller, config):
        with self.lock:
            if config.get("use_laya") and self.laya_agent is None:
                return
        with controller.lock:
            state = controller.snapshot()
            if not state["round_active"] or state["game_over"]:
                return
            identity = (state["round_serial"], state["piece_serial"])
            with self.lock:
                if self.pending_identity is not None:
                    if identity == self.pending_identity:
                        return
                    # 当前方块已经变化：使旧后台任务失效，并立即允许新方块规划。
                    # 旧 HTTP/Laya 调用可能仍在其守护线程中返回，但 generation
                    # 校验保证它不能复核、执行或覆盖新任务状态。
                    self.last_discarded_identity = self.pending_identity
                    self.discarded_decisions += 1
                    self.generation += 1
                    self.pending_identity = None
                    self.decision_started_at = None
                if identity == self.processed_identity:
                    return
            if config.get("mode") == "heuristic":
                planning_input = controller.handle({"op": "legal_landings"})
                if not planning_input.get("ok"):
                    return
            else:
                rules_result = controller.handle({"op": "get_rules"})
                if not rules_result.get("ok"):
                    return
                planning_input = {"rules": rules_result["rules"], "state": state}
            with self.lock:
                self.pending_identity = identity
                self.decision_started_at = time.perf_counter()
                self.phase = (
                    "Expectimax 启发式规划中"
                    if config.get("mode") == "heuristic" else "第三方 LLM 规划中"
                )
                self.last_error = ""
                generation = self.generation
        threading.Thread(
            target=self._decide,
            args=(controller, config.copy(), planning_input, identity, generation),
            name=f"decision-{identity[0]}-{identity[1]}",
            daemon=True,
        ).start()

    def _decide(self, controller, config, planning_input, identity, generation):
        try:
            planner_started = time.perf_counter()
            if config.get("mode") == "heuristic":
                candidates, summary = heuristic_module.plan(
                    planning_input, MAX_LLM_CANDIDATES
                )
            else:
                client = llm_module.ThirdPartyLLM(
                    config["base_url"], config["api_key"], config["model"], DEFAULT_TIMEOUT
                )
                candidates, summary = llm_module.plan_strategies(
                    client,
                    planning_input["rules"],
                    planning_input["state"],
                    MAX_LLM_CANDIDATES,
                )
                with self.lock:
                    if generation != self.generation:
                        return
                validated = []
                validated_outcomes = set()
                invalid = []
                for candidate in candidates:
                    check = controller.handle({
                        "op": "validate_plan",
                        "rotation_cw": candidate["rotation_cw"],
                        "target_x": candidate["target_x"],
                        "expected_round_serial": identity[0],
                        "expected_piece_serial": identity[1],
                    })
                    if check.get("ok"):
                        outcome_key = (
                            check["validation"]["final_y"],
                            check["validation"]["board_after"],
                        )
                        if outcome_key in validated_outcomes:
                            invalid.append(f"{candidate['id']}(重复结果)")
                            continue
                        validated_outcomes.add(outcome_key)
                        candidate["validation"] = check["validation"]
                        validated.append(candidate)
                    else:
                        invalid.append(candidate["id"])
                        if check.get("error") in ("piece_changed", "round_changed"):
                            raise RuntimeError("LLM 策略返回前当前方块已经落地")
                minimum_valid = 2 if config.get("use_laya") else 1
                if len(validated) < minimum_valid:
                    raise RuntimeError(
                        f"LLM 仅生成 {len(validated)} 个合法策略；无效策略={invalid}"
                    )
                candidates = validated
                if invalid:
                    summary = f"{summary}；规则层剔除无效策略 {', '.join(invalid)}"
            planner_ms = (time.perf_counter() - planner_started) * 1000
            with self.lock:
                if generation != self.generation:
                    return
                self.phase = (
                    "Laya 最终复核中"
                    if config.get("use_laya")
                    else "规划器自主决策完成"
                )
                self.llm_ms = planner_ms
                self.llm_calls += 1
                self.last_summary = summary

            if config.get("use_laya"):
                laya_started = time.perf_counter()
                if config.get("mode") == "heuristic":
                    choice, selected, confidence = laya_module.review_heuristic_candidates(
                        self.laya_agent, planning_input, candidates
                    )
                else:
                    choice, selected, confidence = laya_module.review_llm_strategies(
                        self.laya_agent,
                        planning_input["rules"],
                        planning_input["state"],
                        candidates,
                    )
                laya_ms = (time.perf_counter() - laya_started) * 1000
                decision_text = (
                    f"Laya {choice} → {selected['id']} · conf={confidence:.3f}"
                )
            else:
                laya_ms = 0.0
                if config.get("mode") == "heuristic":
                    selected = candidates[0]
                    decision_text = f"Expectimax 自主选择 → {selected['id']}"
                else:
                    selected = next(
                        (item for item in candidates if item.get("llm_final_choice")),
                        None,
                    )
                    if selected is None:
                        raise RuntimeError("LLM 的最终选择未通过游戏规则校验")
                    decision_text = f"LLM 自主选择 → {selected['id']}"
            with self.lock:
                if generation != self.generation:
                    return
            execute_started = time.perf_counter()
            if config.get("mode") == "heuristic":
                request = {
                    "op": "fast_place",
                    "landing_id": selected["id"],
                    "expected_round_serial": identity[0],
                    "expected_piece_serial": identity[1],
                }
            else:
                request = {
                    "op": "execute_plan",
                    "rotation_cw": selected["rotation_cw"],
                    "target_x": selected["target_x"],
                    "expected_round_serial": identity[0],
                    "expected_piece_serial": identity[1],
                }
            result = controller.handle(request)
            execute_ms = (time.perf_counter() - execute_started) * 1000
            with self.lock:
                if generation != self.generation:
                    return
                self.laya_ms = laya_ms
                self.execute_ms = execute_ms
                if config.get("use_laya"):
                    self.laya_calls += 1
                self.last_decision = decision_text
                if result.get("ok"):
                    self.phase = "已交互执行并落底"
                else:
                    error = result.get("error", "执行失败")
                    self.phase = "决策已过期" if error in (
                        "piece_changed", "round_changed", "landing_no_longer_reachable",
                        "rotation_blocked", "horizontal_path_blocked",
                        "target_x_not_reached",
                    ) else "执行失败"
                    self.last_error = error
        except Exception as exc:
            with self.lock:
                if generation == self.generation:
                    self.phase = "规划失败；等待该方块自然落地"
                    self.last_error = str(exc)
        finally:
            with self.lock:
                if generation == self.generation:
                    self.processed_identity = identity
                    self.pending_identity = None
                    self.decision_started_at = None


LINE_CLEAR_ANIMATION_SECONDS = 0.26
LINE_CLEAR_PARTICLE_SECONDS = 0.48
PARTICLES_PER_CLEARED_ROW = 24
_clear_particles = []
_particle_game_identity = None
_particle_event_serial = 0
_particle_last_update = time.monotonic()


def draw_clear_particles(screen, game):
    """绘制少量短命方形碎屑；纯视觉效果，不参与或阻塞游戏逻辑。"""
    global _particle_game_identity, _particle_event_serial, _particle_last_update
    game_identity = id(game)
    now = time.monotonic()
    if game_identity != _particle_game_identity:
        _particle_game_identity = game_identity
        _particle_event_serial = 0
        _clear_particles.clear()
        _particle_last_update = now

    if game.clear_event_serial != _particle_event_serial:
        _particle_event_serial = game.clear_event_serial
        rng = random.Random((game.clear_event_serial << 8) ^ game.piece_serial)
        for row in game.last_clear_rows:
            colors = [color for color in game.last_clear_grid[row] if color]
            for _ in range(PARTICLES_PER_CLEARED_ROW):
                color = rng.choice(colors) if colors else (180, 225, 255)
                _clear_particles.append({
                    "x": rng.uniform(4, BOARD_W - 4),
                    "y": row * CELL_SIZE + CELL_SIZE * 0.5,
                    "vx": rng.uniform(-72, 72),
                    "vy": rng.uniform(-115, -35),
                    "size": rng.randint(2, 5),
                    "age": 0.0,
                    "life": rng.uniform(0.34, LINE_CLEAR_PARTICLE_SECONDS),
                    "color": color,
                })

    dt = min(max(now - _particle_last_update, 0.0), 0.05)
    _particle_last_update = now
    if not _clear_particles:
        return

    particle_layer = pygame.Surface((BOARD_W, SCREEN_H), pygame.SRCALPHA)
    alive = []
    for particle in _clear_particles:
        particle["age"] += dt
        if particle["age"] >= particle["life"]:
            continue
        particle["x"] += particle["vx"] * dt
        particle["y"] += particle["vy"] * dt
        particle["vy"] += 260 * dt
        fade = 1.0 - particle["age"] / particle["life"]
        alpha = int(210 * fade)
        size = particle["size"]
        pygame.draw.rect(
            particle_layer,
            (*particle["color"][:3], alpha),
            (int(particle["x"]), int(particle["y"]), size, size),
            border_radius=1,
        )
        alive.append(particle)
    _clear_particles[:] = alive
    screen.blit(particle_layer, (0, 0))


def draw_keycap(screen, label, rect, accent=False):
    fill = (48, 113, 150) if accent else (40, 50, 67)
    pygame.draw.rect(screen, fill, rect, border_radius=7)
    pygame.draw.rect(screen, ACCENT if accent else BORDER, rect, 1, border_radius=7)
    rendered = font(15, True).render(label, True, TEXT)
    screen.blit(rendered, rendered.get_rect(center=rect.center))


def draw_player_guide(screen):
    """玩家自主模式开始前显示的新手说明，开局后自动隐藏。"""
    veil = pygame.Surface((BOARD_W, SCREEN_H), pygame.SRCALPHA)
    veil.fill((8, 12, 19, 180))
    screen.blit(veil, (0, 0))

    guide = pygame.Rect(22, 66, BOARD_W - 44, 630)
    pygame.draw.rect(screen, (27, 35, 49), guide, border_radius=14)
    pygame.draw.rect(screen, BORDER, guide, 1, border_radius=14)
    draw_text(screen, "新手引导", guide.x + 22, guide.y + 20, 24, ACCENT, True)
    draw_text(screen, "填满完整的一横行即可消除并得分", guide.x + 22, guide.y + 58,
              14, TEXT)

    draw_text(screen, "移动与旋转", guide.x + 22, guide.y + 102, 15, MUTED, True)
    key_y = guide.y + 132
    draw_keycap(screen, "←", pygame.Rect(guide.x + 22, key_y, 52, 42))
    draw_keycap(screen, "→", pygame.Rect(guide.x + 82, key_y, 52, 42))
    draw_text(screen, "左右移动（支持长按）", guide.x + 150, key_y + 10, 14, TEXT)
    draw_keycap(screen, "↑", pygame.Rect(guide.x + 22, key_y + 56, 112, 42), True)
    draw_text(screen, "旋转（支持长按）", guide.x + 150, key_y + 66, 14, TEXT)
    draw_keycap(screen, "↓", pygame.Rect(guide.x + 22, key_y + 112, 112, 42))
    draw_text(screen, "约 11 倍加速下落（支持长按）", guide.x + 150, key_y + 122, 13, TEXT)
    draw_keycap(screen, "SPACE", pygame.Rect(guide.x + 22, key_y + 168, 112, 42), True)
    draw_text(screen, "直接落底", guide.x + 150, key_y + 178, 14, TEXT)

    draw_text(screen, "三个小技巧", guide.x + 22, guide.y + 372, 15, MUTED, True)
    draw_text(screen, "1  尽量保持顶部平整", guide.x + 22, guide.y + 404, 14, TEXT)
    draw_text(screen, "2  不要在方块下方留下空洞", guide.x + 22, guide.y + 434, 14, TEXT)
    draw_text(screen, "3  下落会逐渐加速，提前规划位置", guide.x + 22, guide.y + 464, 14, TEXT)
    draw_text(screen, "准备好后，点击右侧「开始新游戏」", guide.x + 22, guide.y + 570,
              14, SUCCESS, True)


def draw_board(screen, controller):
    game = controller.game
    pygame.draw.rect(screen, BOARD_BG, (0, 0, BOARD_W, SCREEN_H))
    animation_age = time.monotonic() - game.last_clear_started_at
    animating_clear = (
        game.last_clear_grid is not None
        and 0 <= animation_age < LINE_CLEAR_ANIMATION_SECONDS
    )
    displayed_grid = game.last_clear_grid if animating_clear else game.grid
    for y in range(game.H):
        for x in range(game.W):
            rect = pygame.Rect(
                x * CELL_SIZE,
                y * CELL_SIZE,
                BLOCK_SIZE - MARGIN,
                BLOCK_SIZE - MARGIN,
            )
            if displayed_grid[y][x]:
                pygame.draw.rect(screen, displayed_grid[y][x], rect, border_radius=3)
            pygame.draw.rect(screen, game_api.COLOR_GRID, rect, 1, border_radius=3)
    if animating_clear:
        # 简短的两段式柔和闪烁；随后直接显示已经完成坍塌的新棋盘。
        progress = animation_age / LINE_CLEAR_ANIMATION_SECONDS
        strength = 0.72 if progress < 0.46 else 0.34
        overlay = pygame.Surface((BOARD_W, SCREEN_H), pygame.SRCALPHA)
        alpha = int(255 * strength)
        for y in game.last_clear_rows:
            row_rect = pygame.Rect(
                0,
                y * CELL_SIZE,
                BOARD_W,
                BLOCK_SIZE - MARGIN,
            )
            pygame.draw.rect(overlay, (224, 244, 255, alpha), row_rect, border_radius=3)
        screen.blit(overlay, (0, 0))
    elif game.piece and game.piece_color:
        for dy, row in enumerate(game.piece):
            for dx, occupied in enumerate(row):
                x, y = game.px + dx, game.py + dy
                if occupied and 0 <= x < game.W and 0 <= y < game.H:
                    rect = pygame.Rect(
                        x * CELL_SIZE,
                        y * CELL_SIZE,
                        BLOCK_SIZE - MARGIN,
                        BLOCK_SIZE - MARGIN,
                    )
                    pygame.draw.rect(screen, game.piece_color, rect, border_radius=3)
                    pygame.draw.rect(screen, game_api.COLOR_GRID, rect, 1, border_radius=3)
    draw_clear_particles(screen, game)


def draw_dashboard(
    screen, controller, coordinator, fields, start_rect, end_rect,
    llm_mode_rect, heuristic_mode_rect, player_mode_rect, laya_toggle_rect,
    planner_mode, use_laya, port,
):
    screen.fill(BG)
    with controller.lock:
        state = controller.snapshot()
        game = controller.game
        draw_board(screen, controller)
    if planner_mode == "player" and not state["round_active"]:
        draw_player_guide(screen)
    info = coordinator.snapshot()
    start_enabled, _ = start_readiness(
        planner_mode, use_laya, fields, info, state
    )
    screen.fill(PANEL_BG, (PANEL_X, 0, PANEL_W + 16, SCREEN_H))
    header = {
        "llm": "LLM + Laya" if use_laya else "LLM",
        "heuristic": "启发式程序 + Laya" if use_laya else "启发式程序",
        "player": "玩家自主",
    }[planner_mode]
    draw_text(screen, header, PANEL_X + 18, 15, 24, ACCENT, True)
    lifecycle_text = {
        "waiting": "等待配置", "running": "自动运行", "ended": "回合结束",
        "game_over": "游戏结束", "terminated": "服务关闭",
    }.get(state["lifecycle"], state["lifecycle"])
    if not state["round_active"] and state["lifecycle"] != "terminated":
        lifecycle_text = "Ready" if start_enabled else "未就绪"
    draw_text(screen, lifecycle_text, PANEL_X + PANEL_W - 118, 20, 15,
              SUCCESS if state["round_active"] or start_enabled else WARNING, True)

    config_rect = pygame.Rect(PANEL_X + 10, 58, PANEL_W - 20, 210)
    card(screen, config_rect, "游戏模式、第三方 LLM 与可选复核")
    editable = not state["round_active"]
    llm_mode_rect.update(config_rect.x + 104, 91, 132, 34)
    heuristic_mode_rect.update(config_rect.x + 244, 91, 190, 34)
    player_mode_rect.update(config_rect.x + 442, 91, 90, 34)
    draw_text(screen, "模式", config_rect.x + 16, 99, 14, MUTED)
    pygame.draw.rect(
        screen, BUTTON if planner_mode == "llm" else (45, 53, 67),
        llm_mode_rect, border_radius=6,
    )
    pygame.draw.rect(
        screen, BUTTON if planner_mode == "heuristic" else (45, 53, 67),
        heuristic_mode_rect, border_radius=6,
    )
    pygame.draw.rect(
        screen, BUTTON if planner_mode == "player" else (45, 53, 67),
        player_mode_rect, border_radius=6,
    )
    draw_text(screen, "LLM", llm_mode_rect.x + 46, llm_mode_rect.y + 7, 14, TEXT, True)
    draw_text(screen, "启发式程序", heuristic_mode_rect.x + 44,
              heuristic_mode_rect.y + 7, 14, TEXT, True)
    draw_text(screen, "玩家自主", player_mode_rect.x + 13, player_mode_rect.y + 7, 14, TEXT, True)
    if planner_mode == "llm":
        label_x, box_x, box_w = config_rect.x + 16, config_rect.x + 104, config_rect.w - 120
        for field, y in zip(fields, (132, 172, 212)):
            field.draw(screen, label_x, box_x, y, box_w, editable)
        laya_toggle_rect.update(config_rect.x + 104, 248, 18, 18)
        checkbox(screen, laya_toggle_rect, use_laya, "使用 Laya 复核（可选）", editable)
    else:
        # 启发式和玩家模式都不需要第三方接口配置。
        for field in fields:
            field.rect.update(0, 0, 0, 0)
            field.active = False
            field.end_mouse_selection()
            field.end_key_repeat()
        if planner_mode == "heuristic":
            draw_text(screen, "无需第三方 API 配置", config_rect.x + 104, 140, 18, SUCCESS, True)
            draw_text(screen, "当前块 + 已知下一块 + 七袋概率下的下下块",
                      config_rect.x + 104, 180, 15, TEXT)
            decision_hint = (
                "生成 4 个候选，由 Laya 最终复核"
                if use_laya else "Expectimax 评分第一名作为最终决策"
            )
            draw_text(screen, decision_hint,
                      config_rect.x + 104, 218, 14, MUTED)
            laya_toggle_rect.update(config_rect.x + 104, 248, 18, 18)
            checkbox(screen, laya_toggle_rect, use_laya, "使用 Laya 复核（可选）", editable)
        else:
            laya_toggle_rect.update(0, 0, 0, 0)
            draw_text(screen, "玩家自主控制，无需 LLM 或 Laya", config_rect.x + 104, 140,
                      18, SUCCESS, True)
            draw_text(screen, "← / → 长按移动     ↑ 长按旋转",
                      config_rect.x + 104, 180, 15, TEXT)
            draw_text(screen, "↓ 长按约 11 倍加速下落     Space 直接落底",
                      config_rect.x + 104, 218, 14, MUTED)

    button_gap = 10
    button_w = (PANEL_W - 20 - button_gap) // 2
    start_rect.update(PANEL_X + 10, 280, button_w, 44)
    end_rect.update(start_rect.right + button_gap, 280, button_w, 44)
    pygame.draw.rect(
        screen, BUTTON if start_enabled else (50, 57, 68), start_rect, border_radius=7
    )
    pygame.draw.rect(screen, (126, 68, 68) if state["round_active"] else (50, 57, 68),
                     end_rect, border_radius=7)
    draw_text(
        screen, "开始新游戏", start_rect.centerx - 48, start_rect.y + 10, 16,
        TEXT if start_enabled else MUTED, True,
    )
    draw_text(screen, "结束当前回合", end_rect.centerx - 56, end_rect.y + 10, 16, TEXT, True)

    stats_rect = pygame.Rect(PANEL_X + 10, 336, PANEL_W - 20, 108)
    card(screen, stats_rect, "游戏状态")
    draw_text(screen, f"分数 {game.score}", stats_rect.x + 16, stats_rect.y + 42, 19, TEXT, True)
    draw_text(screen, f"消行 {game.lines_cleared}", stats_rect.x + 152, stats_rect.y + 42, 18, SUCCESS, True)
    draw_text(screen, f"落块 {game.pieces_locked}", stats_rect.x + 284, stats_rect.y + 42, 18, TEXT, True)
    draw_text(screen, f"耗时 {duration(state['elapsed_seconds'])}", stats_rect.x + 416,
              stats_rect.y + 44, 14, MUTED)
    current, next_name = game.piece_name or "--", game.next_piece_name or "--"
    draw_text(screen, f"Round #{state['round_serial']}  Piece #{game.piece_serial}  当前 {current}  下一块 {next_name}",
              stats_rect.x + 16, stats_rect.y + 78, 14, MUTED)

    decision_rect = pygame.Rect(PANEL_X + 10, 456, PANEL_W - 20, 176)
    if planner_mode == "player":
        card(screen, decision_rect, "玩家操作")
        draw_text(screen, "当前由玩家自主控制", decision_rect.x + 16, decision_rect.y + 42,
                  17, SUCCESS, True)
        draw_text(screen, "左右键长按移动 · 上键长按旋转 · 下键长按 11 倍加速 · 空格直接落底",
                  decision_rect.x + 16, decision_rect.y + 78, 15, TEXT, True)
        draw_text(screen, "方块保持自动下落，速度会随游戏时间逐步提升",
                  decision_rect.x + 16, decision_rect.y + 112, 14, MUTED)
        draw_text(screen, "消行、高亮和粒子效果均保持启用",
                  decision_rect.x + 16, decision_rect.y + 142, 13, MUTED)
    else:
        card(screen, decision_rect, "协作决策" if use_laya else "规划器自主决策")
        if use_laya:
            laya_color = SUCCESS if info["laya_status"] == "已就绪" else WARNING
            laya_text = f"Laya  {info['laya_status']}"
        else:
            laya_color = MUTED
            laya_text = "Laya  未启用"
        draw_text(screen, laya_text, decision_rect.x + 16, decision_rect.y + 42,
                  15, laya_color, True)
        draw_text(screen, shorten(info["phase"], 38), decision_rect.x + 170, decision_rect.y + 42,
                  15, ACCENT, True)
        planner_label = "LLM" if planner_mode == "llm" else "Expectimax"
        planner_ms = (
            info["planning_elapsed_ms"] if info["pending"] else info["llm_ms"]
        )
        timing = f"{planner_label} {planner_ms:.0f} ms"
        if use_laya:
            timing += f" · Laya {info['laya_ms']:.0f} ms"
        timing += f" · 执行 {info['execute_ms']:.1f} ms"
        if info["discarded_decisions"]:
            timing += f" · 丢弃旧请求 {info['discarded_decisions']}"
        draw_text(screen, timing,
                  decision_rect.x + 16, decision_rect.y + 76, 14, MUTED)
        draw_text(screen, shorten(info["last_decision"], 62), decision_rect.x + 16,
                  decision_rect.y + 108, 15, TEXT, True)
        if info["pending"]:
            waiting_message = (
                f"正在等待 {planner_label} 响应 · "
                f"{info['planning_elapsed_ms'] / 1000:.1f}s"
            )
        else:
            waiting_message = (
                "等待第三方 LLM 规划"
                if planner_mode == "llm" else "等待 Expectimax 规划"
            )
        laya_error = info["laya_error"] if use_laya else ""
        message = info["last_error"] or laya_error or info["last_summary"] or waiting_message
        draw_text(screen, shorten(message, 70), decision_rect.x + 16, decision_rect.y + 140,
                  13, ERROR if info["last_error"] or laya_error else MUTED)

    service_rect = pygame.Rect(PANEL_X + 10, 644, PANEL_W - 20, 106)
    card(screen, service_rect, "游戏服务")
    draw_text(screen, f"127.0.0.1:{port}", service_rect.x + 16, service_rect.y + 42, 15, SUCCESS, True)
    speedup = state["seconds_until_speedup"]
    if not state["round_active"]:
        speed_text = f"速度 Lv{state['speed_level']} · {state['fall_ms']} ms · 等待开始"
    else:
        speed_text = (
            f"速度 Lv{state['speed_level']} · {state['fall_ms']} ms · "
            + (f"{speedup:.0f}s后加速" if speedup is not None else "最高速")
        )
    draw_text(screen, speed_text, service_rect.x + 205, service_rect.y + 42, 14, WARNING)
    draw_text(screen, "get_rules · get_state · validate_plan · execute_plan · start/end",
              service_rect.x + 16, service_rect.y + 76, 12, MUTED)


def config_from_fields(fields, planner_mode, use_laya=False):
    values = {field.name: field.value.strip() for field in fields}
    values["mode"] = planner_mode
    values["use_laya"] = bool(use_laya)
    if planner_mode == "llm":
        if not values["base_url"]:
            raise ValueError("请填写 Base URL")
        if not values["model"]:
            raise ValueError("请填写模型名")
    return values


def start_readiness(planner_mode, use_laya, fields, coordinator_info, state):
    """返回当前模式是否满足开始条件，以及未就绪原因。"""
    if state.get("round_active"):
        return False, "当前回合正在运行"
    if state.get("lifecycle") == "terminated" or not state.get("service_running", True):
        return False, "游戏服务已停止"
    if planner_mode == "player":
        return True, "已就绪"
    if use_laya and coordinator_info.get("laya_status") != "已就绪":
        status = coordinator_info.get("laya_status", "未就绪")
        return False, f"本地 Laya {status}"
    try:
        config_from_fields(fields, planner_mode, use_laya)
    except ValueError as exc:
        return False, str(exc)
    return True, "已就绪"


def keyboard_action(planner_mode, key):
    if planner_mode == "player":
        return player_module.action_for_key(key)
    return {
        pygame.K_LEFT: "left",
        pygame.K_RIGHT: "right",
        pygame.K_UP: "rotate_cw",
        pygame.K_DOWN: "soft_drop",
        pygame.K_SPACE: "hard_drop",
    }.get(key)


def run(host, port, fall_ms):
    pygame.init()
    try:
        pygame.scrap.init()
    except pygame.error:
        pass
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Tetris【玩家 / LLM / Expectimax / 可选 Laya】")
    clock = pygame.time.Clock()
    controller = game_api.GameController(fall_ms)
    server = game_api.ControlServer((host, port), controller)
    threading.Thread(target=server.serve_forever, name="tetris-control", daemon=True).start()
    coordinator = DecisionCoordinator()
    fields = [
        InputField("base_url", "Base URL", DEFAULT_BASE_URL),
        InputField("model", "模型", DEFAULT_MODEL),
        InputField("api_key", "API Key", DEFAULT_API_KEY, secret=True),
    ]
    start_rect, end_rect = pygame.Rect(0, 0, 0, 0), pygame.Rect(0, 0, 0, 0)
    llm_mode_rect = pygame.Rect(0, 0, 0, 0)
    heuristic_mode_rect = pygame.Rect(0, 0, 0, 0)
    player_mode_rect = pygame.Rect(0, 0, 0, 0)
    laya_toggle_rect = pygame.Rect(0, 0, 0, 0)
    active_index = None
    player_keys = player_module.KeyRepeater()
    planner_mode = (
        DEFAULT_PLANNER_MODE
        if DEFAULT_PLANNER_MODE in ("llm", "heuristic", "player")
        else "llm"
    )
    use_laya = DEFAULT_USE_LAYA
    config = {
        "mode": planner_mode,
        "use_laya": use_laya,
        "base_url": DEFAULT_BASE_URL,
        "model": DEFAULT_MODEL,
        "api_key": DEFAULT_API_KEY,
    }

    try:
        while not controller.stop_event.is_set():
            if planner_mode != "player" and use_laya:
                coordinator.ensure_laya_loading()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    controller.termination_reason = "window_closed"
                    controller.stop_event.set()
                    continue
                if event.type == pygame.WINDOWFOCUSLOST:
                    # 防止切出窗口时漏掉 KEYUP，造成方向键持续移动。
                    player_keys.clear()
                    for field in fields:
                        field.end_mouse_selection()
                        field.end_key_repeat()
                    continue
                if event.type == pygame.MOUSEMOTION:
                    if (
                        active_index is not None
                        and planner_mode == "llm"
                        and not controller.snapshot()["round_active"]
                    ):
                        fields[active_index].drag_selection(event.pos[0])
                    continue
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    for field in fields:
                        field.end_mouse_selection()
                    continue
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    state = controller.snapshot()
                    if not state["round_active"] and planner_mode == "llm":
                        for index, field in enumerate(fields):
                            if field.rect.collidepoint(event.pos):
                                for other in fields:
                                    other.active = False
                                    other.end_key_repeat()
                                field.active = True
                                active_index = index
                                field.begin_mouse_selection(event.pos[0])
                                field.end_mouse_selection()
                                field.paste(fields)
                                break
                    continue
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    state = controller.snapshot()
                    if not state["round_active"]:
                        if llm_mode_rect.collidepoint(event.pos):
                            planner_mode = "llm"
                        elif heuristic_mode_rect.collidepoint(event.pos):
                            planner_mode = "heuristic"
                        elif player_mode_rect.collidepoint(event.pos):
                            planner_mode = "player"
                        if (
                            planner_mode != "player"
                            and laya_toggle_rect.collidepoint(event.pos)
                        ):
                            use_laya = not use_laya
                            if use_laya:
                                coordinator.ensure_laya_loading()
                        if planner_mode != "player":
                            player_keys.clear()
                        active_index = None
                        for index, field in enumerate(fields):
                            field.active = (
                                planner_mode == "llm" and field.rect.collidepoint(event.pos)
                            )
                            if field.active:
                                active_index = index
                                field.begin_mouse_selection(
                                    event.pos[0],
                                    extend=bool(pygame.key.get_mods() & pygame.KMOD_SHIFT),
                                )
                            else:
                                field.end_key_repeat()
                        if start_rect.collidepoint(event.pos):
                            info = coordinator.snapshot()
                            ready, _ = start_readiness(
                                planner_mode, use_laya, fields, info, state
                            )
                            if ready:
                                try:
                                    config = config_from_fields(
                                        fields, planner_mode, use_laya
                                    )
                                    controller.handle({"op": "start_game"})
                                    coordinator.reset_round(planner_mode)
                                    player_keys.clear()
                                    for field in fields:
                                        field.active = False
                                        field.end_key_repeat()
                                    active_index = None
                                except Exception as exc:
                                    with coordinator.lock:
                                        coordinator.last_error = str(exc)
                                        coordinator.phase = "无法开始"
                    if state["round_active"] and end_rect.collidepoint(event.pos):
                        controller.handle({"op": "end_game"})
                        coordinator.end_round()
                        player_keys.clear()
                    continue
                if event.type == pygame.KEYUP:
                    if active_index is not None:
                        fields[active_index].end_key_repeat(event.key)
                    if planner_mode == "player":
                        player_keys.release(event.key)
                    continue
                if event.type == pygame.KEYDOWN:
                    state = controller.snapshot()
                    if active_index is not None and not state["round_active"] and planner_mode == "llm":
                        if event.key == pygame.K_TAB:
                            fields[active_index].active = False
                            active_index = (active_index + 1) % len(fields)
                            fields[active_index].active = True
                        elif event.key != pygame.K_RETURN:
                            fields[active_index].handle_key(event, fields)
                            fields[active_index].begin_key_repeat(event)
                    elif state["round_active"]:
                        action = keyboard_action(planner_mode, event.key)
                        should_apply = True
                        if planner_mode == "player":
                            should_apply = player_keys.press(
                                event.key, fall_ms=state["fall_ms"]
                            )
                        if action and should_apply:
                            controller.handle({"op": "action", "action": action})

            state = controller.snapshot()
            if (
                active_index is not None
                and planner_mode == "llm"
                and not state["round_active"]
            ):
                fields[active_index].repeat_due(fields)
            if planner_mode == "player" and state["round_active"]:
                for action in player_keys.due_actions(state["fall_ms"]):
                    controller.handle({"op": "action", "action": action})
            else:
                player_keys.clear()
            controller.auto_tick()
            if planner_mode != "player":
                coordinator.maybe_start(controller, config)
            draw_dashboard(
                screen, controller, coordinator, fields, start_rect, end_rect,
                llm_mode_rect, heuristic_mode_rect, player_mode_rect,
                laya_toggle_rect, planner_mode, use_laya, port,
            )
            pygame.display.flip()
            clock.tick(60)
    finally:
        server.shutdown()
        server.server_close()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Tetris with LLM/Expectimax planning and optional local Laya review"
    )
    parser.add_argument("--host", default=game_api.DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=game_api.DEFAULT_PORT)
    parser.add_argument("--fall-ms", type=int, default=game_api.DEFAULT_FALL_MS)
    args = parser.parse_args()
    run(args.host, args.port, args.fall_ms)


if __name__ == "__main__":
    main()
