"""玩家自主模式的键位映射、长按和加速下落逻辑。"""

import time

import pygame


SOFT_DROP_MULTIPLIER = 11.0
SOFT_DROP_MIN_INTERVAL = 0.016


def action_for_key(key):
    return {
        pygame.K_LEFT: "left",
        pygame.K_RIGHT: "right",
        pygame.K_UP: "rotate_cw",
        pygame.K_DOWN: "soft_drop",
        pygame.K_SPACE: "hard_drop",
    }.get(key)


class KeyRepeater:
    """提供可控长按节奏，并屏蔽操作系统的重复 KEYDOWN。"""

    REPEATABLE_KEYS = {pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN}
    PLAYER_KEYS = REPEATABLE_KEYS | {pygame.K_SPACE}

    def __init__(self):
        self.held = set()
        self.next_repeat_at = {}

    def clear(self):
        self.held.clear()
        self.next_repeat_at.clear()

    def press(self, key, now=None, fall_ms=600):
        if key not in self.PLAYER_KEYS or key in self.held:
            return False
        now = time.perf_counter() if now is None else now
        self.held.add(key)
        if key in self.REPEATABLE_KEYS:
            delay = 0.22 if key == pygame.K_UP else 0.18
            if key == pygame.K_DOWN:
                delay = max(
                    SOFT_DROP_MIN_INTERVAL,
                    float(fall_ms) / (1000.0 * SOFT_DROP_MULTIPLIER),
                )
            self.next_repeat_at[key] = now + delay
        return True

    def release(self, key):
        self.held.discard(key)
        self.next_repeat_at.pop(key, None)

    def due_actions(self, fall_ms, now=None):
        now = time.perf_counter() if now is None else now
        actions = []
        for key in tuple(self.REPEATABLE_KEYS & self.held):
            if now < self.next_repeat_at.get(key, now):
                continue
            actions.append(action_for_key(key))
            if key == pygame.K_DOWN:
                interval = max(
                    SOFT_DROP_MIN_INTERVAL,
                    float(fall_ms) / (1000.0 * SOFT_DROP_MULTIPLIER),
                )
            elif key == pygame.K_UP:
                interval = 0.17
            else:
                interval = 0.06
            self.next_repeat_at[key] = now + interval
        return actions
