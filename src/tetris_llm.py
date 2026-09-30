"""第三方 LLM 规划模块。

本模块只接收游戏规则与原始状态，不依赖合法落点枚举、启发式评分或
Expectimax。LLM 必须自行提出多个旋转/位置策略并指定自己的最终选择，
之后由游戏规则层校验；启用 Laya 时，该选择可再交由 Laya 复核。
"""

import json
import re
import urllib.error
import urllib.request


def parse_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError(f"LLM 未返回 JSON object: {text[:250]}")
        return json.loads(text[start : end + 1])


class ThirdPartyLLM:
    """最小 OpenAI-compatible Chat Completions 客户端。"""

    def __init__(self, base_url, api_key, model, timeout=45):
        self.base_url = base_url.strip().rstrip("/")
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout = float(timeout)

    @property
    def endpoint(self):
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def chat_json(self, system_prompt, user_prompt):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 700,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {body[:350]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM 连接失败: {exc.reason}") from exc
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"LLM 响应不兼容 Chat Completions: {str(raw)[:350]}") from exc
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        return parse_json_object(str(content))


def plan_strategies(client, rules, state, count=4):
    """让 LLM 仅根据原始规则和状态自行规划多个策略。"""
    current = state["current_piece"]
    next_piece = state["next_piece"]
    system_prompt = (
        "You are the sole strategic planner for a real-time Tetris game. No heuristic "
        "planner or precomputed landing list is available. Derive placements yourself from "
        "the supplied rules, board, current piece and next piece. Produce distinct strategies "
        "that balance survival, holes, height, surface shape and line clears. rotation_cw is "
        "an integer from 0 to 3. target_x is the zero-based board column of the rotated "
        "shape's left edge. Rank the strategies strongest-first and make your own binding "
        "final choice using final_choice (a 1-based strategy index). The chosen strategy "
        "will be hard-dropped unless optional Laya review is enabled. Return JSON only."
    )
    example = {
        "strategies": [
            {"rotation_cw": 0, "target_x": 3, "reason": "short factual rationale"}
        ],
        "final_choice": 1,
        "summary": "brief overall plan",
    }
    user_prompt = (
        f"Rules:\n{json.dumps(rules, ensure_ascii=False)}\n\n"
        f"Round identity: round={state['round_serial']}, piece={state['piece_serial']}\n"
        f"Current piece: name={current['name']}, shape={current['shape']}, "
        f"x={current['x']}, y={current['y']}\n"
        f"Next piece: name={next_piece['name']}, shape={next_piece['shape']}\n"
        f"Milliseconds until automatic fall: {state['ms_until_auto_fall']}\n"
        f"Board without active piece, top to bottom:\n{state['board']}\n\n"
        f"Return exactly {count} distinct strategies and your binding final_choice in this schema:\n"
        f"{json.dumps(example, ensure_ascii=False)}"
    )
    result = client.chat_json(system_prompt, user_prompt)
    raw_strategies = result.get("strategies")
    if not isinstance(raw_strategies, list) or len(raw_strategies) != count:
        raise RuntimeError(f"LLM 必须返回 {count} 个策略，实际为 {raw_strategies!r}")

    try:
        final_choice = int(result["final_choice"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("LLM 必须用 final_choice 返回自己的最终策略序号") from exc
    if final_choice < 1 or final_choice > count:
        raise RuntimeError(f"LLM final_choice 必须处于 1..{count}，实际为 {final_choice}")

    strategies = []
    seen = set()
    for index, raw in enumerate(raw_strategies, start=1):
        if not isinstance(raw, dict):
            raise RuntimeError(f"LLM 策略 #{index} 不是 object")
        try:
            rotation = int(raw["rotation_cw"])
            target_x = int(raw["target_x"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"LLM 策略 #{index} 缺少合法 rotation_cw/target_x") from exc
        if rotation < 0 or rotation > 3:
            raise RuntimeError(f"LLM 策略 #{index} rotation_cw 必须处于 0..3")
        key = (rotation, target_x)
        if key in seen:
            raise RuntimeError(f"LLM 返回重复策略: rotation={rotation}, x={target_x}")
        seen.add(key)
        strategies.append({
            "id": f"S{index}",
            "rotation_cw": rotation,
            "target_x": target_x,
            "reason": str(raw.get("reason", "")).strip(),
            "llm_rank": index,
            "llm_final_choice": index == final_choice,
        })
    return strategies, str(result.get("summary", "")).strip()
