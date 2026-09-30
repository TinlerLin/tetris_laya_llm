"""第三方 LLM 规划模块。

本模块接收游戏规则、原始状态，以及物理引擎枚举的全部合法落底结果。
这些结果不包含评分、排序、推荐或启发式选择；LLM 负责战略比较、生成
候选短名单并指定自己的最终选择，启用 Laya 时再交由 Laya 复核。
"""

import json
import os
import re
import urllib.error
import urllib.request


def parse_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    # 兼容推理或说明文字包围 JSON 的响应。逐个左花括号尝试解码，
    # 避免把前置推理中的花括号与最终 JSON 粗暴拼接到一起。
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    preview = text[:250] if text else "<empty content>"
    raise RuntimeError(f"LLM 未返回可解析的 JSON object: {preview}")


class ThirdPartyLLM:
    """最小 OpenAI-compatible Chat Completions 客户端。"""

    def __init__(self, base_url, api_key, model, timeout=45, max_tokens=None):
        self.base_url = base_url.strip().rstrip("/")
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout = float(timeout)
        self.max_tokens = int(
            max_tokens or os.getenv("THIRD_PARTY_LLM_MAX_TOKENS", "4096")
        )

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
            "max_tokens": self.max_tokens,
        }
        # DeepSeek 的 JSON Output 需要显式指定 response_format；提示词本身
        # 已多次明确要求返回 JSON，符合其 JSON 模式要求。
        if "deepseek" in self.base_url.lower() or "deepseek" in self.model.lower():
            payload["response_format"] = {"type": "json_object"}
            # deepseek-flash 当前默认启用 high 级别思考；俄罗斯方块规划是
            # 实时短任务，显式关闭思考可避免请求长期停留在推理阶段。
            payload["thinking"] = {"type": "disabled"}
            payload["reasoning_effort"] = "none"
        payload["stream"] = False
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
            choice = raw["choices"][0]
            message = choice["message"]
            content = message.get("content")
        except (AttributeError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"LLM 响应不兼容 Chat Completions: {str(raw)[:350]}") from exc
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        if content is None or not str(content).strip():
            finish_reason = choice.get("finish_reason", "unknown")
            reasoning = message.get("reasoning_content") or ""
            raise RuntimeError(
                "LLM 返回的 content 为空"
                f"（finish_reason={finish_reason}, reasoning_chars={len(str(reasoning))}）。"
                "请提高 THIRD_PARTY_LLM_MAX_TOKENS，或关闭模型的深度思考模式。"
            )
        return parse_json_object(str(content))


def plan_strategies(client, rules, state, count=4, legal_landings=None):
    """让 LLM 比较物理引擎枚举的客观落底结果并规划多个策略。"""
    current = state["current_piece"]
    next_piece = state["next_piece"]
    legal_landings = list(legal_landings or [])
    if len(legal_landings) < count:
        raise RuntimeError(
            f"游戏引擎仅提供 {len(legal_landings)} 个合法落底，无法生成 {count} 个策略"
        )

    landing_map = {item["id"]: item for item in legal_landings}

    def compact_board(board):
        rows = board.splitlines()
        top_y = next((index for index, row in enumerate(rows) if "#" in row), len(rows))
        return {"top_y": top_y, "rows": rows[top_y:]}

    objective_outcomes = []
    for item in legal_landings:
        metrics = item["metrics"]
        objective_outcomes.append({
            "landing_id": item["id"],
            "rotation_cw": item["rotation_cw"],
            "target_x": item["target_x"],
            "final_y": item["final_y"],
            "cleared_lines": metrics["lines"],
            "holes": metrics["holes"],
            "holes_by_column": metrics["holes_by_column"],
            "column_heights": metrics["heights"],
            "max_height": metrics["max_height"],
            "aggregate_height": metrics["aggregate_height"],
            "bumpiness": metrics["bumpiness"],
            "board_after": compact_board(item["board_after"]),
        })

    system_prompt = (
        "You are the sole STRATEGIC planner for a real-time 10x20 Tetris game. The game physics "
        "engine has enumerated every legal current-piece landing and simulated its immediate "
        "result. This list is objective only: it has no score, rank, recommendation, pruning, "
        "or heuristic choice. Never invent a landing and never redo collision geometry; choose "
        "only by landing_id from the supplied list.\n\n"
        "Compare all outcomes, not merely the first few. Use this strict priority: (1) avoid "
        "top-out and dangerous height; (2) minimize holes, especially newly covered or deep "
        "holes; (3) keep aggregate/max height low; (4) keep a reasonably flat, accessible "
        "surface without deep wells or overhangs; (5) use the known NEXT piece as one-piece "
        "lookahead by judging whether it has a safe useful placement on board_after; (6) prefer "
        "line clears when they do not violate the earlier priorities. Never create a hole only "
        "to gain an immediate clear. Treat lower holes, max_height, aggregate_height, and "
        "bumpiness as better unless the board_after and next piece justify a specific exception.\n\n"
        "Return exactly the requested number of distinct landing_ids, strongest first. "
        "final_choice is the 1-based index of the strongest returned strategy and should normally "
        "be 1 after correct ranking. Keep each reason short and factual: cite relevant numeric "
        "outcome metrics and the NEXT-piece outlook. Do not expose chain-of-thought. Respond "
        "immediately with one JSON object only; no Markdown or surrounding prose."
    )
    example = {
        "strategies": [
            {"landing_id": "R0_X3", "reason": "short factual rationale"}
        ],
        "final_choice": 1,
        "summary": "brief overall plan",
    }
    user_prompt = (
        f"Rules:\n{json.dumps(rules, ensure_ascii=False)}\n\n"
        f"Round identity: round={state['round_serial']}, piece={state['piece_serial']}\n"
        f"Current piece: {current['name']} at x={current['x']}, y={current['y']}\n"
        f"Known next piece for lookahead: {next_piece['name']}, shape={next_piece['shape']}\n"
        f"Time until the next gravity step: {state['ms_until_auto_fall']} ms. Planning does "
        "not pause gravity, so answer concisely.\n\n"
        "Objective legal outcomes follow. board_after.top_y is the first non-empty y row after "
        "locking and clearing; board_after.rows contains every 10-cell row from top_y through "
        "y=19, so all omitted rows above top_y are empty. '#' is occupied and '.' is empty. "
        "All metrics describe board_after. Smaller hole/height/bumpiness values are generally "
        "safer. Evaluate the entire JSON array:\n"
        f"{json.dumps(objective_outcomes, ensure_ascii=False, separators=(',', ':'))}\n\n"
        f"Return exactly {count} distinct strategies and your binding final_choice. The single "
        "strategy below illustrates field names only; your strategies array must still contain "
        f"exactly {count} objects:\n{json.dumps(example, ensure_ascii=False)}"
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
        landing_id = str(raw.get("landing_id", "")).strip()
        if landing_id not in landing_map:
            raise RuntimeError(f"LLM 策略 #{index} 返回未知 landing_id={landing_id!r}")
        if landing_id in seen:
            raise RuntimeError(f"LLM 返回重复策略: landing_id={landing_id}")
        seen.add(landing_id)
        landing = landing_map[landing_id]
        strategies.append({
            "id": f"S{index}",
            "landing_id": landing_id,
            "rotation_cw": landing["rotation_cw"],
            "target_x": landing["target_x"],
            # 将物理引擎计算的客观结果继续传给可选 Laya 复核，避免
            # Laya 再从 ASCII 棋盘中自行估算空洞和高度。
            "objective_metrics": dict(landing["metrics"]),
            "reason": str(raw.get("reason", "")).strip(),
            "llm_rank": index,
            "llm_final_choice": index == final_choice,
        })
    return strategies, str(result.get("summary", "")).strip()
