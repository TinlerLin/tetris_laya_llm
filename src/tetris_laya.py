"""本地 Laya 复核模块：只在上游候选中作出最终选择。"""

import os
import re
import time


def load_agent(model=None):
    import laya

    return laya.load(model or os.getenv("LAYA_MODEL", "convaiinnovations/laya"))


def _parse_choice(value, labels):
    text = str(value).strip()
    if text in labels:
        return text
    for label in labels:
        if re.search(rf"(?<![A-Za-z0-9_]){label}(?![A-Za-z0-9_])", text):
            return label
    return None


def review_llm_strategies(agent, rules, state, strategies):
    """从 LLM 规划且经规则校验的候选中选择一个，不生成新策略。"""
    labels = tuple("ABCD"[: len(strategies)])
    option_map = dict(zip(labels, strategies))
    criteria = {}
    for label, strategy in option_map.items():
        validated = strategy["validation"]
        metrics = strategy.get("objective_metrics", {})
        criteria[label] = (
            f"LLM_rank={strategy['llm_rank']}; strategy={strategy['id']}; "
            f"LLM_final_choice={bool(strategy.get('llm_final_choice'))}; "
            f"rotation_cw={strategy['rotation_cw']}; target_x={strategy['target_x']}; "
            f"final_y={validated['final_y']}; cleared_lines={validated['cleared_lines']}; "
            f"score_delta={metrics.get('score_delta', 0)}; "
            f"holes={metrics.get('holes')}; holes_by_column={metrics.get('holes_by_column')}; "
            f"column_heights={metrics.get('heights')}; max_height={metrics.get('max_height')}; "
            f"aggregate_height={metrics.get('aggregate_height')}; "
            f"bumpiness={metrics.get('bumpiness')}; LLM_reason={strategy['reason']}"
        )
    current = state["current_piece"]
    state_text = (
        f"Board size={rules['board_width']}x{rules['board_height']}; "
        f"current={current['name']} at x={current['x']},y={current['y']}; "
        f"next={state['next_piece']['name']}; board={state['board']}"
    )
    questions = {
        "select_plan": {
            "type": "choice",
            "instructions": (
                "Conservatively verify the binding final choice among four rule-valid strategies "
                "shortlisted from the complete legal landing set. The option marked "
                "LLM_final_choice=True is the planner baseline: keep it unless another option "
                "has clear objective evidence of lower survival risk or meaningfully higher "
                "score_delta with comparable safety. score_delta is 0/100/300/500/800 for "
                "clearing 0/1/2/3/4 lines in one placement. Safety comes first: do not accept "
                "top-out, materially higher danger, or newly buried/deep holes just for points. "
                "Among comparably safe choices, prefer more points; then prefer lower "
                "aggregate/max height, an accessible column profile, lower bumpiness, and "
                "compatibility with the known next piece. Do not replace the baseline merely to "
                "be different. Choose exactly one option; do not invent or modify a strategy."
            ),
            "criteria": criteria,
        }
    }
    result = agent.predict(state_text, questions)
    answer = result["answers"]["select_plan"]
    choice = _parse_choice(answer.get("choice"), labels)
    if choice is None:
        raise RuntimeError(f"Laya 返回无法解析的选择: {answer.get('choice')!r}")
    return choice, option_map[choice], float(answer.get("answer_confidence", 0.0) or 0.0)


def review_heuristic_candidates(agent, legal, candidates):
    """Ask Laya to rate each shortlisted placement, then apply a height guard."""
    labels = tuple("ABCD"[: len(candidates)])
    rated = []
    inference_ms = 0.0
    for label, candidate in zip(labels, candidates):
        metrics = candidate["metrics"]
        lines = metrics["lines"]
        line_text = f"消除{('零', '一', '两', '三', '四')[lines]}行" if lines else "不消行"
        holes = metrics.get("holes_delta", metrics.get("holes", 0))
        holes_text = f"埋下{holes}个空洞" if holes else "不留空洞"
        height = metrics.get("height", metrics.get("max_height", 0))
        if height > 15:
            height_text = "堆叠接近板顶"
        elif height > 10:
            height_text = "堆叠变高"
        else:
            height_text = "堆叠保持低位"
        score_delta = metrics.get("score_delta", 0)
        statement = (
            f"这个落点{line_text}，本次得分 +{score_delta} 分，"
            f"{holes_text}，{height_text}。"
        )
        started = time.perf_counter()
        result = agent.predict(
            statement,
            {
                "q": {
                    "type": "noul",
                    "instructions": (
                        "这是一个好的落点吗？先避免危险高度和新埋空洞；安全性相近时，"
                        "本次得分更高的落点更好。不要为了高分接受明显更高的死亡风险。"
                    ),
                }
            },
        )
        inference_ms += (time.perf_counter() - started) * 1000.0
        try:
            p_good = float(result["answers"]["q"]["noul"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Laya 未返回有效的 noul 评分: {result!r}") from exc
        rated.append({"label": label, "candidate": candidate, "p_good": p_good})

    proposed = max(rated, key=lambda item: item["p_good"])
    executed = proposed
    if proposed["candidate"]["metrics"].get("height", 0) > 15:
        safe = [item for item in rated if item["candidate"]["metrics"].get("height", 0) <= 15]
        if safe:
            executed = max(safe, key=lambda item: item["p_good"])

    for item in rated:
        item["candidate"]["laya_p_good"] = item["p_good"]
        item["candidate"]["laya_label"] = item["label"]
    selected = executed["candidate"]
    selected["laya_proposed"] = proposed["candidate"]["id"]
    selected["laya_guard_intervened"] = proposed is not executed
    selected["laya_inference_ms"] = inference_ms
    return executed["label"], selected, executed["p_good"]
