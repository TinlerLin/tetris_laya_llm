"""本地 Laya 复核模块：只在上游候选中作出最终选择。"""

import os
import re


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
        criteria[label] = (
            f"LLM_rank={strategy['llm_rank']}; strategy={strategy['id']}; "
            f"rotation_cw={strategy['rotation_cw']}; target_x={strategy['target_x']}; "
            f"final_y={validated['final_y']}; cleared_lines={validated['cleared_lines']}; "
            f"LLM_reason={strategy['reason']}; resulting_board={validated['board_after']}"
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
                "Make the binding final choice among the LLM-planned, rule-valid strategies. "
                "Prefer survival, fewer inaccessible holes, safe height, useful line clears "
                "and a board compatible with the next piece. Choose exactly one option; do "
                "not invent or modify a strategy."
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
    """启发式模式的独立复核入口；不会被 LLM 模式调用。"""
    labels = tuple("ABCD"[: len(candidates)])
    option_map = dict(zip(labels, candidates))
    criteria = {}
    for rank, (label, item) in enumerate(option_map.items(), start=1):
        metrics = item["metrics"]
        planner = item.get("planner", {})
        criteria[label] = (
            f"planner_rank={rank}; landing={item['id']}; "
            f"rotation_cw={item['rotation_cw']}; x={item['target_x']}; "
            f"lines={metrics['lines']}; holes={metrics['holes']}; "
            f"max_height={metrics['max_height']}; bumpiness={metrics['bumpiness']}; "
            f"expectimax_score={planner.get('score', 0):.3f}"
        )
    questions = {
        "select_plan": {
            "type": "choice",
            "instructions": (
                "Make the binding final choice among the heuristic shortlist. Prefer fewer "
                "holes, safe height, smooth accessible structure and safe line clears."
            ),
            "criteria": criteria,
        }
    }
    state_text = (
        f"Current={legal['current_piece']['name']}; next={legal['next_piece']['name']}; "
        f"board={legal['board']}"
    )
    result = agent.predict(state_text, questions)
    answer = result["answers"]["select_plan"]
    choice = _parse_choice(answer.get("choice"), labels)
    if choice is None:
        raise RuntimeError(f"Laya 返回无法解析的选择: {answer.get('choice')!r}")
    return choice, option_map[choice], float(answer.get("answer_confidence", 0.0) or 0.0)
