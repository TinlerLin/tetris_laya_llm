"""独立的三层 Expectimax 启发式规划模块。"""

import teris as game_api


WEIGHTS = {
    "aggregate_height": -0.510066,
    "lines": 0.760666,
    "holes": -0.35663,
    "bumpiness": -0.184483,
    "max_height": -0.12,
    "danger_height": -1.5,
    "well_depth": -0.08,
}
LOOKAHEAD_DISCOUNT = 0.65
THIRD_PIECE_DISCOUNT = 0.35
NEXT_BEAM_WIDTH = 6
MAX_CANDIDATES = 4


def _rotate_shape(shape):
    return [list(row) for row in zip(*shape[::-1])]


def _collision(shape, x, y, grid):
    for dy, row in enumerate(shape):
        for dx, occupied in enumerate(row):
            if not occupied:
                continue
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= 10 or ny >= 20:
                return True
            if ny >= 0 and grid[ny][nx]:
                return True
    return False


def _clear_grid(grid):
    remaining = [row[:] for row in grid if not all(row)]
    cleared = 20 - len(remaining)
    return [[False] * 10 for _ in range(cleared)] + remaining, cleared


def _evaluate(grid, lines=0):
    heights = []
    holes_by_column = []
    for x in range(10):
        first = next((y for y in range(20) if grid[y][x]), None)
        heights.append(0 if first is None else 20 - first)
        holes_by_column.append(
            0 if first is None else sum(not grid[y][x] for y in range(first, 20))
        )
    holes = sum(holes_by_column)
    max_height = max(heights, default=0)
    aggregate_height = sum(heights)
    bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(9))
    well_depth = 0
    for x, height in enumerate(heights):
        left = 20 if x == 0 else heights[x - 1]
        right = 20 if x == 9 else heights[x + 1]
        well_depth += max(0, min(left, right) - height)
    danger_height = max(0, max_height - 14) ** 2
    score = (
        aggregate_height * WEIGHTS["aggregate_height"]
        + lines * WEIGHTS["lines"]
        + holes * WEIGHTS["holes"]
        + bumpiness * WEIGHTS["bumpiness"]
        + max_height * WEIGHTS["max_height"]
        + danger_height * WEIGHTS["danger_height"]
        + well_depth * WEIGHTS["well_depth"]
    )
    return score, {
        "lines": lines,
        "holes": holes,
        "max_height": max_height,
        "aggregate_height": aggregate_height,
        "bumpiness": bumpiness,
        "well_depth": well_depth,
        "heights": heights,
        "holes_by_column": holes_by_column,
    }


def _piece_landings(shape, grid):
    results = []
    current = [row[:] for row in shape]
    seen = set()
    rotation = 0
    while True:
        key = tuple(tuple(row) for row in current)
        if key in seen:
            break
        seen.add(key)
        for x in range(10 - len(current[0]) + 1):
            if _collision(current, x, 0, grid):
                continue
            y = 0
            while not _collision(current, x, y + 1, grid):
                y += 1
            placed = [row[:] for row in grid]
            for dy, row in enumerate(current):
                for dx, occupied in enumerate(row):
                    if occupied:
                        placed[y + dy][x + dx] = True
            after_clear, lines = _clear_grid(placed)
            score, metrics = _evaluate(after_clear, lines)
            results.append({
                "rotation_cw": rotation,
                "target_x": x,
                "board": after_clear,
                "score": score,
                "metrics": metrics,
            })
        current = _rotate_shape(current)
        rotation += 1
    return results


def plan(legal, max_candidates=MAX_CANDIDATES):
    """基于当前块、下一块和下下块概率执行三层搜索。"""
    probabilities = legal.get("piece_after_next_probabilities") or {
        name: 1.0 / len(game_api.SHAPES) for name in game_api.SHAPES
    }
    ranked = []
    for landing in legal["landings"]:
        first_grid = [
            [cell == "#" for cell in row] for row in landing["board_after"].splitlines()
        ]
        immediate_score, metrics = _evaluate(first_grid, landing["metrics"]["lines"])
        next_candidates = _piece_landings(legal["next_piece"]["shape"], first_grid)
        next_candidates.sort(key=lambda item: item["score"], reverse=True)
        next_candidates = next_candidates[:NEXT_BEAM_WIDTH]

        best_continuation = -1000.0
        best_expected = -1000.0
        best_worst = -1000.0
        for next_item in next_candidates:
            expected = 0.0
            future_scores = []
            for name, probability in probabilities.items():
                future = _piece_landings(game_api.SHAPES[name], next_item["board"])
                piece_best = max((item["score"] for item in future), default=-1000.0)
                expected += float(probability) * piece_best
                future_scores.append(piece_best)
            worst = min(future_scores, default=-1000.0)
            continuation = next_item["score"] + THIRD_PIECE_DISCOUNT * expected
            if continuation > best_continuation:
                best_continuation = continuation
                best_expected = expected
                best_worst = worst

        ranked.append({
            "landing": landing,
            "score": immediate_score + LOOKAHEAD_DISCOUNT * best_continuation,
            "immediate_score": immediate_score,
            "expected_future_score": best_expected,
            "worst_future_score": best_worst,
            "metrics": metrics,
            "board_key": tuple(tuple(row) for row in first_grid),
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)
    for index, item in enumerate(ranked, start=1):
        item["planner_rank"] = index
    if not ranked:
        raise RuntimeError("Expectimax 没有找到合法候选")

    shortlist = ranked[: min(12, len(ranked))]
    selected = []
    selected_keys = set()

    def add(item):
        if len(selected) < max_candidates and item["board_key"] not in selected_keys:
            selected.append(item)
            selected_keys.add(item["board_key"])

    add(ranked[0])
    add(min(shortlist, key=lambda x: (
        x["metrics"]["holes"], x["metrics"]["aggregate_height"], -x["score"]
    )))
    add(min(shortlist, key=lambda x: (
        x["metrics"]["max_height"], x["metrics"]["bumpiness"], -x["score"]
    )))
    add(min(shortlist, key=lambda x: (
        x["metrics"]["bumpiness"], x["metrics"]["well_depth"], -x["score"]
    )))
    for item in ranked:
        add(item)
        if len(selected) == min(max_candidates, len(ranked)):
            break

    candidates = []
    for item in selected:
        candidate = dict(item["landing"])
        candidate["planner"] = {
            "rank": item["planner_rank"],
            "score": item["score"],
            "immediate_score": item["immediate_score"],
            "expected_future_score": item["expected_future_score"],
            "worst_future_score": item["worst_future_score"],
        }
        candidates.append(candidate)
    summary = (
        f"三层 Expectimax 搜索 {len(ranked)} 个当前落点；"
        f"首选 {candidates[0]['id']}，效用 {candidates[0]['planner']['score']:.2f}"
    )
    return candidates, summary
