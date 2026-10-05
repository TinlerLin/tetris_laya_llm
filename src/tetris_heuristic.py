"""One-ply heuristic shortlist used by the local Laya Tetris policy."""

MAX_CANDIDATES = 4
DANGER_HEIGHT = 15


def _board_metrics(board):
    heights = []
    holes = 0
    for x in range(10):
        column = [bool(row[x]) for row in board]
        first = next((y for y, occupied in enumerate(column) if occupied), None)
        height = 0 if first is None else 20 - first
        heights.append(height)
        if first is not None:
            holes += sum(not occupied for occupied in column[first:])
    bumpiness = sum(abs(a - b) for a, b in zip(heights, heights[1:]))
    return heights, holes, bumpiness


def plan(legal, max_candidates=MAX_CANDIDATES):
    """Rank current-piece landings and return a diverse shortlist of up to four."""
    board = [[cell == "#" for cell in row] for row in legal["board"].splitlines()]
    before_heights, before_holes, _ = _board_metrics(board)
    before_aggregate_height = sum(before_heights)

    ranked = []
    for landing in legal["landings"]:
        after = [
            [cell == "#" for cell in row]
            for row in landing["board_after"].splitlines()
        ]
        heights, holes, bumpiness = _board_metrics(after)
        lines = landing["metrics"]["lines"]
        holes_delta = max(0, holes - before_holes)
        aggregate_delta = sum(heights) - before_aggregate_height
        heuristic_score = (
            3.0 * lines
            - 5.0 * holes_delta
            - 0.4 * aggregate_delta
            - 0.2 * bumpiness
        )
        candidate = dict(landing)
        candidate["metrics"] = dict(landing["metrics"])
        candidate["metrics"].update({
            "holes_delta": holes_delta,
            "height": max(heights, default=0),
            "bumpiness": bumpiness,
        })
        candidate["planner"] = {"score": heuristic_score}
        candidate["heuristic"] = heuristic_score
        candidate["dangerous"] = candidate["metrics"]["height"] > DANGER_HEIGHT
        candidate["board_key"] = tuple(tuple(row) for row in after)
        ranked.append(candidate)

    ranked.sort(key=lambda item: item["heuristic"], reverse=True)
    if not ranked:
        raise RuntimeError("启发式规划没有找到合法落点")

    # Match the Ascend example: first prefer distinct outcome categories, then
    # fill any remaining slots from the ranked placements.
    selected = []
    seen = set()
    for candidate in ranked:
        signature = (
            candidate["metrics"]["lines"],
            candidate["metrics"]["holes_delta"],
            candidate["dangerous"],
        )
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(candidate)
        if len(selected) >= min(max_candidates, len(ranked)):
            break
    for candidate in ranked:
        if len(selected) >= min(max_candidates, len(ranked)):
            break
        if candidate not in selected:
            selected.append(candidate)

    for rank, candidate in enumerate(selected, start=1):
        candidate["planner"]["rank"] = rank
        candidate.pop("board_key", None)
    summary = (
        f"单步启发式评估 {len(ranked)} 个合法落点，筛出 {len(selected)} 个候选；"
        f"首选 {selected[0]['id']}，评分 {selected[0]['heuristic']:.2f}"
    )
    return selected, summary
