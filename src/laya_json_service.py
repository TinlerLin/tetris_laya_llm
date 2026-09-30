"""本地 Laya 的通用 JSON 输入/输出适配器。

一次性调用：
    python src/laya_json_service.py --input request.json --pretty
    Get-Content request.json | python src/laya_json_service.py

常驻 JSONL：
    python src/laya_json_service.py --loop
每行输入一个 JSON 请求，每行输出一个 JSON 响应。模型只加载一次。
"""

# 必须先加载 Laya（及其内部 PyTorch）。
import laya

import argparse
import json
import os
import sys
import time


DEFAULT_MODEL = os.getenv("LAYA_MODEL", "convaiinnovations/laya")


class LayaDecisionService:
    def __init__(self, model=DEFAULT_MODEL, device=None):
        self.model = model
        self.device = device
        started = time.perf_counter()
        self.agent = laya.load(model, device=device)
        self.load_ms = (time.perf_counter() - started) * 1000

    def decide(self, request):
        if not isinstance(request, dict):
            raise ValueError("请求必须是 JSON object")
        if "state" not in request:
            raise ValueError("请求缺少 state")
        questions = request.get("questions", request.get("question"))
        if not isinstance(questions, dict) or not questions:
            raise ValueError("请求必须包含非空 questions object")

        started = time.perf_counter()
        result = self.agent.predict(request["state"], questions)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "ok": True,
            "model": self.model,
            "elapsed_ms": round(elapsed_ms, 3),
            "result": result,
        }


def error_response(exc):
    return {
        "ok": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def process_request(service, request):
    try:
        return service.decide(request)
    except Exception as exc:
        return error_response(exc)


def read_request(path):
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    content = sys.stdin.read()
    if not content.strip():
        raise ValueError("标准输入为空；请传入 JSON 或使用 --input")
    return json.loads(content)


def write_json(value, pretty=False):
    if pretty:
        json.dump(value, sys.stdout, ensure_ascii=False, indent=2)
    else:
        json.dump(value, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")
    sys.stdout.flush()


def run_loop(service):
    # ready 事件让调用方确认模型已经加载完毕。
    write_json({
        "event": "ready",
        "model": service.model,
        "load_ms": round(service.load_ms, 3),
    })
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                write_json({"ok": True, "event": "shutdown"})
                return
            response = process_request(service, request)
        except Exception as exc:
            response = error_response(exc)
        write_json(response)


def main():
    parser = argparse.ArgumentParser(description="Local Laya JSON decision adapter")
    parser.add_argument("--input", help="JSON request file; defaults to stdin")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--loop", action="store_true", help="persistent JSONL mode")
    args = parser.parse_args()

    try:
        service = LayaDecisionService(args.model, args.device)
        if args.loop:
            run_loop(service)
            return
        request = read_request(args.input)
        write_json(process_request(service, request), args.pretty)
    except Exception as exc:
        write_json(error_response(exc), args.pretty)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
