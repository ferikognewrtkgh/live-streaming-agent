import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[2]


PROJECT_ROOT = find_project_root()
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "prompts" / "utils" / "live_prompt.txt"
EXIT_COMMANDS = {"/exit", "/quit", "/q", "exit", "quit", "q"}


@dataclass(frozen=True)
class AnswerConfig:
    label: str
    model: str
    thinking: str
    max_keyword: int
    search_limit: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Live Streaming Agent 提示词测试方舟联网搜索的流式多轮对话。"
    )
    parser.add_argument(
        "--fast-model",
        "--model",
        dest="fast_model",
        default="doubao-seed-2-0-mini-260428",
    )
    parser.add_argument(
        "--strong-model", default="doubao-seed-2-1-pro-260628"
    )
    parser.add_argument(
        "--base-url", default="https://ark.cn-beijing.volces.com/api/v3"
    )
    parser.add_argument("--api-key-env", default="ARK_API_KEY")
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE))
    parser.add_argument("--max-history", type=int, default=30)
    parser.add_argument(
        "--fast-max-keyword",
        "--max-keyword",
        dest="fast_max_keyword",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--fast-search-limit",
        "--search-limit",
        dest="fast_search_limit",
        type=int,
        default=2,
    )
    parser.add_argument("--strong-max-keyword", type=int, default=3)
    parser.add_argument("--strong-search-limit", type=int, default=20)
    return parser.parse_args()


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_system_prompt(prompt_file: Path) -> str:
    if not prompt_file.is_file():
        raise FileNotFoundError(f"找不到 Live Streaming Agent 提示词文件: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8").strip()


def trim_history(
    messages: list[dict[str, str]], max_history: int
) -> list[dict[str, str]]:
    """保留 system 提示词和最近 max_history 条对话消息。"""
    system_messages = (
        messages[:1] if messages and messages[0]["role"] == "system" else []
    )
    if max_history <= 0:
        return system_messages
    return system_messages + messages[len(system_messages) :][-max_history:]


def get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def extract_delta_text(event: Any) -> str:
    """从方舟 Responses API 的文本增量事件中提取内容。"""
    event_type = get_value(event, "type")
    if event_type in {
        "response.output_text.delta",
        "response.doubao_app_call_output_text.delta",
    }:
        delta = get_value(event, "delta", "")
        return delta if isinstance(delta, str) else ""
    return ""


def extract_reasoning_delta(event: Any) -> str:
    """提取方舟返回的思考摘要或豆包思考文本增量。"""
    event_type = get_value(event, "type")
    if event_type in {
        "response.reasoning_summary_text.delta",
        "response.reasoning_text.delta",
        "response.doubao_app_call_reasoning_text.delta",
    }:
        delta = get_value(event, "delta", "")
        return delta if isinstance(delta, str) else ""
    return ""


def object_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    legacy_dict = getattr(value, "dict", None)
    if callable(legacy_dict):
        return legacy_dict(exclude_none=True)
    return {}


def add_search_query(item: Any, queries: list[str]) -> bool:
    if get_value(item, "type") != "web_search_call":
        return False
    query = get_value(get_value(item, "action"), "query")
    if isinstance(query, str) and query and query not in queries:
        queries.append(query)
    return True


def add_search_source(annotation: Any, sources: list[dict[str, Any]]) -> None:
    annotation_type = get_value(annotation, "type")
    if annotation_type not in {"url_citation", "doc_citation"}:
        return

    source = object_to_dict(annotation)
    if not source:
        return
    identity = (
        source.get("url"),
        source.get("doc_id"),
        source.get("chunk_id"),
        source.get("title"),
    )
    for existing in sources:
        existing_identity = (
            existing.get("url"),
            existing.get("doc_id"),
            existing.get("chunk_id"),
            existing.get("title"),
        )
        if existing_identity == identity:
            return
    sources.append(source)


def collect_search_details(
    event: Any,
    queries: list[str],
    sources: list[dict[str, Any]],
) -> bool:
    """收集流式事件中的联网搜索关键词和引用来源。"""
    event_type = get_value(event, "type", "")
    search_used = isinstance(event_type, str) and event_type.startswith(
        "response.web_search_call."
    )
    source_count_before = len(sources)

    if event_type in {"response.output_item.added", "response.output_item.done"}:
        search_used = add_search_query(get_value(event, "item"), queries) or search_used

    if event_type == "response.output_text.annotation.added":
        add_search_source(get_value(event, "annotation"), sources)

    if event_type == "response.content_part.done":
        for annotation in get_value(get_value(event, "part"), "annotations", []) or []:
            add_search_source(annotation, sources)

    if event_type == "response.completed":
        response = get_value(event, "response")
        for item in get_value(response, "output", []) or []:
            search_used = add_search_query(item, queries) or search_used
            for content in get_value(item, "content", []) or []:
                for annotation in get_value(content, "annotations", []) or []:
                    add_search_source(annotation, sources)

    return search_used or len(sources) > source_count_before


def print_search_details(
    queries: list[str], sources: list[dict[str, Any]]
) -> None:
    print("\n[联网搜索详情]")
    if queries:
        print("搜索词:")
        for index, query in enumerate(queries, start=1):
            print(f"  {index}. {query}")
    else:
        print("搜索词: API 未返回")

    if not sources:
        print("搜索结果: API 未返回可展示的引用详情")
        return

    print(f"搜索结果（{len(sources)} 条）:")
    for index, source in enumerate(sources, start=1):
        print(f"  {index}. {json.dumps(source, ensure_ascii=False, default=str)}")


def format_seconds(value: float | None) -> str:
    return f"{value:.3f}s" if value is not None else "无"


def format_rate(value: float | None) -> str:
    return f"{value:.1f}字/s" if value is not None else "无"


def run_turn(
    client: Ark,
    config: AnswerConfig,
    messages: list[dict[str, str]],
) -> str:
    print(f"\n========== {config.label} ==========")
    print(
        f"模型={config.model} | 思考={config.thinking} | "
        f"搜索词≤{config.max_keyword} | 搜索结果≤{config.search_limit}"
    )
    request_started_at = time.perf_counter()
    stream = client.responses.create(
        model=config.model,
        input=messages,
        tools=[
            {
                "type": "web_search",
                "max_keyword": config.max_keyword,
                "limit": config.search_limit,
            }
        ],
        thinking={"type": config.thinking},
        stream=True,
    )
    chunks: list[str] = []
    reasoning_chunks: list[str] = []
    first_event_at: float | None = None
    first_reasoning_at: float | None = None
    first_char_at: float | None = None
    reasoning_started = False
    answer_started = False
    event_count = 0
    events_before_first_char = 0
    search_used = False
    search_queries: list[str] = []
    search_sources: list[dict[str, Any]] = []

    for event in stream:
        event_count += 1
        now = time.perf_counter()
        if first_event_at is None:
            first_event_at = now

        search_used = (
            collect_search_details(event, search_queries, search_sources)
            or search_used
        )

        reasoning_delta = extract_reasoning_delta(event)
        if reasoning_delta and config.thinking == "enabled":
            if first_reasoning_at is None:
                first_reasoning_at = now
            if not reasoning_started:
                print("思考过程> ", end="", flush=True)
                reasoning_started = True
            reasoning_chunks.append(reasoning_delta)
            print(reasoning_delta, end="", flush=True)

        delta = extract_delta_text(event)
        if not delta:
            if first_char_at is None:
                events_before_first_char += 1
            continue

        if first_char_at is None:
            first_char_at = now
        if not answer_started:
            if reasoning_started:
                print()
            print(f"{config.label}Live Streaming Agent> ", end="", flush=True)
            answer_started = True
        chunks.append(delta)
        print(delta, end="", flush=True)

    output_finished_at = time.perf_counter()
    assistant_text = "".join(chunks).strip()
    if answer_started:
        print()
    elif reasoning_started:
        print("\n[响应中没有最终回答文本]")
    else:
        print("[响应中没有可显示的文本]")

    first_event_latency = (
        first_event_at - request_started_at if first_event_at is not None else None
    )
    first_char_latency = (
        first_char_at - request_started_at if first_char_at is not None else None
    )
    first_reasoning_latency = (
        first_reasoning_at - request_started_at
        if first_reasoning_at is not None
        else None
    )
    generation_time = (
        output_finished_at - first_char_at if first_char_at is not None else None
    )
    chars_per_second = (
        len(assistant_text) / generation_time
        if generation_time is not None and generation_time > 0
        else None
    )

    print(
        "[耗时] "
        f"首事件={format_seconds(first_event_latency)} | "
        f"思考首字={format_seconds(first_reasoning_latency)} | "
        f"首字={format_seconds(first_char_latency)} | "
        f"流式生成={format_seconds(generation_time)} | "
        f"总耗时={output_finished_at - request_started_at:.3f}s | "
        f"思考字符数={sum(len(chunk) for chunk in reasoning_chunks)} | "
        f"字符数={len(assistant_text)} | "
        f"速度={format_rate(chars_per_second)} | "
        f"事件数={event_count} | "
        f"首字前事件={events_before_first_char}"
    )
    if search_used:
        print_search_details(search_queries, search_sources)
    return assistant_text


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"缺少环境变量 {args.api_key_env}，请在 .env 或系统环境变量中配置。"
        )

    prompt_file = resolve_project_path(args.prompt_file)
    system_prompt = load_system_prompt(prompt_file)
    fast_config = AnswerConfig(
        label="极速配置",
        model=args.fast_model,
        thinking="disabled",
        max_keyword=args.fast_max_keyword,
        search_limit=args.fast_search_limit,
    )
    strong_config = AnswerConfig(
        label="最强配置",
        model=args.strong_model,
        thinking="enabled",
        max_keyword=args.strong_max_keyword,
        search_limit=args.strong_search_limit,
    )
    conversation_states: list[tuple[AnswerConfig, list[dict[str, str]]]] = [
        (fast_config, [{"role": "system", "content": system_prompt}]),
        (strong_config, [{"role": "system", "content": system_prompt}]),
    ]
    client = Ark(base_url=args.base_url, api_key=api_key)

    print("方舟联网搜索双配置对比（Live Streaming Agent 提示词、流式输出）")
    for config, _ in conversation_states:
        print(
            f"{config.label}: 模型={config.model}，思考={config.thinking}，"
            f"搜索词最多 {config.max_keyword} 个，返回最多 {config.search_limit} 条"
        )
    print(f"提示词: {prompt_file}")
    print("输入 /exit、/quit 或 /q 退出。")

    while True:
        try:
            user_input = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            print("已退出。")
            break

        for state_index, (config, messages) in enumerate(conversation_states):
            messages.append({"role": "user", "content": user_input})
            messages = trim_history(messages, args.max_history)
            conversation_states[state_index] = (config, messages)

            try:
                assistant_text = run_turn(client, config, messages)
            except Exception as exc:
                # 单组失败不会影响另一组，也不把失败问题留在该组历史中。
                if messages and messages[-1] == {
                    "role": "user",
                    "content": user_input,
                }:
                    messages.pop()
                print(f"[{config.label}] 请求失败: {exc}")
                continue

            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
