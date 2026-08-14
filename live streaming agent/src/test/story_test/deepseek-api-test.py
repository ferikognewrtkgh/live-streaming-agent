import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[3]


PROJECT_ROOT = find_project_root()
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "resource" / "prompt" / "\u5361\u5e03.txt"


def clean_lines(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def load_system_prompt(prompt_file: Path | None) -> str:
    if prompt_file is None:
        return ""
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return clean_lines(prompt_file.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-turn DeepSeek chat.")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-history", type=int, default=30)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Do not send the DeepSeek thinking-disabled extra_body.",
    )
    return parser.parse_args()


def trim_history(
    messages: list[dict[str, str]], max_history: int
) -> list[dict[str, str]]:
    if max_history <= 0:
        return messages[:1]

    system_messages = (
        messages[:1] if messages and messages[0]["role"] == "system" else []
    )
    chat_messages = messages[len(system_messages) :]
    return system_messages + chat_messages[-max_history:]


def stream_chat_response(
    client: OpenAI,
    args: argparse.Namespace,
    messages: list[dict[str, str]],
) -> str:
    request_kwargs = {
        "model": args.model,
        "messages": messages,
        "stream": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if not args.enable_thinking:
        request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    start_time = time.perf_counter()
    response = client.chat.completions.create(**request_kwargs)

    assistant_chunks: list[str] = []
    first_token_time: float | None = None
    output_chars = 0
    printed_thinking_header = False
    printed_answer_header = False

    for chunk in response:
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        delta = choice.delta
        content = delta.content or ""
        reasoning = getattr(delta, "reasoning_content", None) or ""

        if first_token_time is None and (content or reasoning):
            first_token_time = time.perf_counter()
            print(f"\nfirst token time: {first_token_time - start_time:.3f}s")

        if reasoning:
            if not printed_thinking_header:
                print("thinking:")
                printed_thinking_header = True
            print(reasoning, end="", flush=True)
            output_chars += len(reasoning)
            continue

        if content:
            if not printed_answer_header:
                print("assistant:")
                printed_answer_header = True
            print(content, end="", flush=True)
            assistant_chunks.append(content)
            output_chars += len(content)

        if choice.finish_reason:
            print()
            break

    end_time = time.perf_counter()
    if first_token_time is None:
        print("\nassistant:")
        first_token_time = end_time

    output_time = max(end_time - first_token_time, 0.0)
    speed = output_time / output_chars if output_chars else 0.0
    print(
        f"[stats] startup={first_token_time - start_time:.3f}s "
        f"chars={output_chars} output={output_time:.3f}s sec/char={speed:.4f}"
    )
    return "".join(assistant_chunks).strip()


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")

    prompt_file = Path(args.prompt_file) if args.prompt_file else None
    if prompt_file and not prompt_file.is_absolute():
        prompt_file = PROJECT_ROOT / prompt_file

    system_prompt = load_system_prompt(prompt_file)
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    print("DeepSeek multi-turn chat. Type /exit, /quit, or /q to stop.")
    print(f"model={args.model}")
    if prompt_file:
        print(f"prompt={prompt_file}")

    while True:
        user_input = input("\nyou> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"/exit", "/quit", "/q", "exit", "quit", "q"}:
            break

        messages.append({"role": "user", "content": user_input})
        messages = trim_history(messages, args.max_history)

        assistant_text = stream_chat_response(client, args, messages)
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
