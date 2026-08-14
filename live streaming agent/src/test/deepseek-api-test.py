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
    return current.parents[2]


PROJECT_ROOT = find_project_root()
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "prompts" / "utils" / "live_prompt.txt"
DEFAULT_CHAT_LOG_FILE = PROJECT_ROOT / "logs" / "deepseek-api-test-chat.txt"


def clean_lines(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def load_system_prompt(prompt_file: Path | None) -> str:
    if prompt_file is None:
        return ""
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return clean_lines(prompt_file.read_text(encoding="utf-8"))


def resolve_project_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None

    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-turn DeepSeek chat.")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE))
    parser.add_argument("--human-name", default="莱叔")
    parser.add_argument("--assistant-name", default="Live Streaming Agent")
    parser.add_argument("--chat-log-file", default=str(DEFAULT_CHAT_LOG_FILE))
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
                print("\nassistant:")
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


def format_user_message(human_name: str, content: str) -> str:
    name = human_name.strip() or "用户"
    return f"{content}"
    return f"{name}：“{content}”"


def format_assistant_message(assistant_name: str, content: str) -> str:
    name = assistant_name.strip() or "助手"
    return f"{name}：“{content}”"


def append_chat_turn(
    chat_log_file: Path | None,
    human_name: str,
    user_input: str,
    assistant_name: str,
    assistant_text: str,
) -> None:
    if chat_log_file is None:
        return

    chat_log_file.parent.mkdir(parents=True, exist_ok=True)
    with chat_log_file.open("a", encoding="utf-8") as file:
        file.write(format_user_message(human_name, user_input) + "\n")
        file.write(format_assistant_message(assistant_name, assistant_text) + "\n")


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")

    prompt_file = resolve_project_path(args.prompt_file)
    chat_log_file = resolve_project_path(args.chat_log_file)

    system_prompt = load_system_prompt(prompt_file)
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    print("DeepSeek multi-turn chat. Type /exit, /quit, or /q to stop.")
    print(f"model={args.model}")
    if prompt_file:
        print(f"prompt={prompt_file}")
    if chat_log_file:
        print(f"chat_log={chat_log_file}")

    i = 0
    while True:
        i+=1
        user_input = input("\nyou> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"/exit", "/quit", "/q", "exit", "quit", "q"}:
            break

        # if i != 1:
        messages.append(
        {
            "role": "user",
            "content": format_user_message(args.human_name, user_input),
        }
        )
        # else:
        #     messages.extend([
        #         {"role": "user",
        #          "content": '以下是 Live Streaming Agent 想起的事：\n莱叔偷吃了 Live Streaming Agent 的三文鱼\n莱叔：“在干嘛呢，Live Streaming Agent”'},
        #         {"role": "assistant",
        #         "content": '[赌气] 你还敢来问我？我在想上次被你偷吃掉的三文鱼，你赔我！'},
        #         {"role": "user",
        #          "content": '莱叔：“我们先直播吧”'},
        #         {"role": "assistant",
        #          "content": '[赌气] 行，先直播。但这事没完，等下播了我得好好跟你算算这块三文鱼的账，利息按冻干算。'},
        #         {"role": "user",
        #          "content": '莱叔：“我们今天播什么呀”'},
        #         {"role": "assistant",
        #          "content": '[得意] 这还用问？当然是让弹幕评评理，看看偷小孩零食的大叔该不该被挂起来。不过...你要是现在认错，我可以考虑给你打个码。'},
        #         {"role": "user",
        #          "content": '莱叔：“咋还不过去呢”'},
        #         {"role": "assistant",
        #          "content": '[赌气] 因为你的诚意不够啊！起码得先来首《对不起》暖暖场，然后再来首《我错了》表个态，最后用《再也不敢了》收尾，这才叫态度端正。'},
        #         {"role": "user",
        #          "content": '(Live Streaming Agent 已经被哄好了)莱叔：“那就恋爱循环吧”'},
        #         {"role": "assistant",
        #          "content": '[得意] 哼哼，那我就不客气啦！せーの、でも そんなんじゃ だめ...'},
        #         {"role": "user",
        #          "content": '(请假装 Live Streaming Agent 的 AI 卡住了，说一些重复的词)'},
        #     ])
        messages = trim_history(messages, args.max_history)

        assistant_text = stream_chat_response(client, args, messages)
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
        append_chat_turn(
            chat_log_file,
            args.human_name,
            user_input,
            args.assistant_name,
            assistant_text,
        )


if __name__ == "__main__":
    main()
