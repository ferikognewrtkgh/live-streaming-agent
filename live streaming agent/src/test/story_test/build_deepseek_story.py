import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STORY_PATH = ROOT / "story.txt"
ALL_LINES_PATH = ROOT / "story_all_lines.jsonl"
GU_LINES_PATH = ROOT / "story_gu_qianxu.jsonl"
QIANCHEN_LINES_PATH = ROOT / "story_qianchen.jsonl"
GENERATED_JSONL_PATH = ROOT / "deepseek_generated_dialogue.jsonl"
GENERATED_STORY_PATH = ROOT / "deepseek_generated_story.txt"

LINE_RE = re.compile(r"^(?P<label>[^：:]{1,40})(?P<sep>[：:])(?P<text>.*)$")
LEADING_PARENS_RE = re.compile(r"^((?:[（(][^）)]*[）)]\s*)+)(.*)$")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def normalize_speaker(label: str) -> str | None:
    base = label.strip()
    for marker in ("（", "("):
        if marker in base:
            base = base.split(marker, 1)[0].strip()
    if base == "顾谦虚":
        return "顾谦虚"
    if base in {"千辰", "辰"}:
        return "千辰"
    return None


def output_label(item: dict) -> str:
    label = item["label"].strip()
    return "千辰" if label == "辰" else label


def split_delivery_prefix(text: str) -> tuple[str, str]:
    match = LEADING_PARENS_RE.match(text.strip())
    if not match:
        return "", text.strip()
    return match.group(1).strip(), match.group(2).strip()


def parse_story(path: Path) -> list[dict]:
    items: list[dict] = []
    speaker_counts = {"顾谦虚": 0, "千辰": 0}
    for source_line, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw_line.strip()
        if not raw:
            continue
        match = LINE_RE.match(raw)
        speaker = normalize_speaker(match.group("label")) if match else None
        if speaker is None:
            items.append(
                {
                    "index": len(items) + 1,
                    "source_line": source_line,
                    "type": "stage",
                    "text": raw,
                    "raw": raw,
                }
            )
            continue
        speaker_counts[speaker] += 1
        label = match.group("label").strip()
        text = match.group("text").strip()
        delivery_prefix, speech_hint = split_delivery_prefix(text)
        items.append(
            {
                "index": len(items) + 1,
                "source_line": source_line,
                "type": "dialogue",
                "speaker": speaker,
                "speaker_turn": speaker_counts[speaker],
                "label": label,
                "text": text,
                "delivery_prefix": delivery_prefix,
                "speech_hint": speech_hint,
                "raw": raw,
            }
        )
    return items


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_story(path: Path, rows: list[dict]) -> None:
    lines = []
    for row in rows:
        if row["type"] == "stage":
            lines.append(row["text"])
        else:
            rendered_text = row.get("rendered_text", row["text"])
            lines.append(f"{output_label(row)}：{rendered_text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_persona() -> str:
    persona_files = sorted(ROOT.glob("*人设*.txt"))
    if not persona_files:
        raise FileNotFoundError("没有找到 *人设*.txt")
    parts = []
    for path in persona_files:
        parts.append(f"【{path.name}】\n{path.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(parts)


def build_system_prompt(persona: str) -> str:
    return f"""{persona}

【当前任务】
你只扮演顾谦虚。你正在和千辰直播聊天。
我会给你之前完整的多轮历史、舞台/弹幕提示，以及“原稿提示”。
你需要参考原稿提示，生成顾谦虚下一句台词。

【硬性要求】
- 只输出顾谦虚的台词正文。
- 不要输出角色名、引号、JSON、解释、注释、心理描写或舞台描写。
- 这是逐句微改写，不是自由续写；可以让原稿更像真实直播口语，但不要改剧情方向。
- 必须保留原稿中的地点、物品、事件、真假关系和上下文衔接。
- 不要新增原稿之外的地点、事件、事实和反转包袱；如果原稿只是在敷衍或转移话题，就保持同等含义。
- 尽量保持原句长度；短句就短，别扩写成独白。
"""


def compact_history(rows: list[dict]) -> list[dict]:
    messages = []
    for row in rows:
        if row["type"] == "stage":
            messages.append({"role": "user", "content": f"【舞台/弹幕提示】{row['text']}"})
        elif row["speaker"] == "千辰":
            messages.append({"role": "user", "content": f"千辰：{row.get('rendered_text', row['text'])}"})
        else:
            messages.append({"role": "assistant", "content": row.get("generated_text", row["text"])})
    return messages


def render_source_line(item: dict) -> str:
    if item["type"] == "stage":
        return item["text"]
    return f"{output_label(item)}：{item['text']}"


def build_user_prompt(item: dict, items: list[dict]) -> str:
    hint = item["speech_hint"] or item["text"]
    prefix = item.get("delivery_prefix") or "无"
    index = item["index"] - 1
    start = max(0, index - 3)
    end = min(len(items), index + 3)
    source_context = "\n".join(render_source_line(row) for row in items[start:end])
    return (
        "请生成顾谦虚的下一句。\n"
        "下面是原稿前后文窗口，只用于确保衔接，不要改动千辰台词或舞台提示：\n"
        f"{source_context}\n\n"
        f"原稿标签：{item['label']}\n"
        f"原稿语气/动作提示：{prefix}\n"
        f"原稿台词提示：{hint}\n"
        "只允许做口语化细微改写。不要新增地点、事实或反转。只输出台词正文。"
    )


def sanitize_gu_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json|text)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    cleaned = re.sub(r"^顾谦虚(?:[（(][^）)]*[）)])?[：:]\s*", "", cleaned).strip()
    cleaned = cleaned.strip("\"'“”")
    cleaned = re.sub(r"\s*\n+\s*", " ", cleaned).strip()
    return cleaned


def call_deepseek(
    messages: list[dict],
    api_key: str,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            parsed = json.loads(body)
            print(messages[1:][-3:])
            print(parsed["choices"][0]["message"]["content"])
            return parsed["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
                continue
            break
    raise RuntimeError(f"DeepSeek 请求失败：{last_error}") from last_error


def generate_dialogue(args: argparse.Namespace, items: list[dict]) -> list[dict]:
    load_env_file(ROOT / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY，请在 .env 或环境变量中设置。")

    system_prompt = build_system_prompt(load_persona())
    generated: list[dict] = []
    total_gu = sum(1 for item in items if item.get("speaker") == "顾谦虚")
    gu_done = 0

    for item in items:
        if item["type"] == "stage":
            generated.append(dict(item))
        elif item["speaker"] == "千辰":
            row = dict(item)
            row["generated_text"] = item["text"]
            row["rendered_text"] = item["text"]
            row["generation"] = "fixed_from_story"
            generated.append(row)
        else:
            gu_done += 1
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(compact_history(generated))
            messages.append({"role": "user", "content": build_user_prompt(item, items)})
            print(f"Generating 顾谦虚 {gu_done}/{total_gu}: source line {item['source_line']}", flush=True)
            raw_text = call_deepseek(
                messages=messages,
                api_key=api_key,
                model=args.model,
                base_url=args.base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
            )
            generated_text = sanitize_gu_text(raw_text)
            if not generated_text:
                generated_text = item["speech_hint"] or item["text"]
            row = dict(item)
            row["source_text"] = item["text"]
            row["generated_text"] = generated_text
            row["rendered_text"] = (
                f"{item['delivery_prefix']}{generated_text}"
                if item.get("delivery_prefix")
                else generated_text
            )
            row["generation"] = "deepseek"
            generated.append(row)
        write_jsonl(GENERATED_JSONL_PATH, generated)
        write_story(GENERATED_STORY_PATH, generated)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Split story.txt and generate Gu Qianxu dialogue with DeepSeek.")
    parser.add_argument("--skip-api", action="store_true", help="Only split story.txt into JSONL files.")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    items = parse_story(STORY_PATH)
    write_jsonl(ALL_LINES_PATH, items)
    write_jsonl(GU_LINES_PATH, [item for item in items if item.get("speaker") == "顾谦虚"])
    write_jsonl(QIANCHEN_LINES_PATH, [item for item in items if item.get("speaker") == "千辰"])
    print(
        "Split complete: "
        f"{len(items)} total, "
        f"{sum(1 for item in items if item.get('speaker') == '顾谦虚')} 顾谦虚, "
        f"{sum(1 for item in items if item.get('speaker') == '千辰')} 千辰",
        flush=True,
    )

    if args.skip_api:
        return
    generated = generate_dialogue(args, items)
    print(f"Generated {len(generated)} rows -> {GENERATED_JSONL_PATH.name}, {GENERATED_STORY_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
