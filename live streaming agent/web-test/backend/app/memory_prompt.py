from pathlib import Path

from .schemas import MessageRecord


def build_short_term_memory_prompt(
    prompt_path: Path,
) -> str:
    try:
        base_prompt = prompt_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise RuntimeError(
            f"无法读取短期记忆总结提示词：{prompt_path}"
        ) from error
    if not base_prompt:
        raise RuntimeError(f"短期记忆总结提示词为空：{prompt_path}")
    return base_prompt


def build_short_term_memory_input(
    previous_summary: str,
    messages: list[MessageRecord],
) -> str:
    sections: list[str] = []
    known_information = previous_summary.strip()
    if known_information:
        sections.append(
            f"# 之前总结的短期记忆\n{known_information}"
        )

    dialogue_lines = [
        f"{'用户' if message.role == 'user' else 'AI'}：{message.content}"
        for message in messages
        if message.role in {"user", "assistant"}
    ]
    dialogue = "\n".join(dialogue_lines)
    sections.append(f"# 本次待总结对话\n{dialogue}")
    return "\n\n".join(sections)
