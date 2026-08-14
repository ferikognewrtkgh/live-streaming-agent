"""Load blocked words from a plain text file.

Other modules can import BLOCKED_WORDS directly, or call load_blocked_words()
when they need to reload a specific file.
"""

from pathlib import Path
from functools import lru_cache
import re
from typing import Iterable, Sequence

BLOCKED_WORDS_FILENAME = "违禁词.txt"
COMMENT_PREFIXES = ("#", "//", ";")
TOKEN_SEPARATORS = (
    ",",
    "\uff0c",
    "\u3001",
    "|",
    "\t",
    ";",
    "\uff1b",
)
MIN_BLOCKED_WORD_LENGTH = 2

COLON_RE = re.compile(r"[:\uff1a]")
LEADING_NUMBER_RE = re.compile(
    r"^\s*(?:[\uff08(]?\d+[\uff09)]?|[0-9]+)\s*"
    r"[\.\uff0e\u3001,，:：;；)]\s*"
)
PARENTHETICAL_RE = re.compile(r"[\uff08(][^\uff08\uff09()]*[\uff09)]")
QUOTE_RE = re.compile(r"[\u201c\"']([^'\"]+?)[\u201d\"']")
PURE_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")
PUNCT_STRIP_CHARS = " \t\r\n.。!！?？:：;；,，、|()（）[]【】<>《》\"'“”"

HEADER_TERMS = {
    "\u7edd\u5bf9\u7981\u6b62",
}
HEADER_SUFFIXES = (
    "\u7528\u8bed",
    "\u5ba3\u4f20\u7528\u8bed",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_blocked_words_file(filename: str = BLOCKED_WORDS_FILENAME) -> Path | None:
    root = _project_root()
    candidates = (
        Path.cwd() / filename,
        root / filename,
        root / "resource" / filename,
        root / "prompts" / filename,
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _split_line(line: str) -> Iterable[str]:
    tokens = [line]
    for separator in TOKEN_SEPARATORS:
        split_tokens: list[str] = []
        for token in tokens:
            split_tokens.extend(token.split(separator))
        tokens = split_tokens
    return tokens


def _strip_leading_number(text: str) -> str:
    previous = None
    current = text
    while previous != current:
        previous = current
        current = LEADING_NUMBER_RE.sub("", current, count=1)
    return current


def _strip_parenthetical(text: str) -> str:
    previous = None
    current = text
    while previous != current:
        previous = current
        current = PARENTHETICAL_RE.sub("", current)
    return current


def _strip_category_prefix(line: str) -> str:
    match = COLON_RE.search(line)
    if not match:
        return line

    prefix = line[: match.start()].strip()
    suffix = line[match.end() :].strip()
    if prefix and suffix:
        return suffix
    return line


def _looks_like_header(line: str) -> bool:
    cleaned = _strip_parenthetical(line).strip(PUNCT_STRIP_CHARS)
    if not cleaned:
        return True
    if cleaned in HEADER_TERMS:
        return True
    if cleaned.endswith(HEADER_SUFFIXES) and len(cleaned) >= 4:
        return True
    return False


def _clean_token(token: str) -> str:
    word = _strip_leading_number(token)
    word = _strip_parenthetical(word)
    word = word.strip(PUNCT_STRIP_CHARS)
    if not word or word.startswith(COMMENT_PREFIXES):
        return ""
    if PURE_NUMBER_RE.fullmatch(word):
        return ""
    if _looks_like_header(word):
        return ""
    return word


def _is_valid_blocked_word(word: str) -> bool:
    return len(word.strip()) >= MIN_BLOCKED_WORD_LENGTH


def _quoted_words(line: str) -> list[str]:
    return [match.group(1).strip() for match in QUOTE_RE.finditer(line)]


def parse_blocked_words_line(raw_line: str) -> list[str]:
    line = raw_line.strip()
    if not line or line.startswith(COMMENT_PREFIXES):
        return []

    raw_words: list[str] = []
    quoted_words = _quoted_words(line)
    if quoted_words and ("\u5c4f\u853d" in line or "\u4e00\u8bcd" in line):
        raw_words = quoted_words
    else:
        line = _strip_leading_number(line)
        line = _strip_category_prefix(line)
        if _looks_like_header(line):
            return []
        raw_words = list(_split_line(line))

    words: list[str] = []
    seen: set[str] = set()
    for raw_word in raw_words:
        word = _clean_token(raw_word)
        if not word or not _is_valid_blocked_word(word):
            continue
        key = word.casefold()
        if key in seen:
            continue
        seen.add(key)
        words.append(word)
    return words


def parse_blocked_words_lines(text: str) -> list[list[str]]:
    return [parse_blocked_words_line(line) for line in text.splitlines()]


def format_blocked_words_lines(text: str) -> str:
    return "\n".join(
        ", ".join(repr(word) for word in words)
        for words in parse_blocked_words_lines(text)
    )


def parse_blocked_words_text(text: str) -> list[str]:
    words: list[str] = []
    seen: set[str] = set()

    def add_word(raw_word: str) -> None:
        word = _clean_token(raw_word)
        if not word or not _is_valid_blocked_word(word):
            return
        key = word.casefold()
        if key in seen:
            return
        seen.add(key)
        words.append(word)

    for line_words in parse_blocked_words_lines(text):
        for word in line_words:
            add_word(word)

    return words


def load_blocked_words(path: str | Path | None = None) -> list[str]:
    source = Path(path) if path is not None else find_blocked_words_file()
    if source is None or not source.is_file():
        return []
    return parse_blocked_words_text(_read_text_with_fallback(source))


BLOCKED_WORDS = load_blocked_words()


@lru_cache(maxsize=16)
def _blocked_words_regex(words: tuple[str, ...]) -> re.Pattern[str] | None:
    clean_words = [word for word in words if word and _is_valid_blocked_word(word)]
    if not clean_words:
        return None
    pattern = "|".join(
        re.escape(word)
        for word in sorted(clean_words, key=len, reverse=True)
    )
    return re.compile(pattern, re.IGNORECASE)


def sanitize_blocked_words_text(
    text: str,
    replacement: str = "\u55b5\u55b5",
    blocked_words: list[str] | tuple[str, ...] | None = None,
    ignored_spans: Sequence[tuple[int, int]] | None = None,
) -> str:
    sanitized, _ = sanitize_blocked_words_text_with_matches(
        text,
        replacement=replacement,
        blocked_words=blocked_words,
        ignored_spans=ignored_spans,
    )
    return sanitized


def _normalize_ignored_spans(
    ignored_spans: Sequence[tuple[int, int]] | None,
    text_length: int,
) -> list[tuple[int, int]]:
    if not ignored_spans:
        return []

    normalized: list[tuple[int, int]] = []
    for start, end in ignored_spans:
        clamped_start = max(0, min(int(start), text_length))
        clamped_end = max(0, min(int(end), text_length))
        if clamped_start >= clamped_end:
            continue
        normalized.append((clamped_start, clamped_end))
    normalized.sort()
    return normalized


def _span_overlaps_ignored_spans(
    start: int,
    end: int,
    ignored_spans: Sequence[tuple[int, int]],
) -> bool:
    return any(
        start < ignored_end and end > ignored_start
        for ignored_start, ignored_end in ignored_spans
    )


def sanitize_blocked_words_text_with_matches(
    text: str,
    replacement: str = "\u55b5\u55b5",
    blocked_words: list[str] | tuple[str, ...] | None = None,
    ignored_spans: Sequence[tuple[int, int]] | None = None,
) -> tuple[str, list[str]]:
    if not text:
        return text, []

    words = tuple(blocked_words if blocked_words is not None else BLOCKED_WORDS)
    regex = _blocked_words_regex(words)
    if regex is None:
        return text, []

    matched_words: list[str] = []
    seen: set[str] = set()
    replacement_unit = replacement[:1] if replacement else "\u55b5"
    normalized_ignored_spans = _normalize_ignored_spans(ignored_spans, len(text))

    def replace_match(match: re.Match[str]) -> str:
        if _span_overlaps_ignored_spans(
            match.start(),
            match.end(),
            normalized_ignored_spans,
        ):
            return match.group(0)
        matched_word = match.group(0)
        key = matched_word.casefold()
        if key not in seen:
            seen.add(key)
            matched_words.append(matched_word)
        return replacement_unit * len(matched_word)

    return regex.sub(replace_match, text), matched_words


def find_blocked_words_in_text(
    text: str,
    blocked_words: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    _, matched_words = sanitize_blocked_words_text_with_matches(
        text,
        blocked_words=blocked_words,
    )
    return matched_words


def print_aligned_blocked_words(path: str | Path | None = None) -> None:
    source = Path(path) if path is not None else find_blocked_words_file()
    if source is None or not source.is_file():
        return
    print(format_blocked_words_lines(_read_text_with_fallback(source)))


if __name__ == "__main__":
    print_aligned_blocked_words()
