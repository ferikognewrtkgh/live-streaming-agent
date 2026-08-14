import re
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
from loguru import logger

from .asr_interface import ASRInterface
from ..resource_paths import ASR_CORRECTIONS_PATH


_TOKEN_SPLIT_RE = re.compile(r"[\s,，、]+")
_GROUP_RE = re.compile(r"\[([^\[\]]+)\]")
_GROUP_PATTERN_RE = re.compile(r"(?:\[[^\[\]]+\])+")


def _split_tokens(value: str) -> list[str]:
    return [token.strip() for token in _TOKEN_SPLIT_RE.split(value) if token.strip()]


def _split_group(value: str) -> list[str]:
    tokens = _split_tokens(value)
    if len(tokens) > 1:
        return tokens
    return [char for char in value.strip() if char]


def _expand_source_token(token: str) -> list[str]:
    """Expand [赵詹][梦牧][时师] style source patterns."""
    if not _GROUP_PATTERN_RE.fullmatch(token):
        return [token]

    groups = [_split_group(match.group(1)) for match in _GROUP_RE.finditer(token)]
    if not groups or any(not group for group in groups):
        return []
    return ["".join(parts) for parts in product(*groups)]


def _parse_rule_line(line: str) -> tuple[str, str] | None:
    if "=>" in line:
        sources, target = line.split("=>", 1)
        return target.strip(), sources.strip()

    for separator in ("<=", "<-", "="):
        if separator in line:
            target, sources = line.split(separator, 1)
            return target.strip(), sources.strip()

    return None


def load_asr_corrections(path: str | Path = ASR_CORRECTIONS_PATH) -> dict[str, str]:
    source = Path(path)
    if not source.exists():
        return {}

    corrections: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        parsed = _parse_rule_line(line)
        if parsed is None:
            logger.warning(
                "Invalid ASR correction rule at {}:{}: {}",
                source,
                line_number,
                raw_line,
            )
            continue

        target, sources = parsed
        if not target or not sources:
            logger.warning(
                "Incomplete ASR correction rule at {}:{}: {}",
                source,
                line_number,
                raw_line,
            )
            continue

        for source_token in _split_tokens(sources):
            for expanded_source in _expand_source_token(source_token):
                if expanded_source and expanded_source != target:
                    corrections[expanded_source] = target

    return corrections


@lru_cache(maxsize=1)
def get_asr_corrections() -> dict[str, str]:
    corrections = load_asr_corrections()
    if corrections:
        logger.info(
            "Loaded {} ASR correction aliases from {}",
            len(corrections),
            ASR_CORRECTIONS_PATH,
        )
    return corrections


def normalize_asr_text(text: str, corrections: dict[str, str] | None = None) -> str:
    if not text:
        return text

    active_corrections = (
        corrections if corrections is not None else get_asr_corrections()
    )
    if not active_corrections:
        return text

    normalized = text
    for source, target in sorted(
        active_corrections.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        normalized = re.sub(
            re.escape(source),
            target,
            normalized,
            flags=re.IGNORECASE,
        )
    return normalized


class ASRTextNormalizer(ASRInterface):
    """Wrap an ASR engine and normalize text from a resource-backed correction table."""

    def __init__(self, wrapped: ASRInterface) -> None:
        self._wrapped = wrapped
        self.SAMPLE_RATE = getattr(wrapped, "SAMPLE_RATE", self.SAMPLE_RATE)
        self.NUM_CHANNELS = getattr(wrapped, "NUM_CHANNELS", self.NUM_CHANNELS)
        self.SAMPLE_WIDTH = getattr(wrapped, "SAMPLE_WIDTH", self.SAMPLE_WIDTH)

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)

    async def async_transcribe_np(self, audio: np.ndarray) -> str:
        text = await self._wrapped.async_transcribe_np(audio)
        normalized = normalize_asr_text(text)
        if normalized != text:
            logger.info("ASR text normalized: {!r} -> {!r}", text, normalized)
        return normalized

    def transcribe_np(self, audio: np.ndarray) -> str:
        text = self._wrapped.transcribe_np(audio)
        normalized = normalize_asr_text(text)
        if normalized != text:
            logger.info("ASR text normalized: {!r} -> {!r}", text, normalized)
        return normalized

    def nparray_to_audio_file(
        self,
        audio: np.ndarray,
        sample_rate: int,
        file_path: str,
    ) -> None:
        self._wrapped.nparray_to_audio_file(audio, sample_rate, file_path)
