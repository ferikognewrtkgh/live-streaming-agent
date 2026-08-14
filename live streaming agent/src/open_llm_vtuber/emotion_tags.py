import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable


DEFAULT_GPT_SOVITS_SAMPLE_ROOT = Path(
    r"D:\projects\GSV-TTS-Lite-main\resource\GPT_SoVITS_sample"
)
GPT_SOVITS_SAMPLE_ROOT_ENV = "GPT_SOVITS_SAMPLE_ROOT"

LIVE2D_EMOTION_TAG_ALIASES = {
    "生气": "mad",
    "愤怒": "mad",
    "疑惑": "doubt",
    "困惑": "doubt",
    "开心": "happy",
    "高兴": "happy",
    "咧嘴笑": "happy",
    "害羞": "shy",
    "腮红": "shy",
    "腹黑": "black",
    "喜欢": "like",
    "星星眼": "like",
    "悲伤": "cry",
    "难过": "cry",
    "哭": "cry",
    "哭哭": "cry",
    "兴奋": "exciting",
    "谢礼物": "thanks",
    "醒来": "wake",
}

TTS_EMOTION_TAG_ALIASES = {
    "default": "default",
    "normal": "default",
    "默认": "default",
    "happy": "开心",
    "joy": "开心",
    "开心": "开心",
    "高兴": "开心",
    "咧嘴笑": "开心",
    "doubt": "疑惑",
    "confused": "疑惑",
    "疑惑": "疑惑",
    "困惑": "疑惑",
    "cry": "伤心",
    "sad": "伤心",
    "悲伤": "伤心",
    "难过": "伤心",
    "哭": "伤心",
    "哭哭": "伤心",
    "speechless": "无语",
    "无语": "无语",
    "cute": "装可爱",
    "装可爱": "装可爱",
    "thanks": "谢礼物",
    "gift": "谢礼物",
    "谢礼物": "谢礼物",
    "exciting": "兴奋",
    "excited": "兴奋",
    "兴奋": "兴奋",
    "sleepy": "困倦",
    "sleep": "困倦",
    "困倦": "困倦",
    "wake": "醒来",
    "醒来": "醒来",
    "shy": "别扭",
    "害羞": "别扭",
    "腮红": "别扭",
    "别扭": "别扭",
    "like": "满足",
    "喜欢": "满足",
    "满足": "满足",
    "mad": "赌气",
    "angry": "赌气",
    "生气": "赌气",
    "愤怒": "赌气",
    "赌气": "赌气",
    "disgust": "厌恶",
    "厌恶": "厌恶",
    "embarrassed": "尴尬",
    "awkward": "尴尬",
    "尴尬": "尴尬",
}

PREFERRED_DEFAULT_TTS_EMOTION_TAG = "default"

BRACKET_TAG_RE = re.compile(r"\[([^\[\]]{1,40})\]")


def _clean_tag(value: object) -> str:
    return str(value or "").strip().strip("[]").strip()


def normalize_live2d_emotion_tag(value: object) -> str:
    tag = _clean_tag(value).casefold()
    return LIVE2D_EMOTION_TAG_ALIASES.get(tag, tag)


def _configured_tts_sample_root() -> Path:
    configured = os.getenv(GPT_SOVITS_SAMPLE_ROOT_ENV)
    if configured:
        return Path(configured)
    return DEFAULT_GPT_SOVITS_SAMPLE_ROOT


@lru_cache(maxsize=4)
def load_tts_emotion_tags(sample_root: str | None = None) -> dict[str, str]:
    root = Path(sample_root) if sample_root else _configured_tts_sample_root()
    tags: dict[str, str] = {}
    if root.exists():
        for child in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
            if child.is_dir():
                name = child.name.strip()
                if name:
                    tags[name.casefold()] = name
    return tags


def normalize_tts_emotion_tag(value: object) -> str:
    tag = _clean_tag(value)
    if not tag:
        return ""

    tts_tags = load_tts_emotion_tags()
    canonical = tts_tags.get(tag.casefold())
    if canonical:
        return canonical

    mapped = TTS_EMOTION_TAG_ALIASES.get(tag.casefold())
    if mapped:
        canonical = tts_tags.get(mapped.casefold())
        if canonical:
            return canonical

    live2d_tag = normalize_live2d_emotion_tag(tag)
    if live2d_tag != tag.casefold():
        mapped = TTS_EMOTION_TAG_ALIASES.get(live2d_tag.casefold())
        if mapped:
            canonical = tts_tags.get(mapped.casefold())
            if canonical:
                return canonical

    return ""


def default_tts_emotion_tag(sample_root: str | None = None) -> str:
    tts_tags = load_tts_emotion_tags(sample_root)
    preferred = tts_tags.get(PREFERRED_DEFAULT_TTS_EMOTION_TAG.casefold())
    if preferred:
        return preferred
    if not tts_tags:
        return ""
    return sorted(tts_tags.values(), key=str.casefold)[0]


def resolve_tts_emotion_tag(
    value: object,
    fallback: object = PREFERRED_DEFAULT_TTS_EMOTION_TAG,
) -> str:
    normalized = normalize_tts_emotion_tag(value)
    if normalized:
        return normalized

    normalized_fallback = normalize_tts_emotion_tag(fallback)
    if normalized_fallback:
        return normalized_fallback

    return default_tts_emotion_tag()


def extract_emotion_tags_from_text(
    text: str,
    live2d_tags: Iterable[str] | None = None,
) -> list[str]:
    live2d_tag_set = {
        normalize_live2d_emotion_tag(tag) for tag in (live2d_tags or [])
    }
    emotions: list[str] = []

    for match in BRACKET_TAG_RE.finditer(str(text or "")):
        raw_tag = _clean_tag(match.group(1))
        if not raw_tag:
            continue

        tts_tag = normalize_tts_emotion_tag(raw_tag)
        live2d_tag = normalize_live2d_emotion_tag(raw_tag)
        if tts_tag:
            emotion = tts_tag
        elif live2d_tag in live2d_tag_set:
            emotion = live2d_tag
        else:
            continue

        if emotion not in emotions:
            emotions.append(emotion)

    return emotions
