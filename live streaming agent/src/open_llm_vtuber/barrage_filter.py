"""
barrage_filter.py - 弹幕过滤与去重系统

从 barrage_adapter.py 提取的过滤逻辑，并新增：
  - 纯 emoji 过滤
  - 基于字符 bigram Jaccard 的文本相似度去重
  - 回复频率限制器（输出侧，同一用户5分钟内最多被回复N条）
"""

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional
from loguru import logger

from .blocked_words_loader import find_blocked_words_in_text


# ============================================================
# Config（由 BarrageConfig 传入）
# ============================================================

@dataclass
class FilterConfig:
    min_content_length: int = 2
    max_content_length: int = 200
    dedup_window_seconds: float = 30.0
    max_per_user_in_window: int = 3
    blocked_words: list = field(default_factory=list)
    ignore_exact: list = field(default_factory=lambda: [
        "666", "1", "11", "111",
        "dd", "DD",
    ])

    # 文本相似度去重
    semantic_dedup_window: float = 180.0
    semantic_dedup_threshold: float = 0.55

    # 礼物过滤
    gift_trigger_enabled: bool = True
    gift_trigger_min_diamonds: int = 10000


# ============================================================
# Emoji 检测
# ============================================================

_EMOJI_PATTERN = re.compile(
    r'^['
    r'\U0001F600-\U0001F64F'   # emoticons
    r'\U0001F300-\U0001F5FF'   # symbols & pictographs
    r'\U0001F680-\U0001F6FF'   # transport & map
    r'\U0001F900-\U0001F9FF'   # supplemental symbols
    r'\U0001FA00-\U0001FA6F'   # chess symbols
    r'\U0001FA70-\U0001FAFF'   # symbols extended-A
    r'\U00002702-\U000027B0'   # dingbats
    r'\U0000FE00-\U0000FE0F'   # variation selectors
    r'\U0000200D'              # zero width joiner
    r'\U000020E3'              # combining enclosing keycap
    r'\s'
    r']+$',
    re.UNICODE,
)
_BRACKETED_CONTENT_PATTERN = re.compile(
    r"\[[^\[\]]*\]"
    r"|\{[^{}]*\}"
    r"|\([^()]*\)"
    r"|\uff08[^\uff08\uff09]*\uff09"
)


def is_pure_emoji(text: str) -> bool:
    return bool(_EMOJI_PATTERN.match(text))


def strip_barrage_bracketed_content(text: str) -> str:
    if not text:
        return text

    current = text
    while True:
        cleaned = _BRACKETED_CONTENT_PATTERN.sub("", current)
        if cleaned == current:
            return cleaned.strip()
        current = cleaned


# ============================================================
# 文本相似度去重（bigram Jaccard）
# ============================================================

class TextSimilarityDedup:
    """基于字符 bigram Jaccard 相似度的弹幕去重"""

    def __init__(self, window: float = 180.0, threshold: float = 0.55):
        self.window = window
        self.threshold = threshold
        self._recent: deque = deque()  # (timestamp, content, bigram_set)

    @staticmethod
    def _bigrams(text: str) -> set:
        text = text.strip().lower()
        if len(text) < 2:
            return {text}
        return {text[i:i+2] for i in range(len(text) - 1)}

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union else 0.0

    def _purge_expired(self, now: float):
        cutoff = now - self.window
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()

    def is_duplicate(self, content: str) -> bool:
        now = time.time()
        self._purge_expired(now)

        new_bigrams = self._bigrams(content)

        for _, old_content, old_bigrams in self._recent:
            sim = self._jaccard(new_bigrams, old_bigrams)
            if sim >= self.threshold:
                logger.debug(
                    f"[BarrageFilter] 文本相似度去重: "
                    f"'{content[:20]}' ≈ '{old_content[:20]}' (J={sim:.2f})"
                )
                return True

        self._recent.append((now, content, new_bigrams))
        return False

    def detect_all_same_topic(self) -> bool:
        """检测当前窗口内是否 >80% 弹幕在讨论同一话题"""
        now = time.time()
        self._purge_expired(now)

        if len(self._recent) < 5:
            return False

        items = list(self._recent)
        similar_count = 0
        total_pairs = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                total_pairs += 1
                sim = self._jaccard(items[i][2], items[j][2])
                if sim >= self.threshold:
                    similar_count += 1

        if total_pairs == 0:
            return False
        return (similar_count / total_pairs) > 0.8


# ============================================================
# 弹幕过滤器
# ============================================================

class BarrageFilter:
    """弹幕入队过滤链"""

    def __init__(self, config: FilterConfig):
        self.config = config
        self._user_history: Dict[str, deque] = {}
        self._content_cache: Dict[str, float] = {}
        self._semantic_dedup = TextSimilarityDedup(
            window=config.semantic_dedup_window,
            threshold=config.semantic_dedup_threshold,
        )

    def should_accept(self, msg_type: int, data: dict) -> bool:
        if msg_type == 5:
            return self._check_gift(data)
        if msg_type != 1:
            return False

        content = data.get("Content", "").strip()
        user = data.get("User") or {}
        user_id = user.get("SecUid", "") or user.get("DisplayId", "unknown")

        if len(content) < self.config.min_content_length:
            return False
        if len(content) > self.config.max_content_length:
            return False
        if content in self.config.ignore_exact:
            return False

        if is_pure_emoji(content):
            logger.debug(f"[BarrageFilter] 纯emoji过滤: {content}")
            return False

        blocked_words = find_blocked_words_in_text(
            content,
            blocked_words=self.config.blocked_words,
        )
        if blocked_words:
            logger.info(
                "[BarrageFilter] Blocked words filtered: "
                f"words={blocked_words!r} content={content!r}"
            )
            return False

        now = time.time()
        if not self._check_user_rate(user_id, now):
            return False
        if not self._check_content_dedup(content, now):
            return False
        if self._semantic_dedup.is_duplicate(content):
            return False

        return True

    def _check_gift(self, data: dict) -> bool:
        if not self.config.gift_trigger_enabled:
            return False
        diamonds = data.get("DiamondCount", 0)
        return diamonds >= self.config.gift_trigger_min_diamonds

    def _check_user_rate(self, user_id: str, now: float) -> bool:
        if user_id not in self._user_history:
            self._user_history[user_id] = deque()
        history = self._user_history[user_id]
        cutoff = now - self.config.dedup_window_seconds
        while history and history[0] < cutoff:
            history.popleft()
        if len(history) >= self.config.max_per_user_in_window:
            return False
        history.append(now)
        return True

    def _check_content_dedup(self, content: str, now: float) -> bool:
        if len(self._content_cache) > 1000:
            cutoff = now - self.config.dedup_window_seconds
            self._content_cache = {
                k: v for k, v in self._content_cache.items() if v > cutoff
            }
        content_key = content.strip().lower()
        last_time = self._content_cache.get(content_key, 0)
        if now - last_time < self.config.dedup_window_seconds:
            return False
        self._content_cache[content_key] = now
        return True


# ============================================================
# 回复频率限制器（输出侧）
# ============================================================

class ResponseRateLimiter:
    """同一用户在指定时间窗口内最多被回复 N 条"""

    def __init__(self, max_responses: int = 3, window: float = 300.0):
        self.max_responses = max_responses
        self.window = window
        self._history: Dict[str, deque] = {}

    def can_respond(self, user_id: str) -> bool:
        if not user_id:
            return True
        now = time.time()
        self._purge(user_id, now)
        history = self._history.get(user_id)
        if history and len(history) >= self.max_responses:
            return False
        return True

    def record_response(self, user_id: str):
        if not user_id:
            return
        now = time.time()
        self._purge(user_id, now)
        if user_id not in self._history:
            self._history[user_id] = deque()
        self._history[user_id].append(now)

    def _purge(self, user_id: str, now: float):
        if user_id not in self._history:
            return
        cutoff = now - self.window
        history = self._history[user_id]
        while history and history[0] < cutoff:
            history.popleft()


# ============================================================
# 全局单例
# ============================================================

_rate_limiter_instance: Optional[ResponseRateLimiter] = None


def init_response_rate_limiter(
    max_responses: int = 3, window: float = 300.0
) -> ResponseRateLimiter:
    global _rate_limiter_instance
    _rate_limiter_instance = ResponseRateLimiter(max_responses, window)
    return _rate_limiter_instance


def get_response_rate_limiter() -> Optional[ResponseRateLimiter]:
    return _rate_limiter_instance
