"""
barrage_adapter.py - Douyin barrage adapter

Connects to DouyinBarrageGrab WebSocket, receives live room barrages,
converts them to text-input messages, and injects them into the
conversation pipeline via Orchestrator (if available) or directly.

队列架构:
  gift_queue             — 礼物（开启感谢时按金额重排一次，之后 FIFO）
  barrage_queue_keyword  — 包含关键词的弹幕（不丢弃、不过期）
  barrage_queue_high     — 灯牌 >= 16 的弹幕
  barrage_queue_normal   — 灯牌 < 16 的弹幕

优先级（弹幕模式）: 主播 > 礼物 > 关键词弹幕 > 高级/普通弹幕

模式切换:
  BARRAGE 模式 — 优先消费关键词队列，再双取高/普通弹幕
  GIFT 模式    — 处理礼物，弹幕暂停，最多连续处理2条
"""

import asyncio
import bisect
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Callable, List, Any
from collections import deque
from loguru import logger

from .blocked_words_loader import BLOCKED_WORDS
from .barrage_filter import (
    BarrageFilter,
    FilterConfig,
    strip_barrage_bracketed_content,
)
from .douyin_link_payload import extract_link_anchor_candidates_from_payload_base64
from .orchestrator import get_orchestrator, OrchestratorMessage, MsgPriority, MsgSource


# ============================================================
# Config
# ============================================================

BARRAGE_EVENT_LOG_ROOT = Path("logs") / "barrage"
BARRAGE_AVATAR_CACHE_ROOT = BARRAGE_EVENT_LOG_ROOT / "avatars"
BARRAGE_RAW_USER_DEBUG_ROOT = BARRAGE_EVENT_LOG_ROOT / "raw_user"
BARRAGE_LINK_PROBE_ROOT = BARRAGE_EVENT_LOG_ROOT / "link_probe"
BARRAGE_AVATAR_MAX_BYTES = 5 * 1024 * 1024
BARRAGE_AVATAR_DOWNLOAD_TIMEOUT = 6.0
BARRAGE_EVENT_LOG_FILE_PREFIXES = {
    "barrage": "barrage",
    "gift": "gift",
    "room_stats": "room_stats",
}
LINK_ANCHOR_PATH_HINTS = (
    "link",
    "linked",
    "cohost",
    "co_host",
    "guest",
    "rival",
    "connect",
    "connection",
    "linkmic",
    "link_mic",
    "linker",
    "battle",
    "rival_anchor",
    "guest_anchor",
    "co_anchor",
    "opponent",
    "pk_user",
    "pk_info",
    "mic",
    "pk",
    "\u540c\u5c40",
    "\u5bf9\u6218",
    "\u5bf9\u624b",
    "\u9ea6\u4f4d",
    "\u8fde\u7ebf",
    "\u8fde\u9ea6",
    "\u5609\u5bbe",
    "\u5bf9\u65b9",
)
LINK_ANCHOR_NAME_KEYS = (
    "Nickname",
    "NickName",
    "nickname",
    "Name",
    "name",
    "DisplayName",
    "display_name",
    "DisplayId",
    "DisplayID",
    "display_id",
    "ScreenName",
    "screen_name",
    "Nick",
    "nick",
    "nick_name",
    "open_id",
    "user_id",
    "sec_uid",
)
LINK_ANCHOR_REJECT_NAMES = {
    "",
    "unknown",
    "viewer",
    "\u4e3b\u64ad",
    "\u8fde\u7ebf\u4e3b\u64ad",
    "\u6296\u97f3",
    "\u6296\u97f3\u76f4\u64ad",
    "\u6296\u97f3\u76f4\u64ad\u4f34\u4fa3",
    "\u6211\u65b9\u8d21\u732e\u699c",
    "pk\u8fde\u7ebf",
}
BARRAGE_LINK_PROBE_MAX_DEPTH = 8
BARRAGE_LINK_PROBE_MAX_DICT_KEYS = 80
BARRAGE_LINK_PROBE_MAX_LIST_ITEMS = 30
BARRAGE_LINK_PROBE_MAX_STRING_CHARS = 300
BARRAGE_LINK_PROBE_MAX_HINT_PATHS = 120
BARRAGE_LINK_PROBE_MAX_CANDIDATES = 20


@dataclass
class CustomVariableConfig:
    """自定义筛选/排序变量配置。

    三个内置变量: wealth (财富等级 / PayLevel)
                fan_badge (粉丝牌等级 / FansClub.Level)
                diamond_rank (本场钻石数累计排行)

    - enabled: 是否启用该变量
    - threshold: 筛选门槛, value >= threshold 才能通过 (0 = 不过滤)
    - priority: 数值越小优先级越高 (用于排序键的字段顺序)
    """
    name: str
    enabled: bool = False
    threshold: int = 0
    priority: int = 0


@dataclass
class BarrageConfig:
    """Barrage adapter configuration"""

    # DouyinBarrageGrab WebSocket URL
    ws_url: str = "ws://127.0.0.1:8888"

    # Reconnect interval (seconds)
    reconnect_interval: float = 3.0

    # Max reconnect attempts (0 = infinite)
    max_reconnect_attempts: int = 0

    # ---------- Filter rules ----------
    min_content_length: int = 2
    max_content_length: int = 200
    dedup_window_seconds: float = 30.0
    max_per_user_in_window: int = 3
    blocked_words: list = field(default_factory=lambda: list(BLOCKED_WORDS))
    ignore_exact: list = field(default_factory=lambda: [
        "666", "1", "11", "111",
        "dd", "DD",
    ])

    # ---------- 文本相似度去重 ----------
    semantic_dedup_window: float = 180.0
    semantic_dedup_threshold: float = 0.55

    # ---------- Message queue ----------
    max_queue_size: int = 50
    consume_interval: float = 5.0
    speech_interval_seconds: float = 2.0
    stale_message_max_age: float = 10.0
    high_fan_percent: int = 10
    high_fan_default_level: int = 16

    # ---------- Gift ----------
    gift_trigger_enabled: bool = True
    gift_trigger_min_diamonds: int = 10000
    gift_thanks_enabled: bool = False
    gift_pending_ttl_when_disabled: float = 20.0
    gift_max_consecutive: int = 2
    gift_dedup_window: float = 30.0

    # ---------- Keyword queue ----------
    # 包含这些关键词的弹幕会进入关键词队列（不区分大小写）
    keyword_list: list = field(default_factory=lambda: ["主播", "zhubo"])

    # ---------- Barrage prefix ----------
    prefix_template: str = "[barrage] {nickname}: {content}"

    # ---------- 回复频率限制 ----------
    response_rate_limit_count: int = 3
    response_rate_limit_window: float = 300.0

    # ---------- 连接健康监控 ----------
    # 首次连接后等待第一条消息的超时 (秒)
    # DouyinBarrageGrab 切换直播间后可能需要较长时间才能收到第一条弹幕
    initial_wait_timeout: float = 300.0
    # 已收到过消息后，连续多久没收到新消息则判定为"假活"并强制重连 (秒)
    stale_connection_timeout: float = 180.0
    # 健康监控日志间隔 (秒)
    health_log_interval: float = 30.0

    # ---------- DouyinBarrageGrab 进程监控 ----------
    # 进程可执行文件路径 (设为空字符串禁用进程监控)
    grab_exe_path: str = ""
    # 进程名 (用于检测进程是否存活)
    grab_process_name: str = "WssBarrageServer"
    # 进程崩溃后自动重启 (需要设置 grab_exe_path)
    grab_auto_restart: bool = False
    # 进程健康检查间隔 (秒)
    grab_check_interval: float = 30.0
    # 长时间没有 DouyinBarrageGrab 心跳/消息时，是否重启抓取工具
    grab_restart_on_stale: bool = False
    # 触发抓取工具重启的静默时间 (秒)。正常 Type=6 统计心跳约 10s 一次。
    grab_stale_restart_seconds: float = 180.0
    # 抓取工具自动重启冷却时间 (秒)，避免直播间未打开时反复重启
    grab_restart_cooldown_seconds: float = 180.0
    # 抓取工具启动后等待多久再重连本地 ws
    grab_startup_wait_seconds: float = 5.0

    # ---------- 自定义筛选/排序 (阶段4) ----------
    # 三个独立变量: wealth / fan_badge / diamond_rank
    # 任一 enabled=True 时启用自定义路径; 全部关闭时回退到原有逻辑.
    custom_variables: Dict[str, CustomVariableConfig] = field(
        default_factory=lambda: {
            "wealth": CustomVariableConfig("wealth", priority=0),
            "fan_badge": CustomVariableConfig("fan_badge", priority=1),
            "diamond_rank": CustomVariableConfig("diamond_rank", priority=2),
        }
    )
    # 自定义模式下每条弹幕的存活时间 (秒), 超时自动丢弃
    custom_item_ttl: float = 10.0
    # 自定义模式下回复完成后等待间隔 (秒)
    custom_consume_interval: float = 3.0
    # 自定义队列单队列最大长度 (满时尾部裁剪低优先级)
    custom_queue_max_size: int = 50

    # ---------- B2: 钻石累计 LRU 上限 ----------
    diamond_max_users: int = 10000

    # ---------- B1: 礼物连击合并 ----------
    # 同用户同礼物 N 秒内推送的多条 Type=5 合并为一条 (DiamondCount 累加)
    gift_combo_window: float = 3.0
    # 合并后的礼物消息在缓冲区最长等待时间 (秒), 超过即使还在连击也强制入队
    gift_combo_max_wait: float = 8.0

    # ---------- B5: 假活检测细化 ----------
    # 任意消息静默告警门槛 (秒); 收到 Type=6 直播间统计也算重置
    stale_warn_seconds: float = 30.0
    # 收到 Type=6 直播间统计心跳的最大间隔 (秒), 超过则强制重连
    # DouyinBarrageGrab 每 10s 左右推一次 Type=6, 60s 是非常宽容的阈值
    stale_force_reconnect_seconds: float = 60.0

    # ---------- B3: 进直播间/关注/点赞 ----------
    # 进直播间(Type=3)欢迎语
    welcome_enabled: bool = False
    welcome_batch_window: float = 5.0   # N 秒内的进直播间合并成一条
    welcome_max_batch: int = 5
    welcome_template: str = "欢迎 {names} 来到直播间!"

    # 关注事件(Type=4)
    follow_enabled: bool = False
    follow_batch_window: float = 5.0
    follow_max_batch: int = 3
    follow_template: str = "感谢 {names} 的关注!"

    # 点赞(Type=3): 只计数, 不触发回复
    like_count_track: bool = True

    # ---------- B9: metrics 窗口 ----------
    metrics_window_seconds: float = 60.0


# ============================================================
# Priority
# ============================================================

class Priority:
    """Message priority (lower number = higher priority)"""
    GIFT_BIG = 0
    FANS_HIGH = 1       # fan badge level >= 16
    FANS_MID = 2        # fan badge level >= 10
    FANS_LOW = 3        # fan badge level >= 1
    NORMAL = 4


# ============================================================
# Mode
# ============================================================

class AdapterMode(str, Enum):
    BARRAGE = "barrage"
    GIFT = "gift"


# ============================================================
# Message
# ============================================================

@dataclass(order=True)
class BarrageMessage:
    """Barrage message with priority for queue ordering"""
    priority: int
    timestamp: float = field(compare=False)
    nickname: str = field(compare=False, default="")
    user_id: str = field(compare=False, default="")
    avatar_url: str = field(compare=False, default="")
    avatar_path: str = field(compare=False, default="")
    content: str = field(compare=False, default="")
    raw_data: dict = field(compare=False, default_factory=dict)
    # ---- 自定义筛选/排序 (阶段4) ----
    wealth_level: int = field(compare=False, default=0)
    fan_badge_level: int = field(compare=False, default=0)
    session_diamond_total: int = field(compare=False, default=0)


# ============================================================
# Barrage adapter
# ============================================================

class BarrageAdapter:
    """Douyin barrage adapter with separate gift and barrage queues"""

    def __init__(self, config: Optional[BarrageConfig] = None):
        self.config = config or BarrageConfig()

        # Build filter config from barrage config
        filter_cfg = FilterConfig(
            min_content_length=self.config.min_content_length,
            max_content_length=self.config.max_content_length,
            dedup_window_seconds=self.config.dedup_window_seconds,
            max_per_user_in_window=self.config.max_per_user_in_window,
            blocked_words=self.config.blocked_words,
            ignore_exact=self.config.ignore_exact,
            semantic_dedup_window=self.config.semantic_dedup_window,
            semantic_dedup_threshold=self.config.semantic_dedup_threshold,
            gift_trigger_enabled=self.config.gift_trigger_enabled,
            gift_trigger_min_diamonds=self.config.gift_trigger_min_diamonds,
        )
        self.filter = BarrageFilter(filter_cfg)

        # 弹幕双队列
        self.barrage_queue_high: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=self.config.max_queue_size
        )
        self.barrage_queue_normal: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=self.config.max_queue_size
        )
        # 关键词弹幕队列（无上限、不丢弃）
        self.barrage_queue_keyword: deque = deque()
        # 预编译关键词（小写）供匹配
        self._keywords_lower = [kw.lower() for kw in self.config.keyword_list]
        # 礼物独立队列（FIFO）
        self.gift_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.max_queue_size
        )

        # 模式切换
        self._mode: AdapterMode = AdapterMode.BARRAGE

        # 礼物去重与节流
        self._gift_dedup: Dict[str, float] = {}
        self._consecutive_gift_count: int = 0

        self._running = False
        self._ws = None
        self._ws_handler = None
        self._inject_callback: Optional[Callable] = None
        self._tasks: list = []
        self._last_conversation_done_monotonic: float | None = None

        # 连接健康监控
        self._last_message_time: float = 0.0  # 最后一次收到消息的时间
        self._total_received: int = 0          # 累计收到的原始消息数
        self._reconnect_count: int = 0         # 累计重连次数
        self._grab_process = None              # DouyinBarrageGrab 子进程引用
        self._force_reconnect: bool = False    # 手动触发重连标志
        self._last_grab_restart_time: float = 0.0
        self._stale_grab_restart_count: int = 0

        # ---- 自定义筛选/排序 (阶段4) ----
        # 本场钻石数累计 (LRU OrderedDict, 超出 diamond_max_users 时淘汰最老的)
        from collections import OrderedDict
        self._session_diamond_totals: "OrderedDict[str, int]" = OrderedDict()
        # user_id → 最近一次的昵称 (查询排行时人类可读)
        self._session_diamond_nicks: Dict[str, str] = {}
        # LRU 淘汰计数 (用于诊断)
        self._diamond_evicted_count: int = 0
        # 自定义模式下的有序队列 (按 sort_key 升序; pop(0) 取最高)
        self._custom_keyword_queue: List[BarrageMessage] = []
        self._custom_normal_queue: List[BarrageMessage] = []
        # TTL 丢弃计数 (用于诊断)
        self._custom_dropped_count: int = 0
        # 自定义筛选过滤掉的弹幕计数
        self._custom_filtered_count: int = 0

        # ---- B1: 礼物连击合并缓冲 ----
        # key = f"{user_id}:{gift_name}", value = {"msg": BarrageMessage,
        #   "first_seen": float, "last_seen": float}
        self._gift_combo_buffer: Dict[str, dict] = {}

        # ---- B3: 进直播间/关注 聚合缓冲 ----
        # list of (timestamp, user_id, nickname)
        self._pending_joins: List[tuple] = []
        self._pending_follows: List[tuple] = []
        # 已经在缓冲区里的用户去重, 防止同一用户重复刷
        self._joins_dedup_keys: set = set()
        self._follows_dedup_keys: set = set()
        # 点赞累计 (Type=3, 仅计数)
        self._like_count: int = 0
        self._fan_badge_levels_by_user: Dict[str, int] = {}
        # Type=6 直播间统计最近值
        self._last_room_stats: dict = {}
        self._last_link_anchor_candidate: dict[str, Any] = {}
        # 收到 Type=6 的时间 (用于 B5 假活检测)
        self._last_heartbeat_time: float = 0.0

        # ---- B9: metrics 滑动窗口 ----
        # 每个事件类型一个 deque, 存时间戳; 计算时窗口外的会被剔除
        self._metrics_recv: deque = deque()
        self._metrics_enqueue: deque = deque()
        self._metrics_consume: deque = deque()
        self._metrics_filter_drop: deque = deque()
        self._metrics_combo_merged: deque = deque()
        self._avatar_path_cache: Dict[str, str] = {}
        self._raw_user_debug_signatures: set[str] = set()

    def set_inject_callback(self, callback: Optional[Callable]):
        self._inject_callback = callback

    async def start(self):
        if self._running:
            logger.warning("BarrageAdapter already running")
            return
        self._running = True
        self._last_message_time = time.time()
        logger.info(f"[barrage] starting adapter, connecting to: {self.config.ws_url}")
        self._tasks = [
            asyncio.create_task(self._receiver_loop()),
            asyncio.create_task(self._consumer_loop()),
            asyncio.create_task(self._health_monitor_loop()),
            asyncio.create_task(self._gift_combo_flusher_loop()),
            asyncio.create_task(self._event_batch_flusher_loop()),
        ]
        # 如果配置了 DouyinBarrageGrab 路径，启动进程监控
        if self.config.grab_exe_path:
            self._tasks.append(
                asyncio.create_task(self._grab_process_monitor())
            )

    async def stop(self):
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._ws:
            await self._ws.close()
        logger.info("[barrage] adapter stopped")

    # -------------------- Receiver loop --------------------

    async def _receiver_loop(self):
        attempt = 0
        while self._running:
            try:
                import websockets
                async with websockets.connect(
                    self.config.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    attempt = 0
                    self._last_message_time = time.time()
                    self._reconnect_count += 1
                    got_first_message = False
                    logger.info(
                        f"[barrage] connected to DouyinBarrageGrab: "
                        f"{self.config.ws_url} (连接次数: {self._reconnect_count})"
                    )

                    while True:
                        if not self._running:
                            break
                        # 手动触发重连
                        if self._force_reconnect:
                            self._force_reconnect = False
                            logger.info("[barrage] 收到手动重连指令，断开当前连接")
                            break

                        # B5: 细化超时策略
                        #  - 首次连接: initial_wait_timeout (默认300秒)
                        #  - 已收到过消息: 用更短的 stale_warn_seconds (默认30秒) 做 recv 超时
                        #    每次超时不直接断, 而是检查 Type=6 直播间统计心跳:
                        #    - 距上次 Type=6 < stale_force_reconnect_seconds (60s) → 继续等
                        #    - 否则真的判定假活 → 强制断开重连
                        if got_first_message:
                            timeout = self.config.stale_warn_seconds
                        else:
                            timeout = self.config.initial_wait_timeout

                        try:
                            raw_msg = await asyncio.wait_for(
                                ws.recv(), timeout=timeout
                            )
                            self._last_message_time = time.time()
                            self._total_received += 1
                            if not got_first_message:
                                got_first_message = True
                                logger.info(
                                    "[barrage] ✅ 收到第一条消息，连接正常工作"
                                )
                            await self._process_raw_message(raw_msg)
                        except asyncio.TimeoutError:
                            now = time.time()
                            silent_secs = now - self._last_message_time
                            if not got_first_message:
                                logger.warning(
                                    f"[barrage] ⚠️  等待 {silent_secs:.0f}s 仍未收到任何消息, "
                                    f"尝试重连 (可能需要检查弹幕抓取工具或直播间状态)"
                                )
                                break

                            # B5: got_first_message 后用 Type=6 心跳判断
                            heartbeat_silent = (
                                now - self._last_heartbeat_time
                                if self._last_heartbeat_time
                                else silent_secs
                            )
                            if heartbeat_silent < self.config.stale_force_reconnect_seconds:
                                # 仍在心跳窗口内, 只 WARN 不重连
                                logger.warning(
                                    f"[barrage] ⚠️  {silent_secs:.0f}s 无新消息 "
                                    f"(心跳静默 {heartbeat_silent:.0f}s < "
                                    f"{self.config.stale_force_reconnect_seconds:.0f}s), 继续等待"
                                )
                                continue
                            logger.warning(
                                f"[barrage] ⚠️  连接假活: {silent_secs:.0f}s 无消息 + "
                                f"心跳 (Type=6) 已静默 {heartbeat_silent:.0f}s, 强制重连"
                            )
                            await self._maybe_restart_grab_for_stale(
                                reason="stale-heartbeat",
                                silent_secs=silent_secs,
                                heartbeat_silent=heartbeat_silent,
                            )
                            break
                        except Exception as e:
                            logger.warning(f"[barrage] receive error: {e}")
                            break

            except ImportError:
                logger.error(
                    "[barrage] missing websockets library, run: pip install websockets"
                )
                self._running = False
                return
            except Exception as e:
                attempt += 1
                if (
                    self.config.max_reconnect_attempts > 0
                    and attempt >= self.config.max_reconnect_attempts
                ):
                    logger.error(f"[barrage] max reconnect attempts reached ({attempt}), stopping")
                    self._running = False
                    return
                # 指数退避: 3s → 6s → 12s → 24s → 30s (上限)
                if (
                    self.config.grab_auto_restart
                    and self.config.grab_exe_path
                    and attempt >= 2
                    and self._grab_restart_cooldown_elapsed()
                ):
                    logger.warning(
                        "[barrage] local DouyinBarrageGrab ws connect failed "
                        "{} times ({}), restarting grabber",
                        attempt,
                        e,
                    )
                    restarted = await self._restart_grab_process(
                        reason="local-ws-connect-failed"
                    )
                    if restarted:
                        attempt = 0
                        continue

                backoff = min(
                    self.config.reconnect_interval * (2 ** (attempt - 1)),
                    30.0,
                )
                logger.info(
                    f"[barrage] 连接断开 ({e}), {backoff:.0f}s 后重连 "
                    f"(尝试 #{attempt})"
                )
                await asyncio.sleep(backoff)

    def _barrage_event_log_path(self, now: datetime, event_type: str) -> Path:
        date_str = now.strftime("%Y-%m-%d")
        file_prefix = BARRAGE_EVENT_LOG_FILE_PREFIXES.get(
            event_type,
            "unknown",
        )
        return (
            BARRAGE_EVENT_LOG_ROOT
            / date_str
            / f"{file_prefix}_{date_str}.jsonl"
        )

    @staticmethod
    def _first_non_empty(*values: Any) -> Any:
        for value in values:
            if value not in (None, ""):
                return value
        return None

    @staticmethod
    def _int_or_zero(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _dict_sources(source: dict, *container_keys: str) -> list[dict]:
        sources = [source]
        for key in container_keys:
            nested = source.get(key)
            if isinstance(nested, dict):
                sources.append(nested)
        return sources

    def _first_field(self, sources: list[dict], *keys: str) -> Any:
        for source in sources:
            if not isinstance(source, dict):
                continue
            for key in keys:
                value = source.get(key)
                if value not in (None, ""):
                    return value
        return None

    @staticmethod
    def _normalize_link_anchor_name(value: Any) -> str | None:
        if value is None:
            return None
        name = str(value).strip()
        if not name:
            return None
        name = re.sub(r"[\r\n\t]+", " ", name)
        name = re.sub(r"\s+", " ", name).strip(" @:：|-_/\\")
        name = re.sub(r"(\u7684)?\u76f4\u64ad\u95f4$", "", name).strip()
        name = re.sub(r"(\u6b63\u5728)?\u76f4\u64ad(\u4e2d)?$", "", name).strip()
        if not name:
            return None
        lowered = name.lower()
        if lowered in LINK_ANCHOR_REJECT_NAMES:
            return None
        if "://" in lowered or lowered.startswith("ws"):
            return None
        if len(name) < 2 and not re.fullmatch(r"[\u4e00-\u9fff]", name):
            return None
        if len(name) > 32:
            return None
        if re.fullmatch(r"\d{5,}", name):
            return None
        return name

    @staticmethod
    def _path_has_link_anchor_hint(path: tuple[str, ...]) -> bool:
        joined = ".".join(path).lower()
        return any(hint in joined for hint in LINK_ANCHOR_PATH_HINTS)

    @staticmethod
    def _text_has_link_probe_hint(value: Any) -> bool:
        text = str(value or "").lower()
        return bool(text) and any(hint in text for hint in LINK_ANCHOR_PATH_HINTS)

    def _truncate_link_probe_value(self, value: Any, depth: int = 0) -> Any:
        if depth > BARRAGE_LINK_PROBE_MAX_DEPTH:
            return "<max-depth>"

        if isinstance(value, dict):
            result: dict[str, Any] = {}
            items = list(value.items())
            for key, nested in items[:BARRAGE_LINK_PROBE_MAX_DICT_KEYS]:
                result[str(key)] = self._truncate_link_probe_value(nested, depth + 1)
            if len(items) > BARRAGE_LINK_PROBE_MAX_DICT_KEYS:
                result["<truncated_keys>"] = len(items) - BARRAGE_LINK_PROBE_MAX_DICT_KEYS
            return result

        if isinstance(value, list):
            result = [
                self._truncate_link_probe_value(item, depth + 1)
                for item in value[:BARRAGE_LINK_PROBE_MAX_LIST_ITEMS]
            ]
            if len(value) > BARRAGE_LINK_PROBE_MAX_LIST_ITEMS:
                result.append(f"<truncated_items:{len(value) - BARRAGE_LINK_PROBE_MAX_LIST_ITEMS}>")
            return result

        if isinstance(value, str):
            text = re.sub(r"[\r\n\t]+", " ", value)
            if len(text) > BARRAGE_LINK_PROBE_MAX_STRING_CHARS:
                return text[:BARRAGE_LINK_PROBE_MAX_STRING_CHARS] + "...<truncated>"
            return text

        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return str(value)

    def _collect_link_probe_hint_paths(
        self,
        value: Any,
        path: tuple[str, ...] = (),
        depth: int = 0,
        found: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        if found is None:
            found = []
        if len(found) >= BARRAGE_LINK_PROBE_MAX_HINT_PATHS:
            return found
        if depth > BARRAGE_LINK_PROBE_MAX_DEPTH:
            return found

        path_text = ".".join(path)
        path_hint = bool(path_text) and self._text_has_link_probe_hint(path_text)
        value_hint = (
            isinstance(value, str)
            and self._text_has_link_probe_hint(value)
        )
        if path and (path_hint or value_hint):
            found.append(
                {
                    "path": path_text,
                    "match": "path" if path_hint else "value",
                    "value": self._truncate_link_probe_value(value),
                }
            )

        if isinstance(value, dict):
            for key, nested in list(value.items())[:BARRAGE_LINK_PROBE_MAX_DICT_KEYS]:
                self._collect_link_probe_hint_paths(
                    nested,
                    (*path, str(key)),
                    depth + 1,
                    found,
                )
                if len(found) >= BARRAGE_LINK_PROBE_MAX_HINT_PATHS:
                    break
        elif isinstance(value, list):
            for index, item in enumerate(value[:BARRAGE_LINK_PROBE_MAX_LIST_ITEMS]):
                self._collect_link_probe_hint_paths(
                    item,
                    (*path, str(index)),
                    depth + 1,
                    found,
                )
                if len(found) >= BARRAGE_LINK_PROBE_MAX_HINT_PATHS:
                    break

        return found

    def _iter_link_anchor_candidates(
        self,
        value: Any,
        path: tuple[str, ...] = (),
        depth: int = 0,
        hinted: bool = False,
    ):
        if depth > 8:
            return

        current_hinted = hinted or self._path_has_link_anchor_hint(path)
        if isinstance(value, dict):
            if current_hinted:
                for key in LINK_ANCHOR_NAME_KEYS:
                    candidate = self._normalize_link_anchor_name(value.get(key))
                    if candidate:
                        yield {
                            "name": candidate,
                            "source": "barrage_structured",
                            "path": ".".join((*path, key)),
                            "confidence": 0.86,
                        }

                for key in ("User", "user", "Anchor", "anchor", "Guest", "guest"):
                    nested = value.get(key)
                    if isinstance(nested, dict):
                        for name_key in LINK_ANCHOR_NAME_KEYS:
                            candidate = self._normalize_link_anchor_name(
                                nested.get(name_key)
                            )
                            if candidate:
                                yield {
                                    "name": candidate,
                                    "source": "barrage_structured",
                                    "path": ".".join((*path, key, name_key)),
                                    "confidence": 0.88,
                                }

            for key, nested in value.items():
                next_path = (*path, str(key))
                yield from self._iter_link_anchor_candidates(
                    nested,
                    next_path,
                    depth + 1,
                    current_hinted,
                )
            return

        if isinstance(value, list):
            for index, item in enumerate(value[:20]):
                yield from self._iter_link_anchor_candidates(
                    item,
                    (*path, str(index)),
                    depth + 1,
                    current_hinted,
                )

    def _iter_raw_link_payload_candidates(self, data: dict):
        payload_base64 = data.get("PayloadBase64") or data.get("payload_base64")
        if not payload_base64:
            return

        method = str(data.get("Method") or data.get("method") or "")
        for candidate in extract_link_anchor_candidates_from_payload_base64(
            payload_base64,
            method=method,
        ):
            name = self._normalize_link_anchor_name(candidate.get("name"))
            if not name:
                continue
            result = dict(candidate)
            result["name"] = name
            result["path"] = f"data.PayloadBase64.{method}.{result.get('path')}"
            yield result

    def _update_link_anchor_candidate(self, msg: dict, data: dict) -> None:
        own_anchor_name = self._normalize_link_anchor_name(
            (
                (self._last_room_stats.get("room") or {})
                .get("anchor", {})
                .get("nickname")
            )
        )
        candidates = [
            *list(self._iter_raw_link_payload_candidates(data)),
            *list(self._iter_link_anchor_candidates({"message": msg, "data": data})),
        ]
        for candidate in candidates:
            name = candidate.get("name")
            if not name:
                continue
            if candidate.get("is_host"):
                continue
            if own_anchor_name and name == own_anchor_name:
                continue
            candidate["updated_at"] = time.time()
            self._last_link_anchor_candidate = candidate
            logger.info(
                "[barrage] detected link anchor candidate name={} path={}",
                candidate.get("name"),
                candidate.get("path"),
            )
            return

    def get_link_anchor_candidate(self, max_age_seconds: float = 300.0) -> dict[str, Any]:
        candidate = dict(self._last_link_anchor_candidate or {})
        if not candidate:
            return {
                "found": False,
                "reason": "no live link anchor candidate yet",
            }

        updated_at = float(candidate.get("updated_at") or 0.0)
        age_seconds = time.time() - updated_at if updated_at else None
        if age_seconds is not None and age_seconds > max_age_seconds:
            return {
                "found": False,
                "reason": "live link anchor candidate expired",
                "age_seconds": round(age_seconds, 1),
            }

        candidate["found"] = True
        if age_seconds is not None:
            candidate["age_seconds"] = round(age_seconds, 1)
        return candidate

    @classmethod
    def _first_url_in_value(cls, value: Any, depth: int = 0) -> str:
        """Return the first http(s) URL inside a Douyin avatar-like value."""
        if depth > 5:
            return ""
        if isinstance(value, str):
            text = value.strip()
            if text.startswith(("http://", "https://")):
                return text
            return ""
        if isinstance(value, (list, tuple)):
            for item in value:
                url = cls._first_url_in_value(item, depth + 1)
                if url:
                    return url
            return ""
        if isinstance(value, dict):
            preferred_keys = (
                "UrlList",
                "url_list",
                "URLList",
                "urlList",
                "Urls",
                "urls",
                "Url",
                "url",
                "URL",
                "DownloadUrl",
                "download_url",
            )
            for key in preferred_keys:
                if key in value:
                    url = cls._first_url_in_value(value.get(key), depth + 1)
                    if url:
                        return url
            for nested in value.values():
                url = cls._first_url_in_value(nested, depth + 1)
                if url:
                    return url
        return ""

    def _avatar_url_from_user(self, user: dict) -> str:
        """Extract a displayable avatar URL from DouyinBarrageGrab user data."""
        if not isinstance(user, dict):
            return ""

        avatar_keys = (
            "HeadImgUrl",
            "HeadImageUrl",
            "HeadUrl",
            "headImgUrl",
            "headImageUrl",
            "head_url",
            "headUrl",
            "AvatarThumb",
            "AvatarMedium",
            "AvatarLarger",
            "AvatarLarge",
            "AvatarUrl",
            "AvatarURL",
            "Avatar",
            "avatarThumb",
            "avatarMedium",
            "avatarLarger",
            "avatarLarge",
            "avatar_thumb",
            "avatar_medium",
            "avatar_larger",
            "avatar_large",
            "avatar_url",
            "avatarUrl",
            "avatar",
        )
        for key in avatar_keys:
            if key in user:
                url = self._first_url_in_value(user.get(key))
                if url:
                    return url
        return ""

    @staticmethod
    def _safe_filename_part(value: object, fallback: str = "user") -> str:
        text = str(value or "").strip()
        text = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", text, flags=re.UNICODE)
        text = text.strip("._")
        return (text or fallback)[:40]

    @staticmethod
    def _image_extension_from_bytes(data: bytes, content_type: str = "") -> str:
        content_type = str(content_type or "").split(";")[0].strip().lower()
        if content_type in {"image/jpeg", "image/jpg"}:
            return ".jpg"
        if content_type == "image/png":
            return ".png"
        if content_type == "image/webp":
            return ".webp"
        if content_type == "image/gif":
            return ".gif"

        if data.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
            return ".gif"
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return ".webp"
        return ""

    @staticmethod
    def _image_extension_from_url(url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        suffix = Path(parsed.path).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            return ".jpg" if suffix == ".jpeg" else suffix
        return ""

    def _find_cached_avatar_file(self, url_hash: str) -> str:
        for day_dir in sorted(
            BARRAGE_AVATAR_CACHE_ROOT.glob("*"), reverse=True
        ):
            if not day_dir.is_dir():
                continue
            matches = list(day_dir.glob(f"*_{url_hash}.*"))
            if matches:
                return str(matches[0].resolve())
        return ""

    def _avatar_file_base(
        self,
        url_hash: str,
        user_id: str = "",
        nickname: str = "",
    ) -> Path:
        day = datetime.now().astimezone().strftime("%Y-%m-%d")
        user_part = self._safe_filename_part(user_id or nickname)
        return BARRAGE_AVATAR_CACHE_ROOT / day / f"{user_part}_{url_hash}"

    def _download_avatar_file_sync(
        self,
        avatar_url: str,
        user_id: str = "",
        nickname: str = "",
    ) -> str:
        avatar_url = str(avatar_url or "").strip()
        if not avatar_url.startswith(("http://", "https://")):
            return ""

        url_hash = hashlib.sha256(avatar_url.encode("utf-8")).hexdigest()[:16]
        cached = self._avatar_path_cache.get(avatar_url)
        if cached and Path(cached).exists():
            return cached

        cached_file = self._find_cached_avatar_file(url_hash)
        if cached_file:
            self._avatar_path_cache[avatar_url] = cached_file
            return cached_file

        request = urllib.request.Request(
            avatar_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0 Safari/537.36"
                )
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=BARRAGE_AVATAR_DOWNLOAD_TIMEOUT
            ) as response:
                content_type = response.headers.get("Content-Type", "")
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > BARRAGE_AVATAR_MAX_BYTES:
                        logger.warning(
                            "[barrage][avatar] skip oversized avatar: {} bytes url={}",
                            total,
                            avatar_url,
                        )
                        return ""
                    chunks.append(chunk)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.debug(
                "[barrage][avatar] download failed url={} error={}",
                avatar_url,
                exc,
            )
            self._avatar_path_cache[avatar_url] = ""
            return ""

        data = b"".join(chunks)
        if not data:
            self._avatar_path_cache[avatar_url] = ""
            return ""

        ext = (
            self._image_extension_from_bytes(data, content_type)
            or self._image_extension_from_url(avatar_url)
        )
        if not ext:
            logger.debug(
                "[barrage][avatar] response is not a recognized image: {}",
                avatar_url,
            )
            self._avatar_path_cache[avatar_url] = ""
            return ""
        target = self._avatar_file_base(url_hash, user_id, nickname).with_suffix(ext)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(target)
        except OSError as exc:
            logger.warning(
                "[barrage][avatar] failed to save avatar {}: {}",
                target,
                exc,
            )
            self._avatar_path_cache[avatar_url] = ""
            return ""

        path = str(target.resolve())
        self._avatar_path_cache[avatar_url] = path
        logger.info(
            "[barrage][avatar] saved avatar for {} to {}",
            nickname or user_id or "viewer",
            path,
        )
        return path

    async def _ensure_avatar_file(
        self,
        avatar_url: str,
        user_id: str = "",
        nickname: str = "",
    ) -> str:
        avatar_url = str(avatar_url or "").strip()
        if not avatar_url:
            return ""
        cached = self._avatar_path_cache.get(avatar_url)
        if cached is not None:
            return cached if cached and Path(cached).exists() else ""
        return await asyncio.to_thread(
            self._download_avatar_file_sync,
            avatar_url,
            user_id,
            nickname,
        )

    async def _prepare_avatar_for_message(self, msg: BarrageMessage) -> None:
        if msg.avatar_path and Path(msg.avatar_path).exists():
            return
        if not msg.avatar_url:
            return
        msg.avatar_path = await self._ensure_avatar_file(
            msg.avatar_url,
            msg.user_id,
            msg.nickname,
        )

    def _sender_log_info(self, user: dict) -> dict[str, Any]:
        fans_club = user.get("FansClub") if isinstance(user, dict) else {}
        if not isinstance(fans_club, dict):
            fans_club = {}

        return {
            "user_id": self._first_non_empty(
                user.get("SecUid"),
                user.get("DisplayId"),
                user.get("Id"),
                user.get("IdStr"),
                user.get("UserId"),
            ),
            "sec_uid": user.get("SecUid"),
            "display_id": user.get("DisplayId"),
            "short_id": user.get("ShortId"),
            "nickname": self._first_non_empty(
                user.get("Nickname"),
                user.get("DisplayId"),
            ),
            "avatar_url": self._avatar_url_from_user(user),
            "pay_level": self._int_or_zero(user.get("PayLevel")),
            "fans_club": {
                "level": self._int_or_zero(fans_club.get("Level")),
                "name": self._first_non_empty(
                    fans_club.get("ClubName"),
                    fans_club.get("Name"),
                ),
                "badge": self._first_non_empty(
                    fans_club.get("BadgeName"),
                    fans_club.get("MedalName"),
                ),
            },
        }

    def _append_raw_user_debug_log(
        self,
        msg_type: int,
        user: dict,
        reason: str,
    ) -> None:
        if not isinstance(user, dict):
            return
        keys = sorted(str(key) for key in user.keys())
        signature = f"{msg_type}:{','.join(keys)}:{reason}"
        if signature in self._raw_user_debug_signatures:
            return
        # Avoid unbounded raw dumps if a source changes shape constantly.
        if len(self._raw_user_debug_signatures) >= 50:
            return
        self._raw_user_debug_signatures.add(signature)

        now = datetime.now().astimezone()
        event = {
            "timestamp": now.isoformat(timespec="milliseconds"),
            "msg_type": msg_type,
            "reason": reason,
            "keys": keys,
            "user": user,
        }
        path = BARRAGE_RAW_USER_DEBUG_ROOT / f"{now:%Y-%m-%d}.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.debug("[barrage][raw-user] failed to append debug log: {}", exc)

    def _room_log_info(self, msg: dict, data: dict) -> dict[str, Any]:
        room_sources: list[dict] = []
        for source in (data, msg):
            if not isinstance(source, dict):
                continue
            room_sources.extend(
                self._dict_sources(
                    source,
                    "Room",
                    "RoomInfo",
                    "LiveRoom",
                    "Live",
                    "WebcastRoom",
                    "RoomData",
                    "RoomStats",
                )
            )

        anchor_sources: list[dict] = []
        for source in (data, msg):
            if not isinstance(source, dict):
                continue
            anchor_sources.extend(
                self._dict_sources(
                    source,
                    "Anchor",
                    "AnchorInfo",
                    "Owner",
                    "OwnerInfo",
                    "Streamer",
                    "Author",
                )
            )

        room_id = self._first_field(
            room_sources,
            "RoomId",
            "RoomID",
            "RoomIdStr",
            "room_id",
            "roomId",
            "WebcastRoomId",
            "LiveRoomId",
            "LiveId",
            "Id",
            "ID",
        )
        room_title = self._first_field(
            room_sources,
            "RoomTitle",
            "Title",
            "LiveTitle",
            "title",
            "Name",
        )
        anchor_id = self._first_field(
            anchor_sources,
            "SecUid",
            "UserId",
            "UserID",
            "Id",
            "ID",
            "DisplayId",
        )
        anchor_name = self._first_field(
            anchor_sources,
            "Nickname",
            "NickName",
            "Name",
            "DisplayId",
        )

        return {
            "room_id": room_id,
            "room_title": room_title,
            "anchor": {
                "user_id": anchor_id,
                "nickname": anchor_name,
            },
            "source_ws_url": self.config.ws_url,
            "has_payload_room_info": bool(room_id or room_title or anchor_id or anchor_name),
        }

    def _append_barrage_event_log(self, msg_type: int, msg: dict, data: dict) -> None:
        if msg_type not in (1, 5, 6):
            return

        now = datetime.now().astimezone()
        user = data.get("User") or {}
        if not isinstance(user, dict):
            user = {}

        if msg_type == 1:
            event_type = "barrage"
        elif msg_type == 5:
            event_type = "gift"
        else:
            event_type = "room_stats"
        event: dict[str, Any] = {
            "timestamp": now.isoformat(timespec="milliseconds"),
            "event": event_type,
            "msg_type": msg_type,
            "message_id": self._first_non_empty(
                msg.get("MsgId"),
                msg.get("MsgID"),
                msg.get("MessageId"),
                data.get("MsgId"),
                data.get("MsgID"),
                data.get("MessageId"),
                data.get("Id"),
            ),
            "room": self._room_log_info(msg, data),
        }

        if msg_type == 1:
            sender_info = self._sender_log_info(user)
            event["sender"] = sender_info
            event["content"] = str(data.get("Content") or "")
            if not sender_info.get("avatar_url"):
                self._append_raw_user_debug_log(
                    msg_type, user, "missing-avatar-url"
                )
        elif msg_type == 5:
            sender_info = self._sender_log_info(user)
            event["sender"] = sender_info
            event["gift"] = {
                "name": data.get("GiftName"),
                "count": self._int_or_zero(data.get("GiftCount") or 1),
                "diamond_count": self._int_or_zero(data.get("DiamondCount")),
                "repeat_count": self._int_or_zero(data.get("RepeatCount")),
            }
            if not sender_info.get("avatar_url"):
                self._append_raw_user_debug_log(
                    msg_type, user, "missing-avatar-url"
                )
        else:
            event["stats"] = {
                "online_user_count": self._int_or_zero(data.get("OnlineUserCount")),
                "total_user_count": self._int_or_zero(data.get("TotalUserCount")),
            }

        log_path = self._barrage_event_log_path(now, event_type)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(
                "[barrage][event-log] failed to append {} event: {}",
                event_type,
                exc,
            )

    def _append_link_probe_log(self, msg_type: int, msg: dict, data: dict) -> None:
        message_meta = {
            key: value
            for key, value in msg.items()
            if key not in {"Data", "data"}
        }
        payload = {"message": message_meta, "data": data}
        hint_paths = self._collect_link_probe_hint_paths(payload)
        candidates = [
            *list(self._iter_raw_link_payload_candidates(data)),
            *list(self._iter_link_anchor_candidates(payload)),
        ][:BARRAGE_LINK_PROBE_MAX_CANDIDATES]
        if not hint_paths and not candidates:
            return

        now = datetime.now().astimezone()
        event = {
            "timestamp": now.isoformat(timespec="milliseconds"),
            "event": "link_probe",
            "msg_type": msg_type,
            "message_id": self._first_non_empty(
                msg.get("MsgId"),
                msg.get("MsgID"),
                msg.get("MessageId"),
                data.get("MsgId"),
                data.get("MsgID"),
                data.get("MessageId"),
                data.get("Id"),
            ),
            "room": self._room_log_info(msg, data),
            "candidates": candidates,
            "hint_paths": hint_paths,
            "payload": self._truncate_link_probe_value(payload),
        }

        path = BARRAGE_LINK_PROBE_ROOT / f"{now:%Y-%m-%d}.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")
            if candidates:
                logger.info(
                    "[barrage][link-probe] logged candidates={} msg_type={} path={}",
                    [item.get("name") for item in candidates],
                    msg_type,
                    path,
                )
            else:
                logger.debug(
                    "[barrage][link-probe] logged hint-only msg_type={} path={}",
                    msg_type,
                    path,
                )
        except Exception as exc:
            logger.warning("[barrage][link-probe] failed to append log: {}", exc)

    def _update_room_stats(self, msg: dict, data: dict) -> None:
        online_user_count = self._int_or_zero(
            data.get("OnlineUserCount")
            or data.get("OnlineCount")
            or data.get("MemberCount")
            or 0
        )
        total_user_count = self._int_or_zero(data.get("TotalUserCount"))
        self._last_heartbeat_time = time.time()
        self._last_room_stats = {
            "member_count": online_user_count,
            "online_count": online_user_count,
            "online_user_count": online_user_count,
            "total_user_count": total_user_count,
            "updated_at": self._last_heartbeat_time,
            "room": self._room_log_info(msg, data),
        }

    async def _process_raw_message(self, raw):
        try:
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8', errors='ignore')
            msg = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        if not isinstance(msg, dict):
            return

        msg_type = msg.get("Type", 0)

        data_raw = msg.get("Data", {})
        if isinstance(data_raw, str):
            try:
                data = json.loads(data_raw)
            except json.JSONDecodeError:
                return
        elif isinstance(data_raw, dict):
            data = data_raw
        else:
            return

        # B9 metrics: 接收计数 (除 filter 之前)
        self._append_barrage_event_log(msg_type, msg, data)
        self._append_link_probe_log(msg_type, msg, data)
        self._update_link_anchor_candidate(msg, data)
        self._metric_tick(self._metrics_recv)

        if msg_type == 1:
            original_content = str(data.get("Content") or "")
            cleaned_content = strip_barrage_bracketed_content(
                original_content
            )
            if cleaned_content != original_content.strip():
                logger.debug(
                    "[barrage] stripped bracketed content: raw={!r} "
                    "cleaned={!r}",
                    original_content,
                    cleaned_content,
                )
            data["Content"] = cleaned_content

        # 即使 filter 拒绝, 也要先抽取礼物 DiamondCount 用于钻石累计
        # (累计排行需要看到全部礼物事件, 不受过滤影响)
        if msg_type == 5:
            pre_user = data.get("User") or {}
            pre_uid = pre_user.get("SecUid", "") or pre_user.get(
                "DisplayId", ""
            )
            pre_nick = pre_user.get("Nickname", "") or pre_user.get(
                "DisplayId", ""
            )
            pre_diamonds = data.get("DiamondCount", 0) or 0
            if pre_uid and pre_diamonds > 0:
                self._accumulate_diamond(
                    pre_uid, int(pre_diamonds), pre_nick
                )

        # BarrageFilter 只针对 Type=1/5 做内容/去重过滤; 其他事件类型直通
        if msg_type in (1, 5) and not self.filter.should_accept(
            msg_type, data
        ):
            return

        user = data.get("User") or {}
        user_id = user.get("SecUid", "") or user.get("DisplayId", "unknown")
        nickname = user.get("Nickname", "") or user.get("DisplayId", "viewer")
        avatar_url = self._avatar_url_from_user(user)

        if msg_type == 1:
            # 弹幕
            content = data.get("Content", "").strip()
            if not content:
                return
            formatted = self.config.prefix_template.format(
                nickname=nickname, content=content
            )
            # 抽取自定义筛选所需的字段
            wealth_level = int(user.get("PayLevel") or 0)
            fans_club = user.get("FansClub") or {}
            fan_badge_level = int(fans_club.get("Level") or 0)
            self._record_fan_badge_level(user_id, fan_badge_level)
            priority = self._classify_priority(
                fan_badge_level,
                use_dynamic_high=not self._custom_filter_active(),
            )
            session_diamond_total = self._session_diamond_totals.get(
                user_id, 0
            )
            barrage_msg = BarrageMessage(
                priority=priority,
                timestamp=time.time(),
                nickname=nickname,
                user_id=user_id,
                avatar_url=avatar_url,
                content=formatted,
                raw_data=data,
                wealth_level=wealth_level,
                fan_badge_level=fan_badge_level,
                session_diamond_total=session_diamond_total,
            )
            self._enqueue_barrage(barrage_msg)

        elif msg_type == 5:
            # 礼物 — 走 B1 连击合并缓冲
            if not self._gift_queueing_allowed():
                # logger.debug(
                #     "[barrage][gift] ignore gift thanks because enabled={} "
                #     "barrage_mode={}",
                #     self.config.gift_thanks_enabled,
                #     self._is_barrage_mode(),
                # )
                return
            gift_name = data.get("GiftName", "gift")
            gift_count = data.get("GiftCount", 1) or 1
            diamonds = data.get("DiamondCount", 0) or 0
            self._buffer_gift_combo(
                user_id=user_id,
                nickname=nickname,
                avatar_url=avatar_url,
                gift_name=gift_name,
                gift_count=int(gift_count),
                diamonds=int(diamonds),
                raw_data=data,
            )

        elif msg_type == 2:
            # 点赞 — B3, 只计数
            if self.config.like_count_track:
                inc = int(data.get("Count", 1) or 1)
                self._like_count += inc

        elif msg_type == 3:
            # 进直播间 — B3, 聚合后入队
            if self.config.welcome_enabled and nickname:
                key = user_id or nickname
                if key not in self._joins_dedup_keys:
                    self._joins_dedup_keys.add(key)
                    self._pending_joins.append(
                        (time.time(), user_id, nickname)
                    )

        elif msg_type == 4:
            # 关注 — B3, 聚合后入队
            if self.config.follow_enabled and nickname:
                key = user_id or nickname
                if key not in self._follows_dedup_keys:
                    self._follows_dedup_keys.add(key)
                    self._pending_follows.append(
                        (time.time(), user_id, nickname)
                    )

        elif msg_type == 6:
            self._update_room_stats(msg, data)

        elif msg_type == 8:
            pass
            # 6才是直播间信息
            # # 直播间统计 (心跳信号)
            # # 字段: MemberCount(在线观众), TotalUserCount(累计观众)
            # self._last_heartbeat_time = time.time()
            # self._last_room_stats = {
            #     "member_count": data.get("MemberCount") or data.get("OnlineCount") or 0,
            #     "total_user_count": data.get("TotalUserCount") or 0,
            #     "updated_at": self._last_heartbeat_time,
            #     "room": self._room_log_info(msg, data),
            # }

        elif msg_type == 9:
            # 粉丝团事件 (加入/升级) — 仅记录, 不强制触发
            logger.debug(
                f"[barrage] 粉丝团事件: {nickname} {data.get('Content', '')}"
            )

    def _contains_keyword(self, text: str) -> bool:
        """检查文本是否包含关键词（不区分大小写）"""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self._keywords_lower)

    def _enqueue_barrage(self, msg: BarrageMessage):
        # 提取原始弹幕内容（去掉 prefix 格式化部分）用于关键词匹配
        raw_content = msg.raw_data.get("Content", "") if msg.raw_data else ""

        # ---- 自定义筛选/排序路径 (任一变量启用时生效) ----
        if self._custom_filter_active():
            enabled = self._get_enabled_vars_sorted()
            # 1. 过滤: 所有启用变量都必须 >= 各自阈值
            for var in enabled:
                v = self._get_var_value(var.name, msg)
                if v < var.threshold:
                    self._custom_filtered_count += 1
                    self._metric_tick(self._metrics_filter_drop)
                    logger.debug(
                        f"[barrage][custom] 过滤丢弃 {var.name}={v}<"
                        f"{var.threshold} 用户={msg.nickname}"
                    )
                    return
            # 2. 关键词路由 + 有序插入
            sort_key = self._compute_sort_key(msg)
            if self._contains_keyword(raw_content):
                self._insert_sorted(
                    self._custom_keyword_queue, msg, sort_key
                )
                self._metric_tick(self._metrics_enqueue)
                logger.info(
                    f"[barrage][custom] 关键词弹幕入队 sort_key={sort_key} "
                    f"队列长度={len(self._custom_keyword_queue)}: "
                    f"{msg.content}"
                )
            else:
                self._insert_sorted(
                    self._custom_normal_queue, msg, sort_key
                )
                self._metric_tick(self._metrics_enqueue)
                # logger.info(
                #     f"[barrage][custom] 普通弹幕入队 sort_key={sort_key} "
                #     f"队列长度={len(self._custom_normal_queue)}: "
                #     f"{msg.content}"
                # )
            return

        # ---- 原有逻辑 (三个自定义变量全部关闭时) ----
        if self._contains_keyword(raw_content):
            self.barrage_queue_keyword.append(msg)
            self._metric_tick(self._metrics_enqueue)
            logger.info(
                f"[barrage] 关键词弹幕入队 (不丢弃): {msg.content}"
            )
            return

        if msg.priority <= Priority.FANS_HIGH:
            q = self.barrage_queue_high
        else:
            q = self.barrage_queue_normal
        try:
            q.put_nowait(msg)
            self._metric_tick(self._metrics_enqueue)
            # logger.info(f"[barrage] {label}弹幕入队 (P{msg.priority}): {msg.content}")
        except asyncio.QueueFull:
            pass
            # logger.debug(f"[barrage] {label}弹幕队列满, 丢弃: {msg.content}")

    def _record_fan_badge_level(self, user_id: str, level: int) -> None:
        if not user_id:
            return
        try:
            level = max(0, int(level))
        except (TypeError, ValueError):
            level = 0
        self._fan_badge_levels_by_user[user_id] = level

    def _current_high_fan_cutoff(self) -> int:
        levels = [
            level
            for level in self._fan_badge_levels_by_user.values()
            if level > 0
        ]
        if not levels:
            return self.config.high_fan_default_level

        percent = max(1, min(100, int(self.config.high_fan_percent)))
        top_count = max(1, (len(levels) * percent + 99) // 100)
        levels.sort(reverse=True)
        return max(1, levels[min(top_count, len(levels)) - 1])

    def set_high_fan_percent(self, percent: int | float) -> dict:
        try:
            percent = int(float(percent))
        except (TypeError, ValueError):
            percent = self.config.high_fan_percent
        self.config.high_fan_percent = max(1, min(100, percent))
        rebalance = self._rebalance_barrage_priority_queues(
            "high-fan-percent-updated"
        )
        result = self.high_fan_status()
        result["rebalance"] = rebalance
        logger.info(
            "[barrage] high fan percent updated: percent={} cutoff={} "
            "users={} rebalance={}",
            result["percent"],
            result["level_cutoff"],
            result["sample_users"],
            rebalance,
        )
        return result

    def _drain_priority_queue(
        self,
        queue: asyncio.PriorityQueue,
    ) -> list[BarrageMessage]:
        messages = []
        while not queue.empty():
            try:
                messages.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    def _rebalance_barrage_priority_queues(self, reason: str) -> dict:
        messages = self._drain_priority_queue(
            self.barrage_queue_high
        ) + self._drain_priority_queue(self.barrage_queue_normal)
        dropped = 0
        high = 0
        normal = 0
        for msg in messages:
            msg.priority = self._classify_priority(
                msg.fan_badge_level,
                use_dynamic_high=True,
            )
            target_queue = (
                self.barrage_queue_high
                if msg.priority <= Priority.FANS_HIGH
                else self.barrage_queue_normal
            )
            try:
                target_queue.put_nowait(msg)
                if target_queue is self.barrage_queue_high:
                    high += 1
                else:
                    normal += 1
            except asyncio.QueueFull:
                dropped += 1
        if messages or dropped:
            logger.info(
                "[barrage] rebalanced barrage queues: reason={} high={} "
                "normal={} dropped={} cutoff={}",
                reason,
                high,
                normal,
                dropped,
                self._current_high_fan_cutoff(),
            )
        return {
            "high": high,
            "normal": normal,
            "dropped": dropped,
        }

    def high_fan_status(self) -> dict:
        return {
            "percent": self.config.high_fan_percent,
            "level_cutoff": self._current_high_fan_cutoff(),
            "default_level": self.config.high_fan_default_level,
            "sample_users": len(self._fan_badge_levels_by_user),
            "positive_sample_users": sum(
                1 for level in self._fan_badge_levels_by_user.values()
                if level > 0
            ),
        }

    def _classify_priority(
        self,
        level: int,
        *,
        use_dynamic_high: bool = True,
    ) -> int:
        try:
            level = max(0, int(level))
        except (TypeError, ValueError):
            level = 0

        high_cutoff = (
            self._current_high_fan_cutoff()
            if use_dynamic_high
            else self.config.high_fan_default_level
        )
        if level > 0 and level >= high_cutoff:
            return Priority.FANS_HIGH
        elif level >= 10:
            return Priority.FANS_MID
        elif level >= 1:
            return Priority.FANS_LOW
        return Priority.NORMAL

    # -------------------- 自定义筛选/排序 helpers (阶段4) --------------------

    def _custom_filter_active(self) -> bool:
        """三个变量中只要任一启用, 就走自定义路径"""
        return any(
            v.enabled for v in self.config.custom_variables.values()
        )

    def _get_enabled_vars_sorted(self) -> List[CustomVariableConfig]:
        """启用的变量按 priority 升序 (数值小=高优先级, 排序键的第一项)"""
        enabled = [
            v for v in self.config.custom_variables.values() if v.enabled
        ]
        enabled.sort(key=lambda v: v.priority)
        return enabled

    def _get_var_value(self, name: str, msg: BarrageMessage) -> int:
        """读取消息上对应变量的数值"""
        if name == "wealth":
            return msg.wealth_level
        if name == "fan_badge":
            return msg.fan_badge_level
        if name == "diamond_rank":
            return msg.session_diamond_total
        return 0

    def _compute_sort_key(self, msg: BarrageMessage) -> tuple:
        """构造排序键: 启用变量按 priority 升序, 每项值取负 (= 降序)"""
        enabled = self._get_enabled_vars_sorted()
        return tuple(
            -self._get_var_value(v.name, msg) for v in enabled
        )

    def _insert_sorted(
        self, queue: list, msg: BarrageMessage, sort_key: tuple
    ):
        """按 sort_key 升序插入并裁剪到 custom_queue_max_size."""
        keys = [self._compute_sort_key(m) for m in queue]
        pos = bisect.bisect_right(keys, sort_key)
        queue.insert(pos, msg)
        # 超长则丢弃末尾 (排序键最大 = 实际值最低)
        max_size = self.config.custom_queue_max_size
        while len(queue) > max_size:
            dropped = queue.pop()
            self._custom_dropped_count += 1
            logger.debug(
                f"[barrage][custom] 队列满, 丢弃末尾低优先级: "
                f"{dropped.nickname}"
            )

    def _sweep_custom_queues_stale(self) -> int:
        """根据 custom_item_ttl 清扫两个自定义队列中的过期项. 返回丢弃数."""
        now = time.time()
        ttl = self.config.custom_item_ttl
        dropped = 0
        for queue in (
            self._custom_keyword_queue,
            self._custom_normal_queue,
        ):
            i = 0
            while i < len(queue):
                if now - queue[i].timestamp > ttl:
                    queue.pop(i)
                    dropped += 1
                else:
                    i += 1
        if dropped:
            self._custom_dropped_count += dropped
            logger.debug(
                f"[barrage][custom] TTL 清扫丢弃 {dropped} 条 "
                f"(累计 {self._custom_dropped_count})"
            )
        return dropped

    # -------------------- B2: LRU 钻石累计 --------------------

    def _accumulate_diamond(self, uid: str, amount: int, nick: str):
        """累加钻石数, 使用 LRU 策略防止内存无限增长."""
        cur = self._session_diamond_totals.get(uid, 0)
        self._session_diamond_totals[uid] = cur + amount
        # 移到末尾标记最近活跃
        self._session_diamond_totals.move_to_end(uid)
        if nick:
            self._session_diamond_nicks[uid] = nick
        # 超出上限则淘汰最老的
        max_users = self.config.diamond_max_users
        while len(self._session_diamond_totals) > max_users:
            evicted_uid, evicted_amount = self._session_diamond_totals.popitem(
                last=False
            )
            self._session_diamond_nicks.pop(evicted_uid, None)
            self._diamond_evicted_count += 1
            if self._diamond_evicted_count % 100 == 1:
                logger.warning(
                    f"[barrage][diamond] LRU 淘汰 (累计淘汰 "
                    f"{self._diamond_evicted_count} 个用户) - "
                    f"考虑调大 diamond_max_users (当前={max_users})"
                )

    # -------------------- B9: metrics 滑动窗口 --------------------

    def _metric_tick(self, queue: deque):
        """记录一次事件; 同时清扫窗口外的旧条目."""
        now = time.time()
        queue.append(now)
        cutoff = now - self.config.metrics_window_seconds
        while queue and queue[0] < cutoff:
            queue.popleft()

    def _metric_rate(self, queue: deque) -> float:
        """返回窗口内的速率 (条/秒)."""
        if not queue:
            return 0.0
        now = time.time()
        cutoff = now - self.config.metrics_window_seconds
        while queue and queue[0] < cutoff:
            queue.popleft()
        if not queue:
            return 0.0
        return len(queue) / self.config.metrics_window_seconds

    def get_metrics(self) -> dict:
        """返回 metrics 快照."""
        # 当前队列年龄分布
        now = time.time()
        custom_ages = [
            now - m.timestamp
            for m in self._custom_keyword_queue + self._custom_normal_queue
        ]
        custom_ages.sort()

        def percentile(arr, p):
            if not arr:
                return None
            k = int(len(arr) * p / 100)
            return round(arr[min(k, len(arr) - 1)], 2)

        return {
            "window_s": self.config.metrics_window_seconds,
            "rates_per_s": {
                "recv": round(self._metric_rate(self._metrics_recv), 2),
                "enqueue": round(
                    self._metric_rate(self._metrics_enqueue), 2
                ),
                "consume": round(
                    self._metric_rate(self._metrics_consume), 2
                ),
                "filter_drop": round(
                    self._metric_rate(self._metrics_filter_drop), 2
                ),
                "combo_merged": round(
                    self._metric_rate(self._metrics_combo_merged), 2
                ),
            },
            "queues": {
                "barrage_high": self.barrage_queue_high.qsize(),
                "barrage_normal": self.barrage_queue_normal.qsize(),
                "barrage_keyword": len(self.barrage_queue_keyword),
                "gift": self.gift_queue.qsize(),
                "custom_keyword": len(self._custom_keyword_queue),
                "custom_normal": len(self._custom_normal_queue),
                "gift_combo_buffer": len(self._gift_combo_buffer),
                "pending_joins": len(self._pending_joins),
                "pending_follows": len(self._pending_follows),
            },
            "queue_age_seconds": {
                "custom_p50": percentile(custom_ages, 50),
                "custom_p95": percentile(custom_ages, 95),
                "custom_max": round(custom_ages[-1], 2) if custom_ages else None,
            },
            "counters": {
                "total_received": self._total_received,
                "reconnect_count": self._reconnect_count,
                "like_count": self._like_count,
                "diamond_evicted_total": self._diamond_evicted_count,
                "custom_dropped_total": self._custom_dropped_count,
                "custom_filtered_total": self._custom_filtered_count,
            },
            "room_stats": dict(self._last_room_stats),
        }

    # -------------------- B1: 礼物连击合并 --------------------

    def _buffer_gift_combo(
        self,
        user_id: str,
        nickname: str,
        avatar_url: str,
        gift_name: str,
        gift_count: int,
        diamonds: int,
        raw_data: dict,
    ):
        """缓冲连击礼物, 由 _gift_combo_flusher_loop 定时入队."""
        now = time.time()
        key = f"{user_id}:{gift_name}"
        entry = self._gift_combo_buffer.get(key)
        if entry is None:
            # 新礼物 — 创建缓冲
            msg = BarrageMessage(
                priority=Priority.GIFT_BIG,
                timestamp=now,
                nickname=nickname,
                user_id=user_id,
                avatar_url=avatar_url,
                content="",  # 会在 flush 时生成
                raw_data={
                    "GiftName": gift_name,
                    "_accumulated_count": gift_count,
                    "_accumulated_diamonds": diamonds,
                },
            )
            self._gift_combo_buffer[key] = {
                "msg": msg,
                "first_seen": now,
                "last_seen": now,
            }
            logger.info(
                f"[barrage][gift] 连击缓冲新建: {nickname} {gift_count}x "
                f"{gift_name} ({diamonds}钻)"
            )
        else:
            # 已有连击 — 累加
            entry["msg"].raw_data["_accumulated_count"] += gift_count
            entry["msg"].raw_data["_accumulated_diamonds"] += diamonds
            if avatar_url and not entry["msg"].avatar_url:
                entry["msg"].avatar_url = avatar_url
            entry["last_seen"] = now
            self._metric_tick(self._metrics_combo_merged)
            logger.debug(
                f"[barrage][gift] 连击累加: {nickname} +{gift_count}x "
                f"{gift_name} → 累计 "
                f"{entry['msg'].raw_data['_accumulated_count']}x"
            )

    async def _gift_combo_flusher_loop(self):
        """定期把过期的连击缓冲入队."""
        while self._running:
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                return
            await self._flush_gift_combos()

    async def _flush_gift_combos(self):
        now = time.time()
        combo_window = self.config.gift_combo_window
        max_wait = self.config.gift_combo_max_wait
        to_flush = []
        for key, entry in list(self._gift_combo_buffer.items()):
            silent = now - entry["last_seen"]
            held = now - entry["first_seen"]
            # 连击窗口内安静 N 秒, 或者总缓冲超过 max_wait 都要 flush
            if silent >= combo_window or held >= max_wait:
                to_flush.append(key)

        for key in to_flush:
            entry = self._gift_combo_buffer.pop(key)
            msg = entry["msg"]
            self._queue_gift_combo_message(msg, "combo-window")

    def _gift_diamond_total(self, msg: BarrageMessage) -> int:
        raw_data = msg.raw_data or {}
        return self._int_or_zero(
            raw_data.get("_accumulated_diamonds")
            or raw_data.get("DiamondCount")
            or 0
        )

    def _queue_gift_combo_message(
        self,
        msg: BarrageMessage,
        reason: str,
    ) -> bool:
        raw_data = msg.raw_data or {}
        count = self._int_or_zero(raw_data.get("_accumulated_count") or 1)
        diamonds = self._gift_diamond_total(msg)
        gift_name = raw_data.get("GiftName", "gift")

        # 礼物去重：合并后再去重一次，防止连续窗口重复感谢。
        if not self._gift_queueing_allowed():
            logger.debug(
                "[barrage][gift] drop combo because gift queueing is not "
                "allowed: barrage_mode={} gift={}",
                self._is_barrage_mode(),
                gift_name,
            )
            return False

        if not self._gift_dedup_check(msg.user_id, gift_name):
            logger.debug(
                "[barrage][gift] deduped combo after merge: user={} gift={}",
                msg.nickname,
                gift_name,
            )
            return False

        msg.content = (
            f"[gift] {msg.nickname} sent {count}x {gift_name}"
            f" (worth {diamonds} diamonds), please thank them!"
        )
        try:
            self.gift_queue.put_nowait(msg)
            self._metric_tick(self._metrics_enqueue)
            logger.info(
                "[barrage][gift] combo queued: reason={} count={} "
                "diamonds={} content={}",
                reason,
                count,
                diamonds,
                msg.content,
            )
            return True
        except asyncio.QueueFull:
            logger.debug("[barrage][gift] gift queue full, dropped: {}", msg.content)
            return False

    def _flush_all_pending_gift_combos(self, reason: str) -> int:
        flushed = 0
        for key, entry in list(self._gift_combo_buffer.items()):
            self._gift_combo_buffer.pop(key, None)
            msg = entry.get("msg") if isinstance(entry, dict) else None
            if isinstance(msg, BarrageMessage) and self._queue_gift_combo_message(
                msg,
                reason,
            ):
                flushed += 1
        return flushed

    def _sort_pending_gifts_by_diamonds_once(self, reason: str) -> int:
        pending: list[BarrageMessage] = []
        while not self.gift_queue.empty():
            try:
                pending.append(self.gift_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        if not pending:
            return 0

        sorted_pending = [
            msg
            for _, msg in sorted(
                enumerate(pending),
                key=lambda item: (-self._gift_diamond_total(item[1]), item[0]),
            )
        ]
        dropped = 0
        for msg in sorted_pending:
            try:
                self.gift_queue.put_nowait(msg)
            except asyncio.QueueFull:
                dropped += 1

        logger.info(
            "[barrage][gift] sorted pending gifts once: reason={} count={} "
            "dropped={} top_diamonds={}",
            reason,
            len(sorted_pending),
            dropped,
            self._gift_diamond_total(sorted_pending[0]),
        )
        return len(sorted_pending) - dropped

    # -------------------- B3: 进直播间/关注 聚合 flush --------------------

    async def _event_batch_flusher_loop(self):
        """周期性把进直播间/关注的聚合缓冲转成弹幕消息入队."""
        while self._running:
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                return
            await self._flush_event_batches()

    async def _flush_event_batches(self):
        now = time.time()
        # joins
        if (
            self.config.welcome_enabled
            and self._pending_joins
            and (
                now - self._pending_joins[0][0]
                >= self.config.welcome_batch_window
                or len(self._pending_joins) >= self.config.welcome_max_batch
            )
        ):
            batch = self._pending_joins[: self.config.welcome_max_batch]
            self._pending_joins = self._pending_joins[
                self.config.welcome_max_batch :
            ]
            for _, uid, _ in batch:
                self._joins_dedup_keys.discard(uid)
            names = "、".join(n for _, _, n in batch)
            text = self.config.welcome_template.format(names=names)
            self._enqueue_event_message(text, batch[0][1], batch[0][2])
            logger.info(f"[barrage][welcome] 进直播间合并入队 ({len(batch)} 人): {text}")

        # follows
        if (
            self.config.follow_enabled
            and self._pending_follows
            and (
                now - self._pending_follows[0][0]
                >= self.config.follow_batch_window
                or len(self._pending_follows) >= self.config.follow_max_batch
            )
        ):
            batch = self._pending_follows[: self.config.follow_max_batch]
            self._pending_follows = self._pending_follows[
                self.config.follow_max_batch :
            ]
            for _, uid, _ in batch:
                self._follows_dedup_keys.discard(uid)
            names = "、".join(n for _, _, n in batch)
            text = self.config.follow_template.format(names=names)
            self._enqueue_event_message(text, batch[0][1], batch[0][2])
            logger.info(f"[barrage][follow] 关注合并入队 ({len(batch)} 人): {text}")

    def _enqueue_event_message(self, text: str, user_id: str, nickname: str):
        """把欢迎/感谢消息当作普通弹幕入队
        (享受现有的关键词路由、过滤等)."""
        msg = BarrageMessage(
            priority=Priority.NORMAL,
            timestamp=time.time(),
            nickname=nickname,
            user_id=user_id,
            content=text,
            raw_data={"Content": text, "_event": True},
        )
        self._enqueue_barrage(msg)
        self._metric_tick(self._metrics_enqueue)

    def _gift_dedup_check(self, user_id: str, gift_name: str) -> bool:
        key = f"{user_id}:{gift_name}"
        now = time.time()
        last = self._gift_dedup.get(key, 0)
        if now - last < self.config.gift_dedup_window:
            return False
        self._gift_dedup[key] = now
        # 清理过期条目
        if len(self._gift_dedup) > 200:
            cutoff = now - self.config.gift_dedup_window
            self._gift_dedup = {
                k: v for k, v in self._gift_dedup.items() if v > cutoff
            }
        return True

    # -------------------- Health monitor --------------------

    async def _health_monitor_loop(self):
        """定期检查连接健康状态并输出摘要日志。"""
        interval = self.config.health_log_interval
        while self._running:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return

            if not self._running:
                return

            now = time.time()
            silent = now - self._last_message_time if self._last_message_time else 0
            ws_alive = self._ws is not None and self._ws.open if self._ws else False

            # 短摘要
            status_parts = [
                f"ws={'✅' if ws_alive else '❌'}",
                f"total_recv={self._total_received}",
                f"silent={silent:.0f}s",
                f"reconnects={self._reconnect_count}",
                f"queues(kw={len(self.barrage_queue_keyword)}"
                f"/hi={self.barrage_queue_high.qsize()}"
                f"/lo={self.barrage_queue_normal.qsize()}"
                f"/gift={self.gift_queue.qsize()})",
            ]
            summary = " | ".join(status_parts)

            # 超过已建立连接超时的一半就发 WARNING
            warn_threshold = self.config.stale_connection_timeout * 0.5
            if silent > warn_threshold:
                logger.warning(f"[barrage] 健康监控 ⚠️  {summary}")
            else:
                logger.debug(f"[barrage] 健康监控 {summary}")

    # -------------------- Process monitor --------------------

    async def _maybe_restart_grab_for_stale(
        self,
        reason: str,
        silent_secs: float,
        heartbeat_silent: float | None = None,
    ) -> bool:
        """Restart DouyinBarrageGrab when the upstream barrage stream is stale.

        The local ws connection can stay open even when DouyinBarrageGrab no
        longer receives data from Douyin/browser/live companion. In that state,
        reconnecting only ws://127.0.0.1:8888 is often not enough; restarting
        the grabber forces it to reinstall proxy hooks and resubscribe.
        """
        if not self.config.grab_auto_restart:
            return False
        if not self.config.grab_restart_on_stale:
            return False
        if not self.config.grab_exe_path:
            return False
        if silent_secs < self.config.grab_stale_restart_seconds:
            return False

        if not self._grab_restart_cooldown_elapsed():
            cooldown = max(0.0, float(self.config.grab_restart_cooldown_seconds))
            since_restart = time.time() - self._last_grab_restart_time
            logger.warning(
                "[barrage] skip DouyinBarrageGrab stale restart: "
                "reason={} silent={:.0f}s heartbeat_silent={} cooldown_left={:.0f}s",
                reason,
                silent_secs,
                (
                    f"{heartbeat_silent:.0f}s"
                    if heartbeat_silent is not None
                    else "n/a"
                ),
                cooldown - since_restart,
            )
            return False

        logger.warning(
            "[barrage] restarting DouyinBarrageGrab because stream is stale: "
            "reason={} silent={:.0f}s heartbeat_silent={}",
            reason,
            silent_secs,
            (
                f"{heartbeat_silent:.0f}s"
                if heartbeat_silent is not None
                else "n/a"
            ),
        )
        return await self._restart_grab_process(reason=reason)

    def _grab_restart_cooldown_elapsed(self) -> bool:
        if not self._last_grab_restart_time:
            return True
        cooldown = max(0.0, float(self.config.grab_restart_cooldown_seconds))
        return (time.time() - self._last_grab_restart_time) >= cooldown

    async def _restart_grab_process(self, reason: str = "") -> bool:
        """Terminate then start DouyinBarrageGrab, if auto-restart is enabled."""
        if not self.config.grab_exe_path:
            return False

        stopped = await asyncio.to_thread(self._terminate_grab_process)
        if stopped:
            await asyncio.sleep(1.0)

        started = await asyncio.to_thread(self._start_grab_process)
        if not started:
            return False

        self._last_grab_restart_time = time.time()
        self._stale_grab_restart_count += 1
        wait_seconds = max(0.0, float(self.config.grab_startup_wait_seconds))
        if wait_seconds:
            await asyncio.sleep(wait_seconds)
        self.force_reconnect()
        logger.info(
            "[barrage] DouyinBarrageGrab restarted, reason={} count={}",
            reason or "manual",
            self._stale_grab_restart_count,
        )
        return True

    def _start_grab_process(self) -> bool:
        import os
        import subprocess

        exe_path = os.path.abspath(str(self.config.grab_exe_path or ""))
        if not os.path.isfile(exe_path):
            logger.error("[barrage] cannot start DouyinBarrageGrab: missing {}", exe_path)
            return False

        try:
            self._grab_process = subprocess.Popen(
                [exe_path],
                cwd=os.path.dirname(exe_path) or ".",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == "nt"
                else 0,
            )
        except Exception as exc:
            logger.error("[barrage] failed to start DouyinBarrageGrab {}: {}", exe_path, exc)
            return False

        logger.info(
            "[barrage] DouyinBarrageGrab started: {} (PID: {})",
            exe_path,
            getattr(self._grab_process, "pid", "?"),
        )
        return True

    def _terminate_grab_process(self) -> bool:
        import os
        import subprocess

        proc_name = str(self.config.grab_process_name or "").strip()
        stopped = False

        if os.name == "nt" and proc_name:
            image_name = proc_name if proc_name.lower().endswith(".exe") else f"{proc_name}.exe"
            try:
                result = subprocess.run(
                    ["taskkill", "/IM", image_name, "/T", "/F"],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                stopped = result.returncode == 0
                if stopped:
                    logger.info("[barrage] stopped DouyinBarrageGrab via taskkill: {}", image_name)
                else:
                    logger.debug(
                        "[barrage] taskkill did not stop {}: {} {}",
                        image_name,
                        result.stdout.strip(),
                        result.stderr.strip(),
                    )
            except Exception as exc:
                logger.debug("[barrage] taskkill failed for {}: {}", image_name, exc)

        proc = self._grab_process
        if proc is not None and getattr(proc, "poll", lambda: None)() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
                stopped = True
            except Exception:
                try:
                    proc.kill()
                    stopped = True
                except Exception:
                    pass

        return stopped

    async def _grab_process_monitor(self):
        """监控 DouyinBarrageGrab 进程，崩溃时可自动重启。"""
        import os

        exe_path = self.config.grab_exe_path
        proc_name = self.config.grab_process_name
        interval = self.config.grab_check_interval
        auto_restart = self.config.grab_auto_restart

        logger.info(
            f"[barrage] 进程监控已启动: process={proc_name} "
            f"auto_restart={auto_restart} check_interval={interval}s"
        )

        while self._running:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return

            if not self._running:
                return

            alive = self._is_grab_process_alive(proc_name)

            if alive:
                continue

            logger.warning(
                f"[barrage] ⚠️  DouyinBarrageGrab 进程 ({proc_name}) 未检测到"
            )

            if not auto_restart:
                logger.info(
                    "[barrage] 自动重启未启用 (grab_auto_restart=False)，"
                    "请手动启动 DouyinBarrageGrab"
                )
                continue

            if not exe_path or not os.path.isfile(exe_path):
                logger.error(
                    f"[barrage] 无法自动重启: 找不到 {exe_path}"
                )
                continue

            logger.info(f"[barrage] 🔄 正在自动启动 DouyinBarrageGrab: {exe_path}")
            try:
                started = await asyncio.to_thread(self._start_grab_process)
                if not started:
                    continue
                logger.info(
                    f"[barrage] ✅ DouyinBarrageGrab 已启动 (PID: {self._grab_process.pid})"
                )
                # 启动后触发一次重连，让 adapter 连上新进程
                await asyncio.sleep(5)
                self.force_reconnect()
            except Exception as e:
                logger.error(f"[barrage] 启动 DouyinBarrageGrab 失败: {e}")

    @staticmethod
    def _is_grab_process_alive(proc_name: str) -> bool:
        """检测指定进程名是否正在运行。"""
        import os
        try:
            if os.name == "nt":
                import subprocess
                result = subprocess.run(
                    ["tasklist", "/FI", f"IMAGENAME eq {proc_name}*", "/NH"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return proc_name.lower() in result.stdout.lower()
            else:
                import subprocess
                result = subprocess.run(
                    ["pgrep", "-f", proc_name],
                    capture_output=True,
                    timeout=5,
                )
                return result.returncode == 0
        except Exception:
            return False

    # -------------------- Manual control --------------------

    def force_reconnect(self):
        """手动触发 WebSocket 重连（供控制台命令调用）。"""
        self._force_reconnect = True
        logger.info("[barrage] 已标记强制重连，将在下一次循环中断开并重新连接")

    def get_connection_info(self) -> dict:
        """返回连接诊断信息。"""
        now = time.time()
        ws_alive = self._ws is not None and self._ws.open if self._ws else False
        silent = now - self._last_message_time if self._last_message_time else 0

        grab_alive = self._is_grab_process_alive(self.config.grab_process_name)

        return {
            "ws_connected": ws_alive,
            "ws_url": self.config.ws_url,
            "total_received": self._total_received,
            "silent_seconds": round(silent, 1),
            "reconnect_count": self._reconnect_count,
            "grab_process_alive": grab_alive,
            "grab_process_name": self.config.grab_process_name,
            "grab_auto_restart": self.config.grab_auto_restart,
            "grab_restart_on_stale": self.config.grab_restart_on_stale,
            "grab_stale_restart_seconds": self.config.grab_stale_restart_seconds,
            "grab_restart_cooldown_seconds": self.config.grab_restart_cooldown_seconds,
            "stale_grab_restart_count": self._stale_grab_restart_count,
            "last_grab_restart_time": round(self._last_grab_restart_time, 1),
            "custom": self._snapshot_custom_config(),
        }

    # -------------------- 自定义筛选/排序运行时控制 (阶段4) --------------------

    def update_custom_config(self, variables: List[dict]) -> dict:
        """运行时更新自定义筛选/排序配置.

        输入示例:
          [
            {"name":"wealth","enabled":True,"threshold":15,"priority":0},
            {"name":"fan_badge","enabled":False,"threshold":0,"priority":1},
            {"name":"diamond_rank","enabled":True,"threshold":1000,"priority":2}
          ]
        输出: 当前完整状态 dict (同 _snapshot_custom_config).
        说明:
          - 未在输入中出现的变量保持原样.
          - threshold 为 None / 空字符串 / 未填均按 0 处理 (= 不过滤).
          - 切换状态时清空两个自定义队列, 避免旧 sort_key 产生脏数据.
        """
        if not isinstance(variables, list):
            logger.warning(
                f"[barrage][custom] update_custom_config 入参非 list: "
                f"{type(variables)}"
            )
            return self._snapshot_custom_config()

        for entry in variables:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if name not in self.config.custom_variables:
                logger.warning(
                    f"[barrage][custom] 忽略未知变量: {name}"
                )
                continue
            var = self.config.custom_variables[name]
            if "enabled" in entry:
                var.enabled = bool(entry.get("enabled"))
            if "threshold" in entry:
                # None / "" / 未填 → 0
                raw = entry.get("threshold")
                try:
                    var.threshold = int(raw) if raw not in (None, "") else 0
                except (TypeError, ValueError):
                    var.threshold = 0
            if "priority" in entry:
                try:
                    var.priority = int(entry.get("priority"))
                except (TypeError, ValueError):
                    pass

        # 切换状态时清空两个自定义队列, 避免旧 sort_key 残留
        cleared_kw = len(self._custom_keyword_queue)
        cleared_norm = len(self._custom_normal_queue)
        self._custom_keyword_queue.clear()
        self._custom_normal_queue.clear()

        snapshot = self._snapshot_custom_config()
        logger.info(
            f"[barrage][custom] 配置已更新 (清空旧队列 keyword={cleared_kw} "
            f"normal={cleared_norm}): {snapshot}"
        )
        return snapshot

    def _snapshot_custom_config(self) -> dict:
        """返回自定义筛选/排序的完整状态快照."""
        return {
            "active": self._custom_filter_active(),
            "variables": [
                {
                    "name": v.name,
                    "enabled": v.enabled,
                    "threshold": v.threshold,
                    "priority": v.priority,
                }
                for v in sorted(
                    self.config.custom_variables.values(),
                    key=lambda x: x.priority,
                )
            ],
            "queues": {
                "keyword": len(self._custom_keyword_queue),
                "normal": len(self._custom_normal_queue),
                "ttl_dropped_total": self._custom_dropped_count,
                "filtered_total": self._custom_filtered_count,
            },
            "item_ttl": self.config.custom_item_ttl,
            "consume_interval": self.config.custom_consume_interval,
            "session_diamond_user_count": len(
                self._session_diamond_totals
            ),
        }

    # -------------------- B4: 运行时配置热更新 --------------------

    # 可热更新字段白名单 (字段名 → (类型, 副作用描述))
    _RUNTIME_MUTABLE_FIELDS = {
        # 节奏 / TTL
        "consume_interval": float,
        "speech_interval_seconds": float,
        "custom_consume_interval": float,
        "custom_item_ttl": float,
        "stale_message_max_age": float,
        # 内容过滤
        "high_fan_percent": int,
        "high_fan_default_level": int,
        "min_content_length": int,
        "max_content_length": int,
        "max_per_user_in_window": int,
        "dedup_window_seconds": float,
        "blocked_words": list,
        "ignore_exact": list,
        "semantic_dedup_window": float,
        "semantic_dedup_threshold": float,
        # 关键词
        "keyword_list": list,
        # 礼物
        "gift_trigger_enabled": bool,
        "gift_trigger_min_diamonds": int,
        "gift_pending_ttl_when_disabled": float,
        "gift_max_consecutive": int,
        "gift_dedup_window": float,
        "gift_combo_window": float,
        "gift_combo_max_wait": float,
        # 限流
        "response_rate_limit_count": int,
        "response_rate_limit_window": float,
        # 连接健康
        "stale_warn_seconds": float,
        "stale_force_reconnect_seconds": float,
        "health_log_interval": float,
        "grab_restart_on_stale": bool,
        "grab_stale_restart_seconds": float,
        "grab_restart_cooldown_seconds": float,
        "grab_startup_wait_seconds": float,
        # B3 欢迎/关注
        "welcome_enabled": bool,
        "welcome_batch_window": float,
        "welcome_max_batch": int,
        "welcome_template": str,
        "follow_enabled": bool,
        "follow_batch_window": float,
        "follow_max_batch": int,
        "follow_template": str,
        "like_count_track": bool,
        # 自定义筛选
        "custom_queue_max_size": int,
        # 容量上限
        "diamond_max_users": int,
        # metrics
        "metrics_window_seconds": float,
    }

    def update_runtime_config(self, patch: dict) -> dict:
        """运行时热更新 BarrageConfig 中的允许字段.

        - 只接受白名单字段; 未知字段会被忽略并记录.
        - 类型转换尽力而为; 失败的会被跳过.
        - 部分字段有副作用 (例如 keyword_list 需要重建 _keywords_lower).
        - 返回所有实际生效的字段 dict.
        """
        if not isinstance(patch, dict):
            return {"applied": {}, "skipped": [], "error": "patch must be dict"}

        applied: dict = {}
        skipped: List[str] = []

        for key, value in patch.items():
            if key not in self._RUNTIME_MUTABLE_FIELDS:
                skipped.append(key)
                continue
            expected_type = self._RUNTIME_MUTABLE_FIELDS[key]
            try:
                if expected_type is bool:
                    coerced = bool(value)
                elif expected_type is int:
                    coerced = int(value) if value not in (None, "") else 0
                elif expected_type is float:
                    coerced = float(value) if value not in (None, "") else 0.0
                elif expected_type is list:
                    if not isinstance(value, list):
                        skipped.append(key)
                        continue
                    coerced = list(value)
                elif expected_type is str:
                    coerced = str(value)
                else:
                    coerced = value
            except (TypeError, ValueError):
                skipped.append(key)
                continue

            setattr(self.config, key, coerced)
            applied[key] = coerced

        # ---- 副作用同步 ----
        if "keyword_list" in applied:
            self._keywords_lower = [
                kw.lower() for kw in self.config.keyword_list
            ]

        # 重建 filter (如果任何 filter 相关字段被改)
        if {"high_fan_percent", "high_fan_default_level"} & applied.keys():
            self.config.high_fan_percent = max(
                1,
                min(100, int(self.config.high_fan_percent)),
            )
            self.config.high_fan_default_level = max(
                1,
                int(self.config.high_fan_default_level),
            )
            self._rebalance_barrage_priority_queues(
                "runtime-config-updated"
            )

        filter_keys = {
            "min_content_length",
            "max_content_length",
            "dedup_window_seconds",
            "max_per_user_in_window",
            "blocked_words",
            "ignore_exact",
            "semantic_dedup_window",
            "semantic_dedup_threshold",
            "gift_trigger_enabled",
            "gift_trigger_min_diamonds",
        }
        if filter_keys & applied.keys():
            from .barrage_filter import BarrageFilter, FilterConfig
            self.filter = BarrageFilter(FilterConfig(
                min_content_length=self.config.min_content_length,
                max_content_length=self.config.max_content_length,
                dedup_window_seconds=self.config.dedup_window_seconds,
                max_per_user_in_window=self.config.max_per_user_in_window,
                blocked_words=self.config.blocked_words,
                ignore_exact=self.config.ignore_exact,
                semantic_dedup_window=self.config.semantic_dedup_window,
                semantic_dedup_threshold=self.config.semantic_dedup_threshold,
                gift_trigger_enabled=self.config.gift_trigger_enabled,
                gift_trigger_min_diamonds=self.config.gift_trigger_min_diamonds,
            ))
            logger.info("[barrage][config] BarrageFilter 已重建")

        # response_rate_limiter 重置
        if {"response_rate_limit_count",
            "response_rate_limit_window"} & applied.keys():
            from .barrage_filter import init_response_rate_limiter
            init_response_rate_limiter(
                max_responses=self.config.response_rate_limit_count,
                window=self.config.response_rate_limit_window,
            )
            logger.info("[barrage][config] response_rate_limiter 已重置")

        logger.info(
            f"[barrage][config] 热更新生效: {applied}; 跳过: {skipped}"
        )
        return {"applied": applied, "skipped": skipped}

    def get_runtime_config(self) -> dict:
        """返回所有可热更新字段的当前值."""
        return {
            key: getattr(self.config, key)
            for key in self._RUNTIME_MUTABLE_FIELDS
        }

    def reset_session_diamond_totals(self) -> int:
        """重置钻石累计 (切换直播间时可手动调用). 返回清空的用户数."""
        n = len(self._session_diamond_totals)
        self._session_diamond_totals.clear()
        self._session_diamond_nicks.clear()
        logger.info(f"[barrage][custom] 钻石累计已重置 ({n} 个用户)")
        return n

    def get_diamond_ranking(self, top_n: int = 0) -> dict:
        """返回本场钻石数累计排行.

        top_n: 0 = 全部; 否则只返回前 N 名.
        返回结构:
          {
            "total_users": int,            # 累计用户总数
            "total_diamonds": int,         # 累计钻石总数
            "ranking": [
              {"rank": 1, "nickname": "X", "user_id": "...", "diamonds": 12345},
              ...
            ],
          }
        """
        sorted_users = sorted(
            self._session_diamond_totals.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        if top_n and top_n > 0:
            sorted_users = sorted_users[:top_n]
        total_diamonds = sum(self._session_diamond_totals.values())
        ranking = [
            {
                "rank": i + 1,
                "nickname": self._session_diamond_nicks.get(uid, uid),
                "user_id": uid,
                "diamonds": diamonds,
            }
            for i, (uid, diamonds) in enumerate(sorted_users)
        ]
        return {
            "total_users": len(self._session_diamond_totals),
            "total_diamonds": total_diamonds,
            "ranking": ranking,
        }

    # -------------------- Consumer loop --------------------

    async def _consumer_loop(self):
        """消费循环 — 核心调度逻辑

        优先级: 礼物 > 关键词弹幕 > 高级/普通弹幕

        关键词弹幕特殊处理:
          - 入队后立即消费（不等 consume_interval）
          - 如果正在回复普通弹幕 → 等 TTS 完成后立即消费关键词
          - 如果空闲 → 直接消费
          - 关键词弹幕连续消费，不受 consume_interval 间隔限制
        """
        _loop_iter = 0
        while self._running:
            _loop_iter += 1

            # ---- VTuber 状态机门卫：非弹幕模式暂停消费 ----
            if not self._is_barrage_mode():
                if not self.config.gift_thanks_enabled:
                    self._drop_expired_disabled_gifts("not-barrage-mode")
                if _loop_iter % 100 == 1:
                    logger.debug(
                        f"[barrage] consumer paused (not BARRAGE mode), "
                        f"iter={_loop_iter}"
                    )
                await asyncio.sleep(1)
                continue

            try:
                # ---- Step 1: 礼物最优先 (两种模式共用) ----
                if not self.config.gift_thanks_enabled:
                    self._drop_expired_disabled_gifts("gift-thanks-disabled")
                if self._gift_queue_has_ready_item() and self._mode == AdapterMode.BARRAGE:
                    self._mode = AdapterMode.GIFT
                    self._consecutive_gift_count = 0
                    logger.info("[barrage] 切换到 GIFT 模式")

                if self._mode == AdapterMode.GIFT:
                    await self._consume_gift()
                    # 礼物处理完后不 sleep，立即检查关键词队列
                    continue

                # ---- Step 2: 自定义筛选/排序模式 ----
                if self._custom_filter_active():
                    await self._consume_custom_round(_loop_iter)
                    # custom_consume_interval (3s) 由 _consume_custom_round
                    # 内部处理, 这里直接进入下一轮
                    continue

                # ---- Step 3 (原有逻辑): 关键词弹幕优先 ----
                if self.barrage_queue_keyword and not self._gift_queue_has_ready_item():
                    self._log_queue_snapshot(_loop_iter)
                    consumed = await self._consume_all_keyword_barrages()
                    if consumed > 0:
                        # 关键词弹幕消费后不 sleep，立即进入下一轮
                        # （检查是否有新的关键词弹幕或礼物）
                        continue

                # ---- Step 4 (原有逻辑): 普通弹幕 ----
                self._log_queue_snapshot(_loop_iter)
                if await self._consume_normal_barrages():
                    continue

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"[barrage] consumer error: {e}")
                logger.exception(e)

            # 只有普通弹幕消费后才 sleep（关键词弹幕和礼物不等）
            # 期间如果关键词弹幕到达，提前唤醒
            await self._interruptible_sleep(self.config.consume_interval)

    def _is_barrage_mode(self) -> bool:
        """检查 VTuber 状态机是否处于弹幕模式。"""
        try:
            from .vtuber_state_machine import get_vtuber_state_machine, VTuberMode
            sm = get_vtuber_state_machine()
            return sm is None or sm.mode == VTuberMode.BARRAGE
        except Exception:
            return True  # 状态机未初始化时不阻塞

    def _gift_thanks_allowed(self) -> bool:
        return self.config.gift_thanks_enabled and self._is_barrage_mode()

    def _gift_queueing_allowed(self) -> bool:
        return True

    def _gift_queue_has_ready_item(self) -> bool:
        return self._gift_thanks_allowed() and not self.gift_queue.empty()

    def _has_ready_consumer_item(self) -> bool:
        return (
            bool(self.barrage_queue_keyword)
            or not self.barrage_queue_high.empty()
            or not self.barrage_queue_normal.empty()
            or bool(self._custom_keyword_queue)
            or bool(self._custom_normal_queue)
            or self._gift_queue_has_ready_item()
        )

    def _conversation_metadata(self) -> dict:
        metadata: dict[str, Any] = {}
        interval = max(0.0, float(self.config.speech_interval_seconds))
        if interval <= 0 or self._last_conversation_done_monotonic is None:
            return metadata

        audio_not_before = self._last_conversation_done_monotonic + interval
        remaining = audio_not_before - time.monotonic()
        if remaining <= 0:
            return metadata

        metadata["audio_not_before_monotonic"] = audio_not_before
        metadata["audio_gate_reason"] = "barrage_speech_interval"
        metadata["audio_gate_interval_seconds"] = interval
        logger.info(
            "[barrage] next reply can generate immediately; first audio gated "
            "for {:.3f}s",
            remaining,
        )
        return metadata

    def _mark_conversation_done(self, reason: str) -> None:
        self._last_conversation_done_monotonic = time.monotonic()
        logger.debug("[barrage] conversation done marked: {}", reason)

    def _clear_pending_gifts(self, reason: str) -> int:
        cleared = len(self._gift_combo_buffer)
        self._gift_combo_buffer.clear()
        while not self.gift_queue.empty():
            try:
                self.gift_queue.get_nowait()
                cleared += 1
            except asyncio.QueueEmpty:
                break
        if cleared:
            logger.info(
                "[barrage][gift] cleared pending gift thanks: reason={} count={}",
                reason,
                cleared,
            )
        return cleared

    def _drop_expired_disabled_gifts(self, reason: str) -> int:
        if self.config.gift_thanks_enabled:
            return 0

        ttl = float(self.config.gift_pending_ttl_when_disabled)
        if ttl <= 0:
            return 0

        now = time.time()
        kept = []
        dropped = 0
        while not self.gift_queue.empty():
            try:
                msg = self.gift_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if now - msg.timestamp <= ttl:
                kept.append(msg)
            else:
                dropped += 1

        for msg in kept:
            try:
                self.gift_queue.put_nowait(msg)
            except asyncio.QueueFull:
                dropped += 1

        if dropped:
            logger.info(
                "[barrage][gift] dropped expired pending gift thanks: "
                "reason={} count={} ttl={}",
                reason,
                dropped,
                ttl,
            )
        return dropped

    def set_gift_thanks_enabled(self, enabled: bool) -> dict:
        enabled = bool(enabled)
        expired = 0
        flushed = 0
        sorted_gifts = 0
        if enabled and not self.config.gift_thanks_enabled:
            expired = self._drop_expired_disabled_gifts("gift-thanks-enabled")
            flushed = self._flush_all_pending_gift_combos("gift-thanks-enabled")
            sorted_gifts = self._sort_pending_gifts_by_diamonds_once(
                "gift-thanks-enabled"
            )

        self.config.gift_thanks_enabled = enabled
        if not self.config.gift_thanks_enabled:
            expired = self._drop_expired_disabled_gifts("gift-thanks-disabled")
            self._mode = AdapterMode.BARRAGE
            self._consecutive_gift_count = 0
        logger.info(
            "[barrage][gift] gift thanks {}",
            "enabled" if self.config.gift_thanks_enabled else "disabled",
        )
        return {
            "enabled": self.config.gift_thanks_enabled,
            "cleared": 0,
            "expired": expired,
            "flushed_gift_combos": flushed,
            "sorted_gifts": sorted_gifts,
            "pending_gifts": self.gift_queue.qsize(),
        }

    async def _interruptible_sleep(self, seconds: float):
        """Sleep until the next consumer tick, waking early for ready messages.

        This keeps the long idle polling interval from delaying queued barrages.
        """
        elapsed = 0.0
        interval = 0.5
        while elapsed < seconds and self._running:
            await asyncio.sleep(interval)
            elapsed += interval
            if self._has_ready_consumer_item():
                logger.info(
                    f"[barrage] sleep interrupted ({elapsed:.1f}s/{seconds}s): "
                    f"keyword={len(self.barrage_queue_keyword)} "
                    f"high={self.barrage_queue_high.qsize()} "
                    f"normal={self.barrage_queue_normal.qsize()} "
                    f"custom={len(self._custom_keyword_queue) + len(self._custom_normal_queue)} "
                    f"gift={self.gift_queue.qsize()}"
                )
                return

    def _log_queue_snapshot(self, loop_iter: int):
        """打印队列快照（调试用）。"""
        kw_len = len(self.barrage_queue_keyword)
        queue_snapshot = (
            f"keyword={kw_len} "
            f"high={self.barrage_queue_high.qsize()} "
            f"normal={self.barrage_queue_normal.qsize()} "
            f"gift={self.gift_queue.qsize()}"
        )
        logger.debug(
            f"[barrage] consumer iter={loop_iter}, "
            f"mode={self._mode.value}, queues=[{queue_snapshot}]"
        )
        if kw_len > 0:
            kw_preview = " | ".join(
                f"{m.nickname}:{m.raw_data.get('Content', '')}"
                for m in self.barrage_queue_keyword
            )
            logger.info(
                f"[barrage] 关键词队列当前内容 ({kw_len}条): {kw_preview}"
            )

    async def _consume_all_keyword_barrages(self) -> int:
        """连续消费关键词队列中的所有弹幕。

        每消费一条等 TTS 完成后再消费下一条。
        礼物到达时暂停关键词消费。
        返回成功消费的条数。
        """
        if not self._inject_callback:
            return 0

        consumed = 0
        while self.barrage_queue_keyword and self._running:
            # 礼物可抢占关键词
            if self._gift_queue_has_ready_item():
                logger.info(
                    "[barrage] 关键词消费被礼物抢占, 切换到 GIFT 模式"
                )
                self._mode = AdapterMode.GIFT
                self._consecutive_gift_count = 0
                break

            msg = self.barrage_queue_keyword.popleft()
            try:
                logger.info(
                    f"[barrage] ▶ 处理关键词弹幕 ({consumed + 1}): "
                    f"{msg.content}"
                )
                self._metric_tick(self._metrics_consume)
                await self._prepare_avatar_for_message(msg)
                await self._inject_callback(
                    msg.content, msg.user_id, msg.nickname,
                    avatar_url=msg.avatar_url,
                    avatar_path=msg.avatar_path,
                    barrage_priority=msg.priority,
                    is_keyword=True,
                    metadata=self._conversation_metadata(),
                )
                if await self._wait_for_conversation_done():
                    self._mark_conversation_done("keyword")
                consumed += 1
            except Exception as e:
                logger.error(
                    f"[barrage] keyword barrage inject failed: {e}"
                )

        if consumed > 0:
            logger.info(
                f"[barrage] 关键词弹幕本轮消费完成: {consumed} 条, "
                f"剩余: {len(self.barrage_queue_keyword)} 条"
            )
        return consumed

    async def _consume_normal_barrages(self) -> bool:
        """消费普通弹幕（高级 + 普通队列）。

        每条处理完后检查关键词队列和礼物队列，
        有高优先级消息时立即中断。
        """
        # 再次检查礼物队列
        if self._gift_queue_has_ready_item():
            self._mode = AdapterMode.GIFT
            self._consecutive_gift_count = 0
            logger.info("[barrage] 弹幕消费中发现礼物, 切换到 GIFT 模式")
            return False

        if not self._inject_callback:
            return False

        # ---- 取弹幕: 高级队列 + 普通队列各取一条 ----
        picked = []
        try:
            picked.append(self.barrage_queue_high.get_nowait())
        except asyncio.QueueEmpty:
            pass
        try:
            picked.append(self.barrage_queue_normal.get_nowait())
        except asyncio.QueueEmpty:
            pass

        # 若只有一个队列有消息，尝试从另一个队列再取一条
        if len(picked) == 1:
            other_q = (
                self.barrage_queue_normal
                if picked[0].priority <= Priority.FANS_HIGH
                else self.barrage_queue_high
            )
            try:
                picked.append(other_q.get_nowait())
            except asyncio.QueueEmpty:
                pass

        if not picked:
            return False

        # 顺序处理：每条说完后检查高优先级队列
        consumed = False
        for msg in picked:
            try:
                logger.info(f"[barrage] 处理弹幕: {msg.content}")
                self._metric_tick(self._metrics_consume)
                await self._prepare_avatar_for_message(msg)
                await self._inject_callback(
                    msg.content, msg.user_id, msg.nickname,
                    avatar_url=msg.avatar_url,
                    avatar_path=msg.avatar_path,
                    barrage_priority=msg.priority,
                    metadata=self._conversation_metadata(),
                )
                if await self._wait_for_conversation_done():
                    self._mark_conversation_done("normal")
                consumed = True
            except Exception as e:
                logger.error(f"[barrage] inject failed: {e}")

            # 说完这条后检查高优先级消息
            if self._gift_queue_has_ready_item():
                self._mode = AdapterMode.GIFT
                self._consecutive_gift_count = 0
                logger.info("[barrage] 弹幕处理完发现礼物, 切换到 GIFT 模式")
                return consumed
            if self.barrage_queue_keyword:
                logger.info(
                    "[barrage] 弹幕处理完发现关键词弹幕, "
                    "中断普通弹幕立即处理关键词"
                )
                return consumed

        # 本轮结束后清空过期消息（关键词队列不清理）
        self._clear_stale_messages(self.config.stale_message_max_age)
        return consumed

    # -------------------- 自定义模式消费 (阶段4) --------------------

    async def _consume_custom_round(self, loop_iter: int):
        """自定义模式下的一轮消费.

        优先级:
          1. 礼物 (在 _consumer_loop 中已先处理)
          2. 关键词弹幕 (取 sort_key 最高的一条)
          3. 普通弹幕 (取 sort_key 最高的一条)
        每轮消费 1 条, TTS 完成后等待 custom_consume_interval (默认 3s).
        """
        # 1. TTL 清扫 (两个自定义队列)
        self._sweep_custom_queues_stale()

        # 2. 队列快照日志 (每 5 轮一次, 避免噪音)
        if loop_iter % 5 == 1:
            logger.debug(
                f"[barrage][custom] iter={loop_iter} "
                f"keyword={len(self._custom_keyword_queue)} "
                f"normal={len(self._custom_normal_queue)} "
                f"dropped={self._custom_dropped_count} "
                f"filtered={self._custom_filtered_count}"
            )

        # 3. 关键词优先于普通
        if self._custom_keyword_queue:
            msg = self._custom_keyword_queue.pop(0)
            is_keyword = True
        elif self._custom_normal_queue:
            msg = self._custom_normal_queue.pop(0)
            is_keyword = False
        else:
            # 没东西可消费, 浅睡眠避免空转
            await asyncio.sleep(0.5)
            return

        # 4. 注入并等回复完成
        await self._inject_one(msg, is_keyword=is_keyword)
        if await self._wait_for_conversation_done():
            self._mark_conversation_done("custom")

    async def _inject_one(
        self, msg: BarrageMessage, is_keyword: bool
    ):
        """复用现有 _inject_callback 调用模式 (参考 _consume_all_keyword_barrages
        和 _consume_normal_barrages 的调用方式)."""
        if not self._inject_callback:
            return
        try:
            label = "关键词" if is_keyword else "普通"
            logger.info(
                f"[barrage][custom] ▶ 处理{label}弹幕 "
                f"wealth={msg.wealth_level} badge={msg.fan_badge_level} "
                f"diamond={msg.session_diamond_total}: {msg.content}"
            )
            self._metric_tick(self._metrics_consume)
            await self._prepare_avatar_for_message(msg)
            await self._inject_callback(
                msg.content, msg.user_id, msg.nickname,
                avatar_url=msg.avatar_url,
                avatar_path=msg.avatar_path,
                barrage_priority=msg.priority,
                is_keyword=is_keyword,
                metadata=self._conversation_metadata(),
            )
        except Exception as e:
            logger.error(f"[barrage][custom] inject failed: {e}")

    async def _consume_gift(self):
        if not self._inject_callback:
            return

        # 队列空或超出节流上限 → 恢复弹幕模式
        if not self._gift_thanks_allowed():
            if self._gift_queueing_allowed():
                self._drop_expired_disabled_gifts("gift-thanks-not-allowed")
            else:
                self._clear_pending_gifts("gift-thanks-not-allowed")
            self._mode = AdapterMode.BARRAGE
            self._consecutive_gift_count = 0
            return

        if self.gift_queue.empty() or self._consecutive_gift_count >= self.config.gift_max_consecutive:
            if self._consecutive_gift_count >= self.config.gift_max_consecutive:
                dropped = 0
                while not self.gift_queue.empty():
                    try:
                        self.gift_queue.get_nowait()
                        dropped += 1
                    except asyncio.QueueEmpty:
                        break
                if dropped:
                    logger.info(f"[barrage] 礼物节流, 丢弃 {dropped} 条")
            self._mode = AdapterMode.BARRAGE
            self._consecutive_gift_count = 0
            logger.info("[barrage] 切换回 BARRAGE 模式")
            return

        try:
            msg = self.gift_queue.get_nowait()
        except asyncio.QueueEmpty:
            self._mode = AdapterMode.BARRAGE
            self._consecutive_gift_count = 0
            return

        try:
            logger.info(f"[barrage] 处理礼物: {msg.content}")
            self._metric_tick(self._metrics_consume)
            await self._prepare_avatar_for_message(msg)
            await self._inject_callback(
                msg.content, msg.user_id, msg.nickname,
                avatar_url=msg.avatar_url,
                avatar_path=msg.avatar_path,
                barrage_priority=Priority.GIFT_BIG,
                is_gift=True,
                metadata=self._conversation_metadata(),
            )
            if await self._wait_for_conversation_done():
                self._mark_conversation_done("gift")
            self._consecutive_gift_count += 1
        except Exception as e:
            logger.error(f"[barrage] gift inject failed: {e}")

    def _clear_stale_messages(self, max_age_seconds: float = 10.0):
        now = time.time()
        for q in (self.barrage_queue_high, self.barrage_queue_normal):
            kept = []
            while not q.empty():
                try:
                    msg = q.get_nowait()
                    if (now - msg.timestamp) < max_age_seconds:
                        kept.append(msg)
                except asyncio.QueueEmpty:
                    break
            for msg in kept:
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    break
            # only log if we actually dropped something
        logger.debug("[barrage] 清空过期消息完成")

    async def _wait_for_conversation_done(self, timeout: float = 60.0) -> bool:
        """Wait for the current conversation task to finish.

        Two phases:
          1. Wait for a *running* task to appear in current_conversation_tasks
             (the Orchestrator needs time to dequeue the message we just posted
             and create the conversation task).
          2. Once a running task is observed, poll until it completes.
        """
        if not self._ws_handler:
            return False

        start = time.time()

        def _has_active_task() -> bool:
            for uid, task in self._ws_handler.current_conversation_tasks.items():
                if task and not task.done():
                    return True
            return False

        # Phase 1 — wait for a running task to appear (up to 5 s)
        task_appeared = False
        while self._running and (time.time() - start) < 3.0:
            if _has_active_task():
                task_appeared = True
                break
            await asyncio.sleep(0.1)

        if not task_appeared:
            logger.debug(
                "[barrage] no active conversation task appeared within 5 s, "
                "proceeding"
            )
            return False

        # Phase 2 — wait for the running task to finish
        while self._running and (time.time() - start) < timeout:
            if not _has_active_task():
                return True
            await asyncio.sleep(0.1)

        logger.debug("[barrage] wait for conversation done timed out, proceeding")
        return False

    # -------------------- Status --------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def queue_size(self) -> int:
        return (
            self.barrage_queue_high.qsize()
            + self.barrage_queue_normal.qsize()
            + len(self.barrage_queue_keyword)
            + self.gift_queue.qsize()
        )

    @property
    def mode(self) -> AdapterMode:
        return self._mode

    def get_queue_status(self) -> dict:
        """返回所有队列的详细状态（含关键词队列完整内容），供调试用。"""
        now = time.time()

        keyword_items = []
        for msg in self.barrage_queue_keyword:
            keyword_items.append({
                "nickname": msg.nickname,
                "content": msg.content,
                "age_s": round(now - msg.timestamp, 1),
                "priority": msg.priority,
            })

        ws_alive = False
        try:
            ws_alive = self._ws is not None and self._ws.open
        except Exception:
            pass

        return {
            "mode": self._mode.value,
            "running": self._running,
            "connection": {
                "ws_alive": ws_alive,
                "total_received": self._total_received,
                "silent_seconds": round(now - self._last_message_time, 1)
                if self._last_message_time
                else None,
                "reconnect_count": self._reconnect_count,
            },
            "keyword_queue": {
                "size": len(self.barrage_queue_keyword),
                "items": keyword_items,
            },
            "high_queue_size": self.barrage_queue_high.qsize(),
            "normal_queue_size": self.barrage_queue_normal.qsize(),
            "gift_queue_size": self.gift_queue.qsize(),
            "total": self.queue_size,
            "gift_thanks_enabled": self.config.gift_thanks_enabled,
            "high_fan": self.high_fan_status(),
            "custom": self._snapshot_custom_config(),
        }


# ============================================================
# Integration helpers
# ============================================================

def create_barrage_inject_callback(ws_handler, target_client_uid: str):
    """Create inject callback that routes through Orchestrator if available"""

    async def inject(
        text: str,
        user_id: str = "",
        nickname: str = "",
        avatar_url: str = "",
        avatar_path: str = "",
        barrage_priority: int = Priority.NORMAL,
        is_gift: bool = False,
        is_keyword: bool = False,
        metadata: Optional[dict] = None,
    ):
        orch = get_orchestrator()
        if orch:
            # 灯牌优先级 → Orchestrator 优先级映射
            if is_gift:
                orch_priority = MsgPriority.VIP
            elif is_keyword:
                # 关键词弹幕：用户主动触发，给予 CONTINUE 优先级
                # 这样在 Orchestrator 中不会被 DISCARD，而是 MERGE
                orch_priority = MsgPriority.CONTINUE
            elif barrage_priority <= Priority.FANS_HIGH:
                orch_priority = MsgPriority.VIP
            elif barrage_priority <= Priority.FANS_LOW:
                orch_priority = MsgPriority.CONTINUE
            else:
                orch_priority = MsgPriority.BARRAGE

            await orch.put_message(OrchestratorMessage(
                priority=orch_priority,
                source=MsgSource.BARRAGE,
                text=text,
                user_id=user_id,
                nickname=nickname,
                extra={
                    "is_gift": is_gift,
                    "is_keyword": is_keyword,
                    "barrage_priority": barrage_priority,
                    "avatar_url": avatar_url,
                    "avatar_path": avatar_path,
                    "metadata": metadata or {},
                },
            ))
            logger.info(
                f"[barrage] routed via Orchestrator "
                f"(P{orch_priority}"
                f"{'|keyword' if is_keyword else ''}"
                f"{'|gift' if is_gift else ''}): "
                f"{text[:50]}..."
            )
            return

        # Fallback: direct inject if Orchestrator not started
        if target_client_uid not in ws_handler.client_contexts:
            logger.warning(f"[barrage] target client {target_client_uid} not found, skipping")
            return

        websocket = ws_handler.client_connections.get(target_client_uid)
        if not websocket:
            logger.warning(f"[barrage] target client {target_client_uid} has no WebSocket")
            return

        fake_msg = {
            "type": "text-input",
            "text": text,
            "metadata": {
                "human_name": "礼物" if is_gift else "弹幕",
                **(metadata or {}),
            },
        }

        logger.info(f"[barrage] direct inject to {target_client_uid}: {text[:50]}...")

        barrage_content = text or ""
        prefix = f"[barrage] {nickname}: "
        if nickname and barrage_content.startswith(prefix):
            barrage_content = barrage_content[len(prefix):]
        elif barrage_content.startswith("[barrage]"):
            _, sep, tail = barrage_content.partition(": ")
            if sep:
                barrage_content = tail
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "barrage-display",
                        "nickname": nickname or "",
                        "user_id": user_id or "",
                        "avatar_url": avatar_url or "",
                        "avatar_path": avatar_path or "",
                        "content": barrage_content,
                    },
                    ensure_ascii=False,
                )
            )
        except Exception as exc:
            logger.debug(f"[barrage] direct barrage-display failed: {exc}")

        await ws_handler._handle_conversation_trigger(
            websocket, target_client_uid, fake_msg
        )

    return inject


# ============================================================
# Start/stop entry points (called from server.py)
# ============================================================

_adapter_instance: Optional[BarrageAdapter] = None


async def start_barrage_adapter(
    ws_handler,
    config: Optional[BarrageConfig] = None,
) -> BarrageAdapter:
    global _adapter_instance

    adapter = BarrageAdapter(config=config)
    adapter._ws_handler = ws_handler
    _adapter_instance = adapter

    # 初始化回复频率限制器
    from .barrage_filter import init_response_rate_limiter
    init_response_rate_limiter(
        max_responses=adapter.config.response_rate_limit_count,
        window=adapter.config.response_rate_limit_window,
    )

    async def wait_and_bind():
        logger.info("[barrage] waiting for browser client connection...")
        while adapter.is_running:
            if ws_handler.client_connections:
                target_uid = next(iter(ws_handler.client_connections))
                callback = create_barrage_inject_callback(ws_handler, target_uid)
                adapter.set_inject_callback(callback)
                logger.info(f"[barrage] bound to client: {target_uid}")

                while adapter.is_running:
                    await asyncio.sleep(2)
                    if target_uid not in ws_handler.client_connections:
                        logger.info(f"[barrage] client {target_uid} disconnected, rebinding...")
                        break

                adapter.set_inject_callback(None)
                continue

            await asyncio.sleep(1)

    await adapter.start()
    asyncio.create_task(wait_and_bind())

    return adapter


async def stop_barrage_adapter():
    global _adapter_instance
    if _adapter_instance:
        await _adapter_instance.stop()
        _adapter_instance = None


