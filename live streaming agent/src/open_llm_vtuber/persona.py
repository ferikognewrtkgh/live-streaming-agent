"""
persona.py - Persona 状态层

不是决策模块，而是上下文载体。
记录 AI 当前的情绪、对各用户的好感度、行为倾向，
供 Orchestrator 决策时读取，以及注入 LLM prompt。

使用方式:
    persona = Persona()

    # 收到弹幕时更新好感度
    persona.on_barrage(user_id="abc", nickname="小明", content="主播好棒！")

    # 收到礼物时更新情绪和好感度
    persona.on_gift(user_id="abc", nickname="小明", diamonds=100)

    # Orchestrator 读取当前状态注入 prompt
    context = persona.to_prompt_context()

    # LLM 调用前把 context 拼到 system prompt 末尾
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional
from loguru import logger


# ============================================================
# 情绪定义
# ============================================================

class Emotion(str, Enum):
    HAPPY = "开心"
    EXCITED = "兴奋"
    MOVED = "感动"
    ANNOYED = "烦"
    SLACKING = "摆烂"


# 情绪对应的行为倾向
_EMOTION_TO_TENDENCY = {
    Emotion.HAPPY:    "主动",
    Emotion.EXCITED:  "主动",
    Emotion.MOVED:    "哄人",
    Emotion.ANNOYED:  "怼人",
    Emotion.SLACKING: "摆烂",
}

# 情绪对应的 LLM prompt 描述
_EMOTION_PROMPT = {
    Emotion.HAPPY:    "你现在心情很好，说话轻松愉快，偶尔会开玩笑。",
    Emotion.EXCITED:  "你现在非常兴奋，说话节奏快，用词活泼，容易被新话题带走。",
    Emotion.MOVED:    "你现在心里有些触动，说话温柔，更愿意好好听别人说话。",
    Emotion.ANNOYED:  "你现在有点烦，说话会更直接甚至有些刺，但不会真的发火。",
    Emotion.SLACKING: "你现在有点摆烂，不太想认真说话，回复会短一些、随意一些。",
}


# ============================================================
# 好感度系统
# ============================================================

@dataclass
class UserAffinity:
    """单个用户的好感度记录"""
    user_id: str
    nickname: str = ""

    # 好感度分值 (0 ~ 100)
    score: float = 50.0

    # 互动统计
    barrage_count: int = 0       # 累计弹幕数
    gift_count: int = 0          # 累计礼物数
    gift_diamonds: int = 0       # 累计礼物钻石数

    # 最近互动时间戳
    last_seen: float = field(default_factory=time.time)

    # 是否是粉丝团成员
    is_fan_club: bool = False

    def tier(self) -> str:
        """根据好感度返回等级标签"""
        if self.score >= 80:
            return "老铁"
        elif self.score >= 60:
            return "熟人"
        elif self.score >= 40:
            return "路人"
        else:
            return "陌生人"


class AffinitySystem:
    """
    好感度权重系统

    好感度变化规则:
      弹幕         +0.5 (有上限，防刷屏刷好感)
      礼物         +钻石数 * 0.1 (上限 +20)
      粉丝团       首次识别 +10
      长时间未互动  随时间缓慢衰减 (每小时 -1，下限 20)
    """

    # 好感度上下限
    MAX_SCORE = 100.0
    MIN_SCORE = 0.0
    DEFAULT_SCORE = 50.0

    # 衰减配置
    DECAY_RATE_PER_HOUR = 1.0     # 每小时衰减 1 分
    DECAY_MIN_SCORE = 20.0        # 衰减下限

    # 弹幕好感度增益上限 (防止刷屏刷好感)
    BARRAGE_GAIN = 0.5
    BARRAGE_DAILY_CAP = 5.0       # 每自然日最多从弹幕获得 5 分

    def __init__(self):
        self._users: Dict[str, UserAffinity] = {}
        self._barrage_daily: Dict[str, float] = {}  # {user_id: today_gained}

    def get(self, user_id: str) -> UserAffinity:
        """获取用户好感度记录，不存在则创建"""
        if user_id not in self._users:
            self._users[user_id] = UserAffinity(user_id=user_id)
        return self._users[user_id]

    def on_barrage(self, user_id: str, nickname: str, is_fan_club: bool = False):
        """弹幕事件: 增加好感度"""
        u = self.get(user_id)
        u.nickname = nickname
        u.barrage_count += 1
        u.last_seen = time.time()

        # 粉丝团首次识别加分
        if is_fan_club and not u.is_fan_club:
            u.is_fan_club = True
            self._add_score(user_id, 10.0)
            logger.debug(f"[Persona] 粉丝团首次识别 {nickname}: +10")

        # 弹幕加分 (有日上限)
        today_key = f"{user_id}_{self._today()}"
        gained_today = self._barrage_daily.get(today_key, 0.0)
        if gained_today < self.BARRAGE_DAILY_CAP:
            gain = min(self.BARRAGE_GAIN, self.BARRAGE_DAILY_CAP - gained_today)
            self._add_score(user_id, gain)
            self._barrage_daily[today_key] = gained_today + gain

    def on_gift(self, user_id: str, nickname: str, diamonds: int):
        """礼物事件: 按钻石数增加好感度"""
        u = self.get(user_id)
        u.nickname = nickname
        u.gift_count += 1
        u.gift_diamonds += diamonds
        u.last_seen = time.time()

        gain = min(diamonds * 0.1, 20.0)
        self._add_score(user_id, gain)
        logger.debug(f"[Persona] 礼物 {nickname} {diamonds}钻: +{gain:.1f}")

    def decay_all(self):
        """对所有用户执行好感度衰减（建议每小时调用一次）"""
        now = time.time()
        for uid, u in self._users.items():
            hours_since = (now - u.last_seen) / 3600
            if hours_since > 1 and u.score > self.DECAY_MIN_SCORE:
                decay = min(
                    hours_since * self.DECAY_RATE_PER_HOUR,
                    u.score - self.DECAY_MIN_SCORE
                )
                u.score = max(u.score - decay, self.DECAY_MIN_SCORE)

    def top_users(self, n: int = 3) -> list:
        """返回好感度最高的 N 个用户"""
        return sorted(self._users.values(), key=lambda u: u.score, reverse=True)[:n]

    def _add_score(self, user_id: str, delta: float):
        u = self._users[user_id]
        u.score = max(self.MIN_SCORE, min(self.MAX_SCORE, u.score + delta))

    @staticmethod
    def _today() -> str:
        from datetime import date
        return date.today().isoformat()


# ============================================================
# Persona 状态层主类
# ============================================================

class Persona:
    """
    Persona 状态层 — AI 当前状态的实时合集

    字段:
      emotion       : 当前情绪 (Emotion 枚举)
      emotion_value : 情绪强度 0~100
      tendency      : 行为倾向 (主动/怼人/哄人/摆烂)
      affinity      : 好感度系统
    """

    # 情绪衰减配置：一段时间没有新刺激，情绪向中性恢复
    # 注意：直播间有人互动=积极状态，衰减目标应为 HAPPY（低强度）
    EMOTION_DECAY_TO = Emotion.HAPPY       # 默认衰减目标情绪
    EMOTION_DECAY_SECONDS = 600            # 10 分钟后开始往中性衰减（直播间节奏慢是正常的）

    # 情绪触发阈值：情绪强度超过此值才会影响决策
    EMOTION_INFLUENCE_THRESHOLD = 60

    # 连续无人互动多久才允许进入摆烂（秒）
    SLACKING_SILENCE_THRESHOLD = 900       # 15 分钟完全无弹幕才考虑摆烂

    def __init__(self):
        self.emotion: Emotion = Emotion.HAPPY
        self.emotion_value: float = 70.0       # 初始情绪强度
        self.tendency: str = "主动"
        self.affinity = AffinitySystem()

        self._last_emotion_trigger: float = time.time()
        self._last_barrage_time: float = 0.0  # 最近一条弹幕时间（不论是否触发shift）

    # ──────────────────────────────────────────────
    # 事件接口 (外部调用)
    # ──────────────────────────────────────────────

    def on_barrage(
        self,
        user_id: str,
        nickname: str,
        content: str,
        is_fan_club: bool = False,
    ):
        """弹幕事件：更新好感度 + 情绪

        任何弹幕都代表直播间有人在互动，应该产生正向情绪刺激。
        """
        self.affinity.on_barrage(user_id, nickname, is_fan_club)
        self._last_barrage_time = time.time()

        affinity = self.affinity.get(user_id)

        if affinity.score >= 60:
            # 熟人以上发言 → 开心 +5
            self._shift_emotion(Emotion.HAPPY, +5)
        elif len(content) < 3:
            # 无意义刷屏 → 微烦 +2（降低负面影响）
            self._shift_emotion(Emotion.ANNOYED, +2)
        else:
            # 普通用户正常弹幕 → 也给小幅正向刺激
            # 这是最关键的修复：有人互动 = 有人看 = 开心
            self._shift_emotion(Emotion.HAPPY, +2)

    def on_gift(self, user_id: str, nickname: str, diamonds: int):
        """礼物事件：更新好感度 + 情绪"""
        self.affinity.on_gift(user_id, nickname, diamonds)

        if diamonds >= 500:
            self._shift_emotion(Emotion.EXCITED, +20)
        elif diamonds >= 100:
            self._shift_emotion(Emotion.MOVED, +10)
        elif diamonds >= 10:
            self._shift_emotion(Emotion.HAPPY, +5)

    def on_silence(self, seconds: float):
        """沉默事件：直播间一段时间没人说话

        直播间偶尔冷场是正常的，不要太快摆烂。
        只有持续沉默很久才给负面影响。
        """
        if seconds > 300:
            # 5 分钟完全没人，开始有点无聊
            self._shift_emotion(Emotion.SLACKING, +8)
        elif seconds > 120:
            # 2 分钟没人，轻微影响
            self._shift_emotion(Emotion.SLACKING, +3)

    def on_ai_spoke(self, response_text: str):
        """AI 说完话后的状态更新（情绪自然消耗）

        说话本身只造成轻微消耗，不应该快速耗尽情绪。
        """
        # 只消耗 1 点（原来 3 点太激进，14 次回复就接近摆烂阈值）
        self.emotion_value = max(20, self.emotion_value - 1)
        self._sync_tendency()

    # ──────────────────────────────────────────────
    # Orchestrator 决策辅助接口
    # ──────────────────────────────────────────────

    def should_respond(self, user_id: str, content: str) -> bool:
        """
        判断是否应该回复这条消息。
        Orchestrator 调用此方法做第一层过滤。

        规则:
          - 摆烂状态下（真的很久没人了），低好感用户有 30% 概率不回
          - 烦躁状态下，陌生人有 20% 概率被忽略
          - 其他情况都回复（保证互动率）
        """
        import random

        affinity = self.affinity.get(user_id)

        # 摆烂状态：只有真正长时间无人才会到这里，适当降低回复率
        if self.emotion == Emotion.SLACKING:
            if affinity.score < 50:
                if random.random() < 0.3:
                    logger.debug(
                        f"[Persona] 摆烂状态下忽略低好感用户 {affinity.nickname}"
                    )
                    return False

        # 烦躁状态：只对陌生人有小概率忽略
        if self.emotion == Emotion.ANNOYED and self.emotion_value > 70:
            if affinity.score < 40:
                if random.random() < 0.2:
                    logger.debug(
                        f"[Persona] 烦躁状态下忽略陌生用户 {affinity.nickname}"
                    )
                    return False

        return True

    def interrupt_priority_boost(self, user_id: str) -> int:
        """
        根据好感度提升消息的有效优先级。
        好感度高的用户消息可以获得更高的打断权。
        返回值：优先级数值（越小越高）。
        """
        affinity = self.affinity.get(user_id)
        score = affinity.score

        if score >= 80:
            return -1   # 老铁：优先级提升 1 级
        elif score >= 60:
            return 0    # 熟人：不变
        else:
            return +1   # 陌生人：优先级降低 1 级

    # ──────────────────────────────────────────────
    # LLM prompt 注入接口
    # ──────────────────────────────────────────────

    def to_prompt_context(self, current_user_id: Optional[str] = None) -> str:
        """
        生成注入 LLM system prompt 的上下文字符串。

        示例输出:
            [当前状态]
            情绪: 开心 (强度 75/100)
            行为倾向: 主动
            你现在心情很好，说话轻松愉快，偶尔会开玩笑。

            [当前对话用户]
            昵称: 小明 | 好感度: 72 (熟人)
        """
        self._maybe_decay_emotion()

        lines = ["[当前状态]"]
        lines.append(
            f"情绪: {self.emotion.value} (强度 {int(self.emotion_value)}/100)"
        )
        lines.append(f"行为倾向: {self.tendency}")
        lines.append(_EMOTION_PROMPT.get(self.emotion, ""))

        if current_user_id:
            u = self.affinity.get(current_user_id)
            if u.nickname:
                lines.append("")
                lines.append("[当前对话用户]")
                lines.append(
                    f"昵称: {u.nickname} | 好感度: {int(u.score)} ({u.tier()})"
                )

        # 加入最高好感用户（老铁）信息
        top = self.affinity.top_users(3)
        vips = [u for u in top if u.score >= 70 and u.nickname]
        if vips:
            lines.append("")
            lines.append("[直播间老铁]")
            for v in vips:
                lines.append(f"- {v.nickname} (好感度 {int(v.score)})")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """调试用：返回当前 Persona 状态字典"""
        return {
            "emotion": self.emotion.value,
            "emotion_value": round(self.emotion_value, 1),
            "tendency": self.tendency,
            "top_users": [
                {
                    "nickname": u.nickname,
                    "score": round(u.score, 1),
                    "tier": u.tier(),
                }
                for u in self.affinity.top_users(5)
            ],
        }

    # ──────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────

    def _shift_emotion(self, target: Emotion, delta: float):
        """
        向目标情绪靠近，同时调整情绪强度。

        如果目标情绪和当前不同:
          - 先降低当前情绪强度
          - 降到 30 以下时切换到目标情绪
        如果目标情绪和当前相同:
          - 直接增强强度
        """
        self._last_emotion_trigger = time.time()

        if target == self.emotion:
            self.emotion_value = min(100, self.emotion_value + delta)
        else:
            # 不同情绪：先削弱当前，再看要不要切换
            self.emotion_value = max(0, self.emotion_value - delta * 0.5)
            if self.emotion_value < 30:
                old = self.emotion
                self.emotion = target
                self.emotion_value = 40 + delta
                logger.debug(
                    f"[Persona] 情绪切换: {old.value} → {target.value} ({int(self.emotion_value)})"
                )

        self._sync_tendency()

    def _sync_tendency(self):
        """根据当前情绪同步行为倾向"""
        self.tendency = _EMOTION_TO_TENDENCY.get(self.emotion, "主动")

    def _maybe_decay_emotion(self):
        """
        自然衰减：如果长时间没有新事件，情绪强度缓慢下降。

        分两层判断:
          1. 有弹幕但无强刺激 → 缓慢回归 HAPPY(50)，不会摆烂
          2. 完全无人互动 (>15分钟无弹幕) → 才可能衰减到摆烂
        """
        now = time.time()
        elapsed_since_trigger = now - self._last_emotion_trigger
        elapsed_since_barrage = (
            now - self._last_barrage_time if self._last_barrage_time > 0 else float("inf")
        )

        if elapsed_since_trigger < self.EMOTION_DECAY_SECONDS:
            return

        # 缓慢衰减：情绪强度向 50 回归
        decay_rate = (elapsed_since_trigger - self.EMOTION_DECAY_SECONDS) / 600
        decay = min(decay_rate * 5, max(0, self.emotion_value - 40))

        if decay > 0:
            self.emotion_value = max(40, self.emotion_value - decay)

        # 摆烂判断：仅当完全无人互动超过阈值时
        if (
            elapsed_since_barrage > self.SLACKING_SILENCE_THRESHOLD
            and self.emotion != Emotion.SLACKING
        ):
            self.emotion = Emotion.SLACKING
            self.emotion_value = 30
            self.tendency = "摆烂"
            logger.info(
                f"[Persona] {elapsed_since_barrage:.0f}秒无弹幕互动，进入摆烂"
            )
        elif (
            self.emotion not in (Emotion.HAPPY, Emotion.SLACKING)
            and self.emotion_value <= 45
        ):
            # 非摆烂的负面情绪在无刺激时回归 HAPPY
            self.emotion = Emotion.HAPPY
            self.tendency = "主动"
            logger.debug("[Persona] 情绪自然回归到开心")


# ============================================================
# 全局单例
# ============================================================

_persona_instance: Optional[Persona] = None


def get_persona() -> Persona:
    """获取全局 Persona 实例（懒加载）"""
    global _persona_instance
    if _persona_instance is None:
        _persona_instance = Persona()
    return _persona_instance
