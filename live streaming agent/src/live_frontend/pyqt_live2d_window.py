from __future__ import annotations

import base64
import json
import math
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from loguru import logger

import numpy as np
import live2d.v3 as live2d
import pygame
import silero_vad
import torch
import websocket
try:
    from live_frontend.local_douyin_barrage import (
        DEFAULT_LOCAL_BARRAGE_WS_URL,
        detect_local_link_anchor_candidate,
    )
except ModuleNotFoundError:
    from local_douyin_barrage import (
        DEFAULT_LOCAL_BARRAGE_WS_URL,
        detect_local_link_anchor_candidate,
    )
from PyQt5.QtCore import (
    QBuffer,
    QByteArray,
    QElapsedTimer,
    QEvent,
    QIODevice,
    QSize,
    QTimer,
    Qt,
    QUrl,
    pyqtSignal,
)
from PyQt5.QtGui import (
    QColor,
    QFont,
    QImage,
    QIntValidator,
    QPainter,
    QPainterPath,
    QPixmap,
    QSurfaceFormat,
)
from PyQt5.QtMultimedia import QAudio, QAudioDeviceInfo, QAudioFormat, QAudioInput
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QOpenGLWidget,
)

def get_runtime_roots() -> tuple[Path, Path]:
    if getattr(sys, "frozen", False):
        app_root = Path(sys.executable).resolve().parent
        bundle_root = Path(getattr(sys, "_MEIPASS", app_root)).resolve()
        return app_root, bundle_root

    source_root = Path(__file__).resolve().parent
    return source_root.parents[1], source_root


PROJECT_ROOT, LIVE_FRONTEND_BUNDLE_ROOT = get_runtime_roots()
LIVE_FRONTEND_RESOURCE_ROOT = LIVE_FRONTEND_BUNDLE_ROOT / "resource"
LIVE2D_RESOURCE_ROOT = LIVE_FRONTEND_RESOURCE_ROOT / "live2d-models"
MODEL_DICT_PATH = LIVE_FRONTEND_RESOURCE_ROOT / "model_dict.json"
LIVE_STREAMING_AGENT_ICON_ROOT = LIVE_FRONTEND_RESOURCE_ROOT / "live_streaming_agent_icons"
# Agent 字幕窗口默认尺寸 (启动时大小, 用户可随意拖拽缩放)
# 头像和字号会跟随窗口高度按比例响应式调整
LIVE_STREAMING_AGENT_SUBTITLE_DEFAULT_WIDTH = 900
LIVE_STREAMING_AGENT_SUBTITLE_DEFAULT_HEIGHT = 180
# Agent 字幕按钮文字 (开/关状态文字相同, 用颜色区分)
LIVE_STREAMING_AGENT_SUBTITLE_BUTTON_TEXT = "Agent 字幕"
# 弹幕字幕窗口默认尺寸 (展示被回复的弹幕: 用户抖音 id + 弹幕原文)
# 比例参照 psd (回复弹幕界面.psd) 卡片约 2.78:1
BARRAGE_SUBTITLE_DEFAULT_WIDTH = 720
BARRAGE_SUBTITLE_DEFAULT_HEIGHT = 260
# 弹幕字幕按钮文字 (开/关状态文字相同, 用颜色区分)
BARRAGE_SUBTITLE_BUTTON_TEXT = "弹幕字幕"
DEFAULT_MODEL_NAME = "dream"
LIVE2D_PARAM_SILENCE_ID = "ParamSilence"
LIVE2D_WATERMARK_EXPRESSION_NAME = "水印"
LIVE2D_WAKE_VOICE_DELAY_SECONDS = 8.0
LIVE2D_NORMAL_MOTION_GROUP = "normal_motion"
LIVE2D_NORMAL_MOTION_PRIORITY = 3
LIVE2D_NORMAL_MOTION_RETRY_SECONDS = 0.6
# 动作播完 (尤其是表情同步/自动回归动作) 会僵在极端姿态并冻结所有参数。
# 直接 ResetAllParameters 会瞬间归位, 造成明显卡顿。改为在这段时间内把所有参数
# 从冻结值平滑插值回默认值, 消除回归待机时的顿挫感。
LIVE2D_RETURN_FADE_SECONDS = 0.38
LIVE2D_EXPRESSION_SYNC_MOTION_GROUP = "expression_sync_motion"
LIVE2D_EXPRESSION_SYNC_MOTION_PRIORITY = 3
LIVE2D_EXPRESSION_SYNC_IDLE_DELAY_SECONDS = 2.4
LIVE2D_EXPRESSION_SYNC_MOTION_FILES = (
    "motions/左右摇摆.motion3.json",
    "motions/左右张望.motion3.json",
    "motions/向右歪头.motion3.json",
    "motions/摇头.motion3.json",
    "motions/点头.motion3.json",
)
# Live2D 的原生 CubismJson 解析器不接受科学计数法 (例如 1e-7) 和 NaN/Infinity,
# 但 Python 的 json 能读。含这类 token 的 motion3.json 会加载失败: LoadExtraMotion
# 不会报错, 而是返回未变化的分组大小 -> 会缓存错误的 index 并播错动作。加载前先拦截。
LIVE2D_CUBISM_UNSAFE_NUMBER_RE = re.compile(
    r"(?<![\w.])(?:-?\d+\.?\d*[eE][-+]?\d+|NaN|Infinity)(?![\w])"
)
LIVE2D_STATE_MOTION_EMOTION_TAGS = {"sleep", "wake"}
LIVE2D_AUTO_RETURN_MOTION_FILES = {
    "motions/motion4-开心动作.motion3.json",
    "motions/motion5卖萌动作.motion3.json",
    "motions/Exp-1哭哭motion.motion3.json",
}
LIVE2D_NORMAL_MOTION_FILES = (
    "motions/motion1-默认动作.motion3.json",
)
DEFAULT_WS_HOST = "127.0.0.1"
DEFAULT_WS_PORT = "12393"
BACKEND_WS_PATH = "/client-ws"
DEFAULT_WS_URL = f"ws://{DEFAULT_WS_HOST}:{DEFAULT_WS_PORT}{BACKEND_WS_PATH}"
DEFAULT_LIVE2D_WIDTH = 720
DEFAULT_LIVE2D_HEIGHT = 960
CONSOLE_MIN_WIDTH = 760
DIRECTOR_CONSOLE_MIN_WIDTH = 840
CONSOLE_MIN_HEIGHT = 780
DIRECTOR_CONSOLE_MIN_HEIGHT = 900
CONSOLE_STREAMER_MIN_HEIGHT = 430
CONSOLE_BUTTON_WIDTH = 112
CONSOLE_BUTTON_HEIGHT = 44
CONSOLE_WS_HOST_INPUT_WIDTH = 240
CONSOLE_WS_PORT_INPUT_WIDTH = 90
CONSOLE_WS_CONTROL_HEIGHT = 38
CONSOLE_REPLY_PERCENT_COMBO_WIDTH = 43
CONSOLE_COLD_TIME_COMBO_MIN_WIDTH = 58
QT_MAX_SIZE = 16777215
UI_SCALE_OPTIONS = (0.9, 1.0, 1.15, 1.3)
DIRECTOR_METRIC_PANEL_WIDTH = 270
DIRECTOR_STORY_PANEL_WIDTH = 780
DIRECTOR_STORY_PANEL_MIN_HEIGHT = 540
DIRECTOR_STORY_ROW_MIN_HEIGHT = 104
DIRECTOR_METRIC_INPUT_WIDTH = 96
PERFORMANCE_MONITOR_MAX_TURNS = 200
PERFORMANCE_METRIC_FIELDS = (
    ("user_speech_seconds", "用户说话用时"),
    ("asr_seconds", "ASR 用时"),
    ("knowledge_seconds", "知识库用时"),
    ("web_search_seconds", "联网搜索用时"),
    ("llm_first_token_seconds", "大模型首字用时"),
    ("llm_first_sentence_seconds", "大模型首句用时"),
    ("llm_total_seconds", "大模型完整输出用时"),
    ("tts_first_audio_seconds", "TTS 首音用时"),
    ("tts_total_seconds", "TTS 完整用时"),
    ("speech_end_to_audio_start_seconds", "用户说完到首音"),
)
PERFORMANCE_WARNING_DELTA_SECONDS = 0.2
PERFORMANCE_PROGRESS_MAX_SECONDS = 2.0
PERFORMANCE_COLOR_FAST = "#34c759"
PERFORMANCE_COLOR_WARNING = "#ffcc00"
PERFORMANCE_COLOR_SLOW = "#ff3b30"
PERFORMANCE_COLOR_UNRATED = "#5e8fd8"
PERFORMANCE_HIGHLIGHTED_METRICS = {
    "asr_seconds",
    "knowledge_seconds",
    "web_search_seconds",
    "llm_first_token_seconds",
    "llm_first_sentence_seconds",
    "tts_first_audio_seconds",
    "speech_end_to_audio_start_seconds",
}
PERFORMANCE_ALWAYS_BLUE_METRICS = {
    "user_speech_seconds",
    "llm_total_seconds",
    "tts_total_seconds",
}
PERFORMANCE_STATE_LABELS = {
    "idle": "等待",
    "thinking": "思考",
    "speaking": "说话",
    "interrupting": "打断",
}
VISION_IMAGE_MAX_BYTES = 10 * 1024 * 1024
VISION_IMAGE_PREVIEW_HEIGHT = 150
VISION_CONTEXT_MODE_PERSISTENT = "vision_persistent"
IMAGE_MODE_BUTTON_TEXT = "图片模式"
IMAGE_MODE_BUTTON_ACTIVE_TEXT = "关闭图片"
GAME_VISION_BUTTON_TEXT = "\u6e38\u620f\u8bc6\u56fe"
GAME_VISION_COLD_IDLE_SECONDS = 10
GAME_VISION_SCREENSHOT_MAX_EDGE = 1600
GAME_VISION_SCREENSHOT_JPEG_QUALITY = 85
PAINT_BUTTON_INACTIVE_TEXT = "开启画图"
PAINT_BUTTON_ACTIVE_TEXT = "关闭画图"
PAINT_WINDOW_DEFAULT_WIDTH = 720
PAINT_WINDOW_DEFAULT_HEIGHT = 560
PAINT_LOADING_INTERVAL_MS = 350
GAME_VISION_COLD_PROMPT = (
    "\u3010\u6e38\u620f\u8bc6\u56fe\u51b7\u573a\u3011\u4e3b\u64ad\u5df2\u7ecf"
    f"{GAME_VISION_COLD_IDLE_SECONDS}\u79d2\u6ca1\u6709\u8bf4\u8bdd\u3002"
    "\u8bf7\u89c2\u5bdf\u5f53\u524d\u6e38\u620f\u753b\u9762\uff0c\u4e3b\u52a8"
    "\u63a5\u4e00\u53e5\u9002\u5408\u76f4\u64ad\u95f4\u7684\u56de\u590d\uff1a"
    "\u53ef\u4ee5\u70b9\u8bc4\u5c40\u52bf\u3001\u63d0\u9192\u64cd\u4f5c\u3001"
    "\u5410\u69fd\u753b\u9762\u6216\u629b\u4e00\u4e2a\u8f7b\u677e\u95ee\u9898\u3002"
    "\u4e0d\u8981\u8bf4\u4f60\u5728\u770b\u622a\u56fe\uff0c\u4e0d\u8981\u590d\u8ff0"
    "\u63d0\u793a\u8bcd\uff0c\u63a7\u5236\u57281-2\u53e5\uff0c\u8bed\u6c14\u7b26\u5408"
    "\u5361\u5e03\u7684\u4eba\u8bbe\u3002"
)
VISION_IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
LOG_ROOT = PROJECT_ROOT / "logs" / "live_frontend"
WINDOW_STATE_PATH = LOG_ROOT / "window_state.json"
CONNECTION_STATE_PATH = LOG_ROOT / "connection_state.json"
DISPLAY_MODE_PATH = LOG_ROOT / "display_mode.json"
UI_SCALE_STATE_PATH = LOG_ROOT / "ui_scale.json"
LIVE2D_TRANSFORM_STATE_PATH = LOG_ROOT / "live2d_transform.json"
# 游戏识图绑定的窗口标题 (为空=未绑定; 游戏识图不允许截整个屏幕)
GAME_WINDOW_STATE_PATH = LOG_ROOT / "game_window.json"
LINK_NAME_WINDOW_STATE_PATH = LOG_ROOT / "link_name_window.json"
LINK_NAME_ROI_STATE_PATH = LOG_ROOT / "link_name_roi.json"
LINK_NAME_OCR_ROOT = LOG_ROOT / "link_name_ocr"
LINK_NAME_VISION_CAPTURE_ROOT = LOG_ROOT / "link_name_vision"
LINK_NAME_PROBE_DEBUG_ROOT = LOG_ROOT / "link_name_probe"
DISPLAY_MODE_STREAMER = "streamer"
DISPLAY_MODE_DIRECTOR = "director"
MACOS_APP_STYLE = """
QWidget {
    background: #f5f5f7;
    color: #1d1d1f;
    font-family: "Segoe UI", "Microsoft YaHei UI", "PingFang SC", Arial;
    font-size: 14px;
}
QLabel {
    background: transparent;
    color: #515154;
    font-weight: 600;
}
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #d2d2d7;
    border-radius: 10px;
    color: #1d1d1f;
    padding: 0 12px;
    selection-background-color: #0a84ff;
}
QComboBox {
    padding: 0 24px 0 10px;
}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {
    border: 1px solid #0a84ff;
    background: #ffffff;
}
QFrame#visionImagePanel {
    background: rgba(255, 255, 255, 0.76);
    border: 1px solid #d2d2d7;
    border-radius: 12px;
}
QFrame#visionImagePanel[imageModeActive="false"] {
    background: rgba(242, 242, 247, 0.72);
    border: 1px solid #d8d8de;
}
QFrame#consoleSectionCard {
    background: rgba(255, 255, 255, 0.76);
    border: 1px solid #d2d2d7;
    border-radius: 14px;
}
QPushButton#sectionToggleButton {
    background: transparent;
    border: 0;
    border-radius: 7px;
    color: #1d1d1f;
    font-family: "Segoe UI Symbol", "Microsoft YaHei UI", Arial;
    font-size: 15px;
    font-weight: 900;
    padding: 0;
}
QPushButton#sectionToggleButton:hover {
    background: rgba(0, 0, 0, 0.07);
    border: 0;
}
QLabel#visionImagePreview {
    background: rgba(245, 245, 247, 0.9);
    border: 1px dashed #c7c7cc;
    border-radius: 8px;
    color: #8e8e93;
    font-weight: 500;
}
QLabel#visionImagePreview:disabled {
    background: rgba(235, 235, 240, 0.9);
    border: 1px dashed #d1d1d6;
    color: #a1a1a6;
}
QLabel#visionImageStatus {
    color: #6e6e73;
    font-size: 12px;
    font-weight: 500;
}
QComboBox::drop-down {
    background: transparent;
    border: 0;
    width: 18px;
}
QComboBox#replyPercentCombo {
    padding: 0 6px;
}
QComboBox#replyPercentCombo::drop-down {
    background: transparent;
    border: 0;
    width: 0;
}
QComboBox#replyPercentCombo::down-arrow {
    image: none;
    width: 0;
    height: 0;
}
QComboBox#coldTimeCombo {
    padding: 0 6px;
}
QComboBox#coldTimeCombo::drop-down {
    background: transparent;
    border: 0;
    width: 0;
}
QComboBox#coldTimeCombo::down-arrow {
    image: none;
    width: 0;
    height: 0;
}
QPushButton {
    background: rgba(255, 255, 255, 0.9);
    color: #1d1d1f;
    border: 1px solid #d2d2d7;
    border-radius: 14px;
    font-weight: 600;
    padding: 0 14px;
}
QPushButton:hover {
    background: #ffffff;
    border-color: #b9b9c0;
}
QPushButton:pressed {
    background: #e8e8ed;
}
"""
CONNECTED_BUTTON_STYLE = "QPushButton { background: #c7f9cc; color: #0f5132; border: 1px solid #a7e8af; border-radius: 14px; font-weight: 600; padding: 0 14px; } QPushButton:hover { background: #b7f3c0; } QPushButton:pressed { background: #a7e8af; }"
INACTIVE_BUTTON_STYLE = "QPushButton { background: rgba(255, 255, 255, 0.9); color: #1d1d1f; border: 1px solid #d2d2d7; border-radius: 14px; font-weight: 600; padding: 0 14px; } QPushButton:hover { background: #ffffff; border-color: #b9b9c0; } QPushButton:pressed { background: #e8e8ed; }"
WARNING_BUTTON_STYLE = "QPushButton { background: #fee2e2; color: #7f1d1d; border: 1px solid #fecaca; border-radius: 14px; font-weight: 600; padding: 0 14px; } QPushButton:hover { background: #fff1f2; } QPushButton:pressed { background: #fecaca; }"
STORY_MODE_BUTTON_STYLE = "QPushButton { background: #dbeafe; color: #1e3a8a; border: 1px solid #bfdbfe; border-radius: 14px; font-weight: 600; padding: 0 14px; } QPushButton:hover { background: #eff6ff; } QPushButton:pressed { background: #bfdbfe; }"
BARRAGE_MODE_BUTTON_STYLE = "QPushButton { background: #fef3c7; color: #713f12; border: 1px solid #fde68a; border-radius: 14px; font-weight: 600; padding: 0 14px; } QPushButton:hover { background: #fffbeb; } QPushButton:pressed { background: #fde68a; }"
WAKE_PENDING_BUTTON_STYLE = "QPushButton { background: #fef3c7; color: #713f12; border: 1px solid #fde68a; border-radius: 14px; font-weight: 600; padding: 0 14px; } QPushButton:hover { background: #fef3c7; } QPushButton:pressed { background: #fef3c7; }"
SECTION_TITLE_STYLE = "QLabel { color: #6e6e73; font-size: 13px; font-weight: 700; padding: 2px 0 0 2px; }"
COLLAPSIBLE_SECTION_TITLE_STYLE = "QLabel { color: #1d1d1f; font-size: 15px; font-weight: 900; padding: 2px 0 0 2px; }"
STORY_PANEL_STYLE = "QFrame { background: rgba(255, 255, 255, 0.78); border: 1px solid #d2d2d7; border-radius: 16px; }"
STORY_ROW_STYLE = "QLabel { background: #ffffff; color: #1d1d1f; border: 1px solid #e5e5ea; border-radius: 12px; padding: 8px 10px; font-weight: 400; }"
STORY_ROW_HIGHLIGHT_STYLE = "QLabel { background: #fff4c2; color: #3f2f00; border: 1px solid #f5d76e; border-radius: 12px; padding: 8px 10px; font-weight: 400; }"
DIRECTOR_METRIC_ROW_STYLE = """
QFrame#directorMetricRow {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid #d2d2d7;
    border-radius: 12px;
}
QLabel#directorMetricDragHandle {
    background: transparent;
    border: 0;
    color: #8e8e93;
    font-size: 18px;
    font-weight: 700;
}
QLabel#directorMetricDragHandle:hover {
    color: #0a84ff;
}
QCheckBox {
    background: transparent;
    border: 0;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
}
"""


def scaled_int(value: int | float, scale: float) -> int:
    return max(1, int(round(float(value) * float(scale))))


def scaled_console_style(scale: float) -> str:
    return MACOS_APP_STYLE.replace(
        "font-size: 14px;",
        f"font-size: {scaled_int(14, scale)}px;",
        1,
    )


def scaled_section_title_style(scale: float) -> str:
    return SECTION_TITLE_STYLE.replace(
        "font-size: 13px;",
        f"font-size: {scaled_int(13, scale)}px;",
    )


def scaled_collapsible_section_title_style(scale: float) -> str:
    return COLLAPSIBLE_SECTION_TITLE_STYLE.replace(
        "font-size: 15px;",
        f"font-size: {scaled_int(15, scale)}px;",
    )


def scaled_director_metric_row_style(scale: float) -> str:
    return DIRECTOR_METRIC_ROW_STYLE.replace(
        "font-size: 18px;",
        f"font-size: {scaled_int(18, scale)}px;",
    )
STORY_EMPTY_TEXT = "未加载剧本"
MIC_SAMPLE_RATE = 16000
MIC_CHANNELS = 1
MIC_SAMPLE_SIZE_BITS = 16
MIC_VAD_CHUNK_SAMPLES = 512
MIC_SEND_CHUNK_SAMPLES = 4096
BACKEND_WS_CONNECT_TIMEOUT_SECONDS = 8.0
BACKEND_WS_RECV_POLL_TIMEOUT_SECONDS = 0.05
BACKEND_WS_SEND_TIMEOUT_SECONDS = 10.0
MIC_VAD_PROB_THRESHOLD = 0.4
MIC_VAD_DB_THRESHOLD = 40
MIC_VAD_REQUIRED_HITS = 3
MIC_VAD_REQUIRED_MISSES = 4
MIC_VAD_SMOOTHING_WINDOW = 5
MIC_VAD_PRE_BUFFER_CHUNKS = 20
MIC_VAD_MIN_UTTERANCE_CHUNKS = 40
MIC_VAD_PAUSE_CHUNKS = max(1, math.ceil(MIC_VAD_MIN_UTTERANCE_CHUNKS / 3))
MIC_VAD_MAX_UTTERANCE_SECONDS = 15.0
MIC_HEALTH_CHECK_INTERVAL_MS = 1000
MIC_HEALTH_START_GRACE_SECONDS = 3.0
MIC_HEALTH_NO_DATA_TIMEOUT_SECONDS = 5.0
MIC_HEALTH_RESTART_COOLDOWN_SECONDS = 3.0
AUDIO_OUTPUT_REINIT_COOLDOWN_SECONDS = 1.0
MOUTH_OPEN_ATTACK_SECONDS = 0.08
MOUTH_OPEN_RELEASE_SECONDS = 0.16
MOUTH_OPEN_DEADZONE = 0.015
MIC_ON_TEXT = "\u9ea6\u514b\u98ce\u5f00"
MIC_OFF_TEXT = "\u9ea6\u514b\u98ce\u5173"
MIC_ERROR_TEXT = "\u9ea6\u514b\u98ce\u5f02\u5e38"
LINK_MIC_ON_TEXT = "\u8fde\u7ebf\u9ea6\u514b\u98ce\u5f00"
LINK_MIC_OFF_TEXT = "\u8fde\u7ebf\u9ea6\u514b\u98ce\u5173"
LINK_MIC_PENDING_TEXT = "\u8fde\u7ebf\u5f00\u542f\u4e2d"
LINK_MIC_ERROR_TEXT = "\u8fde\u7ebf\u5f02\u5e38"
MIC_ERROR_TOOLTIP = "\u9ea6\u514b\u98ce\u8bbe\u5907\u4e0d\u53ef\u7528\uff0c\u6b63\u5728\u540e\u53f0\u91cd\u8bd5"
LINK_MIC_ERROR_TOOLTIP = "\u8fde\u7ebf\u9ea6\u514b\u98ce\u8bbe\u5907\u4e0d\u53ef\u7528\uff0c\u6b63\u5728\u540e\u53f0\u91cd\u8bd5"
DEFAULT_LINK_HUMAN_NAME = "\u8fde\u7ebf\u4e3b\u64ad"
LINK_HUMAN_NAME_INPUT_WIDTH = 148
LINK_HUMAN_NAME_AUTO_BUTTON_TEXT = "\u81ea\u52a8\u8bc6\u522b"
LINK_HUMAN_NAME_FAILED_BUTTON_TEXT = "\u8bc6\u522b\u5931\u8d25"
LINK_HUMAN_NAME_AUTO_BUTTON_WIDTH = 84
LINK_HUMAN_NAME_SAVE_DEBOUNCE_MS = 500
LINK_HUMAN_NAME_DETECT_TIMEOUT_MS = 60_000
LINK_HUMAN_NAME_PROBE_DURATION_SECONDS = 55.0
LINK_HUMAN_NAME_WS_FIRST_STAGE_MS = 55_000
LINK_HUMAN_NAME_WS_POLL_INTERVAL_MS = 700
DEFAULT_LINK_NAME_TARGET_ROI = (0.58, 0.70, 0.73, 0.90)
ANCHOR_HUMAN_NAME = "\u4e3b\u64ad"
ANCHOR_TEXT_INPUT_WIDTH = 260
ANCHOR_TEXT_SEND_BUTTON_WIDTH = 68
OUTPUT_AUDIBLE_TEXT = "\u8f93\u51fa\u6709\u58f0"
OUTPUT_MUTED_TEXT = "\u8f93\u51fa\u9759\u97f3"
SLEEP_AWAKE_TEXT = "点击休眠"
SLEEP_SLEEPING_TEXT = "点击唤醒"
SLEEP_WAKING_TEXT = "唤醒中"
PUNISH_INACTIVE_TEXT = "点击罚站"
PUNISH_ACTIVE_TEXT = "取消罚站"
GIFT_THANKS_INACTIVE_TEXT = "感谢礼物"
GIFT_THANKS_ACTIVE_TEXT = "取消感谢"
BARRAGE_REPLY_TEXT = "关闭读弹幕"
BARRAGE_IGNORE_TEXT = "开启读弹幕"
DIRECTOR_METRIC_FIELDS = (
    ("wealth_level", "财富等级"),
    ("fan_badge_level", "粉丝牌等级"),
    ("session_diamonds", "本场钻石"),
)


DEFAULT_LIVE2D_TRANSFORM = {
    "scale": 1.0,
    "offset_x": 0.0,
    "offset_y": 0.0,
}
LIVE_STREAMING_AGENT_LIVE2D_TRANSFORM = {
    "scale": 0.9,
    "offset_x": 0.0,
    "offset_y": 0.5,
}


def normalize_display_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"director", "producer", "control", "controller", "编导", "导播"}:
        return DISPLAY_MODE_DIRECTOR
    if mode in {"streamer", "anchor", "host", "live", "主播"}:
        return DISPLAY_MODE_STREAMER
    logger.warning("Unknown display mode {!r}; using streamer mode", value)
    return DISPLAY_MODE_STREAMER


def load_display_mode() -> str:
    try:
        with DISPLAY_MODE_PATH.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except FileNotFoundError:
        DISPLAY_MODE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DISPLAY_MODE_PATH.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "mode": DISPLAY_MODE_STREAMER,
                    "comment": "mode 可填 streamer/主播 或 director/编导",
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("Created default display mode config: {}", DISPLAY_MODE_PATH)
        return DISPLAY_MODE_STREAMER
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read display mode config: {}", exc)
        return DISPLAY_MODE_STREAMER

    if not isinstance(config, dict):
        logger.warning("Ignoring invalid display mode config: {}", config)
        return DISPLAY_MODE_STREAMER

    mode = normalize_display_mode(
        config.get("mode")
        or config.get("role")
        or config.get("display_mode")
    )
    logger.info("Loaded display mode from {}: {}", DISPLAY_MODE_PATH, mode)
    return mode


def normalize_ui_scale(value: Any) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError):
        return 1.0
    return min(UI_SCALE_OPTIONS, key=lambda option: abs(option - scale))


def load_ui_scale() -> float:
    try:
        with UI_SCALE_STATE_PATH.open("r", encoding="utf-8") as file:
            state = json.load(file)
    except FileNotFoundError:
        return 1.0
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read UI scale state: {}", exc)
        return 1.0

    if not isinstance(state, dict):
        logger.warning("Ignoring invalid UI scale state: {}", state)
        return 1.0

    scale = normalize_ui_scale(state.get("scale"))
    logger.info("Loaded UI scale from {}: {}", UI_SCALE_STATE_PATH, scale)
    return scale


def save_ui_scale(scale: float) -> None:
    scale = normalize_ui_scale(scale)
    try:
        UI_SCALE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with UI_SCALE_STATE_PATH.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "scale": scale,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("Saved UI scale: {}", scale)
    except OSError as exc:
        logger.warning("Failed to save UI scale: {}", exc)


def load_game_window_binding() -> str | None:
    """读取游戏识图绑定的窗口标题。空/缺失表示未绑定，不允许截整个屏幕。"""
    try:
        with GAME_WINDOW_STATE_PATH.open("r", encoding="utf-8") as file:
            state = json.load(file)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read game window binding: {}", exc)
        return None
    if not isinstance(state, dict):
        return None
    title = state.get("window_title")
    if isinstance(title, str) and title.strip():
        logger.info("Loaded bound game window: {}", title)
        return title
    return None


def save_game_window_binding(title: str | None) -> None:
    """持久化游戏识图绑定的窗口标题。None/空=未绑定，不允许截整个屏幕。"""
    try:
        GAME_WINDOW_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with GAME_WINDOW_STATE_PATH.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "window_title": title or "",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("Saved bound game window: {}", title or "(未绑定)")
    except OSError as exc:
        logger.warning("Failed to save game window binding: {}", exc)


def _coerce_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_link_name_window_binding() -> dict[str, Any] | None:
    try:
        with LINK_NAME_WINDOW_STATE_PATH.open("r", encoding="utf-8") as file:
            state = json.load(file)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read link name window binding: {}", exc)
        return None
    if isinstance(state, str):
        title = state.strip()
        return {"window_title": title, "hwnd": None} if title else None
    if not isinstance(state, dict):
        return None
    title = state.get("window_title")
    if not isinstance(title, str):
        title = ""
    title = title.strip()
    hwnd = _coerce_optional_int(state.get("hwnd"))
    if title or hwnd:
        logger.info("Loaded bound link name window: title={} hwnd={}", title, hwnd)
        return {
            "window_title": title,
            "hwnd": hwnd,
            "rect": state.get("rect") if isinstance(state.get("rect"), dict) else None,
        }
    return None


def save_link_name_window_binding(
    title: str | None,
    hwnd: int | None = None,
    window_info: dict[str, Any] | None = None,
) -> None:
    try:
        LINK_NAME_WINDOW_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "window_title": title or "",
            "hwnd": hwnd,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        if window_info:
            state.update(
                {
                    key: value
                    for key, value in window_info.items()
                    if key not in {"window_title", "hwnd", "updated_at"}
                }
            )
        with LINK_NAME_WINDOW_STATE_PATH.open("w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)
        logger.info("Saved bound link name window: title={} hwnd={}", title or "(unbound)", hwnd)
    except OSError as exc:
        logger.warning("Failed to save link name window binding: {}", exc)


LINK_NAME_REJECT_WORDS = (
    "\u8fde\u7ebf\u4e3b\u64ad",
    "\u4e3b\u64ad",
    "\u6296\u97f3\u76f4\u64ad\u4f34\u4fa3",
    "\u6296\u97f3\u76f4\u64ad",
    "\u76f4\u64ad\u4f34\u4fa3",
    "\u76f4\u64ad\u95f4",
    "\u8fde\u9ea6",
    "\u8fde\u7ebf",
    "pk\u8fde\u7ebf",
    "\u8fde\u7ebfpk",
    "\u4e0e",
    "\u548c",
    "\u9000\u51fapk",
    "\u6bd4\u62fc\u65b9\u5f0f",
    "\u5e38\u89c4pk",
    "\u8fdb\u884c\u4e2d",
    "\u66f4\u591a\u73a9\u6cd5",
    "\u7acb\u5373\u5339\u914d",
    "\u968f\u673a\u5339\u914d",
    "\u8bf4\u70b9\u4ec0\u4e48",
    "\u53d1\u9001",
    "\u5bfc\u64ad",
    "\u573a\u666f",
    "\u573a\u666f\u4e00",
    "\u573a\u666f\u4e8c",
    "\u573a\u666f\u4e09",
    "\u5e38\u89c4\u6a21\u5f0f",
    "\u4e92\u52a8\u73a9\u6cd5",
    "\u89c2\u4f17\u8fde\u7ebf",
    "ai\u5609\u5bbe",
    "\u798f\u888b",
    "\u793c\u7269\u83dc\u5355",
    "\u5ba0\u7c89\u7ea2\u5305",
    "\u5ba0\u7c89",
    "\u5fc3\u613f",
    "\u793c\u7269\u6295\u7968",
    "\u6e38\u620f\u80fd\u529b",
    "\u7559\u8a00\u4e0a\u5899",
    "\u7559\u8a00",
    "\u4eba\u6c14\u4efb\u52a1",
    "\u793c\u7269\u5c55\u9986",
    "\u8d77\u6d41\u6311\u6218",
    "\u76f4\u64ad\u5de5\u5177",
    "\u76f4\u64ad\u8bbe\u7f6e",
    "\u4e2d\u63a7\u53f0",
    "\u7d20\u6750\u5e93",
    "\u6e38\u620f\u73a9\u6cd5",
    "\u4e92\u52a8\u5de5\u5177",
    "\u865a\u62df\u5f62\u8c61",
    "\u7eff\u5e55\u76f4\u64ad",
    "ai\u7ecf\u7eaa\u4eba",
    "\u6296\u97f3\u5c0f\u52a9\u624b",
    "\u6211\u65b9\u8d21\u732e\u699c",
    "\u8d21\u732e\u699c",
    "pk\u8d21\u732e\u699c",
    "\u5728\u7ebf\u89c2\u4f17\u699c",
    "\u672c\u573a\u89c2\u4f17\u699c",
    "\u518d\u6765\u4e00\u5c40",
    "\u7ed9ta\u70b9\u70b9",
    "\u6444\u50cf\u5934\u5e03\u5c40",
    "\u8fde\u7ebf\u8bbe\u7f6e",
    "\u672a\u5206\u7c7b",
    "\u4eba\u6c14\u699c",
    "\u5c0f\u8377\u699c",
    "pk\u7ed3\u675f",
    "\u5e73\u5c40",
    "\u5173\u64ad",
    "\u4e3b\u64ad\u4e2d\u5fc3",
    "\u663e\u793a\u5668",
    "\u6dfb\u52a0\u7d20\u6750",
    "obs64.exe",
    "dream maker live console",
    "chrome legacy window",
    "websocket\u8fde\u63a5",
    "\u7a97\u53e3\u63a7\u5236",
    "\u529f\u80fd\u63a7\u5236",
    "\u5a31\u4e50\u529f\u80fd",
    "live",
    "tiktok",
    "douyin",
)
LINK_NAME_REJECT_SUBSTRINGS = (
    "\u8981\u83b7\u53d6\u7f3a\u5931\u7684\u56fe\u7247\u8bf4\u660e",
    "\u6b22\u8fce\u6765\u5230\u76f4\u64ad\u95f4",
    "\u5e73\u53f0\u4e25\u7981",
    "\u8bf7\u6587\u660epk",
    "pk\u8fde\u7ebf",
    "\u8fde\u7ebf",
    "\u9000\u51fapk",
    "\u6bd4\u62fc\u65b9\u5f0f",
    "\u5e38\u89c4pk",
    "\u8fdb\u884c\u4e2d",
    "\u66f4\u591a\u73a9\u6cd5",
    "\u7acb\u5373\u5339\u914d",
    "\u968f\u673a\u5339\u914d",
    "\u8bf4\u70b9\u4ec0\u4e48",
    "\u8d21\u732e\u699c",
    "\u793c\u7269\u83dc\u5355",
    "\u5ba0\u7c89",
    "\u798f\u888b",
    "\u6e38\u620f\u80fd\u529b",
    "\u7559\u8a00",
    "\u7ed9ta\u70b9",
    "websocket",
    "chrome legacy",
)


def normalize_link_human_name_candidate(value: Any) -> str | None:
    if value is None:
        return None
    name = str(value).strip()
    if not name:
        return None
    name = re.sub(r"[\r\n\t]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip(" @:：|-_/\\")
    name = re.sub(
        r"^(\u5bf9\u65b9|\u5609\u5bbe|\u4e3b\u64ad|\u7528\u6237|\u6635\u79f0)[:：\s]+",
        "",
        name,
    )
    name = re.sub(r"(\u7684)?\u76f4\u64ad\u95f4$", "", name).strip()
    name = re.sub(r"(\u6b63\u5728)?\u76f4\u64ad(\u4e2d)?$", "", name).strip()
    name = name.strip(" @:：|-_/\\")
    if len(name) > 32:
        return None
    if len(name) < 2 and not re.fullmatch(r"[\u4e00-\u9fff]", name):
        return None
    lowered = name.lower()
    if lowered in {"unknown", "viewer"}:
        return None
    if any(word in lowered for word in ("http://", "https://", "ws://")):
        return None
    if any(word.lower() == lowered for word in LINK_NAME_REJECT_WORDS):
        return None
    if any(word.lower() in lowered for word in LINK_NAME_REJECT_SUBSTRINGS):
        return None
    if re.fullmatch(r"\d{5,}", name):
        return None
    return name


def link_name_candidate_from_title(title: str) -> str | None:
    title = str(title or "").strip()
    if not title:
        return None
    lowered_title = title.lower()
    if any(
        marker in lowered_title
        for marker in (
            "google chrome",
            "microsoft edge",
            "chrome",
            "\u6296\u97f3\u76f4\u64ad",
            "\u6296\u97f3\u76f4\u64ad\u95f4",
            "\u76f4\u64ad\u4f34\u4fa3",
        )
    ):
        return None

    parts = [title]
    parts.extend(
        part.strip()
        for part in re.split(r"\s*[-|_—–·:：]\s*", title)
        if part.strip()
    )
    parts.extend(
        match.group(1).strip()
        for match in re.finditer(r"(.+?)\u7684\u76f4\u64ad\u95f4", title)
        if match.group(1).strip()
    )

    seen: set[str] = set()
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        candidate = normalize_link_human_name_candidate(part)
        if candidate and not any(
            word.lower() == candidate.lower() for word in LINK_NAME_REJECT_WORDS
        ):
            return candidate
    return None


def _win_enumerate_windows() -> list[tuple[int, str]]:
    """枚举当前可见、未最小化的顶层窗口 (hwnd, 标题)。仅 Windows 有效。

    会排除本程序自身的窗口 (Live2D / 控制台), 避免误绑定。
    """
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL

    self_hwnds: set[int] = set()
    try:
        for widget in QApplication.topLevelWidgets():
            try:
                self_hwnds.add(int(widget.winId()))
            except Exception:
                pass
    except Exception:
        pass

    results: list[tuple[int, str]] = []
    enum_proc = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
    )

    def _callback(hwnd, _lparam):
        try:
            if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
                return True
            if int(hwnd) in self_hwnds:
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = (buffer.value or "").strip()
            if title:
                results.append((int(hwnd), title))
        except Exception:
            pass
        return True

    try:
        user32.EnumWindows(enum_proc(_callback), 0)
    except Exception as exc:
        logger.warning("EnumWindows failed: {}", exc)
    return results


def _win_find_window_by_title(title: str) -> int | None:
    """按标题重新解析窗口 hwnd。窗口可能重启导致 hwnd 变化, 故每次按标题重找。

    优先精确匹配, 其次包含匹配 (处理标题含动态后缀, 如分数/关卡)。
    """
    if not title or sys.platform != "win32":
        return None
    windows = _win_enumerate_windows()
    for hwnd, wtitle in windows:
        if wtitle == title:
            return hwnd
    for hwnd, wtitle in windows:
        if title in wtitle or wtitle in title:
            return hwnd
    return None


def _win_get_window_title(hwnd: int | None) -> str | None:
    if not hwnd or sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int

    hwnd_value = wintypes.HWND(int(hwnd))
    if not user32.IsWindow(hwnd_value):
        return None
    length = user32.GetWindowTextLengthW(hwnd_value)
    if length <= 0:
        return None
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd_value, buffer, length + 1)
    return (buffer.value or "").strip() or None


def _win_window_rect(hwnd: int | None) -> tuple[int, int, int, int] | None:
    if not hwnd or sys.platform != "win32":
        return None
    import ctypes
    from ctypes import byref, wintypes

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    user32 = ctypes.windll.user32
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    rect = RECT()
    if not user32.GetWindowRect(wintypes.HWND(int(hwnd)), byref(rect)):
        return None
    return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))


def _win_window_is_usable(hwnd: int | None, expected_title: str | None = None) -> bool:
    if not hwnd or sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL

    hwnd_value = wintypes.HWND(int(hwnd))
    if (
        not user32.IsWindow(hwnd_value)
        or not user32.IsWindowVisible(hwnd_value)
        or user32.IsIconic(hwnd_value)
    ):
        return False
    title = _win_get_window_title(hwnd)
    if not title:
        return False
    expected = (expected_title or "").strip()
    if not expected:
        return True
    return title == expected or expected in title or title in expected


def _win_grab_window_pixmap(
    hwnd: int,
    *,
    use_print_window: bool = True,
) -> "QPixmap | None":
    """截取指定窗口所在屏幕区域 (只含该窗口矩形, 屏幕其它内容不入画)。

    从合成后的屏幕 DC 按窗口矩形 BitBlt, 因此 DirectX / 无边框全屏游戏也能截到;
    坐标与截图源同为物理像素, 避免 Qt 逻辑坐标/DPI 换算。仅 Windows 有效。
    """
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import byref, sizeof, wintypes

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", ctypes.c_uint32),
            ("biWidth", ctypes.c_int32),
            ("biHeight", ctypes.c_int32),
            ("biPlanes", ctypes.c_uint16),
            ("biBitCount", ctypes.c_uint16),
            ("biCompression", ctypes.c_uint32),
            ("biSizeImage", ctypes.c_uint32),
            ("biXPelsPerMeter", ctypes.c_int32),
            ("biYPelsPerMeter", ctypes.c_int32),
            ("biClrUsed", ctypes.c_uint32),
            ("biClrImportant", ctypes.c_uint32),
        ]

    class BITMAPINFO(ctypes.Structure):
        _fields_ = [
            ("bmiHeader", BITMAPINFOHEADER),
            ("bmiColors", ctypes.c_uint32 * 3),
        ]

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.GetDC.restype = ctypes.c_void_p
    user32.ReleaseDC.argtypes = [wintypes.HWND, ctypes.c_void_p]
    user32.ReleaseDC.restype = ctypes.c_int
    user32.PrintWindow.argtypes = [wintypes.HWND, ctypes.c_void_p, ctypes.c_uint]
    user32.PrintWindow.restype = wintypes.BOOL
    gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
    gdi32.CreateCompatibleDC.restype = ctypes.c_void_p
    gdi32.CreateCompatibleBitmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    gdi32.CreateCompatibleBitmap.restype = ctypes.c_void_p
    gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    gdi32.SelectObject.restype = ctypes.c_void_p
    gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
    gdi32.DeleteDC.argtypes = [ctypes.c_void_p]
    gdi32.BitBlt.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint32,
    ]
    gdi32.BitBlt.restype = wintypes.BOOL
    gdi32.GetDIBits.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    gdi32.GetDIBits.restype = ctypes.c_int

    rect = RECT()
    if not user32.GetWindowRect(wintypes.HWND(hwnd), byref(rect)):
        return None
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None

    screen_dc = user32.GetDC(None)
    if not screen_dc:
        return None
    mem_dc = gdi32.CreateCompatibleDC(screen_dc)
    bitmap = gdi32.CreateCompatibleBitmap(screen_dc, width, height)
    old_obj = gdi32.SelectObject(mem_dc, bitmap)
    pixmap = None
    try:
        srccopy = 0x00CC0020
        captureblt = 0x40000000
        captured = False
        if use_print_window:
            try:
                # Prefer PrintWindow so an overlapping control panel does not leak into
                # the target-window capture. Some GPU/DirectX windows return blank,
                # so BitBlt remains the fallback.
                captured = bool(
                    user32.PrintWindow(wintypes.HWND(hwnd), mem_dc, 0x00000002)
                )
                if not captured:
                    captured = bool(user32.PrintWindow(wintypes.HWND(hwnd), mem_dc, 0))
            except Exception as exc:
                logger.debug("PrintWindow capture failed: {}", exc)
        if not captured:
            captured = bool(
                gdi32.BitBlt(
                    mem_dc,
                    0,
                    0,
                    width,
                    height,
                    screen_dc,
                    rect.left,
                    rect.top,
                    srccopy | captureblt,
                )
            )
        if captured:
            info = BITMAPINFO()
            info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER)
            info.bmiHeader.biWidth = width
            info.bmiHeader.biHeight = -height  # 负高=top-down, 行序与 QImage 一致
            info.bmiHeader.biPlanes = 1
            info.bmiHeader.biBitCount = 32
            info.bmiHeader.biCompression = 0  # BI_RGB
            buffer = (ctypes.c_char * (width * height * 4))()
            if gdi32.GetDIBits(mem_dc, bitmap, 0, height, buffer, byref(info), 0):
                raw = bytes(buffer)
                image = QImage(raw, width, height, QImage.Format_RGB32)
                pixmap = QPixmap.fromImage(image.copy())
    except Exception as exc:
        logger.warning("Window region capture failed: {}", exc)
    finally:
        gdi32.SelectObject(mem_dc, old_obj)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(None, screen_dc)
    return pixmap


def _link_name_target_roi() -> tuple[float, float, float, float]:
    raw = os.environ.get("LINK_NAME_TARGET_ROI", "").strip()
    if raw:
        ratio = _parse_link_name_roi(raw)
        if ratio:
            return ratio
        logger.warning("Invalid LINK_NAME_TARGET_ROI: {}", raw)
    ratio = load_link_name_roi()
    if ratio:
        return ratio
    return DEFAULT_LINK_NAME_TARGET_ROI


def _parse_link_name_roi(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        raw_values = [
            value.get("left"),
            value.get("top"),
            value.get("right"),
            value.get("bottom"),
        ]
    elif isinstance(value, (list, tuple)):
        raw_values = list(value)
    else:
        raw_values = str(value or "").replace("，", ",").split(",")
    if len(raw_values) != 4:
        return None
    try:
        left, top, right, bottom = [float(part) for part in raw_values]
    except (TypeError, ValueError):
        return None
    if 0 <= left < right <= 1 and 0 <= top < bottom <= 1:
        return left, top, right, bottom
    return None


def load_link_name_roi() -> tuple[float, float, float, float] | None:
    try:
        with LINK_NAME_ROI_STATE_PATH.open("r", encoding="utf-8") as file:
            state = json.load(file)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read link name ROI config: {}", exc)
        return None
    ratio = _parse_link_name_roi(state.get("roi") if isinstance(state, dict) else state)
    if not ratio:
        logger.warning("Ignoring invalid link name ROI config: {}", state)
        return None
    return ratio


def save_link_name_roi(ratio: tuple[float, float, float, float] | None) -> None:
    try:
        LINK_NAME_ROI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if ratio is None:
            if LINK_NAME_ROI_STATE_PATH.exists():
                LINK_NAME_ROI_STATE_PATH.unlink()
            logger.info("Reset link name ROI config")
            return
        left, top, right, bottom = ratio
        with LINK_NAME_ROI_STATE_PATH.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "roi": {
                        "left": left,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                    },
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "comment": "Relative crop for co-host nickname: left, top, right, bottom in 0..1.",
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("Saved link name ROI config: {}", ratio)
    except OSError as exc:
        logger.warning("Failed to save link name ROI config: {}", exc)


def _format_link_name_roi(ratio: tuple[float, float, float, float]) -> str:
    return ",".join(f"{value:.4g}" for value in ratio)


def _crop_pixmap_by_ratio(
    pixmap: QPixmap,
    ratio: tuple[float, float, float, float],
) -> tuple[QPixmap, dict[str, Any]]:
    width = pixmap.width()
    height = pixmap.height()
    left_ratio, top_ratio, right_ratio, bottom_ratio = ratio
    left = max(0, min(width - 1, int(round(width * left_ratio))))
    top = max(0, min(height - 1, int(round(height * top_ratio))))
    right = max(left + 1, min(width, int(round(width * right_ratio))))
    bottom = max(top + 1, min(height, int(round(height * bottom_ratio))))
    cropped = pixmap.copy(left, top, right - left, bottom - top)
    return cropped, {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": right - left,
        "height": bottom - top,
        "ratio": {
            "left": left_ratio,
            "top": top_ratio,
            "right": right_ratio,
            "bottom": bottom_ratio,
        },
        "source_width": width,
        "source_height": height,
    }


def _pixmap_content_stats(pixmap: QPixmap) -> dict[str, Any]:
    if pixmap is None or pixmap.isNull():
        return {
            "sample_count": 0,
            "mean": 0.0,
            "dynamic_range": 0,
            "nonblack_ratio": 0.0,
            "is_blank": True,
        }
    image = pixmap.toImage().convertToFormat(QImage.Format_RGB32)
    width = image.width()
    height = image.height()
    if width <= 0 or height <= 0:
        return {
            "sample_count": 0,
            "mean": 0.0,
            "dynamic_range": 0,
            "nonblack_ratio": 0.0,
            "is_blank": True,
        }

    target_samples = 1200
    stride = max(1, int(math.sqrt((width * height) / target_samples)))
    luminance_min = 255
    luminance_max = 0
    luminance_total = 0
    nonblack_count = 0
    sample_count = 0
    for y in range(stride // 2, height, stride):
        for x in range(stride // 2, width, stride):
            color = QColor(image.pixel(x, y))
            red = color.red()
            green = color.green()
            blue = color.blue()
            luminance = int(round((red * 299 + green * 587 + blue * 114) / 1000))
            luminance_min = min(luminance_min, luminance)
            luminance_max = max(luminance_max, luminance)
            luminance_total += luminance
            if max(red, green, blue) > 18:
                nonblack_count += 1
            sample_count += 1

    if sample_count <= 0:
        return {
            "sample_count": 0,
            "mean": 0.0,
            "dynamic_range": 0,
            "nonblack_ratio": 0.0,
            "is_blank": True,
        }

    nonblack_ratio = nonblack_count / sample_count
    dynamic_range = luminance_max - luminance_min
    mean = luminance_total / sample_count
    return {
        "sample_count": sample_count,
        "mean": round(mean, 2),
        "dynamic_range": dynamic_range,
        "nonblack_ratio": round(nonblack_ratio, 4),
        "is_blank": nonblack_ratio < 0.002 and dynamic_range < 8,
    }


def _window_rect_payload(
    rect: tuple[int, int, int, int] | None,
) -> dict[str, int] | None:
    if not rect:
        return None
    left, top, right, bottom = rect
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": max(0, right - left),
        "height": max(0, bottom - top),
    }


def default_live2d_transform_state() -> dict[str, Any]:
    return {
        "default": DEFAULT_LIVE2D_TRANSFORM,
        "models": {
            "Live Streaming Agent": LIVE_STREAMING_AGENT_LIVE2D_TRANSFORM,
            "尤里": LIVE_STREAMING_AGENT_LIVE2D_TRANSFORM,
        },
        "comment": (
            "scale controls model size. offset_x/offset_y are normalized model "
            "offsets; positive offset_y moves most Live2D models upward."
        ),
    }


def default_link_name_roi_state() -> dict[str, Any]:
    left, top, right, bottom = DEFAULT_LINK_NAME_TARGET_ROI
    return {
        "roi": {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
        },
        "comment": (
            "Legacy co-host nickname crop setting. Current auto-detect uses only "
            "local DouyinBarrage WebSocket capture."
        ),
        "example": "0.58,0.70,0.73,0.90",
    }


def ensure_json_file(path: Path, default_data: dict[str, Any]) -> None:
    if path.exists():
        return

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(default_data, file, ensure_ascii=False, indent=2)
        logger.info("Created default frontend state file: {}", path)
    except OSError as exc:
        logger.warning("Failed to create frontend state file {}: {}", path, exc)


def ensure_frontend_state_files(
    default_url: str,
    default_width: int,
    default_height: int,
) -> None:
    host, port = split_backend_url(default_url)
    ensure_json_file(
        CONNECTION_STATE_PATH,
        {
            "url": build_backend_url(host, port),
            "host": host,
            "port": port,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    ensure_json_file(
        DISPLAY_MODE_PATH,
        {
            "mode": DISPLAY_MODE_STREAMER,
            "comment": "mode 可填 streamer/主播 或 director/编导",
        },
    )
    ensure_json_file(
        WINDOW_STATE_PATH,
        {
            "width": default_width,
            "height": default_height,
        },
    )
    ensure_json_file(
        LIVE2D_TRANSFORM_STATE_PATH,
        default_live2d_transform_state(),
    )
    ensure_json_file(
        UI_SCALE_STATE_PATH,
        {
            "scale": 1.0,
        },
    )
    ensure_json_file(
        LINK_NAME_ROI_STATE_PATH,
        default_link_name_roi_state(),
    )


def init_logger() -> Path:
    diagnose_setting = True
    log_dir = LOG_ROOT / datetime.now().strftime("%Y-%m-%d")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pyqt_live2d_window_{time}.log"

    logger.remove()
    console_sink = sys.stdout or sys.stderr
    if console_sink:
        logger.add(
            console_sink,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS}|{level: <8}|{name}:{function}:{line}|{message}",
            enqueue=False,
            backtrace=True,
            diagnose=diagnose_setting,
        )
    logger.add(
        log_file,
        level="DEBUG",
        rotation="10 MB",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=diagnose_setting,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS}|{level: <8}|{name}:{function}:{line}|{message}",
    )

    logger.info("PyQt Live2D logger initialized: {}", log_file)
    return log_file


def load_live2d_window_size(default_width: int, default_height: int) -> tuple[int, int]:
    try:
        with WINDOW_STATE_PATH.open("r", encoding="utf-8") as file:
            state = json.load(file)
    except FileNotFoundError:
        return default_width, default_height
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read Live2D window state: {}", exc)
        return default_width, default_height

    if not isinstance(state, dict):
        logger.warning("Ignoring invalid Live2D window state: {}", state)
        return default_width, default_height

    try:
        width = int(state.get("width") or default_width)
        height = int(state.get("height") or default_height)
    except (TypeError, ValueError) as exc:
        logger.warning("Ignoring invalid Live2D window size: {}", exc)
        return default_width, default_height

    width = max(width, 320)
    height = max(height, 420)
    logger.info("Loaded Live2D window size: {}x{}", width, height)
    return width, height


def save_live2d_window_size(width: int, height: int) -> None:
    width = max(int(width), 320)
    height = max(int(height), 420)
    try:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        with WINDOW_STATE_PATH.open("w", encoding="utf-8") as file:
            json.dump({"width": width, "height": height}, file, ensure_ascii=False, indent=2)
        logger.info("Saved Live2D window size: {}x{}", width, height)
    except OSError as exc:
        logger.warning("Failed to save Live2D window size: {}", exc)


def split_backend_url(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    host = parsed.hostname or DEFAULT_WS_HOST
    port = parsed.port or int(DEFAULT_WS_PORT)
    return host, str(port)


def build_backend_url(host: str, port: str) -> str:
    host = host.strip() or DEFAULT_WS_HOST
    port = port.strip() or DEFAULT_WS_PORT
    if ":" in host and not (host.startswith("[") and host.endswith("]")):
        host = f"[{host}]"
    return f"ws://{host}:{port}{BACKEND_WS_PATH}"


def load_backend_ws_url(default_url: str) -> str:
    try:
        with CONNECTION_STATE_PATH.open("r", encoding="utf-8") as file:
            state = json.load(file)
    except FileNotFoundError:
        return default_url
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read backend WebSocket state: {}", exc)
        return default_url

    if not isinstance(state, dict):
        logger.warning("Ignoring invalid backend WebSocket state: {}", state)
        return default_url

    url = str(state.get("url") or "").strip()
    if not url:
        host = str(state.get("host") or "").strip()
        port = str(state.get("port") or "").strip()
        if host and port:
            url = build_backend_url(host, port)

    if not url:
        return default_url

    try:
        host, port = split_backend_url(url)
        normalized_url = build_backend_url(host, port)
    except Exception as exc:
        logger.warning("Ignoring invalid saved backend WebSocket URL: {}", exc)
        return default_url

    logger.info(
        "Loaded backend WebSocket URL from {}: {}",
        CONNECTION_STATE_PATH,
        normalized_url,
    )
    return normalized_url


def save_backend_ws_url(url: str) -> None:
    try:
        host, port = split_backend_url(url)
        normalized_url = build_backend_url(host, port)
        CONNECTION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CONNECTION_STATE_PATH.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "url": normalized_url,
                    "host": host,
                    "port": port,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("Saved backend WebSocket URL: {}", normalized_url)
    except Exception as exc:
        logger.warning("Failed to save backend WebSocket URL: {}", exc)


@dataclass
class ModelConfig:
    name: str
    model_path: Path
    emotion_map: dict[str, Any] = field(default_factory=dict)
    motion_map: dict[str, Any] = field(default_factory=dict)


@dataclass
class Live2DTransform:
    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0


def live2d_transform_from_mapping(value: Any) -> Live2DTransform:
    if not isinstance(value, dict):
        value = {}

    def read_float(key: str, default: float) -> float:
        try:
            return float(value.get(key, default))
        except (TypeError, ValueError):
            return default

    return Live2DTransform(
        scale=max(read_float("scale", 1.0), 0.01),
        offset_x=read_float("offset_x", 0.0),
        offset_y=read_float("offset_y", 0.0),
    )


def load_live2d_transform(model_config: ModelConfig) -> Live2DTransform:
    ensure_json_file(
        LIVE2D_TRANSFORM_STATE_PATH,
        default_live2d_transform_state(),
    )
    try:
        with LIVE2D_TRANSFORM_STATE_PATH.open("r", encoding="utf-8") as file:
            state = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read Live2D transform state: {}", exc)
        return live2d_transform_from_mapping(DEFAULT_LIVE2D_TRANSFORM)

    if not isinstance(state, dict):
        logger.warning("Ignoring invalid Live2D transform state: {}", state)
        return live2d_transform_from_mapping(DEFAULT_LIVE2D_TRANSFORM)

    models = state.get("models")
    if not isinstance(models, dict):
        models = {}

    model_keys = (
        model_config.name,
        model_config.model_path.stem,
        model_config.model_path.parent.name,
    )
    for key in model_keys:
        transform = models.get(key)
        if isinstance(transform, dict):
            loaded = live2d_transform_from_mapping(transform)
            logger.info(
                "Loaded Live2D transform for {}: scale={} offset_x={} offset_y={}",
                key,
                loaded.scale,
                loaded.offset_x,
                loaded.offset_y,
            )
            return loaded

    loaded = live2d_transform_from_mapping(state.get("default"))
    logger.info(
        "Loaded default Live2D transform: scale={} offset_x={} offset_y={}",
        loaded.scale,
        loaded.offset_x,
        loaded.offset_y,
    )
    return loaded


@dataclass
class AudioJob:
    path: Path
    display_text: dict[str, Any] | None = None
    actions: dict[str, Any] | None = None
    subtitle_payload: dict[str, Any] | None = None
    volumes: list[float] = field(default_factory=list)
    slice_length_ms: int = 20
    delete_after_play: bool = False
    notify_backend_done: bool = False

class WaveMouthTracker:
    """Small RMS-based mouth tracker for local wav files."""

    def __init__(self, path: Path) -> None:
        self.duration = 0.0
        self.frame_rate = 0
        self.chunk_size = 0
        self.audio_data: np.ndarray | None = None

        try:
            with wave.open(str(path), "rb") as wav_file:
                self.frame_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
                sample_width = wav_file.getsampwidth()
                channels = wav_file.getnchannels()
                self.duration = wav_file.getnframes() / max(self.frame_rate, 1)
        except (wave.Error, OSError) as exc:
            logger.debug("Cannot read wav data for mouth tracking: {}", exc)
            return

        if sample_width != 2:
            logger.debug("Unsupported sample width for mouth tracking: {}", sample_width)
            return

        raw = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            raw = raw.reshape(-1, channels).mean(axis=1).astype(np.int16)

        self.audio_data = raw.astype(np.float32)
        self.chunk_size = max(self.frame_rate // 60, 1)

    def mouth_open_at(self, elapsed: float) -> float:
        if self.audio_data is None or elapsed > self.duration:
            return 0.0

        start = int(elapsed * self.frame_rate)
        end = min(start + self.chunk_size, len(self.audio_data))
        pad = self.chunk_size * 2
        chunk = self.audio_data[max(0, start - pad) : min(len(self.audio_data), end + pad)]
        if len(chunk) == 0:
            return 0.0

        rms = math.sqrt(float(np.mean(chunk**2)))
        return min(1.0, rms / 5000.0)


class AudioPlayer:
    """Pygame-backed single-track player with queue and mouth timing state."""

    def __init__(
        self,
        on_job_done: Callable[[AudioJob], None],
        on_job_start: Callable[[AudioJob], None] | None = None,
    ) -> None:
        self._queue: queue.Queue[AudioJob] = queue.Queue()
        self._on_job_done = on_job_done
        self._on_job_start = on_job_start
        self.current_job: AudioJob | None = None
        self.started_at = 0.0
        self.wave_tracker: WaveMouthTracker | None = None
        self._busy_last_tick = False
        self._muted = False
        self._paused = False
        self._paused_at = 0.0
        self._start_block_reasons: set[str] = set()
        self._mouth_open_smoothed = 0.0
        self._mouth_last_update = 0.0
        self._mixer_reinit_requested = False
        self._mixer_reinit_reason = ""
        self._mixer_last_reinit_at = 0.0
        self._mixer_ready = self._init_mixer()

    def _init_mixer(self) -> bool:
        attempts: list[tuple[str, str | None]] = [("default", None)]
        if sys.platform.startswith("win"):
            attempts.extend(
                [
                    ("directsound", "directsound"),
                    ("winmm", "winmm"),
                ]
            )

        original_driver = os.environ.get("SDL_AUDIODRIVER")
        errors: list[str] = []
        for label, driver in attempts:
            if driver is None:
                if original_driver is None:
                    os.environ.pop("SDL_AUDIODRIVER", None)
                else:
                    os.environ["SDL_AUDIODRIVER"] = original_driver
            else:
                os.environ["SDL_AUDIODRIVER"] = driver

            try:
                pygame.mixer.quit()
            except pygame.error:
                pass

            try:
                pygame.mixer.init(frequency=48000, size=-16, channels=2, buffer=512)
                logger.info(
                    "Pygame mixer initialized: audio_driver={}",
                    os.environ.get("SDL_AUDIODRIVER") or "default",
                )
                return True
            except pygame.error as exc:
                errors.append(f"{label}: {exc}")
                logger.warning(
                    "Pygame mixer init failed with audio_driver={}: {}",
                    os.environ.get("SDL_AUDIODRIVER") or "default",
                    exc,
                )

        if original_driver is None:
            os.environ.pop("SDL_AUDIODRIVER", None)
        else:
            os.environ["SDL_AUDIODRIVER"] = original_driver

        logger.error(
            "Pygame mixer unavailable; local audio playback disabled. attempts={}",
            "; ".join(errors),
        )
        return False

    def request_reinitialize(self, reason: str) -> None:
        self._mixer_reinit_requested = True
        self._mixer_reinit_reason = reason
        logger.error("Pygame mixer reinitialize requested: reason={}", reason)

    def _reinitialize_mixer(self, reason: str, *, force: bool = False) -> bool:
        now = time.monotonic()
        if (
            not force
            and now - self._mixer_last_reinit_at < AUDIO_OUTPUT_REINIT_COOLDOWN_SECONDS
        ):
            return self._mixer_ready

        self._mixer_last_reinit_at = now
        logger.error("Reinitializing pygame mixer: reason={}", reason)
        if self._mixer_ready:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except pygame.error as exc:
                logger.debug("Failed to stop pygame mixer before reinit: {}", exc)

        self._mixer_ready = self._init_mixer()
        if self._mixer_ready:
            self._mixer_reinit_requested = False
            self._mixer_reinit_reason = ""
            try:
                pygame.mixer.music.set_volume(0.0 if self._muted else 1.0)
            except pygame.error as exc:
                logger.warning("Failed to set mixer volume after reinit: {}", exc)
            logger.info("Pygame mixer reinitialized successfully")
        return self._mixer_ready

    def _ensure_mixer_ready(self, reason: str) -> bool:
        if self._mixer_reinit_requested:
            return self._reinitialize_mixer(self._mixer_reinit_reason or reason)
        if not self._mixer_ready:
            return self._reinitialize_mixer(reason)
        return True

    def _finish_unplayed_job(self, job: AudioJob, reason: str) -> None:
        logger.error("Dropping local audio playback: reason={} path={}", reason, job.path)
        if self._on_job_start:
            try:
                self._on_job_start(job)
            except Exception as exc:
                logger.warning("Audio job start callback failed: {}", exc)
        self._on_job_done(job)
        if job.delete_after_play:
            try:
                job.path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to delete temp audio file {}: {}", job.path, exc)

    def enqueue(self, job: AudioJob) -> None:
        if not self._mixer_ready:
            self._ensure_mixer_ready("enqueue")
            if not self._mixer_ready:
                self._finish_unplayed_job(job, "pygame-mixer-unavailable")
                return

        self._queue.put(job)
        self._start_next_if_idle()

    def set_start_blocked(self, reason: str, blocked: bool) -> None:
        was_blocked = bool(self._start_block_reasons)
        if blocked:
            self._start_block_reasons.add(reason)
        else:
            self._start_block_reasons.discard(reason)

        is_blocked = bool(self._start_block_reasons)
        if was_blocked == is_blocked:
            return

        logger.debug(
            "Audio playback start gate {}: reasons={}",
            "blocked" if is_blocked else "released",
            sorted(self._start_block_reasons),
        )
        if not is_blocked:
            self._start_next_if_idle()

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        if not self._mixer_ready:
            logger.info("Audio output muted while mixer is unavailable: {}", muted)
            return
        pygame.mixer.music.set_volume(0.0 if muted else 1.0)
        logger.info("Audio output muted: {}", muted)

    def stop(self) -> None:
        jobs_to_cleanup: list[AudioJob] = []
        if self.current_job:
            jobs_to_cleanup.append(self.current_job)
        if self._mixer_ready:
            pygame.mixer.music.stop()
            try:
                pygame.mixer.music.unload()
            except pygame.error as exc:
                logger.debug("Failed to unload stopped audio: {}", exc)
        self.current_job = None
        self.wave_tracker = None
        self._busy_last_tick = False
        self._paused = False
        self._paused_at = 0.0
        self._reset_mouth_smoothing()
        while not self._queue.empty():
            try:
                jobs_to_cleanup.append(self._queue.get_nowait())
            except queue.Empty:
                break

        for job in jobs_to_cleanup:
            if job.delete_after_play:
                try:
                    job.path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("Failed to delete temp audio file {}: {}", job.path, exc)

    def is_idle(self) -> bool:
        return self.current_job is None and self._queue.empty()

    def is_playing(self) -> bool:
        if not self._mixer_ready:
            return False
        return self.current_job is not None and (self._paused or pygame.mixer.music.get_busy())

    def pause(self) -> bool:
        if not self._mixer_ready:
            return False
        if self.current_job is None:
            return False
        if self._paused:
            return True
        if not pygame.mixer.music.get_busy():
            return False

        pygame.mixer.music.pause()
        self._paused = True
        self._paused_at = time.monotonic()
        logger.debug("Paused audio playback: {}", self.current_job.path)
        return True

    def resume(self) -> bool:
        if not self._mixer_ready:
            return False
        if not self._paused:
            return False

        paused_duration = time.monotonic() - self._paused_at
        self.started_at += max(paused_duration, 0.0)
        pygame.mixer.music.unpause()
        self._paused = False
        self._paused_at = 0.0
        self._busy_last_tick = True
        logger.debug(
            "Resumed audio playback after microphone rejection: {}",
            self.current_job.path if self.current_job else None,
        )
        return True

    def tick(self) -> None:
        if self.current_job is None and (
            self._mixer_reinit_requested or not self._mixer_ready
        ):
            self._ensure_mixer_ready("audio-player-idle")

        if not self._mixer_ready:
            return

        if self._paused:
            return

        try:
            busy = pygame.mixer.music.get_busy()
        except pygame.error as exc:
            logger.error("Pygame mixer get_busy failed: {}", exc)
            self.request_reinitialize(f"get-busy-failed:{exc}")
            busy = False

        if self._busy_last_tick and not busy:
            finished_job = self.current_job
            try:
                pygame.mixer.music.unload()
            except pygame.error as exc:
                logger.debug("Failed to unload finished audio: {}", exc)
            self.current_job = None
            self.wave_tracker = None
            self._busy_last_tick = False
            if finished_job.delete_after_play:
                try:
                    finished_job.path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("Failed to delete temp audio file {}: {}", finished_job.path, exc)
            self._on_job_done(finished_job)
            self._start_next_if_idle()
        self._busy_last_tick = busy

        if self.current_job is None:
            self._start_next_if_idle()

    def _reset_mouth_smoothing(self) -> None:
        self._mouth_open_smoothed = 0.0
        self._mouth_last_update = 0.0

    def _clamp_mouth_open(self, value: Any) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return 0.0

    def _mouth_target_at(self, elapsed: float) -> float:
        if self.current_job is None:
            return 0.0

        volumes = self.current_job.volumes
        if volumes:
            slice_seconds = max(self.current_job.slice_length_ms / 1000.0, 0.001)
            position = elapsed / slice_seconds
            index = int(position)
            if index < 0 or index >= len(volumes):
                return 0.0

            current = self._clamp_mouth_open(volumes[index])
            next_index = index + 1
            if next_index >= len(volumes):
                return current

            next_value = self._clamp_mouth_open(volumes[next_index])
            fraction = max(0.0, min(position - index, 1.0))
            return current + (next_value - current) * fraction

        if self.wave_tracker:
            return self.wave_tracker.mouth_open_at(elapsed)
        return 0.0

    def _smooth_mouth_open(self, target: float, now: float) -> float:
        target = self._clamp_mouth_open(target)
        if target < MOUTH_OPEN_DEADZONE:
            target = 0.0

        if self._mouth_last_update <= 0:
            self._mouth_last_update = now
            self._mouth_open_smoothed = target
            return self._mouth_open_smoothed

        dt = max(0.0, min(now - self._mouth_last_update, 0.1))
        self._mouth_last_update = now
        tau = (
            MOUTH_OPEN_ATTACK_SECONDS
            if target > self._mouth_open_smoothed
            else MOUTH_OPEN_RELEASE_SECONDS
        )
        alpha = 1.0 - math.exp(-dt / max(tau, 0.001))
        self._mouth_open_smoothed += (target - self._mouth_open_smoothed) * alpha
        if self._mouth_open_smoothed < MOUTH_OPEN_DEADZONE:
            self._mouth_open_smoothed = 0.0
        return self._mouth_open_smoothed

    def mouth_open_y(self) -> float:
        now = time.monotonic()
        if self.current_job is None or self._paused:
            return self._smooth_mouth_open(0.0, now)

        elapsed = now - self.started_at
        return self._smooth_mouth_open(self._mouth_target_at(elapsed), now)

    def _start_next_if_idle(self) -> None:
        if (
            self.current_job is not None
            or self._queue.empty()
            or self._start_block_reasons
        ):
            return
        if not self._ensure_mixer_ready("start-next-audio"):
            job = self._queue.get_nowait()
            self._finish_unplayed_job(job, "pygame-mixer-unavailable")
            return

        self.current_job = self._queue.get_nowait()
        self.started_at = time.monotonic()
        self.wave_tracker = WaveMouthTracker(self.current_job.path)
        self._paused = False
        self._paused_at = 0.0
        try:
            pygame.mixer.music.load(str(self.current_job.path))
            pygame.mixer.music.set_volume(0.0 if self._muted else 1.0)
            pygame.mixer.music.play()
            if self._on_job_start:
                try:
                    self._on_job_start(self.current_job)
                except Exception as exc:
                    logger.warning("Audio job start callback failed: {}", exc)
        except pygame.error as exc:
            failed_job = self.current_job
            logger.error("Pygame audio playback failed: path={} error={}", failed_job.path, exc)
            self.current_job = None
            self.wave_tracker = None
            self._busy_last_tick = False
            self.request_reinitialize(f"playback-failed:{exc}")
            if self._reinitialize_mixer("playback-failed", force=True):
                with self._queue.mutex:
                    self._queue.queue.appendleft(failed_job)
            else:
                self._finish_unplayed_job(failed_job, f"playback-failed:{exc}")
            return
        self._busy_last_tick = True
        logger.debug("Started audio playback: {}", self.current_job.path)


class BackendWebSocketClient(threading.Thread):
    """Backend WebSocket client with automatic reconnect and queued sends."""

    def __init__(
        self,
        url: str,
        on_message: Callable[[dict[str, Any]], None],
        on_state: Callable[[bool], None],
        on_error: Callable[[str], None],
        reconnect_interval: float = 3.0,
        heartbeat_interval: float = 30.0,
        connect_timeout: float = BACKEND_WS_CONNECT_TIMEOUT_SECONDS,
        recv_poll_timeout: float = BACKEND_WS_RECV_POLL_TIMEOUT_SECONDS,
        send_timeout: float = BACKEND_WS_SEND_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(daemon=True)
        self.url = url
        self.on_message = on_message
        self.on_state = on_state
        self.on_error = on_error
        self.reconnect_interval = reconnect_interval
        self.heartbeat_interval = heartbeat_interval
        self.connect_timeout = connect_timeout
        self.recv_poll_timeout = recv_poll_timeout
        self.send_timeout = send_timeout
        self._stopped = threading.Event()
        self._wake = threading.Event()
        self._send_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._ws_lock = threading.Lock()
        self._ws: websocket.WebSocket | None = None
        self._connected = False
        self._last_heartbeat = 0.0
        self._url_lock = threading.Lock()
        self._audio_payload_counts: dict[str, int] = {}
        self._audio_payload_samples: dict[str, int] = {}

    def run(self) -> None:
        while not self._stopped.is_set():
            try:
                url = self.current_url()
                logger.info("Connecting to backend WebSocket: {}", url)
                ws = websocket.create_connection(url, timeout=self.connect_timeout)
                ws.settimeout(self.recv_poll_timeout)
                with self._ws_lock:
                    self._ws = ws
                self._set_connected(True)
                self._last_heartbeat = 0.0
                self._send_immediately({"type": "request-init-config"})
                self._receive_loop(ws)
            except Exception as exc:
                if not self._stopped.is_set():
                    self.on_error(str(exc))
                    logger.warning("Backend WebSocket disconnected: {}", exc)
            finally:
                self._close_ws()
                self._set_connected(False)

            if not self._stopped.is_set():
                self._wake.wait(self.reconnect_interval)
                self._wake.clear()

    def send_json(self, payload: dict[str, Any]) -> None:
        with self._ws_lock:
            ws = self._ws
        if not self._connected or ws is None:
            logger.info(
                "Dropping WebSocket payload while backend is offline: type={}",
                payload.get("type"),
            )
            return
        self._send_queue.put(payload)
        self._wake.set()

    def close(self) -> None:
        self._stopped.set()
        self._wake.set()
        self._close_ws()

    def reconnect(self, url: str | None = None) -> None:
        if url:
            with self._url_lock:
                self.url = url
        logger.info("Reconnecting backend WebSocket: {}", self.current_url())
        self._close_ws()
        self._set_connected(False)
        self._wake.set()

    def current_url(self) -> str:
        with self._url_lock:
            return self.url

    def is_connected(self) -> bool:
        return self._connected

    def _receive_loop(self, ws: websocket.WebSocket) -> None:
        while not self._stopped.is_set():
            self._drain_send_queue()
            self._send_heartbeat_if_needed()
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except websocket.WebSocketConnectionClosedException:
                break

            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug("Ignoring non-json WebSocket message: {}", raw)
                continue
            if isinstance(data, dict):
                self.on_message(data)

    def _drain_send_queue(self) -> None:
        while not self._send_queue.empty():
            try:
                payload = self._send_queue.get_nowait()
            except queue.Empty:
                return
            self._send_immediately(payload)

    def _discard_send_queue(self, reason: str) -> None:
        discarded = 0
        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
                discarded += 1
            except queue.Empty:
                break
        if discarded:
            logger.info(
                "Discarded {} queued WebSocket payload(s): reason={}",
                discarded,
                reason,
            )

    def _send_heartbeat_if_needed(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat < self.heartbeat_interval:
            return
        self._send_immediately({"type": "heartbeat"})
        self._last_heartbeat = now

    def _send_immediately(self, payload: dict[str, Any]) -> None:
        with self._ws_lock:
            ws = self._ws
        if not ws:
            return
        try:
            ws.settimeout(self.send_timeout)
            ws.send(json.dumps(payload, ensure_ascii=False))
        finally:
            try:
                ws.settimeout(self.recv_poll_timeout)
            except Exception as exc:
                logger.debug("Failed to restore WebSocket receive timeout: {}", exc)
        payload_type = payload["type"]
        if payload_type in ("mic-audio-data", "raw-audio-data"):
            self._record_audio_payload(payload)
        elif payload_type in ("mic-audio-segment-end", "mic-audio-end"):
            self._flush_audio_payload_log(payload_type)
            logger.debug("Sent WebSocket payload: {}", truncate_data(payload))
        elif payload_type not in ("heartbeat", ):
            self._flush_audio_payload_log(payload_type)
            logger.debug("Sent WebSocket payload: {}", truncate_data(payload))

    def _record_audio_payload(self, payload: dict[str, Any]) -> None:
        payload_type = str(payload.get("type") or "unknown")
        payload_source = str(payload.get("mic_source") or payload.get("audio_source") or "")
        payload_key = f"{payload_type}:{payload_source}" if payload_source else payload_type
        self._audio_payload_counts[payload_key] = (
            self._audio_payload_counts.get(payload_key, 0) + 1
        )
        self._audio_payload_samples[payload_key] = (
            self._audio_payload_samples.get(payload_key, 0)
            + len(payload.get("audio") or [])
        )

    def _flush_audio_payload_log(self, reason: str) -> None:
        if not self._audio_payload_counts:
            return

        summary = ", ".join(
            (
                f"{payload_type}: chunks={count} "
                f"samples={self._audio_payload_samples.get(payload_type, 0)}"
            )
            for payload_type, count in sorted(self._audio_payload_counts.items())
        )
        logger.debug(
            "Sent WebSocket audio payload summary before {}: {}",
            reason,
            summary,
        )
        self._audio_payload_counts.clear()
        self._audio_payload_samples.clear()

    def _close_ws(self) -> None:
        self._flush_audio_payload_log("websocket-close")
        with self._ws_lock:
            ws = self._ws
            self._ws = None
        if ws:
            try:
                ws.close()
            except Exception as exc:
                logger.debug("Error closing backend WebSocket: {}", exc)

    def _set_connected(self, connected: bool) -> None:
        if self._connected == connected:
            return
        if not connected:
            self._discard_send_queue("backend-disconnected")
        self._connected = connected
        self.on_state(connected)
        logger.info("Backend WebSocket state changed: {}", "connected" if connected else "disconnected")


class MicrophoneVadWorker(threading.Thread):
    """Run Silero VAD away from the Qt UI thread and send finished utterances."""

    _RESET_MARKER = object()

    def __init__(
        self,
        on_speech_candidate_start: Callable[[], None],
        on_speech_start: Callable[[], None],
        on_speech_cancelled: Callable[[], None],
        on_audio_detected: Callable[[np.ndarray], None],
        on_audio_confirmed: Callable[[], None],
        on_error: Callable[[str], None],
    ) -> None:
        super().__init__(daemon=True)
        self.on_speech_candidate_start = on_speech_candidate_start
        self.on_speech_start = on_speech_start
        self.on_speech_cancelled = on_speech_cancelled
        self.on_audio_detected = on_audio_detected
        self.on_audio_confirmed = on_audio_confirmed
        self.on_error = on_error
        self._queue: queue.Queue[np.ndarray | object | None] = queue.Queue(maxsize=64)
        self._stopped = threading.Event()
        self._buffer = np.array([], dtype=np.float32)
        self._vad_state = "idle"
        self._vad_hit_count = 0
        self._vad_miss_count = 0
        self._vad_probs: list[float] = []
        self._vad_dbs: list[float] = []
        self._vad_bytes = bytearray()
        self._vad_prob_window = deque(maxlen=MIC_VAD_SMOOTHING_WINDOW)
        self._vad_db_window = deque(maxlen=MIC_VAD_SMOOTHING_WINDOW)
        self._vad_pre_buffer = deque(maxlen=MIC_VAD_PRE_BUFFER_CHUNKS)
        self._vad_state_started_at = time.monotonic()
        self._vad_candidate_start_emitted = False
        self._vad_speech_start_emitted = False
        self._vad_segments_sent = 0

    def submit(self, samples: np.ndarray) -> None:
        if self._stopped.is_set() or samples.size == 0:
            return
        try:
            self._queue.put_nowait(samples.astype(np.float32, copy=True))
        except queue.Full:
            logger.warning("Dropping microphone samples because VAD queue is full")

    def reset(self) -> None:
        if self._stopped.is_set():
            return

        discarded = 0
        while True:
            try:
                self._queue.get_nowait()
                discarded += 1
            except queue.Empty:
                break

        try:
            self._queue.put_nowait(self._RESET_MARKER)
        except queue.Full:
            logger.debug("Display microphone VAD reset marker dropped because queue is full")
        logger.debug(
            "Display microphone VAD reset requested; discarded queued chunks={}",
            discarded,
        )

    def stop(self) -> None:
        self._stopped.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def run(self) -> None:
        try:
            vad_model = self._create_vad_model()
            logger.info(
                "Display microphone Silero VAD worker started with {}",
                type(vad_model).__name__,
            )
            while not self._stopped.is_set():
                try:
                    samples = self._queue.get(timeout=0.2)
                except queue.Empty:
                    for event, audio_bytes in self._finish_stale_vad_if_needed():
                        if event == "reset":
                            logger.debug("Display microphone VAD reset marker received")
                            self._reset_vad_model(vad_model)
                        elif event == "cancel":
                            self._safe_speech_cancelled()
                        elif event == "confirm":
                            self._safe_audio_confirmed()
                        elif event == "audio" and audio_bytes:
                            audio = (
                                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                                / 32767.0
                            )
                            self._safe_audio_detected(audio)
                    continue
                if samples is None:
                    break

                if samples is self._RESET_MARKER:
                    had_pending_speech = self._reset_processing_state("external-reset")
                    self._reset_vad_model(vad_model)
                    if had_pending_speech:
                        self._safe_speech_cancelled()
                    continue

                self._buffer = np.concatenate((self._buffer, samples))
                usable = (len(self._buffer) // MIC_VAD_CHUNK_SAMPLES) * MIC_VAD_CHUNK_SAMPLES
                if usable <= 0:
                    continue

                chunk = self._buffer[:usable]
                self._buffer = self._buffer[usable:]
                for start in range(0, len(chunk), MIC_VAD_CHUNK_SAMPLES):
                    frame = chunk[start : start + MIC_VAD_CHUNK_SAMPLES]
                    speech_prob = self._infer_speech_probability(vad_model, frame)
                    for event, audio_bytes in self._process_vad_frame(
                        speech_prob,
                        frame,
                    ):
                        if event == "candidate":
                            self._safe_speech_candidate_start()
                        elif event == "pause":
                            self._safe_speech_start()
                        elif event == "reset":
                            logger.debug("Display microphone VAD reset marker received")
                            self._reset_vad_model(vad_model)
                        elif event == "cancel":
                            self._safe_speech_cancelled()
                        elif event == "confirm":
                            self._safe_audio_confirmed()
                        elif event == "audio" and audio_bytes:
                            audio = (
                                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                                / 32767.0
                            )
                            self._safe_audio_detected(audio)
        except Exception as exc:
            logger.exception("Display microphone VAD worker failed")
            self._safe_error(str(exc))

    def _create_vad_model(self) -> Any:
        return silero_vad.load_silero_vad()

    def _reset_vad_model(self, vad_model: Any) -> None:
        try:
            vad_model.reset_states()
        except Exception as exc:
            logger.debug("Failed to reset Silero VAD model state: {}", exc)

    def _infer_speech_probability(self, vad_model: Any, frame: np.ndarray) -> float:
        frame = frame.astype(np.float32, copy=False)
        model_input = torch.from_numpy(frame)
        return self._probability_to_float(vad_model(model_input, MIC_SAMPLE_RATE))

    @staticmethod
    def _probability_to_float(value: Any) -> float:
        if isinstance(value, dict):
            for key in ("prob", "probability", "speech_prob", "isSpeech"):
                if key in value:
                    value = value[key]
                    break
        elif isinstance(value, (list, tuple)) and value:
            value = value[0]
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            return float(value.item())
        if isinstance(value, np.ndarray):
            return float(value.reshape(-1)[0])
        return float(value)

    @staticmethod
    def _calculate_db(audio_data: np.ndarray) -> float:
        rms = np.sqrt(np.mean(np.square(audio_data)))
        return 20 * np.log10(rms + 1e-7) if rms > 0 else -np.inf

    def _reset_vad_buffers(self) -> None:
        self._vad_probs.clear()
        self._vad_dbs.clear()
        self._vad_bytes.clear()
        self._vad_prob_window.clear()
        self._vad_db_window.clear()
        self._vad_candidate_start_emitted = False
        self._vad_speech_start_emitted = False

    def _reset_vad_utterance_state(self) -> None:
        self._vad_hit_count = 0
        self._vad_miss_count = 0
        self._reset_vad_buffers()
        self._vad_pre_buffer.clear()
        self._vad_segments_sent = 0

    def _reset_processing_state(self, reason: str) -> bool:
        had_pending_speech = (
            self._vad_state != "idle"
            or bool(self._vad_probs)
            or self._vad_speech_start_emitted
            or self._buffer.size > 0
        )
        self._buffer = np.array([], dtype=np.float32)
        self._set_vad_state("idle", reason)
        self._reset_vad_utterance_state()
        return had_pending_speech

    def _set_vad_state(
        self,
        state: str,
        reason: str,
        prob: float | None = None,
        db: float | None = None,
    ) -> None:
        if self._vad_state == state:
            return
        logger.debug(
            "Display microphone VAD state: {} -> {} ({}, prob={}, db={})",
            self._vad_state,
            state,
            reason,
            f"{prob:.3f}" if prob is not None else "-",
            f"{db:.1f}" if db is not None else "-",
        )
        self._vad_state = state
        self._vad_state_started_at = time.monotonic()

    def _append_vad_frame(self, frame_bytes: bytes, prob: float, db: float) -> None:
        self._vad_probs.append(prob)
        self._vad_dbs.append(db)
        self._vad_bytes.extend(frame_bytes)

    def _maybe_emit_speech_start(
        self,
        events: list[tuple[str, bytes | None]],
    ) -> None:
        if self._vad_speech_start_emitted:
            return
        if len(self._vad_probs) < MIC_VAD_PAUSE_CHUNKS:
            return

        self._vad_speech_start_emitted = True
        events.append(("pause", None))

    def _emit_current_segment(
        self,
        events: list[tuple[str, bytes | None]],
        reason: str,
    ) -> bool:
        if not self._vad_bytes:
            return False
        if not self._vad_speech_start_emitted:
            logger.debug(
                "Skipping microphone segment during {} before speech-start emit: chunks={}",
                reason,
                len(self._vad_probs),
            )
            return False

        prefix = b"".join(self._vad_pre_buffer) if self._vad_segments_sent == 0 else b""
        audio_bytes = prefix + bytes(self._vad_bytes)
        logger.info(
            "Display microphone segment detected: reason={} segment={} chunks={} bytes={} max_prob={:.3f} max_db={:.1f}",
            reason,
            self._vad_segments_sent,
            len(self._vad_probs),
            len(audio_bytes),
            max(self._vad_probs) if self._vad_probs else 0.0,
            max(self._vad_dbs) if self._vad_dbs else float("-inf"),
        )
        events.append(("audio", audio_bytes))
        self._vad_segments_sent += 1
        self._vad_bytes.clear()
        self._vad_pre_buffer.clear()
        return True

    def _finish_current_utterance(self, reason: str) -> list[tuple[str, bytes | None]]:
        events: list[tuple[str, bytes | None]] = []
        has_valid_utterance = (
            self._vad_segments_sent > 0
            or len(self._vad_probs) >= MIC_VAD_MIN_UTTERANCE_CHUNKS
        )
        if has_valid_utterance:
            self._maybe_emit_speech_start(events)
            self._emit_current_segment(events, reason)
            events.append(("confirm", None))
            logger.info(
                "Display microphone utterance confirmed: reason={} chunks={} segments={} max_prob={:.3f} max_db={:.1f}",
                reason,
                len(self._vad_probs),
                self._vad_segments_sent,
                max(self._vad_probs) if self._vad_probs else 0.0,
                max(self._vad_dbs) if self._vad_dbs else float("-inf"),
            )
        else:
            logger.debug(
                "Dropping short microphone utterance during {}: chunks={} required_chunks={} max_prob={:.3f} max_db={:.1f}",
                reason,
                len(self._vad_probs),
                MIC_VAD_MIN_UTTERANCE_CHUNKS,
                max(self._vad_probs) if self._vad_probs else 0.0,
                max(self._vad_dbs) if self._vad_dbs else float("-inf"),
            )
            if self._vad_candidate_start_emitted:
                events.append(("cancel", None))

        events.append(("reset", None))
        self._set_vad_state("idle", reason)
        self._reset_vad_utterance_state()
        return events

    def _finish_stale_vad_if_needed(self) -> list[tuple[str, bytes | None]]:
        if self._vad_state == "idle":
            return []

        elapsed = time.monotonic() - self._vad_state_started_at
        if elapsed < MIC_VAD_MAX_UTTERANCE_SECONDS:
            return []

        logger.warning(
            "Display microphone VAD stayed in {} for {:.1f}s; forcing utterance reset",
            self._vad_state,
            elapsed,
        )
        return self._finish_current_utterance("stale-vad-timeout")

    def _process_vad_frame(
        self,
        speech_prob: float,
        frame: np.ndarray,
    ) -> list[tuple[str, bytes | None]]:
        int_frame = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
        frame_bytes = int_frame.tobytes()
        db = self._calculate_db(int_frame.astype(np.float32))
        self._vad_prob_window.append(speech_prob)
        self._vad_db_window.append(db)
        smoothed_prob = float(np.mean(self._vad_prob_window))
        smoothed_db = float(np.mean(self._vad_db_window))
        is_speech = (
            smoothed_prob >= MIC_VAD_PROB_THRESHOLD
            and smoothed_db >= MIC_VAD_DB_THRESHOLD
        )

        events: list[tuple[str, bytes | None]] = []
        if self._vad_state == "idle":
            self._vad_pre_buffer.append(frame_bytes)
            if is_speech:
                self._vad_hit_count += 1
                if self._vad_hit_count >= MIC_VAD_REQUIRED_HITS:
                    self._set_vad_state(
                        "active",
                        "speech-start",
                        smoothed_prob,
                        smoothed_db,
                    )
                    self._vad_candidate_start_emitted = True
                    events.append(("candidate", None))
                    self._append_vad_frame(frame_bytes, smoothed_prob, smoothed_db)
                    self._maybe_emit_speech_start(events)
                    self._vad_hit_count = 0
            else:
                self._vad_hit_count = 0

        elif self._vad_state == "active":
            self._append_vad_frame(frame_bytes, smoothed_prob, smoothed_db)
            self._maybe_emit_speech_start(events)
            if is_speech:
                self._vad_miss_count = 0
            else:
                self._vad_miss_count += 1
                logger.debug(
                    "Display microphone VAD active miss streak: misses={}/{} "
                    "prob={:.3f}/{:.3f} db={:.1f}/{:.1f}",
                    self._vad_miss_count,
                    MIC_VAD_REQUIRED_MISSES,
                    smoothed_prob,
                    MIC_VAD_PROB_THRESHOLD,
                    smoothed_db,
                    MIC_VAD_DB_THRESHOLD,
                )
                if self._vad_miss_count >= MIC_VAD_REQUIRED_MISSES:
                    self._set_vad_state(
                        "inactive",
                        "silence-candidate",
                        smoothed_prob,
                        smoothed_db,
                    )
                    self._emit_current_segment(events, "inactive")
                    self._vad_miss_count = 0

        elif self._vad_state == "inactive":
            if is_speech:
                self._append_vad_frame(frame_bytes, smoothed_prob, smoothed_db)
                self._maybe_emit_speech_start(events)
                self._vad_hit_count += 1
                if self._vad_hit_count >= MIC_VAD_REQUIRED_HITS:
                    self._set_vad_state(
                        "active",
                        "speech-resumed",
                        smoothed_prob,
                        smoothed_db,
                    )
                    self._vad_hit_count = 0
                    self._vad_miss_count = 0
            else:
                self._vad_hit_count = 0
                self._vad_miss_count += 1
                if self._vad_miss_count >= MIC_VAD_REQUIRED_MISSES:
                    events.extend(self._finish_current_utterance("speech-end"))

        return events

    def _safe_speech_start(self) -> None:
        try:
            self.on_speech_start()
        except Exception as exc:
            logger.debug("Microphone speech-start callback failed: {}", exc)

    def _safe_speech_candidate_start(self) -> None:
        try:
            self.on_speech_candidate_start()
        except Exception as exc:
            logger.debug("Microphone speech-candidate callback failed: {}", exc)

    def _safe_speech_cancelled(self) -> None:
        try:
            self.on_speech_cancelled()
        except Exception as exc:
            logger.debug("Microphone speech-cancelled callback failed: {}", exc)

    def _safe_audio_detected(self, audio: np.ndarray) -> None:
        try:
            self.on_audio_detected(audio)
        except Exception as exc:
            logger.debug("Microphone audio callback failed: {}", exc)

    def _safe_audio_confirmed(self) -> None:
        try:
            self.on_audio_confirmed()
        except Exception as exc:
            logger.debug("Microphone audio confirm callback failed: {}", exc)

    def _safe_error(self, message: str) -> None:
        try:
            self.on_error(message)
        except Exception as exc:
            logger.debug("Microphone error callback failed: {}", exc)


class _ShrinkableLabel(QLabel):
    """QLabel 的 minimumSizeHint 会跟着字号 + word-wrap 后的内容变大,
    并被父布局当成窗口的最小尺寸 -> 文字长了就再也缩不回去.
    这里只强制 minimumSizeHint=(0,0), 让窗口能自由缩到用户设的下限;
    sizeHint 保留 QLabel 默认 (font 高度), 这样名字栏在 VBox 里仍能拿到正确的预留高度."""

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(0, 0)


class LiveStreamingAgentSubtitleWindow(QOpenGLWidget):
    """Live Streaming Agent 回复字幕窗口

    显示 Live Streaming Agent 最新一句回复 + 根据 emotion 标签切换左上角头像 icon.

    用法:
        win = LiveStreamingAgentSubtitleWindow()
        win.show()
        win.set_subtitle("你好啊~", emotion="happy")

    Icon 命名约定:
        放在 LIVE_FRONTEND_RESOURCE_ROOT / "live_streaming_agent_icons" 目录下,
        文件名 = emotion 标签 + ".png" (例: happy.png, mad.png, cry.png).
        找不到对应 emotion 时回退到 default.png; 再没有就显示空头像位.

    样式:
        米色卡片 + 棕色边框 + 圆角, 仿 mockup 图风格.
        头像是圆形 (用 QPainterPath clip).
    """

    closed = pyqtSignal()  # 用户手动关窗时通知 console 把按钮变灰

    # 卡片样式色 (从 mockup 图取色)
    # 卡片内部保持米色不透明; 卡片"外面四角"靠无边框透明窗口透出背景
    _CARD_BG = QColor("#FAEDD0")          # 米色背景
    _CARD_BORDER = QColor("#C49356")      # 棕色边框
    _AVATAR_BORDER = QColor("#A6743E")    # 头像描边色 (略深的棕)
    _AVATAR_BORDER_WIDTH = 3
    _TEXT_COLOR = QColor("#6B3D14")       # 深棕文字

    # 响应式比例 (基于窗口高度计算)
    # 边框粗细 ~ 高度的 1.6%, 圆角 ~ 高度的 13%
    # 边距 ~ 高度的 17%, 头像 ~ 高度的 78%
    # 名字字号 ~ 高度的 10%, 文字字号 ~ 高度的 7%
    _BORDER_WIDTH_RATIO = 0.016
    _CORNER_RADIUS_RATIO = 0.13
    _MARGIN_RATIO = 0.17
    _AVATAR_RATIO = 0.78
    _NAME_FONT_RATIO = 0.10
    # 正文字号跟名字字号一样大 (用户要求)
    _BODY_FONT_RATIO = 0.10

    # 用 ratio 算出的下限, 极小窗口 (~50px) 下也要能渲染
    _MIN_BORDER_WIDTH = 1
    _MIN_CORNER_RADIUS = 3
    _MIN_AVATAR_SIZE = 12
    _MIN_NAME_FONT = 5
    _MIN_BODY_FONT = 5

    # 打字机动画参数
    # 关键设计:
    # 流式 TTS (GPT-SoVITS 等) 的 audio chunk 严重低估整句时长 —— 第一帧只带
    # ~200ms 音频, 但 display_text 已经是完整一句话 (例如 6 个字). 直接拿
    # chunk 时长除以字数, per_char 就被算成 ~33ms, 字幕跑得比语音快好几倍.
    # 而且后续 chunk 的 text 跟首帧相同, 被前端 dedup 直接 return, 没机会
    # "看到更多音频" 后修正速率. 所以光指望 chunk 时长不行.
    #
    # 解决: 用一个 "自然中文语速" 的默认值兜底, 并强制 per_char = max(chunk估算, 默认).
    # chunk 时长只能"放慢"字幕, 不能"加快".
    #
    # 调速 (用户反馈字幕慢于 TTS 语音): 把默认/下限调快, 让字幕更贴语音.
    # 150ms/字 = ~6.7 字/秒, 略快于自然朗读, 抵消打字机相对语音的固有滞后感,
    # 使最终一句话基本跟语音同步收尾, 而不是语音念完字幕还在补字.
    _DEFAULT_PER_CHAR_MS = 10
    # 即便有靠谱的整句时长 (非流式分支算出来), 也强制不快于这个下限.
    # 110ms/字 = ~9 字/秒, 接近人眼可舒适跟读的上限, 再快就太跳了.
    _MIN_PER_CHAR_MS = 110
    # 上限只防御异常大的 duration_ms (例如 metadata 出错). 800ms/字 已经够慢.
    _MAX_PER_CHAR_MS = 800

    def __init__(
        self,
        icon_dir: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.icon_dir = Path(icon_dir)
        self._icon_cache: dict[str, QPixmap | None] = {}
        self._current_emotion: str = "default"

        # 打字机动画状态
        self._full_text: str = ""
        self._typed_index: int = 0
        self._typing_timer = QTimer(self)
        self._typing_timer.setSingleShot(False)
        self._typing_timer.timeout.connect(self._on_type_tick)
        self._typing_paused_by_audio = False

        # 整轮回复累积状态
        # 同一个 turn 内的所有 segment 文字都拼接进 _full_text 继续打字机播放;
        # 下一轮 turn_id 变化时才重置. 这样 Live Streaming Agent 说完整段后字幕保留, 直到新一轮.
        self._current_turn_id: str | None = None
        # 用于 dedup: 同一 segment 因流式 TTS 会重复来很多 chunk, 防止拼接重复
        self._last_appended_segment: str = ""

        # 窗口属性: 跟 Live2DWindow 一样走 OpenGL 表面 + 透明背景.
        #  - QOpenGLWidget + alpha 格式 + WA_TranslucentBackground: GL framebuffer
        #    带 alpha 通道, 圆角卡片外的四角清成真透明 (不会被合成成黑色),
        #    OBS "窗口捕获" 能拿到带 alpha 的圆角卡片.
        #  - 保留系统标题栏: 可拖动 + 边缘原生拉伸缩放.
        gl_format = QSurfaceFormat()
        gl_format.setAlphaBufferSize(8)
        self.setFormat(gl_format)

        flags = (
            Qt.Window
            | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowCloseButtonHint
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)

        self.setWindowTitle("Agent 字幕")
        self.resize(
            LIVE_STREAMING_AGENT_SUBTITLE_DEFAULT_WIDTH,
            LIVE_STREAMING_AGENT_SUBTITLE_DEFAULT_HEIGHT,
        )
        # 极限缩放下限 (用户要求最小 50x50)
        self.setMinimumSize(50, 50)

        self._build_ui()
        self._apply_emotion("default")

    # -------------------- UI 构建 --------------------

    def _build_ui(self) -> None:
        # 头像区 (大小不写死, 由 _rescale_layout 动态设置)
        self._avatar_label = QLabel()
        self._avatar_label.setAlignment(Qt.AlignCenter)
        self._avatar_label.setStyleSheet(
            "background: transparent; border: none;"
        )

        # 名字 (字号由 _rescale_layout 动态设置)
        # 用 _ShrinkableLabel 而不是裸 QLabel: 否则窗口放大时名字字号变大,
        # minimumSizeHint 也跟着变大, 把窗口宽度下限拉高 -> 横向缩不回去.
        # 但 sizeHint 保留默认, 这样在 VBox 里仍能拿到正确的预留高度,
        # 不会被 stretch=1 的正文挤掉.
        self._name_label = _ShrinkableLabel("Live Streaming Agent")
        self._name_label.setStyleSheet(
            f"color: {self._TEXT_COLOR.name()}; "
            "background: transparent; border: none;"
        )

        # 文字 (字号动态)
        # 关键点: QLabel + setWordWrap(True) 默认会把 word-wrap 后的高度当成
        # minimumSizeHint, layout 把它累加到窗口最小尺寸 -> 放大后就缩不回去.
        # 解决: 用 _ShrinkableLabel 把 minimumSizeHint 强制成 (0,0),
        # 这样窗口能一路缩到 self.setMinimumSize(50, 50) 这个用户层下限.
        # SizePolicy 保留 Expanding/Expanding: 正文占满右侧剩余空间.
        self._text_label = _ShrinkableLabel("")
        self._text_label.setWordWrap(True)
        self._text_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._text_label.setStyleSheet(
            f"color: {self._TEXT_COLOR.name()}; "
            "background: transparent; border: none;"
        )
        self._text_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        # 右侧 (名字 + 文字)
        self._right_layout = QVBoxLayout()
        self._right_layout.setSpacing(6)
        self._right_layout.setContentsMargins(0, 2, 0, 0)
        self._right_layout.addWidget(self._name_label)
        self._right_layout.addWidget(self._text_label, 1)

        # 左侧 (头像)
        self._left_layout = QVBoxLayout()
        self._left_layout.setSpacing(0)
        self._left_layout.setContentsMargins(0, 0, 0, 0)
        self._left_layout.addWidget(
            self._avatar_label, 0, Qt.AlignTop | Qt.AlignLeft
        )
        self._left_layout.addStretch(1)

        # 卡片内部主布局
        self._card_layout = QHBoxLayout()
        self._card_layout.addLayout(self._left_layout)
        self._card_layout.addLayout(self._right_layout, 1)
        self.setLayout(self._card_layout)

        # 先初始化一次响应式尺寸
        self._rescale_layout()

    # -------------------- 响应式: 跟随窗口高度 --------------------

    def _current_border_width(self) -> int:
        return max(
            self._MIN_BORDER_WIDTH,
            int(self.height() * self._BORDER_WIDTH_RATIO),
        )

    def _current_corner_radius(self) -> int:
        return max(
            self._MIN_CORNER_RADIUS,
            int(self.height() * self._CORNER_RADIUS_RATIO),
        )

    def _rescale_layout(self) -> None:
        """根据当前窗口高度重设头像大小、字号、边距."""
        h = max(self.height(), 1)
        # 边距和 spacing 下限给 2px (极小窗口时尽量给内容腾空间)
        margin = max(2, int(h * self._MARGIN_RATIO))
        spacing = max(2, int(margin * 0.85))
        # 卡片内边距; 右边距留多一点让文字不顶圆角
        self._card_layout.setContentsMargins(
            margin, margin, int(margin * 1.4), margin
        )
        self._card_layout.setSpacing(spacing)

        # 头像大小: 高度比例, 受最小值兜底, 上限留够边距
        avatar = max(
            self._MIN_AVATAR_SIZE,
            min(int(h * self._AVATAR_RATIO), h - 2 * margin),
        )
        # setFixedSize 会把 min/max 都钉死, 当窗口放大到 h=400 时 avatar=264,
        # 之后 WM 检查窗口最小尺寸时就被这 264 卡住, 整个窗口缩不下去.
        # 解决: 先 setFixedSize 给出"目标显示尺寸" + Fixed 策略 (避免被横向拉伸),
        # 然后立刻把 minimumSize 放回到 _MIN_AVATAR_SIZE, 这样:
        #   - 当前几何/preferred = avatar (视觉跟以前一样)
        #   - max = avatar (策略仍是 Fixed, 不会被无意义拉伸)
        #   - min = 12 (用户缩窗时头像也能跟着缩, 不阻挡窗口收缩)
        self._avatar_label.setFixedSize(avatar, avatar)
        self._avatar_label.setMinimumSize(
            self._MIN_AVATAR_SIZE, self._MIN_AVATAR_SIZE
        )

        # 字号
        name_size = max(self._MIN_NAME_FONT, int(h * self._NAME_FONT_RATIO))
        body_size = max(self._MIN_BODY_FONT, int(h * self._BODY_FONT_RATIO))

        name_font = QFont()
        name_font.setPointSize(name_size)
        name_font.setBold(True)
        self._name_label.setFont(name_font)

        body_font = QFont()
        body_font.setPointSize(body_size)
        self._text_label.setFont(body_font)

        # 头像也要按新尺寸重画 (重新圆形裁剪 + 描边)
        self._apply_emotion(self._current_emotion)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale_layout()

    # -------------------- 卡片背景绘制 (OpenGL 表面) --------------------

    def paintGL(self) -> None:
        # 在 GL framebuffer 上自绘米色圆角卡片 + 棕色边框.
        # 子 QLabel (头像/名字/正文) 会在 GL 内容之上合成显示.
        # 边框宽度和圆角半径都跟随当前窗口高度变化 (响应式)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # 先把整个 GL 表面清成真透明 (CompositionMode_Source 覆盖掉旧像素的 alpha),
        # 这样圆角卡片外的四角是透明的, OBS 抓窗口能保留 alpha.
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        border_width = self._current_border_width()
        corner_radius = self._current_corner_radius()

        half_border = border_width / 2.0
        rect = self.rect().adjusted(
            int(half_border),
            int(half_border),
            -int(half_border),
            -int(half_border),
        )
        path = QPainterPath()
        path.addRoundedRect(
            rect.x(),
            rect.y(),
            rect.width(),
            rect.height(),
            corner_radius,
            corner_radius,
        )
        painter.fillPath(path, self._CARD_BG)
        pen = painter.pen()
        pen.setColor(self._CARD_BORDER)
        pen.setWidth(border_width)
        painter.setPen(pen)
        painter.drawPath(path)
        painter.end()

    # -------------------- 头像加载 / emotion 切换 --------------------

    def _load_pixmap_for(self, emotion: str) -> QPixmap | None:
        if emotion in self._icon_cache:
            return self._icon_cache[emotion]
        candidate = self.icon_dir / f"{emotion}.png"
        if not candidate.exists():
            # 回退 default.png
            candidate = self.icon_dir / "default.png"
        if candidate.exists():
            pm = QPixmap(str(candidate))
            if pm.isNull():
                logger.warning(
                    "Live Streaming Agent subtitle: failed to load icon {}", candidate
                )
                self._icon_cache[emotion] = None
                return None
            self._icon_cache[emotion] = pm
            return pm
        # 缓存空结果避免反复探测磁盘
        self._icon_cache[emotion] = None
        return None

    def _render_circular(self, pixmap: QPixmap, size: int) -> QPixmap:
        """把任意比例 pixmap 裁成圆形, 加棕色描边."""
        result = QPixmap(size, size)
        result.fill(Qt.transparent)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # 中心圆形 clip
        path = QPainterPath()
        path.addEllipse(0, 0, size, size)
        painter.setClipPath(path)

        scaled = pixmap.scaled(
            size,
            size,
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )
        # 居中绘制
        offset_x = (size - scaled.width()) // 2
        offset_y = (size - scaled.height()) // 2
        painter.drawPixmap(offset_x, offset_y, scaled)
        painter.setClipping(False)

        # 描边
        pen = painter.pen()
        pen.setColor(self._AVATAR_BORDER)
        pen.setWidth(self._AVATAR_BORDER_WIDTH)
        painter.setPen(pen)
        painter.drawEllipse(
            self._AVATAR_BORDER_WIDTH // 2,
            self._AVATAR_BORDER_WIDTH // 2,
            size - self._AVATAR_BORDER_WIDTH,
            size - self._AVATAR_BORDER_WIDTH,
        )
        painter.end()
        return result

    def _apply_emotion(self, emotion: str) -> None:
        pm = self._load_pixmap_for(emotion)
        if pm is None or pm.isNull():
            # 显示一个棕色空圆作为占位
            placeholder = QPixmap(self._avatar_label.width(), self._avatar_label.height())
            placeholder.fill(Qt.transparent)
            painter = QPainter(placeholder)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setBrush(self._CARD_BORDER)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(0, 0, placeholder.width(), placeholder.height())
            painter.end()
            self._avatar_label.setPixmap(placeholder)
            return
        size = min(self._avatar_label.width(), self._avatar_label.height())
        self._avatar_label.setPixmap(self._render_circular(pm, size))
        self._current_emotion = emotion

    # -------------------- 对外 API --------------------

    def _compute_per_char(
        self,
        segment_text: str,
        duration_ms: int | None,
    ) -> int:
        """根据音频时长 + segment 字数算每字间隔, 但永远不会比 DEFAULT 更快.

        流式 TTS 的 chunk 时长严重低估整句 (chunk 只是整句一小段, 但 text 是
        完整段), 直接除会算出 30-50ms/字 的飞奔速度. 用 max() 让 chunk 估算
        只能"放慢"字幕, 不能"加快".
        """
        char_count = max(1, len(segment_text or ""))
        if duration_ms is not None:
            from_audio = duration_ms // char_count
        else:
            from_audio = 0
        per_char = max(from_audio, self._DEFAULT_PER_CHAR_MS)
        per_char = max(self._MIN_PER_CHAR_MS, min(per_char, self._MAX_PER_CHAR_MS))
        return per_char

    def set_subtitle(
        self,
        text: str | None,
        emotion: str | None = None,
        duration_ms: int | None = None,
        turn_id: str | None = None,
    ) -> None:
        """打字机式累积字幕; 同一轮回复持续追加, 下一轮 turn_id 变化时才清空.

        Args:
            text: 这一段 segment 的文字 (不是整轮累积值).
                  None  -> 不动文字 (仅更新 emotion / turn_id)
                  ""    -> 不动文字 (例如剥完表情标签为空)
                  非空  -> 跟现有累积文字拼接 (同 turn 内) 或开启新累积 (新 turn)
            emotion: 切换头像情绪; None 保持当前.
            duration_ms: 这段音频时长. 用来推算打字机速率, 但只能比默认慢.
            turn_id: 当前对话回合 id. 跟上次记录的 turn_id 不同就视为新一轮,
                     清空累积字幕重头开始. None 时不触发清空, 视为同 turn.
        """
        is_new_turn = turn_id is not None and turn_id != self._current_turn_id
        if emotion:
            self._apply_emotion(emotion)
        elif is_new_turn:
            self._apply_emotion("default")

        # 检测新一轮回复 (turn_id 变化) -> 清空累积, 准备重新打字机
        if is_new_turn:
            self._stop_typing()
            self._full_text = ""
            self._typed_index = 0
            self._last_appended_segment = ""
            self._text_label.setText("")
            self._current_turn_id = turn_id

        # text=None 或剥完为空 -> 仅 emotion / turn 切换, 保留现有累积字幕
        if not text:
            return

        # 同一 segment 的重复 chunk (流式 TTS 后续帧 text 跟首帧相同) -> dedup
        if text == self._last_appended_segment:
            return
        self._last_appended_segment = text

        # 把这一段追加到累积末尾 (turn 内多段拼成一整片字幕, 两三行慢慢展开)
        self._full_text = self._full_text + text

        # 用新 segment 的音频时长重新算节拍 (但永远不快于默认值)
        per_char = self._compute_per_char(text, duration_ms)
        self._typing_timer.setInterval(per_char)

        # turn 内第一段的首字立刻显示, 减少首字延迟感
        if self._typed_index == 0:
            self._typed_index = 1
            self._text_label.setText(self._full_text[:1])

        # 还有字没打完 -> 启动/继续 timer 继续推进
        if self._typed_index < len(self._full_text):
            if not self._typing_timer.isActive():
                if not self._typing_paused_by_audio:
                    self._typing_timer.start()

    def _on_type_tick(self) -> None:
        """打字机定时器回调: 每次推进一字, 完成时自动停."""
        if self._typing_paused_by_audio:
            self._stop_typing()
            return
        if self._typed_index >= len(self._full_text):
            self._stop_typing()
            return
        self._typed_index += 1
        self._text_label.setText(self._full_text[: self._typed_index])
        if self._typed_index >= len(self._full_text):
            self._stop_typing()

    def _stop_typing(self) -> None:
        if self._typing_timer.isActive():
            self._typing_timer.stop()

    def pause_subtitle_progress(self) -> None:
        """音频暂停时同步暂停字幕打字机."""
        self._typing_paused_by_audio = True
        self._stop_typing()

    def resume_subtitle_progress(self) -> None:
        """音频恢复播放时同步恢复字幕打字机."""
        if not self._typing_paused_by_audio:
            return
        self._typing_paused_by_audio = False
        if self._typed_index < len(self._full_text):
            self._typing_timer.start()

    def stop_subtitle_progress(self) -> None:
        """停止Agent 字幕继续打字, 并丢弃还没显示出来的后续文本."""
        self._typing_paused_by_audio = False
        self._stop_typing()
        self._full_text = self._full_text[: self._typed_index]
        self._last_appended_segment = ""
        self._text_label.setText(self._full_text)

    def reload_icons(self) -> None:
        """清空图标缓存并重新加载当前 emotion (用户替换图片后调用)."""
        self._icon_cache.clear()
        self._apply_emotion(self._current_emotion)

    # -------------------- 关闭事件 --------------------

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)


class BarrageSubtitleWindow(QOpenGLWidget):
    """弹幕字幕窗口

    展示"被回复的那条弹幕": 左上角是用户的抖音 id (@xxx),
    下方直接显示弹幕原文 (不走打字机, 消费哪条就整条放上去),
    展示区可占两三行, 超出自动换行.

    样式取自 psd (回复弹幕界面.psd):
        米色圆角卡片 + 深棕边框, 文字深棕.
    """

    closed = pyqtSignal()  # 用户手动关窗时通知 console 把按钮变灰

    # 卡片样式色 (从 psd 取色)
    # 卡片内部保持米色不透明; 卡片"外面四角"靠 OpenGL 透明表面透出背景
    _CARD_BG = QColor("#FDF5EB")          # 米色背景
    _CARD_BORDER = QColor("#7B3E1C")      # 深棕边框
    _TEXT_COLOR = QColor("#7B3E1C")       # 深棕文字 (id 和正文同色)

    # 响应式比例 (基于窗口高度计算)
    _BORDER_WIDTH_RATIO = 0.025           # 边框 ~ 高度 2.5% (psd 10/439)
    _CORNER_RADIUS_RATIO = 0.14
    _MARGIN_RATIO = 0.12
    _ID_FONT_RATIO = 0.13                 # 抖音 id 字号 ~ 高度 13%
    _BODY_FONT_RATIO = 0.12               # 弹幕正文字号 ~ 高度 12%
    _AVATAR_RATIO = 0.18                  # 用户头像直径 ~ 高度 18%

    _MIN_BORDER_WIDTH = 1
    _MIN_CORNER_RADIUS = 3
    _MIN_ID_FONT = 6
    _MIN_BODY_FONT = 6
    _MIN_AVATAR_SIZE = 10
    _AVATAR_BORDER_WIDTH = 2
    _SHOW_AVATAR = False

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # 窗口属性: 跟 Live2DWindow / Agent 字幕一致, 走 OpenGL 表面 + 透明背景.
        # QOpenGLWidget + alpha 格式 + WA_TranslucentBackground: 圆角卡片外四角
        # 清成真透明, OBS 窗口捕获能保留 alpha. 保留标题栏可拖动 + 边缘原生缩放.
        gl_format = QSurfaceFormat()
        gl_format.setAlphaBufferSize(8)
        self.setFormat(gl_format)

        flags = (
            Qt.Window
            | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowCloseButtonHint
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)

        self.setWindowTitle("弹幕字幕")
        self.resize(
            BARRAGE_SUBTITLE_DEFAULT_WIDTH,
            BARRAGE_SUBTITLE_DEFAULT_HEIGHT,
        )
        self.setMinimumSize(50, 50)

        self._avatar_url = ""
        self._avatar_cache: dict[str, QPixmap | None] = {}
        self._avatar_file_cache: dict[str, QPixmap | None] = {}
        self._avatar_network = QNetworkAccessManager(self)

        self._build_ui()

    # -------------------- UI 构建 --------------------

    def _build_ui(self) -> None:
        # 用户头像 (异步下载; 无头像/下载失败时隐藏)
        self._avatar_label = QLabel()
        self._avatar_label.setAlignment(Qt.AlignCenter)
        self._avatar_label.setStyleSheet(
            "background: transparent; border: none;"
        )
        self._avatar_label.hide()

        # 抖音 id (左上角); 用 _ShrinkableLabel 避免放大后窗口缩不回去
        self._id_label = _ShrinkableLabel("@用户ID")
        self._id_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._id_label.setStyleSheet(
            f"color: {self._TEXT_COLOR.name()}; "
            "background: transparent; border: none;"
        )

        # 弹幕正文 (下方, 自动换行, 两三行)
        self._text_label = _ShrinkableLabel("")
        self._text_label.setWordWrap(True)
        self._text_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._text_label.setStyleSheet(
            f"color: {self._TEXT_COLOR.name()}; "
            "background: transparent; border: none;"
        )
        self._text_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        self._header_layout = QHBoxLayout()
        self._header_layout.setContentsMargins(0, 0, 0, 0)
        self._header_layout.addWidget(
            self._avatar_label, 0, Qt.AlignLeft | Qt.AlignVCenter
        )
        self._header_layout.addWidget(self._id_label, 1)

        self._card_layout = QVBoxLayout()
        self._card_layout.addLayout(self._header_layout)
        self._card_layout.addWidget(self._text_label, 1)
        self.setLayout(self._card_layout)

        self._rescale_layout()

    # -------------------- 响应式 --------------------

    def _current_border_width(self) -> int:
        return max(
            self._MIN_BORDER_WIDTH,
            int(self.height() * self._BORDER_WIDTH_RATIO),
        )

    def _current_corner_radius(self) -> int:
        return max(
            self._MIN_CORNER_RADIUS,
            int(self.height() * self._CORNER_RADIUS_RATIO),
        )

    def _rescale_layout(self) -> None:
        h = max(self.height(), 1)
        margin = max(2, int(h * self._MARGIN_RATIO))
        self._card_layout.setContentsMargins(
            int(margin * 1.2), margin, int(margin * 1.2), margin
        )
        self._card_layout.setSpacing(max(2, int(margin * 0.5)))
        self._header_layout.setSpacing(max(2, int(margin * 0.45)))

        id_size = max(self._MIN_ID_FONT, int(h * self._ID_FONT_RATIO))
        body_size = max(self._MIN_BODY_FONT, int(h * self._BODY_FONT_RATIO))
        avatar_size = max(
            self._MIN_AVATAR_SIZE,
            min(int(h * self._AVATAR_RATIO), h - 2 * margin),
        )
        self._avatar_label.setFixedSize(avatar_size, avatar_size)
        self._avatar_label.setMinimumSize(
            self._MIN_AVATAR_SIZE, self._MIN_AVATAR_SIZE
        )

        id_font = QFont()
        id_font.setPointSize(id_size)
        id_font.setBold(True)
        self._id_label.setFont(id_font)

        body_font = QFont()
        body_font.setPointSize(body_size)
        self._text_label.setFont(body_font)
        self._rerender_current_avatar()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale_layout()

    # -------------------- 卡片背景绘制 (OpenGL 表面) --------------------

    def paintGL(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # 先把整个 GL 表面清成真透明 (圆角外四角保留 alpha)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        border_width = self._current_border_width()
        corner_radius = self._current_corner_radius()

        half_border = border_width / 2.0
        rect = self.rect().adjusted(
            int(half_border),
            int(half_border),
            -int(half_border),
            -int(half_border),
        )
        path = QPainterPath()
        path.addRoundedRect(
            rect.x(),
            rect.y(),
            rect.width(),
            rect.height(),
            corner_radius,
            corner_radius,
        )
        painter.fillPath(path, self._CARD_BG)
        pen = painter.pen()
        pen.setColor(self._CARD_BORDER)
        pen.setWidth(border_width)
        painter.setPen(pen)
        painter.drawPath(path)
        painter.end()

    # -------------------- 用户头像加载 / 渲染 --------------------

    def _normalize_avatar_url(self, avatar_url: str | None) -> str:
        url = str(avatar_url or "").strip()
        if not url.startswith(("http://", "https://")):
            return ""
        return url

    def _normalize_avatar_path(self, avatar_path: str | None) -> str:
        raw_path = str(avatar_path or "").strip()
        if not raw_path:
            return ""
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        try:
            path = path.resolve()
        except OSError:
            return ""
        if not path.exists() or not path.is_file():
            return ""
        return str(path)

    def _render_circular_avatar(self, pixmap: QPixmap, size: int) -> QPixmap:
        result = QPixmap(size, size)
        result.fill(Qt.transparent)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        path = QPainterPath()
        path.addEllipse(0, 0, size, size)
        painter.setClipPath(path)

        scaled = pixmap.scaled(
            size,
            size,
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )
        painter.drawPixmap(
            (size - scaled.width()) // 2,
            (size - scaled.height()) // 2,
            scaled,
        )
        painter.setClipping(False)

        pen = painter.pen()
        pen.setColor(self._CARD_BORDER)
        pen.setWidth(self._AVATAR_BORDER_WIDTH)
        painter.setPen(pen)
        half = self._AVATAR_BORDER_WIDTH // 2
        painter.drawEllipse(
            half,
            half,
            max(1, size - self._AVATAR_BORDER_WIDTH),
            max(1, size - self._AVATAR_BORDER_WIDTH),
        )
        painter.end()
        return result

    def _apply_avatar_pixmap(self, pixmap: QPixmap | None) -> None:
        if pixmap is None or pixmap.isNull():
            self._avatar_label.clear()
            self._avatar_label.hide()
            return
        size = min(self._avatar_label.width(), self._avatar_label.height())
        if size <= 0:
            return
        self._avatar_label.setPixmap(self._render_circular_avatar(pixmap, size))
        self._avatar_label.show()

    def _rerender_current_avatar(self) -> None:
        if not self._avatar_url:
            return
        self._apply_avatar_pixmap(self._avatar_cache.get(self._avatar_url))

    def _set_avatar_url(self, avatar_url: str | None) -> None:
        url = self._normalize_avatar_url(avatar_url)
        self._avatar_url = url
        if not url:
            self._apply_avatar_pixmap(None)
            return

        if url in self._avatar_cache:
            self._apply_avatar_pixmap(self._avatar_cache.get(url))
            return

        self._apply_avatar_pixmap(None)
        request = QNetworkRequest(QUrl(url))
        request.setRawHeader(b"User-Agent", b"Mozilla/5.0")
        reply = self._avatar_network.get(request)
        reply.setProperty("avatar_url", url)
        reply.finished.connect(
            lambda reply=reply: self._on_avatar_download_finished(reply)
        )

    def _set_avatar_path(self, avatar_path: str | None) -> bool:
        path = self._normalize_avatar_path(avatar_path)
        if not path:
            return False

        self._avatar_url = ""
        if path in self._avatar_file_cache:
            self._apply_avatar_pixmap(self._avatar_file_cache.get(path))
            return True

        pixmap = QPixmap(path)
        if pixmap.isNull():
            logger.debug("Barrage subtitle avatar file decode failed: {}", path)
            self._avatar_file_cache[path] = None
            self._apply_avatar_pixmap(None)
            return False

        self._avatar_file_cache[path] = pixmap
        self._apply_avatar_pixmap(pixmap)
        return True

    def _on_avatar_download_finished(self, reply: QNetworkReply) -> None:
        url = str(reply.property("avatar_url") or "")
        try:
            if reply.error() != QNetworkReply.NoError:
                logger.debug(
                    "Barrage subtitle avatar download failed: {} ({})",
                    url,
                    reply.errorString(),
                )
                self._avatar_cache[url] = None
                if url == self._avatar_url:
                    self._apply_avatar_pixmap(None)
                return

            pixmap = QPixmap()
            if not pixmap.loadFromData(bytes(reply.readAll())):
                logger.debug("Barrage subtitle avatar decode failed: {}", url)
                self._avatar_cache[url] = None
                if url == self._avatar_url:
                    self._apply_avatar_pixmap(None)
                return

            self._avatar_cache[url] = pixmap
            if url == self._avatar_url:
                self._apply_avatar_pixmap(pixmap)
        finally:
            reply.deleteLater()

    # -------------------- 对外 API --------------------

    def set_barrage(
        self,
        nickname: str | None,
        content: str | None,
        avatar_url: str | None = "",
        avatar_path: str | None = "",
    ) -> None:
        """直接展示一条被回复的弹幕 (无打字机, 整条放上去).

        Args:
            nickname: 用户抖音 id; None/空则保留当前 id.
            content: 弹幕原文; None/空则保留当前正文.
            avatar_url: 保留后端头像 URL 字段，当前弹幕字幕不展示头像.
            avatar_path: 保留已落盘头像路径字段，当前弹幕字幕不展示头像.
        """
        if self._SHOW_AVATAR:
            if not self._set_avatar_path(avatar_path):
                self._set_avatar_url(avatar_url)
        else:
            self._avatar_url = ""
            self._apply_avatar_pixmap(None)
        if nickname:
            self._id_label.setText(f"@{nickname}")
        if content:
            self._text_label.setText(content)

    # -------------------- 关闭事件 --------------------

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)


class Live2DWindow(QOpenGLWidget):
    playback_complete = pyqtSignal(object)
    window_closed = pyqtSignal()
    wake_animation_state_changed = pyqtSignal(bool)
    audio_started = pyqtSignal(object)

    def __init__(
        self,
        model_config: ModelConfig,
        width: int = 720,
        height: int = 960,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.model_transform = load_live2d_transform(model_config)
        self.model: live2d.Model | None = None
        self.elapsed = QElapsedTimer()
        self.pending_backend_done = False
        self.pending_backend_done_turn_id: str | None = None
        self.interrupted = False
        self.active_turn_id: str | None = None
        self.blocked_turn_ids: set[str] = set()
        self.awaiting_turn_initial_expression = False
        self.current_response_parts: list[str] = []
        self._first_frame_drawn = False
        self._live2d_resources_released = False
        self.audio_player = AudioPlayer(
            on_job_done=self._handle_audio_job_done,
            on_job_start=self._handle_audio_job_start,
        )
        self.last_expressions = []
        self.last_motions: list[str] = []
        self.loaded_extra_motions: dict[tuple[str, str], int] = {}
        self.sleep_motion_enabled = False
        self.sleep_motion_key: str | None = None
        self.sleep_motion_started_at = 0.0
        self.sleep_motion_duration = 0.0
        self.sleep_motion_loop = False
        self.wake_motion_active = False
        self.wake_voice_released = False
        self.wake_motion_started_at = 0.0
        self.wake_motion_duration = 0.0
        self.speech_motion_active = False
        self.action_motion_active = False
        self.return_motion_active = False
        self.active_motion_requires_return = False
        self.action_motion_idle_delay_seconds = LIVE2D_NORMAL_MOTION_RETRY_SECONDS
        self.next_speech_motion_attempt_at = 0.0
        # 平滑回归待机: 动作播完后, 把参数从冻结值渐变回默认值, 避免 ResetAllParameters 硬切。
        self.param_return_fade_active = False
        self.param_return_fade_start = 0.0
        self.param_return_frozen: list[float] = []
        self.param_return_defaults: list[float] = []
        self.deferred_wake_audio_payloads: list[dict[str, Any]] = []
        self.persistent_expressions: set[str] = set()

        flags = (
            Qt.Window
            | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowCloseButtonHint
        )
        # if always_on_top:
        #     flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        # self.setWindowOpacity(1)
        self.setWindowTitle(f"Dream Maker Live2D - {model_config.name}")
        self.setMinimumSize(QSize(320, 420))
        self.resize(width, height)

        self.tick_timer = QTimer(self)
        self.tick_timer.setInterval(16)
        self.tick_timer.timeout.connect(self._tick)

    def initializeGL(self) -> None:
        self._live2d_resources_released = False
        live2d.init()
        self.model = live2d.Model()
        self.model.LoadModelJson(str(self.model_config.model_path))
        live2d.glInit()
        self.model.CreateRenderer(3)
        self.model.Resize(self.width(), self.height())
        self._apply_model_transform()
        self._set_parameter_if_exists(LIVE2D_PARAM_SILENCE_ID, 1.0)
        self.elapsed.start()
        self.tick_timer.start()
        expressions = self.model.GetExpressions()
        logger.info("Live2D model loaded: {}", self.model_config.model_path)
        logger.info("Live2D model expressions: {}", expressions)
        self._apply_watermark_expression(expressions)
        if self.sleep_motion_enabled:
            self._start_sleep_motion(force=True)

    def resizeGL(self, width: int, height: int) -> None:
        if self.model:
            self.model.Resize(width, height)
            self._apply_model_transform()

    def paintGL(self) -> None:
        if not self.model:
            return

        try:
            live2d.clearBuffer()

            dt = self.elapsed.restart() / 1000
            self.model.LoadParameters()
            if not self.model.IsMotionFinished():
                self.model.UpdateMotion(dt)

            self.model.SaveParameters()
            if not self.sleep_motion_enabled and not self.wake_motion_active:
                self.model.UpdateBlink(dt)
            self.model.UpdateBreath(dt)
            self.model.UpdateExpression(dt)
            self.model.UpdatePhysics(dt)
            self.model.UpdatePose(dt)
            mouth_open_y = self.audio_player.mouth_open_y()
            self._set_parameter_if_exists(
                live2d.StandardParams.ParamMouthOpenY,
                mouth_open_y,
            )
            if self.audio_player.is_playing():
                self._set_parameter_if_exists(LIVE2D_PARAM_SILENCE_ID, 1.0)
            self.model.Draw()
        except Exception as exc:
            logger.exception("Live2D paintGL failed: {}", exc)

    def closeEvent(self, event: Any) -> None:
        self.tick_timer.stop()
        self.audio_player.stop()
        self._release_live2d_resources("closeEvent")
        self.window_closed.emit()
        super().closeEvent(event)

    def _release_live2d_resources(self, reason: str) -> None:
        if self._live2d_resources_released:
            return

        self._live2d_resources_released = True
        model = self.model
        self.model = None
        self.last_expressions = []
        self.last_motions = []
        self.sleep_motion_enabled = False
        self.sleep_motion_key = None
        self.sleep_motion_started_at = 0.0
        self.sleep_motion_duration = 0.0
        self.sleep_motion_loop = False
        self.speech_motion_active = False
        self.action_motion_active = False
        self.return_motion_active = False
        self.active_motion_requires_return = False
        self.action_motion_idle_delay_seconds = LIVE2D_NORMAL_MOTION_RETRY_SECONDS
        self.next_speech_motion_attempt_at = 0.0
        self._cancel_param_return_fade()
        self._clear_wake_motion_state(clear_deferred=True)
        self.loaded_extra_motions.clear()
        self.persistent_expressions.clear()

        context_active = False
        if model:
            try:
                self.makeCurrent()
                context_active = True
            except Exception as exc:
                logger.warning(
                    "Failed to activate Live2D OpenGL context for release: {}",
                    exc,
                )

        if model and context_active:
            try:
                try:
                    model.ResetExpressions()
                except Exception as exc:
                    logger.debug("Failed to reset Live2D expressions on release: {}", exc)

                try:
                    model.DestroyRenderer()
                except Exception as exc:
                    logger.warning("Failed to destroy Live2D renderer: {}", exc)

                try:
                    live2d.glRelease()
                except Exception as exc:
                    logger.warning("Failed to release Live2D OpenGL resources: {}", exc)
            finally:
                try:
                    self.doneCurrent()
                except Exception as exc:
                    logger.debug("Failed to release current OpenGL context: {}", exc)

        try:
            live2d.dispose()
        except Exception as exc:
            logger.warning("Failed to dispose Live2D framework: {}", exc)

        logger.info("Live2D resources released by {}", reason)

    def _message_turn_id(self, data: dict[str, Any]) -> str | None:
        return data.get("turn_id") or data.get("request_id")

    def _is_wake_turn_protected(self) -> bool:
        return self.wake_motion_active or bool(self.deferred_wake_audio_payloads)

    def _prepare_wake_turn(self) -> None:
        if self.active_turn_id or self.blocked_turn_ids:
            logger.debug(
                "Clearing stale Live2D turn before wake voice: "
                "active_turn_id={} blocked_turns={}",
                self.active_turn_id,
                list(self.blocked_turn_ids),
            )
        self.active_turn_id = None
        self.blocked_turn_ids.clear()
        self.interrupted = False
        self.pending_backend_done = False
        self.pending_backend_done_turn_id = None

    def _start_turn(self, turn_id: str | None) -> bool:
        if (
            self._is_wake_turn_protected()
            and self.active_turn_id
            and turn_id
            and turn_id != self.active_turn_id
        ):
            logger.debug(
                "Keeping active Live2D wake turn during wake animation: "
                "active_turn_id={} incoming_turn_id={}",
                self.active_turn_id,
                turn_id,
            )
            return False
        self.blocked_turn_ids.clear()
        self.active_turn_id = turn_id
        self.awaiting_turn_initial_expression = True
        return True

    def _block_turn(self, turn_id: str | None) -> None:
        target_turn_id = turn_id or self.active_turn_id
        self.blocked_turn_ids.clear()
        if target_turn_id:
            self.blocked_turn_ids.add(target_turn_id)

    def _finish_turn(self, turn_id: str | None) -> None:
        if not turn_id or self.active_turn_id == turn_id:
            self.active_turn_id = None
        self.blocked_turn_ids.clear()

    def _should_accept_turn(self, turn_id: str | None) -> bool:
        if not turn_id:
            return True
        return self.active_turn_id == turn_id and turn_id not in self.blocked_turn_ids

    def _is_tts_failure_sleep_payload(self, data: dict[str, Any]) -> bool:
        return data.get("source") == "tts_failure_sleep_voice"

    def handle_backend_message(self, data: dict[str, Any]) -> None:
        msg_type = data.get("type")
        turn_id = self._message_turn_id(data)
        if msg_type == "audio":
            if self.interrupted or not self._should_accept_turn(turn_id):
                logger.debug(
                    "Skipping audio payload because Live2D playback is interrupted or turn is stale: {}",
                    turn_id,
                )
                return
            if (
                self.wake_motion_active
                and not self.wake_voice_released
                and not self._is_tts_failure_sleep_payload(data)
            ):
                self.deferred_wake_audio_payloads.append(dict(data))
                logger.debug(
                    "Deferring audio payload until Live2D wake voice delay elapses: turn_id={} deferred_count={}",
                    turn_id,
                    len(self.deferred_wake_audio_payloads),
                )
                return
            if self.wake_motion_active and not self.wake_voice_released:
                logger.info(
                    "Playing TTS failure sleep voice immediately during wake motion: turn_id={}",
                    turn_id,
                )
            self._handle_backend_audio(data)
        elif msg_type == "backend-synth-complete":
            if not self._should_accept_turn(turn_id):
                logger.debug("Dropping backend synth completion for stale turn: {}", turn_id)
                self._send_frontend_playback_complete(
                    turn_id,
                    skipped=True,
                    reason="stale-turn",
                    force=True,
                )
                return
            if self.interrupted:
                self._send_frontend_playback_complete(turn_id)
            elif (
                self.wake_motion_active
                and not self.wake_voice_released
            ) or self.deferred_wake_audio_payloads:
                self.pending_backend_done = True
                self.pending_backend_done_turn_id = turn_id
                logger.debug(
                    "Deferring playback completion until Live2D wake voice is released: turn_id={}",
                    turn_id,
                )
            elif self.audio_player.is_idle():
                self._send_frontend_playback_complete(turn_id)
            else:
                self.pending_backend_done = True
                self.pending_backend_done_turn_id = turn_id
        elif msg_type == "full-text" and data.get("text"):
            logger.info("Backend text: {}", data["text"])
        elif msg_type == "control":
            self._handle_backend_control(str(data.get("text") or ""), turn_id)
        elif msg_type == "interrupt-signal":
            if self._should_accept_turn(turn_id):
                self._interrupt_playback(reason="backend interrupt-signal", turn_id=turn_id)
        elif msg_type == "force-new-message":
            if self._should_accept_turn(turn_id):
                self.current_response_parts.clear()
                self.interrupted = False
        elif msg_type == "error":
            logger.error("Backend error: {}", data.get("message"))

    def play_audio_file(
        self,
        audio_path: Path,
        display_text: dict[str, Any] | None = None,
        actions: dict[str, Any] | None = None,
        subtitle_payload: dict[str, Any] | None = None,
        volumes: list[float] | None = None,
        slice_length_ms: int = 20,
        delete_after_play: bool = False,
        notify_backend_done: bool = False,
    ) -> None:
        self.audio_player.enqueue(
            AudioJob(
                path=audio_path,
                display_text=display_text,
                actions=actions,
                subtitle_payload=subtitle_payload,
                volumes=volumes or [],
                slice_length_ms=slice_length_ms,
                delete_after_play=delete_after_play,
                notify_backend_done=notify_backend_done,
            )
        )

    def _tick(self) -> None:
        self.audio_player.tick()
        self._tick_sleep_motion()
        self._tick_wake_motion()
        self._tick_param_return_fade()
        self._tick_motion_return()
        self._tick_normal_motion()
        if (
            self.pending_backend_done
            and self.audio_player.is_idle()
            and (not self.wake_motion_active or self.wake_voice_released)
            and not self.deferred_wake_audio_payloads
        ):
            self._send_frontend_playback_complete(self.pending_backend_done_turn_id)
        self.update()

    def _apply_model_transform(self) -> None:
        if not self.model:
            return

        if hasattr(self.model, "SetScale"):
            self.model.SetScale(self.model_transform.scale)
        else:
            logger.warning("Current Live2D binding does not support SetScale")

        if hasattr(self.model, "SetOffset"):
            self.model.SetOffset(
                self.model_transform.offset_x,
                self.model_transform.offset_y,
            )
        else:
            logger.warning("Current Live2D binding does not support SetOffset")

    def _apply_watermark_expression(self, expressions: Any) -> None:
        if not self.model:
            return

        expression_names = {str(expression) for expression in expressions or []}
        if LIVE2D_WATERMARK_EXPRESSION_NAME not in expression_names:
            logger.debug(
                "Live2D watermark expression not found: {}",
                LIVE2D_WATERMARK_EXPRESSION_NAME,
            )
            return

        try:
            self.model.AddExpression(LIVE2D_WATERMARK_EXPRESSION_NAME)
            self.persistent_expressions.add(LIVE2D_WATERMARK_EXPRESSION_NAME)
            logger.info(
                "Applied Live2D watermark expression: {}",
                LIVE2D_WATERMARK_EXPRESSION_NAME,
            )
        except Exception as exc:
            logger.warning(
                "Failed to apply Live2D watermark expression {}: {}",
                LIVE2D_WATERMARK_EXPRESSION_NAME,
                exc,
            )

    def _set_parameter_if_exists(self, parameter_id: str, value: float) -> None:
        if not self.model:
            return
        try:
            self.model.SetParameterValueById(parameter_id, value)
        except Exception:
            pass

    def _reset_all_parameters(self, reason: str) -> bool:
        if not self.model:
            logger.info(
                "Skip Live2D ResetAllParameters: reason={} model_missing=True",
                reason,
            )
            return False
        if not hasattr(self.model, "ResetAllParameters"):
            logger.info(
                "Skip Live2D ResetAllParameters: reason={} unsupported=True",
                reason,
            )
            return False
        logger.info("Calling Live2D ResetAllParameters: reason={}", reason)
        try:
            self.model.ResetAllParameters()
        except Exception as exc:
            logger.warning(
                "Failed Live2D ResetAllParameters: reason={} error={}",
                reason,
                exc,
            )
            return False
        logger.info(
            "Called Live2D ResetAllParameters: reason={} success=True",
            reason,
        )
        return True

    def _begin_param_return_fade(self, reason: str) -> None:
        """把所有参数从当前(冻结)值平滑渐变回默认值, 代替瞬间 ResetAllParameters。

        动作 (点头/摇头/歪头等表情同步动作, 以及开心/卖萌/哭哭等自动回归动作) 播完后
        会僵在极端姿态并冻结全部约 500 个参数; 直接 ResetAllParameters 会一帧内归位,
        造成明显卡顿。这里在 LIVE2D_RETURN_FADE_SECONDS 内用 smoothstep 插值平滑回归。
        缺少所需底层 API 时退回硬重置, 保证功能正确。
        """
        if not self.model:
            return
        get_value = getattr(self.model, "GetParameterValue", None)
        get_default = getattr(self.model, "GetParameterDefaultValue", None)
        set_save = getattr(self.model, "SetAndSaveParameterValue", None)
        get_ids = getattr(self.model, "GetParameterIds", None)
        if not (
            callable(get_value)
            and callable(get_default)
            and callable(set_save)
            and callable(get_ids)
        ):
            self._reset_all_parameters(reason)
            return
        try:
            count = len(get_ids())
            frozen = [float(get_value(i)) for i in range(count)]
            defaults = [float(get_default(i)) for i in range(count)]
        except Exception as exc:
            logger.warning(
                "Failed to snapshot Live2D params for return fade; hard reset: {}",
                exc,
            )
            self._reset_all_parameters(reason)
            return
        if all(abs(f - d) < 1e-4 for f, d in zip(frozen, defaults)):
            # 已在默认位, 无需过渡。
            self.param_return_fade_active = False
            return
        self.param_return_frozen = frozen
        self.param_return_defaults = defaults
        self.param_return_fade_start = time.monotonic()
        self.param_return_fade_active = True
        logger.info(
            "Begin Live2D smooth return-to-default fade: reason={} params={}",
            reason,
            count,
        )

    def _tick_param_return_fade(self) -> None:
        if not self.param_return_fade_active:
            return
        if not self.model or self.sleep_motion_enabled or self.wake_motion_active:
            self._cancel_param_return_fade()
            return
        set_save = getattr(self.model, "SetAndSaveParameterValue", None)
        if not callable(set_save):
            self._finish_param_return_fade()
            return
        if LIVE2D_RETURN_FADE_SECONDS > 0.0:
            alpha = (time.monotonic() - self.param_return_fade_start) / (
                LIVE2D_RETURN_FADE_SECONDS
            )
        else:
            alpha = 1.0
        alpha = max(0.0, min(1.0, alpha))
        done = alpha >= 1.0
        eased = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep, 收尾更柔和
        frozen = self.param_return_frozen
        defaults = self.param_return_defaults
        n = min(len(frozen), len(defaults))
        try:
            for i in range(n):
                f = frozen[i]
                d = defaults[i]
                set_save(i, f + (d - f) * eased)
        except Exception as exc:
            logger.warning("Live2D return fade failed; hard reset: {}", exc)
            done = True
        if done:
            self._finish_param_return_fade()

    def _finish_param_return_fade(self) -> None:
        """渐变结束: 精确归位并清理状态 (此时参数已基本在默认位, 无可见跳变)。"""
        was_active = self.param_return_fade_active
        self.param_return_fade_active = False
        self.param_return_frozen = []
        self.param_return_defaults = []
        if was_active and self.model:
            self._reset_all_parameters("return-fade-complete")

    def _cancel_param_return_fade(self) -> None:
        """被新动作/打断/休眠等接管时, 只清理状态, 不做归位 (交给接管方驱动参数)。"""
        self.param_return_fade_active = False
        self.param_return_frozen = []
        self.param_return_defaults = []

    def _action_emotions(self, actions: dict[str, Any]) -> list[str]:
        raw_emotions = actions.get("emotions")
        if raw_emotions is None and actions.get("emotion"):
            raw_emotions = [actions.get("emotion")]
        if isinstance(raw_emotions, str):
            raw_emotions = [raw_emotions]
        if not isinstance(raw_emotions, list):
            return []

        return list(
            dict.fromkeys(
                normalize_emotion_tag(emotion)
                for emotion in raw_emotions
                if normalize_emotion_tag(emotion)
            )
        )

    def _motion_from_mapping(self, mapping: Any) -> dict[str, Any] | None:
        if isinstance(mapping, str):
            motion_text = mapping.strip()
            if not motion_text:
                return None
            return {"group": motion_text}

        if not isinstance(mapping, dict):
            return None

        motion = mapping.get("motion")
        if isinstance(motion, dict):
            result = dict(motion)
        else:
            result = dict(mapping)

        group = result.get("group") or result.get("name")
        file_path = result.get("file") or result.get("path")
        if not group and file_path:
            group = Path(str(file_path)).stem
        if not group:
            return None

        result["group"] = str(group)
        if file_path:
            result["file"] = str(file_path)
        return result

    def _expression_from_mapping(self, mapping: Any) -> str | None:
        if isinstance(mapping, str):
            return mapping.strip() or None
        if isinstance(mapping, dict):
            expression = (
                mapping.get("expression")
                or mapping.get("name")
                or mapping.get("id")
            )
            if expression:
                return str(expression).strip() or None
        return None

    def _actions_from_emotions(
        self,
        emotions: list[str],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        expressions: list[str] = []
        motions: list[dict[str, Any]] = []

        for emotion in emotions:
            if emotion in LIVE2D_STATE_MOTION_EMOTION_TAGS:
                logger.debug(
                    "Ignoring Live2D state motion emotion tag from actions: {}",
                    emotion,
                )
                continue

            motion_mapping = self.model_config.motion_map.get(emotion)
            if motion_mapping is not None:
                motion = self._motion_from_mapping(motion_mapping)
                if motion:
                    motions.append(motion)
                    continue

            mapping = self.model_config.emotion_map.get(emotion)
            if mapping is None:
                logger.debug("No Live2D mapping for emotion tag: {}", emotion)
                continue

            mappings = mapping if isinstance(mapping, list) else [mapping]
            for item in mappings:
                item_type = str(item.get("type") or "").lower() if isinstance(item, dict) else ""
                if item_type == "motion" or (
                    isinstance(item, dict)
                    and any(key in item for key in ("motion", "group", "file", "path"))
                    and not any(key in item for key in ("expression", "id"))
                ):
                    motion = self._motion_from_mapping(item)
                    if motion:
                        motions.append(motion)
                    continue

                expression = self._expression_from_mapping(item)
                if expression:
                    expressions.append(expression)

        return (
            list(dict.fromkeys(expressions)),
            self._dedupe_motions(motions),
        )

    def _dedupe_motions(self, motions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for motion in motions:
            key = self._motion_key(motion)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(motion)
        return deduped

    def _motion_key(self, motion: dict[str, Any]) -> str:
        return "|".join(
            [
                str(motion.get("group") or ""),
                str(motion.get("index") or motion.get("no") or ""),
                str(motion.get("file") or motion.get("path") or ""),
            ]
        )

    def _legacy_motion_specs(self, actions: dict[str, Any]) -> list[dict[str, Any]]:
        raw_motions = actions.get("motions") or []
        if isinstance(raw_motions, (str, dict)):
            raw_motions = [raw_motions]
        if not isinstance(raw_motions, list):
            return []

        motions = []
        for item in raw_motions:
            motion = self._motion_from_mapping(item)
            if motion:
                motions.append(motion)
        return self._dedupe_motions(motions)

    def set_sleeping(self, sleeping: bool) -> None:
        if self.sleep_motion_enabled == sleeping:
            return

        self.sleep_motion_enabled = sleeping
        self.sleep_motion_key = None
        self._cancel_param_return_fade()
        self.speech_motion_active = False
        self.action_motion_active = False
        self.return_motion_active = False
        self.active_motion_requires_return = False
        self.action_motion_idle_delay_seconds = LIVE2D_NORMAL_MOTION_RETRY_SECONDS
        self.next_speech_motion_attempt_at = 0.0
        if sleeping:
            self._clear_wake_motion_state(clear_deferred=True)
            self._set_auto_blink_enabled(False)
            self._clear_expressions()
            self._start_sleep_motion(force=True)
            logger.info("Live2D sleep motion loop enabled")
            return

        self.last_motions = []
        self.sleep_motion_key = None
        self.sleep_motion_started_at = 0.0
        self.sleep_motion_duration = 0.0
        self.sleep_motion_loop = False
        self.speech_motion_active = False
        self.action_motion_active = False
        self.return_motion_active = False
        self.active_motion_requires_return = False
        self.action_motion_idle_delay_seconds = LIVE2D_NORMAL_MOTION_RETRY_SECONDS
        self.next_speech_motion_attempt_at = 0.0
        self._set_auto_blink_enabled(False)
        self._reset_all_parameters("sleep-motion-disabled")
        logger.info("Live2D sleep motion loop disabled")
        if not self._start_wake_motion():
            self._set_auto_blink_enabled(True)

    def _sleep_motion_spec(self) -> dict[str, Any] | None:
        mapping = self.model_config.motion_map.get("sleep")
        if mapping is None:
            return None
        return self._motion_from_mapping(mapping)

    def _wake_motion_spec(self) -> dict[str, Any] | None:
        mapping = self.model_config.motion_map.get("wake")
        if mapping is None:
            return None
        return self._motion_from_mapping(mapping)

    def _start_sleep_motion(self, *, force: bool = False) -> None:
        if not self.sleep_motion_enabled or not self.model:
            return

        motion = self._sleep_motion_spec()
        if not motion:
            logger.debug("No Live2D sleep motion mapping configured")
            return

        motion_key = self._motion_key(motion)
        if not force and motion_key == self.sleep_motion_key and not self.model.IsMotionFinished():
            return

        if self._start_motion(motion):
            self.sleep_motion_key = motion_key
            self.sleep_motion_started_at = time.monotonic()
            meta = self._motion_meta(motion)
            self.sleep_motion_duration = max(float(meta.get("duration") or 0.0), 0.0)
            self.sleep_motion_loop = bool(meta.get("loop"))

    def _tick_sleep_motion(self) -> None:
        if not self.sleep_motion_enabled or not self.model:
            return

        if not self.model.IsMotionFinished():
            return

        elapsed = time.monotonic() - self.sleep_motion_started_at
        logger.info(
            "Live2D sleep motion finished, restarting: elapsed={:.3f}s duration={}s key={}",
            elapsed,
            self.sleep_motion_duration,
            self.sleep_motion_key,
        )
        self._start_sleep_motion(force=True)

    def _start_wake_motion(self) -> bool:
        self._clear_wake_motion_state(clear_deferred=True)
        if not self.model:
            return False

        motion = self._wake_motion_spec()
        if not motion:
            logger.debug("No Live2D wake motion mapping configured")
            return False

        self._prepare_wake_turn()
        self._set_auto_blink_enabled(False)
        if not self._start_motion(motion):
            return False

        self.last_motions = [self._motion_key(motion)]
        meta = self._motion_meta(motion)
        self.wake_motion_active = True
        self.wake_animation_state_changed.emit(True)
        self.wake_motion_started_at = time.monotonic()
        self.wake_motion_duration = max(float(meta.get("duration") or 0.0), 0.0)
        logger.info(
            "Live2D wake motion started before wake voice: duration={}s voice_delay={}s",
            self.wake_motion_duration,
            LIVE2D_WAKE_VOICE_DELAY_SECONDS,
        )
        return True

    def _tick_wake_motion(self) -> None:
        if not self.wake_motion_active:
            return

        elapsed = time.monotonic() - self.wake_motion_started_at
        if (
            not self.wake_voice_released
            and elapsed >= LIVE2D_WAKE_VOICE_DELAY_SECONDS
        ):
            self._release_deferred_wake_audio_payloads("wake-voice-delay-elapsed")

        if elapsed < 0.5:
            return
        if self.wake_motion_duration > 0 and elapsed < self.wake_motion_duration:
            return
        if self.wake_motion_duration <= 0 and self.model and not self.model.IsMotionFinished():
            return

        self._finish_wake_motion()

    def _finish_wake_motion(self) -> None:
        if not self.wake_voice_released:
            self._release_deferred_wake_audio_payloads("wake-motion-finished")
        self._clear_wake_motion_state(clear_deferred=False)
        self._set_auto_blink_enabled(True)
        self._reset_all_parameters("wake-motion-finished")
        self.next_speech_motion_attempt_at = (
            time.monotonic() + LIVE2D_NORMAL_MOTION_RETRY_SECONDS
        )
        logger.info("Live2D wake motion finished")

    def _clear_wake_motion_state(self, *, clear_deferred: bool) -> None:
        was_active = self.wake_motion_active
        self.wake_motion_active = False
        self.wake_voice_released = False
        self.wake_motion_started_at = 0.0
        self.wake_motion_duration = 0.0
        if clear_deferred:
            self.deferred_wake_audio_payloads.clear()
        if was_active:
            self.wake_animation_state_changed.emit(False)

    def _release_deferred_wake_audio_payloads(self, reason: str) -> None:
        if self.wake_voice_released:
            return

        deferred_count = len(self.deferred_wake_audio_payloads)
        self.wake_voice_released = True
        elapsed = (
            time.monotonic() - self.wake_motion_started_at
            if self.wake_motion_started_at
            else 0.0
        )
        logger.info(
            "Live2D wake voice released by {} after {:.3f}s "
            "(target={}s): deferred_audio_payloads={}",
            reason,
            elapsed,
            LIVE2D_WAKE_VOICE_DELAY_SECONDS,
            deferred_count,
        )
        self._flush_deferred_wake_audio_payloads()

    def _flush_deferred_wake_audio_payloads(self) -> None:
        payloads = self.deferred_wake_audio_payloads
        self.deferred_wake_audio_payloads = []
        for payload in payloads:
            turn_id = self._message_turn_id(payload)
            if self.interrupted or not self._should_accept_turn(turn_id):
                logger.debug(
                    "Dropping deferred wake audio because turn is interrupted or stale: {}",
                    turn_id,
                )
                continue
            self._handle_backend_audio(payload)

    def _motion_meta(self, motion: dict[str, Any]) -> dict[str, Any]:
        loop = motion.get("loop")
        duration = motion.get("duration")
        file_path = motion.get("file") or motion.get("path")
        if file_path and (loop is None or duration is None):
            motion_path = self._resolve_motion_path(str(file_path))
            try:
                with motion_path.open("r", encoding="utf-8") as file:
                    data = json.load(file)
                meta = data.get("Meta") if isinstance(data, dict) else {}
                if isinstance(meta, dict):
                    if loop is None:
                        loop = meta.get("Loop")
                    if duration is None:
                        duration = meta.get("Duration")
            except Exception as exc:
                logger.debug("Failed to read Live2D motion meta {}: {}", motion_path, exc)

        return {
            "loop": bool(loop),
            "duration": duration or 0.0,
        }

    def _set_auto_blink_enabled(self, enabled: bool) -> None:
        if not self.model or not hasattr(self.model, "SetAutoBlinkEnable"):
            return
        try:
            self.model.SetAutoBlinkEnable(enabled)
        except Exception as exc:
            logger.debug("Failed to set Live2D auto blink to {}: {}", enabled, exc)

    def _random_expression_sync_motion(self) -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        for file_path in LIVE2D_EXPRESSION_SYNC_MOTION_FILES:
            try:
                if not self._resolve_motion_path(file_path).exists():
                    logger.debug(
                        "Ignoring missing expression sync motion: {}", file_path
                    )
                    continue
            except ValueError as exc:
                logger.debug(
                    "Ignoring invalid expression sync motion path {}: {}",
                    file_path,
                    exc,
                )
                continue
            candidates.append(
                {
                    "group": LIVE2D_EXPRESSION_SYNC_MOTION_GROUP,
                    "file": file_path,
                    "priority": LIVE2D_EXPRESSION_SYNC_MOTION_PRIORITY,
                }
            )

        if not candidates:
            logger.debug(
                "No Live2D expression sync motion files found for model {}",
                self.model_config.name,
            )
            return None
        return random.choice(candidates)

    def _apply_actions(self, actions: dict[str, Any] | None) -> None:
        if not self.model:
            return
        actions = actions or {}

        emotion_expressions, emotion_motions = self._actions_from_emotions(
            self._action_emotions(actions)
        )
        legacy_expressions = actions.get("expressions") or []
        if isinstance(legacy_expressions, str):
            legacy_expressions = [legacy_expressions]
        if not isinstance(legacy_expressions, list):
            legacy_expressions = []

        requested_expressions = list(
            dict.fromkeys(
                [
                    *emotion_expressions,
                    *[
                        str(expression)
                        for expression in legacy_expressions
                        if expression
                    ],
                ]
            )
        )
        requested_motions = self._dedupe_motions(
            [*emotion_motions, *self._legacy_motion_specs(actions)]
        )
        is_initial_turn_actions = self.awaiting_turn_initial_expression
        self.awaiting_turn_initial_expression = False

        exclusive_motion = self._exclusive_auto_return_motion(requested_motions)
        if exclusive_motion is not None:
            if is_initial_turn_actions and not requested_expressions:
                self._clear_active_expressions("new-turn-without-expression")
            if requested_expressions or len(requested_motions) > 1:
                logger.debug(
                    "Live2D exclusive motion requested; dropping expressions={} "
                    "and extra motions={}",
                    requested_expressions,
                    [
                        self._motion_key(motion)
                        for motion in requested_motions
                        if motion is not exclusive_motion
                    ],
                )
            self._apply_motions([exclusive_motion])
            return

        if not requested_expressions:
            if is_initial_turn_actions:
                self._clear_active_expressions("new-turn-without-expression")
            elif self.last_expressions:
                logger.debug(
                    "Keeping Live2D expressions until new expression or new turn: {}",
                    self.last_expressions,
                )
            self._apply_motions(requested_motions)
            return

        active_expression_set = set(self.last_expressions) | self.persistent_expressions
        requested_expression_set = set(requested_expressions)
        expressions_to_remove = [
            expression
            for expression in self.last_expressions
            if expression not in requested_expression_set
        ]
        expressions_to_add = [
            expression
            for expression in requested_expressions
            if expression not in active_expression_set
        ]
        skipped_expressions = [
            expression
            for expression in requested_expressions
            if expression in active_expression_set
        ]

        should_play_expression_sync_motion = bool(requested_expressions) and bool(
            is_initial_turn_actions or expressions_to_remove or expressions_to_add
        )
        if should_play_expression_sync_motion:
            sync_motion = self._random_expression_sync_motion()
            if sync_motion:
                requested_motions = self._dedupe_motions(
                    [*requested_motions, sync_motion]
                )
                logger.debug(
                    "Starting random Live2D expression sync motion before expression: "
                    "expressions={} motion={}",
                    requested_expressions,
                    sync_motion.get("file"),
                )

        # 有表情标签时先启动动作，再切换表情。
        # 这样 _apply_motions() 里可能发生的 ResetAllParameters 不会把新表情擦掉。
        try:
            self._apply_motions(requested_motions)
        except Exception as exc:
            logger.warning(
                "Failed to start Live2D expression sync motion; "
                "expression will still be applied: expressions={} error={}",
                requested_expressions,
                exc,
            )

        if should_play_expression_sync_motion and skipped_expressions:
            for expression in skipped_expressions:
                try:
                    self.model.RemoveExpression(expression)
                except Exception as exc:
                    logger.debug(
                        "Failed to refresh Live2D expression {} before reapply: {}",
                        expression,
                        exc,
                    )
            expressions_to_add = list(
                dict.fromkeys([*expressions_to_add, *skipped_expressions])
            )
            skipped_expressions = []

        for expression in expressions_to_remove:
            try:
                self.model.RemoveExpression(expression)
            except Exception as exc:
                logger.debug("Failed to clear Live2D expression {}: {}", expression, exc)
        if expressions_to_remove:
            logger.debug("Cleared stale Live2D expressions: {}", expressions_to_remove)

        added_expressions: list[str] = []
        for expression in expressions_to_add:
            try:
                self.model.AddExpression(expression)
                added_expressions.append(expression)
                logger.debug("Applied Live2D expression: {}", expression)
            except Exception as exc:
                logger.debug("Failed to apply Live2D expression {}: {}", expression, exc)

        if skipped_expressions:
            logger.debug(
                "Live2D expressions already applied, skipping: {}",
                skipped_expressions,
            )

        successfully_active = set(skipped_expressions) | set(added_expressions)
        self.last_expressions = [
            expression
            for expression in requested_expressions
            if expression in successfully_active
            and expression not in self.persistent_expressions
        ]

    def _apply_motions(self, motions: list[dict[str, Any]]) -> None:
        if not self.model or not motions:
            return

        # 新动作接管: 取消尚未结束的平滑回归渐变, 由新动作驱动参数。
        self._cancel_param_return_fade()

        exclusive_motion = self._exclusive_auto_return_motion(motions)
        if exclusive_motion is not None:
            if len(motions) > 1:
                logger.debug(
                    "Live2D exclusive motion will play alone; dropped motions={}",
                    [
                        self._motion_key(motion)
                        for motion in motions
                        if motion is not exclusive_motion
                    ],
                )
            motions = [exclusive_motion]
            self._clear_active_expressions("before-exclusive-motion")

        motion_keys = [self._motion_key(motion) for motion in motions]
        if motion_keys == self.last_motions:
            logger.debug("Live2D motions already applied, skipping: {}", motion_keys)
            return

        if exclusive_motion is not None:
            if (
                self.speech_motion_active
                or self.return_motion_active
                or self.action_motion_active
                or self.last_motions
            ):
                self._reset_all_parameters("before-exclusive-motion")
            self.speech_motion_active = False
            self.return_motion_active = False
            self.action_motion_active = False
            self.active_motion_requires_return = False
            self.action_motion_idle_delay_seconds = LIVE2D_NORMAL_MOTION_RETRY_SECONDS
            self.last_motions = []
        elif self.speech_motion_active or self.return_motion_active:
            self._reset_all_parameters("before-action-motion")
            self.speech_motion_active = False
            self.return_motion_active = False

        applied_motions: list[str] = []
        applied_motion_specs: list[dict[str, Any]] = []
        for motion in motions:
            if self._start_motion(motion):
                applied_motions.append(self._motion_key(motion))
                applied_motion_specs.append(motion)
        if applied_motions:
            has_expression_sync_motion = any(
                self._motion_is_expression_sync(motion)
                for motion in applied_motion_specs
            )
            self.action_motion_active = True
            self.speech_motion_active = False
            self.return_motion_active = False
            self.active_motion_requires_return = any(
                self._motion_requires_reset_after_finish(motion)
                for motion in applied_motion_specs
            )
            self.action_motion_idle_delay_seconds = (
                LIVE2D_EXPRESSION_SYNC_IDLE_DELAY_SECONDS
                if has_expression_sync_motion
                else LIVE2D_NORMAL_MOTION_RETRY_SECONDS
            )
        else:
            self.action_motion_idle_delay_seconds = LIVE2D_NORMAL_MOTION_RETRY_SECONDS
        self.last_motions = applied_motions

    def _motion_file_key(self, file_path: Any) -> str:
        return str(file_path or "").strip().replace("\\", "/")

    def _motion_requires_return_to_default(self, motion: dict[str, Any]) -> bool:
        return (
            self._motion_file_key(motion.get("file") or motion.get("path"))
            in LIVE2D_AUTO_RETURN_MOTION_FILES
        )

    def _motion_is_expression_sync(self, motion: dict[str, Any]) -> bool:
        key = self._motion_file_key(motion.get("file") or motion.get("path"))
        return (
            str(motion.get("group") or "") == LIVE2D_EXPRESSION_SYNC_MOTION_GROUP
            or key in LIVE2D_EXPRESSION_SYNC_MOTION_FILES
        )

    def _motion_requires_reset_after_finish(self, motion: dict[str, Any]) -> bool:
        # 表情同步动作 (点头/摇头/歪头/摇摆/张望) 播完会停在非中立姿态并冻结,
        # 必须触发 auto-return 清理才能平滑回到常态待机, 否则模型会僵在歪头姿势。
        key = self._motion_file_key(motion.get("file") or motion.get("path"))
        return (
            key in LIVE2D_AUTO_RETURN_MOTION_FILES
            or key in LIVE2D_EXPRESSION_SYNC_MOTION_FILES
        )

    def _exclusive_auto_return_motion(
        self,
        motions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        for motion in motions:
            if self._motion_requires_return_to_default(motion):
                return motion
        return None

    def _tick_motion_return(self) -> None:
        if not self.model or self.sleep_motion_enabled or self.wake_motion_active:
            self.return_motion_active = False
            self.active_motion_requires_return = False
            return

        # 平滑回归渐变进行中: 让 _tick_param_return_fade 独立驱动, 这里不介入。
        if self.param_return_fade_active:
            return

        if self.return_motion_active:
            if self.model.IsMotionFinished():
                self.return_motion_active = False
                self.last_motions = []
                logger.debug("Live2D return-to-default motion finished")
            return

        if not self.active_motion_requires_return:
            return

        if not (self.action_motion_active or self.speech_motion_active):
            self.active_motion_requires_return = False
            return

        if not self.model.IsMotionFinished():
            return

        # 动作播完且需要回归: 用平滑渐变代替瞬间 ResetAllParameters, 消除卡顿。
        # idle 待机动作在渐变结束 + 常规间隔后再开始 (下方 next_speech_motion_attempt_at)。
        self._begin_param_return_fade("auto-cleanup-finished-motion")
        self.next_speech_motion_attempt_at = (
            time.monotonic()
            + LIVE2D_RETURN_FADE_SECONDS
            + LIVE2D_NORMAL_MOTION_RETRY_SECONDS
        )
        self.action_motion_active = False
        self.speech_motion_active = False
        self.return_motion_active = False
        self.active_motion_requires_return = False
        self.action_motion_idle_delay_seconds = LIVE2D_NORMAL_MOTION_RETRY_SECONDS
        self.last_motions = []
        logger.debug("Stopped finished Live2D motion that requires auto cleanup")

    def _tick_normal_motion(self) -> None:
        if not self.model or self.sleep_motion_enabled or self.wake_motion_active:
            self.speech_motion_active = False
            self.next_speech_motion_attempt_at = 0.0
            return

        # 平滑回归渐变期间不启动待机动作, 等渐变结束再进入 idle。
        if self.param_return_fade_active:
            return

        if self.return_motion_active:
            return

        if self.action_motion_active:
            if self.model.IsMotionFinished():
                idle_delay = max(
                    float(self.action_motion_idle_delay_seconds),
                    LIVE2D_NORMAL_MOTION_RETRY_SECONDS,
                )
                self.action_motion_active = False
                self.last_motions = []
                self.active_motion_requires_return = False
                self.next_speech_motion_attempt_at = (
                    time.monotonic() + idle_delay
                )
                self.action_motion_idle_delay_seconds = (
                    LIVE2D_NORMAL_MOTION_RETRY_SECONDS
                )
                return
            else:
                return

        if self.speech_motion_active:
            if not self.model.IsMotionFinished():
                return
            self.speech_motion_active = False
            self.last_motions = []
            self.active_motion_requires_return = False
            self.next_speech_motion_attempt_at = (
                time.monotonic() + LIVE2D_NORMAL_MOTION_RETRY_SECONDS
            )
            return

        if self.model.IsMotionFinished():
            self._start_random_normal_motion()

    def _start_random_normal_motion(self) -> bool:
        if not self.model or self.sleep_motion_enabled or self.wake_motion_active:
            return False
        if self.param_return_fade_active:
            return False
        now = time.monotonic()
        if now < self.next_speech_motion_attempt_at:
            return False
        if self.return_motion_active:
            if self.model.IsMotionFinished():
                self.return_motion_active = False
                self.last_motions = []
                self.next_speech_motion_attempt_at = (
                    time.monotonic() + LIVE2D_NORMAL_MOTION_RETRY_SECONDS
                )
                return False
            else:
                return False
        if self.action_motion_active:
            if self.model.IsMotionFinished():
                idle_delay = max(
                    float(self.action_motion_idle_delay_seconds),
                    LIVE2D_NORMAL_MOTION_RETRY_SECONDS,
                )
                self.action_motion_active = False
                self.last_motions = []
                self.next_speech_motion_attempt_at = (
                    time.monotonic() + idle_delay
                )
                self.action_motion_idle_delay_seconds = (
                    LIVE2D_NORMAL_MOTION_RETRY_SECONDS
                )
                return False
            else:
                return False
        if self.speech_motion_active:
            if not self.model.IsMotionFinished():
                return False
            self.speech_motion_active = False
            self.last_motions = []
            self.next_speech_motion_attempt_at = (
                time.monotonic() + LIVE2D_NORMAL_MOTION_RETRY_SECONDS
            )
            return False
        if not self.model.IsMotionFinished():
            return False

        candidates: list[dict[str, Any]] = []
        for index, file_path in enumerate(LIVE2D_NORMAL_MOTION_FILES):
            try:
                if not self._resolve_motion_path(file_path).exists():
                    continue
            except ValueError as exc:
                logger.debug("Ignoring invalid normal motion path {}: {}", file_path, exc)
                continue

            group = f"{LIVE2D_NORMAL_MOTION_GROUP}_{index}"
            candidates.append(
                {
                    "group": group,
                    "file": file_path,
                    "priority": LIVE2D_NORMAL_MOTION_PRIORITY,
                }
            )

        if not candidates:
            logger.debug(
                "No Live2D normal motion files found for model {}",
                self.model_config.name,
            )
            return False

        motion = random.choice(candidates)
        self.next_speech_motion_attempt_at = (
            now + LIVE2D_NORMAL_MOTION_RETRY_SECONDS
        )
        if self._start_motion(motion):
            self.speech_motion_active = True
            self.return_motion_active = False
            self.active_motion_requires_return = self._motion_requires_return_to_default(
                motion
            )
            logger.debug("Started random Live2D normal motion: {}", motion["file"])
            return True
        self.speech_motion_active = False
        self.active_motion_requires_return = False
        return False

    def _start_motion(self, motion: dict[str, Any]) -> bool:
        if not self.model:
            return False

        group = str(motion.get("group") or "").strip()
        if not group:
            return False

        priority = self._safe_int(motion.get("priority"), 3)
        index = motion.get("index", motion.get("no"))
        file_path = motion.get("file") or motion.get("path")

        try:
            if file_path:
                loaded_index = self._ensure_extra_motion_loaded(group, str(file_path))
                if index is None:
                    index = loaded_index

            if index is None:
                result = self.model.StartRandomMotion(group, priority)
                if result is False:
                    logger.debug("Live2D random motion rejected: {}", motion)
                    return False
                logger.debug(
                    "Started Live2D random motion: group={} priority={}",
                    group,
                    priority,
                )
            else:
                motion_index = self._safe_int(index, 0)
                result = self.model.StartMotion(group, motion_index, priority)
                if result is False:
                    logger.debug("Live2D motion rejected: {}", motion)
                    return False
                logger.debug(
                    "Started Live2D motion: group={} index={} priority={}",
                    group,
                    motion_index,
                    priority,
                )
            return True
        except Exception as exc:
            logger.warning("Failed to start Live2D motion {}: {}", motion, exc)
            return False

    def _ensure_extra_motion_loaded(self, group: str, file_path: str) -> int:
        if not self.model:
            return 0

        motion_path = self._resolve_motion_path(file_path)
        cache_key = (group, str(motion_path))
        if cache_key in self.loaded_extra_motions:
            return self.loaded_extra_motions[cache_key]

        # Live2D 无法解析科学计数法/NaN/Infinity: 这类文件加载失败时 LoadExtraMotion
        # 会静默返回未变化的分组大小, 若不拦截就会缓存错误 index 并播错动作。
        self._verify_cubism_motion_json(motion_path)

        # LoadExtraMotion 返回的是新加入动作的 0 基下标 (即本次加载前该分组的大小),
        # 不是累计计数, 因此直接用作 index, 不再减一。
        loaded_index = self.model.LoadExtraMotion(group, str(motion_path))
        if loaded_index is None or loaded_index < 0:
            raise RuntimeError(f"LoadExtraMotion returned {loaded_index}")

        motion_index = int(loaded_index)
        self.loaded_extra_motions[cache_key] = motion_index
        logger.info(
            "Loaded Live2D extra motion: group={} index={} file={}",
            group,
            motion_index,
            motion_path,
        )
        return motion_index

    def _verify_cubism_motion_json(self, motion_path: Path) -> None:
        """Motion 文件含 Live2D 原生解析器无法处理的 token 时抛错。

        CubismJson 不支持科学计数法 (如 1e-7) 与 NaN/Infinity。Python 的 json
        能正常读, 但 LoadExtraMotion 会解析失败并静默返回错误 index, 因此加载前
        先扫描拦截。
        """
        try:
            text = motion_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                f"Cannot read Live2D motion file {motion_path}: {exc}"
            ) from exc
        if LIVE2D_CUBISM_UNSAFE_NUMBER_RE.search(text):
            raise RuntimeError(
                "Live2D motion file uses scientific notation / NaN / Infinity "
                f"that the native parser cannot load: {motion_path}"
            )

    def _resolve_motion_path(self, file_path: str) -> Path:
        motion_path = (self.model_config.model_path.parent / file_path).resolve()
        try:
            motion_path.relative_to(self.model_config.model_path.parent.resolve())
        except ValueError as exc:
            raise ValueError(f"Live2D motion path escapes model root: {file_path}") from exc
        return motion_path

    def _safe_int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _reset_expressions_to_default(self, reason: str) -> None:
        if not self.model or not hasattr(self.model, "ResetExpressions"):
            return
        try:
            self.model.ResetExpressions()
            logger.debug("Reset Live2D expressions to default: reason={}", reason)
        except Exception as exc:
            logger.debug(
                "Failed to reset Live2D expressions to default: reason={} error={}",
                reason,
                exc,
            )
            return

        # ResetExpressions 会清掉所有 expression；水印这类常驻 expression 需要补回。
        for expression in list(self.persistent_expressions):
            try:
                self.model.AddExpression(expression)
            except Exception as exc:
                logger.debug(
                    "Failed to reapply persistent Live2D expression {} after reset: {}",
                    expression,
                    exc,
                )

    def _clear_active_expressions(self, reason: str) -> None:
        if not self.last_expressions:
            return

        expressions = list(self.last_expressions)
        self.last_expressions = []
        if not self.model:
            return

        for expression in expressions:
            try:
                self.model.RemoveExpression(expression)
            except Exception as exc:
                logger.debug("Failed to clear Live2D expression {}: {}", expression, exc)
        logger.debug(
            "Cleared Live2D expressions: reason={} expressions={}",
            reason,
            expressions,
        )
        self._reset_expressions_to_default(reason)

    def _clear_expressions(self) -> None:
        self.last_motions = []
        self._clear_active_expressions("clear-expressions")

    def clear_expressions(self, reason: str) -> None:
        self._clear_expressions()
        logger.debug("Live2D expressions reset by {}", reason)

    def _handle_backend_control(self, control_text: str, turn_id: str | None = None) -> None:
        logger.info("Backend control: {}", control_text)
        if control_text == "conversation-chain-start":
            if not self._start_turn(turn_id):
                return
            self.interrupted = False
            self.pending_backend_done = False
            self.pending_backend_done_turn_id = None
            self.current_response_parts.clear()
            return

        if control_text == "interrupt":
            if turn_id:
                if self._should_accept_turn(turn_id):
                    self._interrupt_playback(reason=control_text, turn_id=turn_id)
                return
            if self.audio_player.is_idle() and not self.active_turn_id:
                logger.debug("Ignoring backend interrupt without active local turn")
                return
            self._interrupt_playback(reason=control_text, turn_id=turn_id)
            return

        if control_text == "conversation-chain-end":
            if self._should_accept_turn(turn_id):
                self.interrupted = False
                self._finish_turn(turn_id)

    def _interrupt_playback(self, reason: str, turn_id: str | None = None) -> str:
        heard_response = "".join(self.current_response_parts)
        self._block_turn(turn_id)
        self.interrupted = True
        self.pending_backend_done = False
        self.pending_backend_done_turn_id = None
        self._clear_wake_motion_state(clear_deferred=True)
        self._set_auto_blink_enabled(True)
        self.audio_player.stop()
        self._cancel_param_return_fade()
        self.speech_motion_active = False
        self.action_motion_active = False
        self.return_motion_active = False
        self.active_motion_requires_return = False
        self.action_motion_idle_delay_seconds = LIVE2D_NORMAL_MOTION_RETRY_SECONDS
        self.next_speech_motion_attempt_at = 0.0
        self._clear_expressions()
        logger.info(
            "Live2D playback interrupted by {}. Heard response length: {}",
            reason,
            len(heard_response),
        )
        self.current_response_parts.clear()
        return heard_response

    def interrupt_from_console(self) -> str:
        return self._interrupt_playback(
            reason="console voice cutoff",
            turn_id=self.active_turn_id,
        )

    def is_playing_audio(self) -> bool:
        return self.audio_player.is_playing()

    def set_audio_start_blocked(self, reason: str, blocked: bool) -> None:
        self.audio_player.set_start_blocked(reason, blocked)

    def pause_audio_for_microphone(self) -> bool:
        return self.audio_player.pause()

    def resume_audio_after_microphone_cancelled(self) -> bool:
        return self.audio_player.resume()

    def set_output_muted(self, muted: bool) -> None:
        self.audio_player.set_muted(muted)

    def stop_voice_processing(self, reason: str) -> None:
        self.interrupted = False
        self.pending_backend_done = False
        self.pending_backend_done_turn_id = None
        self._clear_wake_motion_state(clear_deferred=True)
        self._set_auto_blink_enabled(True)
        self.audio_player.stop()
        self._cancel_param_return_fade()
        self.speech_motion_active = False
        self.action_motion_active = False
        self.return_motion_active = False
        self.active_motion_requires_return = False
        self.action_motion_idle_delay_seconds = LIVE2D_NORMAL_MOTION_RETRY_SECONDS
        self.next_speech_motion_attempt_at = 0.0
        self._clear_expressions()
        self.current_response_parts.clear()
        logger.info("Live2D voice processing stopped by {}", reason)

    def _handle_backend_audio(self, data: dict[str, Any]) -> None:
        display_text = data.get("display_text")
        actions = data.get("actions")
        audio_base64 = data.get("audio")
        volumes = data.get("volumes") or []
        slice_length = int(data.get("slice_length") or 20)

        if not audio_base64:
            self._apply_actions(actions)
            if display_text and display_text.get("text"):
                logger.info("{}: {}", display_text.get("name", "AI"), display_text["text"])
            return

        audio_bytes = base64.b64decode(audio_base64)
        fd, temp_path = tempfile.mkstemp(prefix="live2d_audio_", suffix=".wav")
        with os.fdopen(fd, "wb") as temp_file:
            temp_file.write(audio_bytes)

        self.play_audio_file(
            audio_path=Path(temp_path),
            display_text=display_text,
            actions=actions,
            subtitle_payload={
                "type": "audio",
                "display_text": display_text,
                "actions": actions,
                "volumes": volumes,
                "slice_length": slice_length,
                "turn_id": self._message_turn_id(data),
            },
            volumes=volumes,
            slice_length_ms=slice_length,
            delete_after_play=True,
            notify_backend_done=True,
        )

    def _handle_audio_job_start(self, job: AudioJob) -> None:
        self._apply_actions(job.actions)
        if job.subtitle_payload:
            self.audio_started.emit(dict(job.subtitle_payload))
        if job.display_text and job.display_text.get("text"):
            text = str(job.display_text["text"])
            self.current_response_parts.append(text)
            logger.info("{}: {}", job.display_text.get("name", "AI"), text)

    def _handle_audio_job_done(self, job: AudioJob) -> None:
        if job.notify_backend_done and self.pending_backend_done and self.audio_player.is_idle():
            self._send_frontend_playback_complete(self.pending_backend_done_turn_id)

    def _send_frontend_playback_complete(
        self,
        turn_id: str | None = None,
        *,
        skipped: bool = False,
        reason: str | None = None,
        force: bool = False,
    ) -> None:
        if not force and not self._should_accept_turn(turn_id):
            logger.debug("Skipping playback completion for stale turn: {}", turn_id)
            return
        payload: dict[str, Any] = {"turn_id": turn_id}
        if skipped:
            payload["skipped"] = True
        if reason:
            payload["reason"] = reason

        if skipped:
            self.playback_complete.emit(payload)
            logger.debug(
                "Live2D playback completion acknowledged without playback: turn_id={} reason={}",
                turn_id,
                reason,
            )
            return

        keep_return_flow = self.return_motion_active or (
            self.active_motion_requires_return
            and (self.action_motion_active or self.speech_motion_active)
        )
        if not keep_return_flow:
            self.speech_motion_active = False
            self.action_motion_active = False
            self.return_motion_active = False
            self.active_motion_requires_return = False
            self.action_motion_idle_delay_seconds = LIVE2D_NORMAL_MOTION_RETRY_SECONDS
        self.next_speech_motion_attempt_at = 0.0
        self.pending_backend_done = False
        self.pending_backend_done_turn_id = None
        self.playback_complete.emit(payload)
        logger.debug("Live2D playback complete")

def truncate_data(data: Any, max_len: int = 20) -> Any:
    """
    递归处理字典/列表，截断超长字符串
    """
    if isinstance(data, dict):
        return {k: truncate_data(v, max_len) for k, v in data.items()}
    elif isinstance(data, list):
        return [truncate_data(item, max_len) for item in data[:max_len]]
    elif isinstance(data, str):
        if len(data) > max_len:
            return data[:max_len] + f"...(len:{len(data)})"
        return data
    return data


class VolumeSignalIndicator(QWidget):
    """Small local-only microphone level indicator."""

    def __init__(self, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._level = 0.0
        self._active = False
        self.setFixedSize(34, 28)
        self.setToolTip(tooltip)
        self.decay_timer = QTimer(self)
        self.decay_timer.setInterval(50)
        self.decay_timer.timeout.connect(self._decay_level)
        self.decay_timer.start()

    def set_ui_scale(self, scale: float) -> None:
        self.setFixedSize(scaled_int(34, scale), scaled_int(28, scale))

    def set_active(self, active: bool) -> None:
        if self._active == active:
            return
        self._active = active
        if not active:
            self._level = 0.0
        self.update()

    def set_level(self, level: float) -> None:
        if not self._active:
            return
        level = max(0.0, min(float(level), 1.0))
        self._level = max(level, self._level * 0.55)
        self.update()

    def _decay_level(self) -> None:
        if self._level <= 0.001:
            self._level = 0.0
            return
        self._level *= 0.86
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#4fbd77") if self._active else QColor("#c8ced7"))

        bar_width = 4
        gap = 3
        base_x = 5
        bottom = self.height() - 5
        max_height = self.height() - 8
        min_height = 4
        level = self._level if self._active else 0.0
        for index in range(4):
            strength = min(1.0, max(0.0, level * 1.3 - index * 0.18))
            height = min_height + int(max_height * (0.2 + 0.8 * strength))
            x = base_x + index * (bar_width + gap)
            y = bottom - height
            painter.drawRoundedRect(x, y, bar_width, height, 2, 2)


class CollapsibleSection(QWidget):
    """带三角按钮的控制台折叠分组。"""

    def __init__(
        self,
        title: str,
        content_layout: QVBoxLayout | QHBoxLayout,
        *,
        expanded: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._expanded = bool(expanded)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("consoleSectionTitle")
        self.title_label.setStyleSheet(COLLAPSIBLE_SECTION_TITLE_STYLE)

        self.toggle_button = QPushButton()
        self.toggle_button.setObjectName("sectionToggleButton")
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self.toggle_button.setFocusPolicy(Qt.NoFocus)
        self.toggle_button.setToolTip("展开/收起")
        self.toggle_button.clicked.connect(self.toggle)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(2, 0, 0, 0)
        header_layout.setSpacing(7)
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.toggle_button)
        header_layout.addStretch(1)

        self.card = QFrame()
        self.card.setObjectName("consoleSectionCard")
        content_layout.setContentsMargins(18, 14, 18, 14)
        content_layout.setSpacing(12)
        self.card.setLayout(content_layout)

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(6)
        root_layout.addLayout(header_layout)
        root_layout.addWidget(self.card)
        self.setLayout(root_layout)

        self.set_expanded(self._expanded)

    def toggle(self) -> None:
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = bool(expanded)
        self.card.setVisible(self._expanded)
        self.toggle_button.setText("▼" if self._expanded else "▶")

    def apply_ui_scale(self, scale: float) -> None:
        self.title_label.setStyleSheet(scaled_collapsible_section_title_style(scale))
        font = QFont("Segoe UI Symbol")
        font.setBold(True)
        font.setPixelSize(scaled_int(15, scale))
        self.toggle_button.setFont(font)
        size = scaled_int(26, scale)
        self.toggle_button.setFixedSize(size, size)
        content_layout = self.card.layout()
        if content_layout:
            content_layout.setContentsMargins(
                scaled_int(18, scale),
                scaled_int(14, scale),
                scaled_int(18, scale),
                scaled_int(14, scale),
            )
            content_layout.setSpacing(scaled_int(12, scale))


class VisionImageDropLabel(QLabel):
    file_dropped = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__("暂无图片\n点击选择或拖入图片")
        self.setObjectName("visionImagePreview")
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(VISION_IMAGE_PREVIEW_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def dragEnterEvent(self, event: Any) -> None:
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if urls and Path(urls[0].toLocalFile()).suffix.lower() in VISION_IMAGE_MIME_TYPES:
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: Any) -> None:
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if not urls:
            event.ignore()
            return
        self.file_dropped.emit(urls[0].toLocalFile())
        event.acceptProposedAction()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton:
            self.file_dropped.emit("")
        super().mousePressEvent(event)


class PaintWindow(QWidget):
    closed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Agent 画板")
        self.resize(PAINT_WINDOW_DEFAULT_WIDTH, PAINT_WINDOW_DEFAULT_HEIGHT)
        self.setMinimumSize(360, 280)
        self._original_pixmap: QPixmap | None = None
        self._loading_step = 0

        self.image_label = QLabel("等待画图")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet(
            "QLabel { background: #ffffff; color: #6e6e73; "
            "border: 1px solid #d2d2d7; border-radius: 12px; }"
        )

        self.status_label = QLabel("开启画图后，Live Streaming Agent 会把画好的图显示在这里")
        self.status_label.setTextFormat(Qt.PlainText)
        self.status_label.setStyleSheet("color: #6e6e73; font-size: 13px;")

        layout = QVBoxLayout()
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.loading_timer = QTimer(self)
        self.loading_timer.setInterval(PAINT_LOADING_INTERVAL_MS)
        self.loading_timer.timeout.connect(self._tick_loading)

    def set_loading(self, prompt: str) -> None:
        self._original_pixmap = None
        self._loading_step = 0
        self.image_label.clear()
        self._tick_loading()
        self.status_label.setStyleSheet("color: #6e6e73; font-size: 13px;")
        self.status_label.setText(f"正在画：{prompt}")
        self.loading_timer.start()
        self.show()
        self.raise_()

    def set_error(self, message: str) -> None:
        self.loading_timer.stop()
        self._original_pixmap = None
        self.image_label.setText("画图失败")
        self.status_label.setText(message or "画图失败")
        self.status_label.setStyleSheet("color: #b42318; font-size: 13px;")
        self.show()
        self.raise_()

    def set_image(self, image_base64: str, prompt: str = "") -> None:
        self.loading_timer.stop()
        image_bytes = base64.b64decode(str(image_base64 or ""), validate=True)
        pixmap = QPixmap()
        if not pixmap.loadFromData(image_bytes):
            raise ValueError("图片无法解码")
        self._original_pixmap = pixmap
        self.status_label.setStyleSheet("color: #6e6e73; font-size: 13px;")
        self.status_label.setText(f"完成：{prompt}" if prompt else "画图完成")
        self._apply_scaled_pixmap()
        self.show()
        self.raise_()

    def _tick_loading(self) -> None:
        dots = "." * (self._loading_step % 4)
        self._loading_step += 1
        self.image_label.setText(f"画图中{dots}")

    def _apply_scaled_pixmap(self) -> None:
        if not self._original_pixmap:
            return
        target_size = self.image_label.size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            return
        scaled = self._original_pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._apply_scaled_pixmap()

    def closeEvent(self, event: Any) -> None:
        self.loading_timer.stop()
        self.closed.emit()
        super().closeEvent(event)


class DirectorMetricRow(QFrame):
    def __init__(
        self,
        key: str,
        title: str,
        on_changed: Callable[..., None],
        on_drag_release: Callable[[str, int], None],
    ) -> None:
        super().__init__()
        self.key = key
        self.title = title
        self._on_drag_release = on_drag_release
        self._dragging = False
        self._drag_start_global_y = 0
        self._drag_origin_y = 0
        self.drag_handle: QLabel | None = None
        self.value_input: QLineEdit | None = None
        self.setObjectName("directorMetricRow")
        self.setFocusPolicy(Qt.ClickFocus)
        self.setFixedHeight(44)
        self.setStyleSheet(DIRECTOR_METRIC_ROW_STYLE)
        self.installEventFilter(self)

        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(10, 5, 10, 5)
        row_layout.setSpacing(8)
        self.setLayout(row_layout)

        self.drag_handle = QLabel("☰")
        self.drag_handle.setObjectName("directorMetricDragHandle")
        self.drag_handle.setFixedWidth(22)
        self.drag_handle.setAlignment(Qt.AlignCenter)
        self.drag_handle.setCursor(Qt.OpenHandCursor)
        self.drag_handle.installEventFilter(self)

        self.checkbox = QCheckBox()
        self.label = QLabel(title)
        self.label.setMinimumWidth(84)
        self.label.setFocusPolicy(Qt.ClickFocus)
        self.label.installEventFilter(self)
        self.value_input = QLineEdit()
        self.value_input.setText("0")
        self.value_input.setFixedSize(DIRECTOR_METRIC_INPUT_WIDTH, 32)
        self.value_input.setValidator(QIntValidator(0, 999999999, self.value_input))
        self.checkbox.installEventFilter(self)

        row_layout.addWidget(self.drag_handle)
        row_layout.addWidget(self.checkbox)
        row_layout.addWidget(self.label)
        row_layout.addWidget(self.value_input)

        self.checkbox.stateChanged.connect(on_changed)
        self.value_input.editingFinished.connect(self._finish_value_edit)
        self.value_input.editingFinished.connect(on_changed)

    def apply_ui_scale(self, scale: float) -> None:
        row_height = scaled_int(44, scale)
        self.setMinimumHeight(row_height)
        self.setMaximumHeight(row_height)
        self.setStyleSheet(scaled_director_metric_row_style(scale))
        row_layout = self.layout()
        if row_layout:
            row_layout.setContentsMargins(
                scaled_int(10, scale),
                scaled_int(5, scale),
                scaled_int(10, scale),
                scaled_int(5, scale),
            )
            row_layout.setSpacing(scaled_int(8, scale))
        if self.drag_handle:
            self.drag_handle.setFixedWidth(scaled_int(22, scale))
        if self.label:
            self.label.setMinimumWidth(scaled_int(84, scale))
        if self.value_input:
            self.value_input.setMinimumSize(
                scaled_int(DIRECTOR_METRIC_INPUT_WIDTH, scale),
                scaled_int(32, scale),
            )
            self.value_input.setMaximumSize(
                scaled_int(DIRECTOR_METRIC_INPUT_WIDTH, scale),
                scaled_int(32, scale),
            )

    def _finish_value_edit(self) -> None:
        if not self.value_input:
            return
        if not self.value_input.text().strip():
            self.value_input.setText("0")

    def eventFilter(self, source: Any, event: Any) -> bool:
        value_input = self.value_input
        drag_handle = self.drag_handle

        if (
            value_input is not None
            and event.type() == QEvent.MouseButtonPress
            and source is not value_input
        ):
            value_input.clearFocus()

        if drag_handle is not None and source is drag_handle:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._dragging = True
                self._drag_start_global_y = event.globalY()
                self._drag_origin_y = self.y()
                drag_handle.setCursor(Qt.ClosedHandCursor)
                drag_handle.grabMouse()
                self.raise_()
                return True
            if event.type() == QEvent.MouseMove and self._dragging:
                delta_y = event.globalY() - self._drag_start_global_y
                self.move(self.x(), self._drag_origin_y + delta_y)
                return True
            if event.type() == QEvent.MouseButtonRelease and self._dragging:
                self._dragging = False
                drag_handle.setCursor(Qt.OpenHandCursor)
                drag_handle.releaseMouse()
                self._on_drag_release(self.key, event.globalY())
                return True
        return super().eventFilter(source, event)


class ProjectConfigDialog(QDialog):
    save_requested = pyqtSignal(dict)
    test_requested = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("项目配置")
        self.setMinimumSize(590, 520)
        self.setModal(False)
        self._catalog: dict[str, dict[str, Any]] = {}
        self._applying_state = False

        title = QLabel("基础模型配置")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        description = QLabel(
            "用于普通对话和记忆总结；游戏识图与图片识别模型保持不变。"
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #6e6e73;")
        self.active_model_status = QLabel("当前使用：等待后端同步…")
        self.active_model_status.setWordWrap(True)
        self.active_model_status.setStyleSheet(
            "color: #5e8fd8; font-weight: 600;"
        )

        self.provider_combo = QComboBox()
        self.model_combo = QComboBox()
        self.model_id_input = QLineEdit()
        self.model_id_input.setReadOnly(True)
        self.temperature_input = QDoubleSpinBox()
        self.temperature_input.setRange(0.0, 2.0)
        self.temperature_input.setDecimals(2)
        self.temperature_input.setSingleStep(0.1)
        self.temperature_input.setValue(0.8)
        self.web_search_checkbox = QCheckBox("联网搜索")
        self.web_search_checkbox.setCursor(Qt.PointingHandCursor)
        self.web_search_forced_checkbox = QCheckBox("强制搜索")
        self.web_search_forced_checkbox.setCursor(Qt.PointingHandCursor)
        self.web_search_max_tool_calls_input = QSpinBox()
        self.web_search_max_tool_calls_input.setRange(1, 10)
        self.web_search_max_tool_calls_input.setValue(1)
        self.web_search_result_limit_input = QSpinBox()
        self.web_search_result_limit_input.setRange(1, 20)
        self.web_search_result_limit_input.setValue(3)

        self.web_search_parameters = QWidget()
        web_search_parameters_layout = QGridLayout()
        web_search_parameters_layout.setContentsMargins(20, 0, 0, 0)
        web_search_parameters_layout.setHorizontalSpacing(14)
        web_search_parameters_layout.setVerticalSpacing(8)
        web_search_parameters_layout.addWidget(QLabel("最大搜索次数"), 0, 0)
        web_search_parameters_layout.addWidget(
            self.web_search_max_tool_calls_input,
            0,
            1,
        )
        web_search_parameters_layout.addWidget(QLabel("搜索网页数"), 1, 0)
        web_search_parameters_layout.addWidget(
            self.web_search_result_limit_input,
            1,
            1,
        )
        web_search_parameters_layout.setColumnStretch(2, 1)
        self.web_search_parameters.setLayout(web_search_parameters_layout)

        self.web_search_container = QWidget()
        web_search_layout = QVBoxLayout()
        web_search_layout.setContentsMargins(0, 0, 0, 0)
        web_search_layout.setSpacing(8)
        web_search_layout.addWidget(self.web_search_checkbox)
        web_search_layout.addWidget(self.web_search_forced_checkbox)
        web_search_layout.addWidget(self.web_search_parameters)
        self.web_search_container.setLayout(web_search_layout)

        form = QGridLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(12)
        fields = (
            ("Provider", self.provider_combo),
            ("Model", self.model_combo),
            ("模型 ID", self.model_id_input),
            ("温度 (0-2)", self.temperature_input),
        )
        for row, (label_text, widget) in enumerate(fields):
            label = QLabel(label_text)
            label.setMinimumWidth(112)
            form.addWidget(label, row, 0)
            form.addWidget(widget, row, 1)
        form.setColumnStretch(1, 1)

        self.credential_status = QLabel("等待后端配置…")
        self.credential_status.setWordWrap(True)
        self.credential_status.setStyleSheet("color: #6e6e73;")
        self.test_status = QLabel("等待测试")
        self.test_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.test_status.setStyleSheet("color: #6e6e73;")
        self.test_button = QPushButton("测试连接")
        self.save_button = QPushButton("保存并应用")
        for button in (self.test_button, self.save_button):
            button.setMinimumSize(118, 40)
            button.setCursor(Qt.PointingHandCursor)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        actions.addWidget(self.test_button)
        actions.addWidget(self.save_button)
        actions.addStretch(1)
        actions.addWidget(self.test_status)

        layout = QVBoxLayout()
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(self.active_model_status)
        layout.addLayout(form)
        layout.addWidget(self.web_search_container)
        layout.addWidget(self.credential_status)
        layout.addStretch(1)
        layout.addLayout(actions)
        self.setLayout(layout)

        self.provider_combo.currentIndexChanged.connect(
            self._handle_provider_changed
        )
        self.model_combo.currentIndexChanged.connect(self._sync_model_id)
        self.web_search_checkbox.stateChanged.connect(
            self._update_web_search_controls
        )
        self.test_button.clicked.connect(self._request_test)
        self.save_button.clicked.connect(self._request_save)

    def apply_state(self, data: dict[str, Any]) -> None:
        catalog = data.get("catalog")
        if not isinstance(catalog, list):
            return
        self._applying_state = True
        try:
            self._catalog = {
                str(item.get("id")): item
                for item in catalog
                if isinstance(item, dict) and item.get("id")
            }
            selected_provider = str(data.get("provider") or "")
            selected_model = str(data.get("model") or "")
            self.provider_combo.clear()
            for item in catalog:
                if not isinstance(item, dict):
                    continue
                self.provider_combo.addItem(
                    str(item.get("name") or item.get("id") or ""),
                    str(item.get("id") or ""),
                )
            provider_index = self.provider_combo.findData(selected_provider)
            self.provider_combo.setCurrentIndex(max(provider_index, 0))
            self._populate_models(selected_model)
            self._update_active_model_status(selected_provider, selected_model)
            self.temperature_input.setValue(
                min(2.0, max(0.0, float(data.get("temperature", 0.8))))
            )
            supported = self._selected_provider_supports_search()
            self.web_search_checkbox.setEnabled(supported)
            self.web_search_checkbox.setChecked(
                supported and bool(data.get("web_search_enabled"))
            )
            self.web_search_forced_checkbox.setChecked(
                supported
                and bool(data.get("web_search_enabled"))
                and bool(data.get("web_search_forced"))
            )
            self.web_search_max_tool_calls_input.setValue(
                min(
                    10,
                    max(1, int(data.get("web_search_max_tool_calls", 1))),
                )
            )
            self.web_search_result_limit_input.setValue(
                min(
                    20,
                    max(1, int(data.get("web_search_result_limit", 3))),
                )
            )
            self._update_web_search_controls()
            self._update_credential_status()
            self.test_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.test_status.setText("配置已同步")
            self.test_status.setStyleSheet("color: #6e6e73;")
        finally:
            self._applying_state = False

    def show_test_result(self, data: dict[str, Any]) -> None:
        self.test_button.setEnabled(True)
        if data.get("ok"):
            latency = data.get("latency_ms")
            latency_text = f"，首字 {float(latency):.0f}ms" if latency is not None else ""
            self.test_status.setText(f"连接成功{latency_text}")
            self.test_status.setStyleSheet("color: #34c759;")
            return
        self.test_status.setText(f"连接失败：{data.get('message') or '未知错误'}")
        self.test_status.setStyleSheet("color: #ff3b30;")

    def show_config_error(self, message: str) -> None:
        self.test_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.test_status.setText(f"配置失败：{message}")
        self.test_status.setStyleSheet("color: #ff3b30;")

    def _handle_provider_changed(self, _index: int) -> None:
        if self._applying_state:
            return
        provider = str(self.provider_combo.currentData() or "")
        saved_config = self._catalog.get(provider, {}).get("saved_config") or {}
        self._populate_models(str(saved_config.get("model") or ""))
        self.temperature_input.setValue(
            min(2.0, max(0.0, float(saved_config.get("temperature", 0.8))))
        )
        supported = self._selected_provider_supports_search()
        enabled = supported and bool(saved_config.get("web_search_enabled"))
        self.web_search_checkbox.setChecked(enabled)
        self.web_search_forced_checkbox.setChecked(
            enabled and bool(saved_config.get("web_search_forced"))
        )
        self.web_search_max_tool_calls_input.setValue(
            min(
                10,
                max(1, int(saved_config.get("web_search_max_tool_calls", 1))),
            )
        )
        self.web_search_result_limit_input.setValue(
            min(
                20,
                max(1, int(saved_config.get("web_search_result_limit", 3))),
            )
        )
        self._update_web_search_controls()
        self._update_credential_status()

    def _update_web_search_controls(self, _state: int = 0) -> None:
        provider = str(self.provider_combo.currentData() or "")
        supported = provider in {"doubao", "qwen"}
        enabled = supported and self.web_search_checkbox.isChecked()
        self.web_search_container.setVisible(supported)
        self.web_search_checkbox.setEnabled(supported)
        self.web_search_forced_checkbox.setVisible(enabled)
        self.web_search_forced_checkbox.setEnabled(enabled)
        self.web_search_parameters.setVisible(enabled and provider == "doubao")
        if not enabled:
            self.web_search_forced_checkbox.setChecked(False)

    def _populate_models(self, preferred_model: str = "") -> None:
        provider = str(self.provider_combo.currentData() or "")
        provider_info = self._catalog.get(provider, {})
        models = provider_info.get("models") or []
        self.model_combo.clear()
        for model in models:
            if not isinstance(model, dict):
                continue
            self.model_combo.addItem(
                str(model.get("label") or model.get("id") or ""),
                str(model.get("id") or ""),
            )
        target = preferred_model or str(provider_info.get("default_model") or "")
        model_index = self.model_combo.findData(target)
        self.model_combo.setCurrentIndex(max(model_index, 0))
        self._sync_model_id()

    def _sync_model_id(self, _index: int = -1) -> None:
        self.model_id_input.setText(str(self.model_combo.currentData() or ""))

    def _selected_provider_supports_search(self) -> bool:
        provider = str(self.provider_combo.currentData() or "")
        return bool(self._catalog.get(provider, {}).get("web_search_supported"))

    def _update_credential_status(self) -> None:
        provider = str(self.provider_combo.currentData() or "")
        info = self._catalog.get(provider, {})
        provider_name = str(info.get("name") or provider)
        if info.get("has_api_key"):
            self.credential_status.setText(
                f"已载入 {provider_name}，服务端凭据已配置"
            )
            self.credential_status.setStyleSheet("color: #5e8fd8;")
        else:
            self.credential_status.setText(
                f"{provider_name} 的 API Key 尚未在服务端配置"
            )
            self.credential_status.setStyleSheet("color: #ff3b30;")

    def _update_active_model_status(self, provider: str, model: str) -> None:
        provider_info = self._catalog.get(provider, {})
        provider_name = str(provider_info.get("name") or provider or "未知供应商")
        model_label = model
        for model_info in provider_info.get("models") or []:
            if not isinstance(model_info, dict):
                continue
            if str(model_info.get("id") or "") == model:
                model_label = str(model_info.get("label") or model)
                break
        if model_label and model_label != model:
            model_text = f"{model_label}（{model}）"
        else:
            model_text = model or "未知模型"
        self.active_model_status.setText(
            f"当前使用：{provider_name} / {model_text}"
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "provider": str(self.provider_combo.currentData() or ""),
            "model": str(self.model_combo.currentData() or ""),
            "temperature": float(self.temperature_input.value()),
            "web_search_enabled": bool(self.web_search_checkbox.isChecked()),
            "web_search_forced": bool(
                self.web_search_forced_checkbox.isChecked()
            ),
            "web_search_max_tool_calls": int(
                self.web_search_max_tool_calls_input.value()
            ),
            "web_search_result_limit": int(
                self.web_search_result_limit_input.value()
            ),
        }

    def _request_test(self) -> None:
        self.test_button.setEnabled(False)
        self.test_status.setText("正在测试…")
        self.test_status.setStyleSheet("color: #ff9f0a;")
        self.test_requested.emit(self._payload())

    def _request_save(self) -> None:
        self.save_button.setEnabled(False)
        self.test_status.setText("正在应用配置…")
        self.test_status.setStyleSheet("color: #ff9f0a;")
        self.save_requested.emit(self._payload())


class PerformanceMonitorDialog(QDialog):
    average_reset_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("性能监控")
        self.setMinimumWidth(760)
        self.setModal(False)
        self._records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._latest_turn_id: str | None = None
        self._showing_average = False
        self._metric_rows: dict[str, tuple[QProgressBar, QLabel]] = {}
        self._status_labels: dict[str, QLabel] = {}
        self._active_status = "idle"
        self._log_entries: deque[dict[str, str]] = deque(maxlen=200)

        self.summary_label = QLabel("等待主播语音输入")
        self.summary_label.setTextFormat(Qt.PlainText)

        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(4)
        for status_key in PERFORMANCE_STATE_LABELS:
            status_label = QLabel()
            status_label.setTextFormat(Qt.RichText)
            self._status_labels[status_key] = status_label
            status_layout.addWidget(status_label)
        self.set_status("idle")

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumWidth(500)
        self.log_output.setFixedHeight(125)
        self.log_output.document().setMaximumBlockCount(200)
        self.log_output.setStyleSheet(
            "QPlainTextEdit { background: #f7f7f8; color: #1d1d1f; "
            "border: 1px solid #d2d2d7; border-radius: 6px; "
            "padding: 7px; font-size: 13px; }"
        )

        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(6)
        log_layout.addWidget(self.summary_label)
        log_layout.addWidget(self.log_output)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.addLayout(status_layout)
        top_row.addSpacing(18)
        top_row.addLayout(log_layout, 1)

        metric_grid = QGridLayout()
        metric_grid.setContentsMargins(0, 4, 0, 4)
        metric_grid.setHorizontalSpacing(14)
        metric_grid.setVerticalSpacing(9)
        metric_grid.setColumnMinimumWidth(0, 180)
        metric_grid.setColumnStretch(1, 1)
        metric_grid.setColumnMinimumWidth(2, 225)
        metric_grid.addWidget(QLabel("指标"), 0, 0)
        duration_header = QLabel("耗时")
        duration_header.setAlignment(Qt.AlignCenter)
        metric_grid.addWidget(duration_header, 0, 1)
        metric_grid.addWidget(QLabel("当前 / 平均"), 0, 2)

        for grid_row, (key, title) in enumerate(PERFORMANCE_METRIC_FIELDS, start=1):
            title_label = QLabel(title)
            title_color = (
                PERFORMANCE_COLOR_FAST
                if key in PERFORMANCE_HIGHLIGHTED_METRICS
                else (
                    PERFORMANCE_COLOR_UNRATED
                    if key in PERFORMANCE_ALWAYS_BLUE_METRICS
                    else "#3a3a3c"
                )
            )
            title_label.setStyleSheet(f"color: {title_color};")

            progress_bar = QProgressBar()
            progress_bar.setRange(0, 1000)
            progress_bar.setValue(0)
            progress_bar.setTextVisible(False)
            progress_bar.setFixedHeight(18)
            self._set_progress_color(progress_bar, PERFORMANCE_COLOR_UNRATED)

            value_label = QLabel("--")
            value_label.setTextFormat(Qt.RichText)
            value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            value_label.setMinimumWidth(225)

            self._metric_rows[key] = (progress_bar, value_label)
            metric_grid.addWidget(title_label, grid_row, 0)
            metric_grid.addWidget(progress_bar, grid_row, 1)
            metric_grid.addWidget(value_label, grid_row, 2)

        self.average_button = QPushButton("显示多轮平均")
        self.average_button.setCursor(Qt.PointingHandCursor)
        self.average_button.clicked.connect(self.toggle_average)
        self.reset_average_button = QPushButton("重置平均值")
        self.reset_average_button.setCursor(Qt.PointingHandCursor)
        self.reset_average_button.clicked.connect(self._request_average_reset)
        close_button = QPushButton("关闭")
        close_button.setCursor(Qt.PointingHandCursor)
        close_button.clicked.connect(self.hide)

        button_row = QHBoxLayout()
        button_row.addWidget(self.average_button)
        button_row.addWidget(self.reset_average_button)
        button_row.addStretch(1)
        button_row.addWidget(close_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)
        layout.addLayout(top_row)
        layout.addLayout(metric_grid)
        layout.addLayout(button_row)
        self.setLayout(layout)
        self.append_log("性能监控已就绪，等待主播说话。")

    def append_log(self, message: str, timestamp: str | None = None) -> None:
        readable_message = str(message or "").strip()
        if not readable_message:
            return
        readable_timestamp = str(timestamp or "").strip()
        if not readable_timestamp:
            readable_timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3]
        self._log_entries.append(
            {"timestamp": readable_timestamp, "message": readable_message}
        )
        self.log_output.appendPlainText(
            f"{readable_timestamp}  {readable_message}"
        )
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def export_snapshot(self) -> dict[str, Any]:
        return {
            "records": [
                {
                    "turn_id": turn_id,
                    "metrics": dict(record.get("metrics") or {}),
                    "source": record.get("source"),
                    "completed": bool(record.get("completed")),
                    "average_eligible": bool(
                        record.get("average_eligible", True)
                    ),
                }
                for turn_id, record in self._records.items()
            ],
            "logs": list(self._log_entries),
            "status": self._active_status,
        }

    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        records = snapshot.get("records")
        logs = snapshot.get("logs")
        status = str(snapshot.get("status") or "").strip().lower()
        if status in PERFORMANCE_STATE_LABELS:
            self.set_status(status)
        if isinstance(records, list):
            self._records.clear()
            self._latest_turn_id = None
            for record in records[-PERFORMANCE_MONITOR_MAX_TURNS:]:
                if not isinstance(record, dict):
                    continue
                self.update_turn(
                    str(record.get("turn_id") or ""),
                    record.get("metrics") or {},
                    source=record.get("source"),
                    completed=bool(record.get("completed")),
                    average_eligible=bool(
                        record.get("average_eligible", True)
                    ),
                )
            if not self._records:
                self._render_latest()
        if isinstance(logs, list):
            self._log_entries.clear()
            self.log_output.clear()
            for entry in logs[-200:]:
                if isinstance(entry, dict):
                    self.append_log(
                        str(entry.get("message") or ""),
                        str(entry.get("timestamp") or "") or None,
                    )

    def set_status(self, active_status: str) -> None:
        if active_status not in PERFORMANCE_STATE_LABELS:
            active_status = "idle"
        self._active_status = active_status
        for status_key, label in self._status_labels.items():
            selected = status_key == active_status
            indicator = "●" if selected else "○"
            indicator_color = "#ff3b30" if selected else "#6e6e73"
            status_text = PERFORMANCE_STATE_LABELS[status_key]
            label.setText(
                f'<span style="color:{indicator_color}; font-weight:600">'
                f"{indicator}</span> "
                f'<span style="color:#1d1d1f">{status_text}</span>'
            )

    def update_turn(
        self,
        turn_id: str,
        metrics: dict[str, Any],
        *,
        source: str | None = None,
        completed: bool | None = None,
        average_eligible: bool | None = None,
    ) -> None:
        if not turn_id:
            return
        record = self._records.setdefault(
            turn_id,
            {
                "metrics": {},
                "source": source,
                "completed": False,
                "average_eligible": (
                    True if average_eligible is None else average_eligible
                ),
            },
        )
        known_metrics = dict(PERFORMANCE_METRIC_FIELDS)
        for key, value in metrics.items():
            if key not in known_metrics:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric) and numeric >= 0:
                record["metrics"][key] = numeric
        if source:
            record["source"] = source
        if completed is not None:
            record["completed"] = bool(completed)
        if average_eligible is not None:
            record["average_eligible"] = bool(average_eligible)

        self._records.move_to_end(turn_id)
        while len(self._records) > PERFORMANCE_MONITOR_MAX_TURNS:
            self._records.popitem(last=False)
        self._latest_turn_id = turn_id
        self._render_average() if self._showing_average else self._render_latest()

    def toggle_average(self) -> None:
        self._showing_average = not self._showing_average
        self.average_button.setText(
            "显示最新一轮" if self._showing_average else "显示多轮平均"
        )
        self._render_average() if self._showing_average else self._render_latest()

    def _request_average_reset(self) -> None:
        self.reset_averages()
        self.average_reset_requested.emit()

    def reset_averages(self) -> None:
        latest_turn_id = self._latest_turn_id
        latest_record = self._records.get(latest_turn_id or "")
        self._records.clear()
        if latest_turn_id and latest_record:
            latest_record["average_eligible"] = False
            self._records[latest_turn_id] = latest_record
        else:
            self._latest_turn_id = None
        self._showing_average = False
        self.average_button.setText("显示多轮平均")
        self._render_latest()

    def _render_latest(self) -> None:
        turn_id = self._latest_turn_id
        record = self._records.get(turn_id or "")
        if not turn_id or not record:
            self.summary_label.setText("等待主播语音输入")
            self._set_values({}, {})
            return
        historical_records = [
            item
            for record_turn_id, item in self._records.items()
            if (
                record_turn_id != turn_id
                and item.get("completed")
                and item.get("average_eligible", True)
            )
        ]
        historical_averages = self._calculate_averages(historical_records)
        source = self._source_label(record.get("source"))
        state = "已完成" if record.get("completed") else "进行中"
        baseline_text = (
            f"{len(historical_records)} 轮历史平均"
            if historical_records
            else "暂无历史平均"
        )
        self.summary_label.setText(
            f"最新一轮 · {source} · {state} · {baseline_text} · {turn_id[:8]}"
        )
        self._set_values(record["metrics"], historical_averages)

    def _render_average(self) -> None:
        completed_records = [
            record
            for record in self._records.values()
            if record.get("completed") and record.get("average_eligible", True)
        ]
        if not completed_records:
            self.summary_label.setText("暂无完整播放结束的语音轮次")
            self._set_values({}, {})
            return
        averages = self._calculate_averages(completed_records)
        self.summary_label.setText(f"多轮平均 · {len(completed_records)} 轮完整语音")
        self._set_values(
            averages,
            {},
            average_mode=True,
            average_count=len(completed_records),
        )

    @staticmethod
    def _calculate_averages(
        records: list[dict[str, Any]],
    ) -> dict[str, float]:
        averages: dict[str, float] = {}
        for key, _title in PERFORMANCE_METRIC_FIELDS:
            values = [
                record["metrics"][key]
                for record in records
                if key in record.get("metrics", {})
            ]
            if values:
                averages[key] = sum(values) / len(values)
        return averages

    def _set_values(
        self,
        metrics: dict[str, float],
        averages: dict[str, float],
        *,
        average_mode: bool = False,
        average_count: int = 0,
    ) -> None:
        for key, (progress_bar, label) in self._metric_rows.items():
            value = metrics.get(key)
            if value is None:
                progress_bar.setValue(0)
                self._set_progress_color(progress_bar, PERFORMANCE_COLOR_UNRATED)
                if key in PERFORMANCE_ALWAYS_BLUE_METRICS:
                    label.setText(
                        f'<span style="color:{PERFORMANCE_COLOR_UNRATED}">--</span>'
                    )
                else:
                    label.setText("--")
                continue

            if average_mode:
                progress_bar.setValue(self._progress_value(value))
                self._set_progress_color(progress_bar, PERFORMANCE_COLOR_UNRATED)
                label.setText(
                    f'<span style="color:{PERFORMANCE_COLOR_UNRATED}; '
                    f'font-weight:600">{self._format_seconds(value)}</span> '
                    f'<span style="color:#6e6e73">({average_count} 轮平均)</span>'
                )
                continue

            average = averages.get(key)
            color = (
                PERFORMANCE_COLOR_UNRATED
                if key in PERFORMANCE_ALWAYS_BLUE_METRICS
                else self._comparison_color(value, average)
            )
            progress_bar.setValue(self._progress_value(value))
            self._set_progress_color(progress_bar, color)
            average_text = (
                self._format_seconds(average) if average is not None else "--"
            )
            label.setText(
                f'<span style="color:{color}; font-weight:600">'
                f"{self._format_seconds(value)}</span> "
                f'<span style="color:#6e6e73">(平均 {average_text})</span>'
            )

    @staticmethod
    def _comparison_color(value: float, average: float | None) -> str:
        if average is None:
            return PERFORMANCE_COLOR_UNRATED
        if value <= average:
            return PERFORMANCE_COLOR_FAST
        if value - average <= PERFORMANCE_WARNING_DELTA_SECONDS:
            return PERFORMANCE_COLOR_WARNING
        return PERFORMANCE_COLOR_SLOW

    @staticmethod
    def _progress_value(value: float) -> int:
        ratio = value / PERFORMANCE_PROGRESS_MAX_SECONDS
        return max(0, min(1000, int(round(ratio * 1000))))

    @staticmethod
    def _set_progress_color(progress_bar: QProgressBar, color: str) -> None:
        progress_bar.setStyleSheet(
            "QProgressBar { background: #ffffff; border: 1px solid #d2d2d7; "
            "border-radius: 4px; }"
            f"QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}"
        )

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        if seconds < 1:
            milliseconds = seconds * 1000
            if 0 < milliseconds < 1:
                return "<1 ms"
            return f"{milliseconds:.0f} ms"
        return f"{seconds:.3f} s"

    @staticmethod
    def _source_label(source: Any) -> str:
        if source == "link_microphone":
            return "连线麦克风"
        return "主播麦克风"


class ConsoleWindow(QWidget):
    backend_message = pyqtSignal(dict)
    backend_state = pyqtSignal(bool)
    backend_error = pyqtSignal(str)
    microphone_error = pyqtSignal(str)
    microphone_speech_candidate_started = pyqtSignal()
    microphone_speech_started = pyqtSignal()
    microphone_speech_cancelled = pyqtSignal()
    microphone_audio_detected = pyqtSignal(object)
    microphone_audio_confirmed = pyqtSignal()
    link_microphone_error = pyqtSignal(str)
    link_microphone_speech_candidate_started = pyqtSignal()
    link_microphone_speech_started = pyqtSignal()
    link_microphone_speech_cancelled = pyqtSignal()
    link_microphone_audio_detected = pyqtSignal(object)
    link_microphone_audio_confirmed = pyqtSignal()
    link_human_name_local_ws_finished = pyqtSignal(dict)
    link_human_name_probe_finished = pyqtSignal(dict)

    @staticmethod
    def split_backend_url(url: str) -> tuple[str, str]:
        return split_backend_url(url)

    @staticmethod
    def build_backend_url(host: str, port: str) -> str:
        return build_backend_url(host, port)

    def __init__(
        self,
        url: str,
        fallback_model_name: str,
        fallback_model_path: str | None,
        live2d_width: int,
        live2d_height: int,
        display_mode: str,
    ) -> None:
        super().__init__()
        self.url = url
        self.display_mode = normalize_display_mode(display_mode)
        self.is_director_mode = self.display_mode == DISPLAY_MODE_DIRECTOR
        self.is_streamer_mode = not self.is_director_mode
        self.ui_scale = load_ui_scale()
        self.default_ws_host, self.default_ws_port = self.split_backend_url(url)
        self.fallback_model_name = fallback_model_name
        self.fallback_model_path = fallback_model_path
        self.live2d_width, self.live2d_height = load_live2d_window_size(live2d_width, live2d_height)
        self.model_config: ModelConfig | None = None
        self.live2d_window: Live2DWindow | None = None
        self.user_closed_live2d = False
        self.recreating_live2d = False
        self._closing_console = False
        self.live2d_global_enabled = True
        self.vtuber_mode = "idle"
        self.vtuber_sub_mode = "sleep"
        self.interaction_mode = "co_host"
        self.sleeping = True
        self.punished = False
        self.gift_thanks_enabled = False
        self.paint_enabled = False
        self.paint_window: PaintWindow | None = None
        self.live_streaming_agent_subtitle_enabled = False
        self.live_streaming_agent_subtitle_window: "LiveStreamingAgentSubtitleWindow | None" = None
        self.live_streaming_agent_subtitle_blocked_turn_ids: set[str] = set()
        self.barrage_subtitle_enabled = False
        self.barrage_subtitle_window: "BarrageSubtitleWindow | None" = None
        self.wake_animation_pending = False
        self._syncing_mode_buttons = False
        self.microphone_requested = False
        self.microphone_enabled = False
        self.microphone_faulted = False
        self.link_microphone_requested = False
        self.link_microphone_enabled = False
        self.link_microphone_faulted = False
        self.link_microphone_pending = False
        self.link_microphone_confirmed = False
        self._reported_link_microphone_faulted: bool | None = None
        self.link_human_name = DEFAULT_LINK_HUMAN_NAME
        link_name_window_binding = load_link_name_window_binding() or {}
        self.link_name_window_title: str | None = (
            str(link_name_window_binding.get("window_title") or "").strip() or None
        )
        self.link_name_window_hwnd: int | None = _coerce_optional_int(
            link_name_window_binding.get("hwnd")
        )
        self.link_human_name_detect_pending = False
        self.link_human_name_detect_request_id: str | None = None
        self.link_human_name_vision_requested = False
        self.link_human_name_local_ws_stop: threading.Event | None = None
        self.link_human_name_local_ws_started_at = 0.0
        self.link_human_name_backend_detect_started_at = 0.0
        self.link_human_name_backend_detect_attempts = 0
        self.link_human_name_last_probe_text = ""
        self.link_human_name_last_probe_debug_path: Path | None = None
        latest_probe_debug_path = LINK_NAME_PROBE_DEBUG_ROOT / "link_anchor_probe_latest.txt"
        try:
            if latest_probe_debug_path.exists():
                self.link_human_name_last_probe_text = latest_probe_debug_path.read_text(
                    encoding="utf-8"
                )
                self.link_human_name_last_probe_debug_path = latest_probe_debug_path
        except OSError as exc:
            logger.debug("Failed to load latest link anchor probe debug text: {}", exc)
        self.output_muted = self.is_director_mode
        self.mic_audio_input: QAudioInput | None = None
        self.mic_device = None
        self.mic_format: QAudioFormat | None = None
        self.mic_worker: MicrophoneVadWorker | None = None
        self.mic_byte_buffer = bytearray()
        self.microphone_playback_start_blocked = False
        self.microphone_paused_playback = False
        self.microphone_interrupt_committed = False
        self.mic_started_at = 0.0
        self.mic_last_data_at = 0.0
        self.mic_last_restart_at = 0.0
        self.microphone_restarting = False
        self.link_mic_audio_input: QAudioInput | None = None
        self.link_mic_device = None
        self.link_mic_format: QAudioFormat | None = None
        self.link_mic_worker: MicrophoneVadWorker | None = None
        self.link_mic_byte_buffer = bytearray()
        self.link_microphone_playback_start_blocked = False
        self.link_microphone_paused_playback = False
        self.link_microphone_interrupt_committed = False
        self.link_mic_started_at = 0.0
        self.link_mic_last_data_at = 0.0
        self.link_mic_last_restart_at = 0.0
        self.link_microphone_restarting = False
        self.performance_monitor: PerformanceMonitorDialog | None = None
        self.project_config_dialog: ProjectConfigDialog | None = None
        self.performance_state = "idle"
        self._performance_speech_started_at: dict[str, float] = {}
        self._performance_pending: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._performance_turn_state: dict[str, dict[str, Any]] = {}
        self.story_candidates: list[dict[str, Any]] = []
        self.story_rows: list[QLabel] = []
        self.story_progress_index = 0
        self.story_total = 0
        self.director_metric_list: QWidget | None = None
        self.director_metric_layout: QVBoxLayout | None = None
        self.director_metric_order: list[str] = []
        self.director_metric_widgets: dict[str, dict[str, Any]] = {}
        self.pending_vision_image: dict[str, str] | None = None
        self.vision_image_panel: QFrame | None = None
        self.vision_image_preview: VisionImageDropLabel | None = None
        self.vision_image_status: QLabel | None = None
        self.vision_model_combo: QComboBox | None = None
        self.vision_image_select_button: QPushButton | None = None
        self.vision_image_clear_button: QPushButton | None = None
        self.image_mode_enabled = False
        self.visual_image_context_active = False
        self.visual_image_reply_pending = False
        self.game_vision_enabled = False
        self.game_vision_request_id: str | None = None
        self.game_vision_cold_reply_pending = False
        # 游戏识图绑定的窗口标题 (None=未绑定; 有值=只截该窗口区域)
        self.game_vision_window_title: str | None = load_game_window_binding()
        self.game_vision_cold_timer = QTimer(self)
        self.game_vision_cold_timer.setSingleShot(True)
        self.game_vision_cold_timer.timeout.connect(
            self._handle_game_vision_cold_timeout
        )
        self.link_human_name_detect_timer = QTimer(self)
        self.link_human_name_detect_timer.setSingleShot(True)
        self.link_human_name_detect_timer.setInterval(
            LINK_HUMAN_NAME_DETECT_TIMEOUT_MS
        )
        self.link_human_name_detect_timer.timeout.connect(
            self._handle_link_human_name_detect_timeout
        )
        self.link_human_name_backend_retry_timer = QTimer(self)
        self.link_human_name_backend_retry_timer.setSingleShot(True)
        self.link_human_name_backend_retry_timer.setInterval(
            LINK_HUMAN_NAME_WS_POLL_INTERVAL_MS
        )
        self.link_human_name_backend_retry_timer.timeout.connect(
            self._handle_link_human_name_backend_retry
        )

        mode_name = "编导模式" if self.is_director_mode else "主播模式"
        self.setWindowTitle(f"Dream Maker Live Console - {mode_name}")
        console_min_height = (
            DIRECTOR_CONSOLE_MIN_HEIGHT
            if self.is_director_mode
            else CONSOLE_STREAMER_MIN_HEIGHT
        )
        self.console_min_width = (
            DIRECTOR_CONSOLE_MIN_WIDTH
            if self.is_director_mode
            else CONSOLE_MIN_WIDTH
        )
        self.console_min_height = console_min_height
        initial_width = scaled_int(self.console_min_width, self.ui_scale)
        initial_height = scaled_int(self.console_min_height, self.ui_scale)
        if self.is_director_mode:
            self.setMinimumSize(initial_width, initial_height)
            self.resize(initial_width, initial_height)
        else:
            self.setMinimumSize(initial_width, initial_height)
            self.setMaximumSize(QT_MAX_SIZE, QT_MAX_SIZE)
            self.resize(initial_width, initial_height)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setStyleSheet(scaled_console_style(self.ui_scale))

        self.ws_host_input = QLineEdit(self.default_ws_host)
        self.ws_host_input.setPlaceholderText("IP")
        self.ws_host_input.setFixedSize(
            CONSOLE_WS_HOST_INPUT_WIDTH,
            CONSOLE_WS_CONTROL_HEIGHT,
        )
        self.ws_port_input = QLineEdit(self.default_ws_port)
        self.ws_port_input.setPlaceholderText("端口")
        self.ws_port_input.setFixedSize(
            CONSOLE_WS_PORT_INPUT_WIDTH,
            CONSOLE_WS_CONTROL_HEIGHT,
        )
        self.reconnect_button = QPushButton("重连")
        self.default_ws_button = QPushButton("默认")
        for button in (self.reconnect_button, self.default_ws_button):
            button.setFixedSize(CONSOLE_BUTTON_WIDTH, CONSOLE_WS_CONTROL_HEIGHT)
            button.setCursor(Qt.PointingHandCursor)

        self.connection_button = QPushButton("连接中")
        self.live2d_button = QPushButton("打开Live2D")
        self.sleep_button = QPushButton(SLEEP_SLEEPING_TEXT)
        self.voice_button = QPushButton(PUNISH_INACTIVE_TEXT)
        self.voice_cutoff_button = QPushButton("打断说话")
        self.game_vision_button = QPushButton(GAME_VISION_BUTTON_TEXT)
        self.paint_button = QPushButton(PAINT_BUTTON_INACTIVE_TEXT)
        self.mode_button = QPushButton(BARRAGE_IGNORE_TEXT)
        self.gift_thanks_button = QPushButton(GIFT_THANKS_INACTIVE_TEXT)
        self.live_streaming_agent_subtitle_button = QPushButton(LIVE_STREAMING_AGENT_SUBTITLE_BUTTON_TEXT)
        self.barrage_subtitle_button = QPushButton(BARRAGE_SUBTITLE_BUTTON_TEXT)
        self.project_config_button = QPushButton("项目配置")
        self.performance_button = QPushButton("性能监控")
        self.anchor_text_input = QLineEdit()
        self.anchor_text_input.setPlaceholderText("主播发言：输入后回车发送")
        self.anchor_text_input.setFixedSize(
            ANCHOR_TEXT_INPUT_WIDTH,
            CONSOLE_BUTTON_HEIGHT,
        )
        self.anchor_text_send_button = QPushButton("发送")
        self.anchor_text_send_button.setFixedSize(
            ANCHOR_TEXT_SEND_BUTTON_WIDTH,
            CONSOLE_BUTTON_HEIGHT,
        )
        self.anchor_text_send_button.setCursor(Qt.PointingHandCursor)
        self.anchor_text_send_button.setVisible(False)
        self.image_mode_button = QPushButton(IMAGE_MODE_BUTTON_TEXT)
        self.image_mode_button.setVisible(self.is_streamer_mode)
        self.microphone_button = QPushButton(MIC_OFF_TEXT)
        self.link_microphone_button = QPushButton(LINK_MIC_OFF_TEXT)
        self.microphone_volume_indicator = VolumeSignalIndicator("本地麦克风音量")
        self.link_microphone_volume_indicator = VolumeSignalIndicator("连线麦克风音量")
        self.microphone_volume_indicator.setVisible(self.is_streamer_mode)
        self.link_microphone_volume_indicator.setVisible(self.is_streamer_mode)
        self.link_human_name_input = QLineEdit(self.link_human_name)
        self.link_human_name_input.setPlaceholderText(DEFAULT_LINK_HUMAN_NAME)
        self.link_human_name_input.setFixedSize(
            LINK_HUMAN_NAME_INPUT_WIDTH,
            CONSOLE_BUTTON_HEIGHT,
        )
        self.link_human_name_auto_button = QPushButton(
            LINK_HUMAN_NAME_AUTO_BUTTON_TEXT
        )
        self.link_human_name_auto_button.setFixedSize(
            LINK_HUMAN_NAME_AUTO_BUTTON_WIDTH,
            CONSOLE_BUTTON_HEIGHT,
        )
        self.link_human_name_auto_button.setCursor(Qt.PointingHandCursor)
        self.link_human_name_auto_button.setToolTip(
            "\u4ec5\u4f7f\u7528\u672c\u673a DouyinBarrage WebSocket \u6293\u5305\u8bc6\u522b\u8fde\u7ebf\u4e3b\u64ad"
        )
        self.link_human_name_save_timer = QTimer(self)
        self.link_human_name_save_timer.setSingleShot(True)
        self.link_human_name_save_timer.setInterval(
            LINK_HUMAN_NAME_SAVE_DEBOUNCE_MS
        )
        self.link_human_name_save_timer.timeout.connect(
            self.handle_link_human_name_autosave
        )

        for button in (
            self.microphone_button,
            self.link_microphone_button,
            self.connection_button,
            self.live2d_button,
            self.sleep_button,
            self.voice_button,
            self.voice_cutoff_button,
            self.game_vision_button,
            self.paint_button,
            self.mode_button,
            self.gift_thanks_button,
            self.live_streaming_agent_subtitle_button,
            self.barrage_subtitle_button,
            self.image_mode_button,
            self.project_config_button,
            self.performance_button,
        ):
            button.setFixedSize(CONSOLE_BUTTON_WIDTH, CONSOLE_BUTTON_HEIGHT)
            button.setCursor(Qt.PointingHandCursor)
        self.connection_button.setFixedSize(
            CONSOLE_BUTTON_WIDTH,
            CONSOLE_WS_CONTROL_HEIGHT,
        )
        self.connection_button.setCursor(Qt.ArrowCursor)

        self.reply_probability_combo = QComboBox()
        self.reply_probability_combo.setObjectName("replyPercentCombo")
        for value in (5, 10, 20, 30, 40, 50):
            self.reply_probability_combo.addItem(f"{value}%", value)
        self.reply_probability_combo.setCurrentIndex(
            self.reply_probability_combo.findData(10)
        )
        self.reply_probability_combo.setFixedWidth(
            CONSOLE_REPLY_PERCENT_COMBO_WIDTH
        )

        self.cold_time_combo = QComboBox()
        self.cold_time_combo.setObjectName("coldTimeCombo")
        for value in range(5, 11, 5):
            self.cold_time_combo.addItem(f"{value}s", value)
        self.cold_time_combo.setFixedWidth(CONSOLE_COLD_TIME_COMBO_MIN_WIDTH)
        self.cold_time_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self.ui_scale_label = QLabel("\u754c\u9762")
        self.ui_scale_combo = QComboBox()
        self.ui_scale_combo.setObjectName("uiScaleCombo")
        for scale in UI_SCALE_OPTIONS:
            self.ui_scale_combo.addItem(f"{int(round(scale * 100))}%", scale)
        scale_index = self.ui_scale_combo.findData(self.ui_scale)
        self.ui_scale_combo.setCurrentIndex(max(scale_index, 0))
        self.ui_scale_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self.collapsible_sections: list[CollapsibleSection] = []
        self.section_title_labels: list[QLabel] = []

        websocket_content = QHBoxLayout()
        websocket_content.setSpacing(14)
        websocket_content.addWidget(self.connection_button)
        websocket_content.addWidget(self.ws_host_input)
        websocket_content.addWidget(self.ws_port_input)
        websocket_content.addWidget(self.reconnect_button)
        websocket_content.addWidget(self.default_ws_button)
        websocket_content.addStretch(1)
        websocket_content.addWidget(self.ui_scale_label)
        websocket_content.addWidget(self.ui_scale_combo)
        websocket_section = CollapsibleSection("websocket连接", websocket_content)

        configuration_content = QHBoxLayout()
        configuration_content.setSpacing(14)
        configuration_content.addWidget(self.project_config_button)
        configuration_content.addWidget(self.performance_button)
        configuration_content.addStretch(1)
        configuration_section = CollapsibleSection("配置", configuration_content)

        window_content = QHBoxLayout()
        window_content.setSpacing(14)
        window_content.addWidget(self.live2d_button)
        window_content.addWidget(self.microphone_button)
        if self.is_streamer_mode:
            window_content.addWidget(self.microphone_volume_indicator)
        window_content.addWidget(self.link_microphone_button)
        if self.is_streamer_mode:
            window_content.addWidget(self.link_microphone_volume_indicator)
        window_content.addWidget(self.link_human_name_input)
        window_content.addWidget(self.link_human_name_auto_button)
        window_content.addStretch(1)
        window_section = CollapsibleSection("窗口控制", window_content)

        function_content = QHBoxLayout()
        function_content.setSpacing(14)
        function_content.addWidget(self.sleep_button)
        function_content.addWidget(self.voice_button)
        function_content.addWidget(self.mode_button)
        function_content.addWidget(self.gift_thanks_button)
        function_content.addWidget(self.voice_cutoff_button)
        function_content.addWidget(self.live_streaming_agent_subtitle_button)
        function_content.addWidget(self.barrage_subtitle_button)
        function_content.addStretch(1)
        function_section = CollapsibleSection("功能控制", function_content)

        entertainment_content = QVBoxLayout()
        entertainment_content.setSpacing(12)
        entertainment_button_row = QHBoxLayout()
        entertainment_button_row.setSpacing(14)
        entertainment_button_row.addWidget(self.game_vision_button)
        entertainment_button_row.addWidget(self.paint_button)
        entertainment_button_row.addWidget(self.image_mode_button)
        entertainment_button_row.addStretch(1)
        anchor_text_row = QHBoxLayout()
        anchor_text_row.setSpacing(14)
        anchor_text_row.addWidget(self.anchor_text_input, 1)
        entertainment_content.addLayout(entertainment_button_row)
        entertainment_content.addLayout(anchor_text_row)
        if self.is_streamer_mode:
            self.vision_image_panel = self._create_vision_image_panel()
            self.vision_image_panel.setVisible(True)
            self._set_vision_image_panel_enabled(False)
            self._set_vision_image_status(
                "图片模式已关闭：视觉识别图片输入栏已禁用"
            )
            entertainment_content.addWidget(self.vision_image_panel)
        entertainment_section = CollapsibleSection("娱乐功能", entertainment_content)

        self.collapsible_sections = [
            websocket_section,
            configuration_section,
            window_section,
            function_section,
            entertainment_section,
        ]
        self.section_title_labels = [
            section.title_label for section in self.collapsible_sections
        ]

        selection_row = QHBoxLayout()
        selection_row.setSpacing(10)
        selection_row.addWidget(QLabel("前"))
        selection_row.addWidget(self.reply_probability_combo)
        selection_row.addWidget(QLabel("观众划分为高等级"))
        selection_row.addWidget(QLabel("冷场时间"))
        selection_row.addWidget(self.cold_time_combo)
        selection_row.addStretch(1)

        if self.is_director_mode:
            self.director_metric_list = self._create_director_metric_list()

        story_title = QLabel("剧本预览")
        self.story_title = story_title
        self.story_title.setStyleSheet(scaled_section_title_style(self.ui_scale))
        self.section_title_labels.append(self.story_title)
        self.story_panel = QFrame()
        self.story_panel.setStyleSheet(STORY_PANEL_STYLE)
        self.story_panel.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.MinimumExpanding,
        )
        self.story_panel.setMinimumHeight(
            scaled_int(DIRECTOR_STORY_PANEL_MIN_HEIGHT, self.ui_scale)
        )
        story_panel_layout = QVBoxLayout()
        story_panel_layout.setContentsMargins(10, 10, 10, 10)
        story_panel_layout.setSpacing(8)
        self.story_panel.setLayout(story_panel_layout)

        self.story_content = QWidget()
        self.story_content.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.MinimumExpanding,
        )
        self.story_content_layout = QVBoxLayout()
        self.story_content_layout.setContentsMargins(0, 0, 0, 0)
        self.story_content_layout.setSpacing(8)
        self.story_content.setLayout(self.story_content_layout)
        for _ in range(4):
            row = QLabel(STORY_EMPTY_TEXT)
            row.setWordWrap(True)
            row.setTextFormat(Qt.PlainText)
            row.setMinimumHeight(DIRECTOR_STORY_ROW_MIN_HEIGHT)
            row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
            row.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            row.setStyleSheet(STORY_ROW_STYLE)
            self.story_rows.append(row)
            self.story_content_layout.addWidget(row)
        self.story_content_layout.addStretch(1)
        story_panel_layout.addWidget(self.story_content)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(10)
        main_layout.addWidget(websocket_section)
        main_layout.addWidget(configuration_section)
        main_layout.addWidget(window_section)
        main_layout.addWidget(function_section)
        main_layout.addWidget(entertainment_section)
        if self.is_director_mode:
            main_layout.addLayout(selection_row)
            if self.director_metric_list:
                main_layout.addWidget(self.director_metric_list)
            main_layout.addWidget(self.story_title)
            main_layout.addWidget(self.story_panel)
        main_layout.addStretch(1)
        self.console_content = QWidget()
        self.console_content.setLayout(main_layout)
        self.console_scroll = QScrollArea()
        self.console_scroll.setWidgetResizable(True)
        self.console_scroll.setFrameShape(QFrame.NoFrame)
        self.console_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.console_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.console_scroll.setWidget(self.console_content)
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self.console_scroll)
        self.setLayout(outer_layout)
        self._apply_ui_scale(self.ui_scale)

        self.reconnect_button.clicked.connect(self.reconnect_backend_from_inputs)
        self.default_ws_button.clicked.connect(self.reconnect_backend_with_default)
        self.ui_scale_combo.currentIndexChanged.connect(self.handle_ui_scale_changed)
        self.ws_host_input.returnPressed.connect(self.reconnect_backend_from_inputs)
        self.ws_port_input.returnPressed.connect(self.reconnect_backend_from_inputs)
        self.live2d_button.clicked.connect(self.toggle_live2d_window)
        self.sleep_button.clicked.connect(self.toggle_sleep_mode)
        self.voice_button.clicked.connect(self.toggle_punish_mode)
        self.microphone_button.clicked.connect(self.toggle_microphone)
        self.link_microphone_button.clicked.connect(self.toggle_link_microphone)
        self.link_human_name_input.textEdited.connect(
            self.schedule_link_human_name_autosave
        )
        self.link_human_name_input.editingFinished.connect(
            self.handle_link_human_name_changed
        )
        self.link_human_name_auto_button.clicked.connect(
            self.handle_link_human_name_auto_clicked
        )
        self.link_human_name_auto_button.setContextMenuPolicy(Qt.CustomContextMenu)
        self.link_human_name_auto_button.customContextMenuRequested.connect(
            self._show_link_human_name_auto_menu
        )
        self.voice_cutoff_button.clicked.connect(self.handle_voice_cutoff_clicked)
        self.game_vision_button.clicked.connect(self._show_game_vision_button_menu)
        self.game_vision_button.setContextMenuPolicy(Qt.CustomContextMenu)
        self.game_vision_button.customContextMenuRequested.connect(
            self._show_game_window_bind_menu
        )
        self.paint_button.clicked.connect(self.toggle_paint)
        self.mode_button.clicked.connect(self.handle_mode_clicked)
        self.gift_thanks_button.clicked.connect(self.toggle_gift_thanks)
        self.live_streaming_agent_subtitle_button.clicked.connect(self.toggle_live_streaming_agent_subtitle)
        self.barrage_subtitle_button.clicked.connect(self.toggle_barrage_subtitle)
        self.project_config_button.clicked.connect(self.show_project_config)
        self.performance_button.clicked.connect(self.show_performance_monitor)
        self.image_mode_button.clicked.connect(self.toggle_image_mode)
        self.anchor_text_input.returnPressed.connect(self.send_anchor_text_input)
        self.anchor_text_send_button.clicked.connect(self.send_anchor_text_input)
        # 按钮初始为灰色 (关闭状态)
        self.live_streaming_agent_subtitle_button.setStyleSheet(INACTIVE_BUTTON_STYLE)
        self.barrage_subtitle_button.setStyleSheet(INACTIVE_BUTTON_STYLE)
        self.project_config_button.setStyleSheet(INACTIVE_BUTTON_STYLE)
        self.performance_button.setStyleSheet(INACTIVE_BUTTON_STYLE)
        self.update_image_mode_button_state(False)
        self.update_game_vision_button_state(False)
        self._apply_game_window_binding_ui()
        self.update_paint_button_state(False)
        self.reply_probability_combo.currentIndexChanged.connect(
            lambda _index: self.send_console_message(
                "reply-probability",
                value=int(self.reply_probability_combo.currentData()),
                unit="percent",
            )
        )
        self.cold_time_combo.currentIndexChanged.connect(
            lambda _index: self.send_console_message(
                "cold-time",
                value=int(self.cold_time_combo.currentData()),
                unit="seconds",
            )
        )
        self.backend_message.connect(self.handle_backend_message)
        self.backend_state.connect(self.update_backend_state)
        self.backend_error.connect(self.handle_backend_error)
        self.microphone_error.connect(self.handle_microphone_error)
        self.microphone_speech_candidate_started.connect(
            self.handle_microphone_speech_candidate_started
        )
        self.microphone_speech_started.connect(self.handle_microphone_speech_started)
        self.microphone_speech_cancelled.connect(self.handle_microphone_speech_cancelled)
        self.microphone_audio_detected.connect(self.send_microphone_audio_segment)
        self.microphone_audio_confirmed.connect(self.confirm_microphone_audio)
        self.link_microphone_error.connect(self.handle_link_microphone_error)
        self.link_microphone_speech_candidate_started.connect(
            self.handle_link_microphone_speech_candidate_started
        )
        self.link_microphone_speech_started.connect(
            self.handle_link_microphone_speech_started
        )
        self.link_microphone_speech_cancelled.connect(
            self.handle_link_microphone_speech_cancelled
        )
        self.link_microphone_audio_detected.connect(
            self.send_link_microphone_audio_segment
        )
        self.link_microphone_audio_confirmed.connect(
            self.confirm_link_microphone_audio
        )
        self.link_human_name_local_ws_finished.connect(
            self._handle_local_douyin_barrage_link_probe_finished
        )
        self.link_human_name_probe_finished.connect(
            self._handle_link_anchor_probe_finished
        )
        self.backend_client = BackendWebSocketClient(
            url=self.url,
            on_message=self.backend_message.emit,
            on_state=self.backend_state.emit,
            on_error=self.backend_error.emit,
        )
        self.update_live2d_button_state(False)
        self.update_vtuber_mode_buttons()
        self.update_gift_thanks_button_state(False)
        self.update_microphone_button_state(False)
        self.update_link_microphone_button_state(False)
        self._apply_link_human_name_auto_tooltip()
        self.update_backend_state(False)
        self.project_config_dialog = ProjectConfigDialog(self)
        self.project_config_dialog.save_requested.connect(
            self._save_project_config
        )
        self.project_config_dialog.test_requested.connect(
            self._test_project_config
        )
        self.performance_monitor = PerformanceMonitorDialog(self)
        self.performance_monitor.average_reset_requested.connect(
            self._handle_performance_average_reset_requested
        )
        self._sync_performance_monitor_status()
        self.microphone_health_timer = QTimer(self)
        self.microphone_health_timer.setInterval(MIC_HEALTH_CHECK_INTERVAL_MS)
        self.microphone_health_timer.timeout.connect(self.check_microphone_health)
        self.microphone_health_timer.start()
        app = QApplication.instance()
        if app:
            app.installEventFilter(self)
        self.backend_client.start()

    def _set_scaled_button_size(
        self,
        button: QPushButton,
        *,
        width: int = CONSOLE_BUTTON_WIDTH,
        height: int = CONSOLE_BUTTON_HEIGHT,
    ) -> None:
        button.setMinimumSize(
            scaled_int(width, self.ui_scale),
            scaled_int(height, self.ui_scale),
        )
        button.setMaximumSize(QT_MAX_SIZE, scaled_int(height, self.ui_scale))
        button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

    def _set_scaled_line_edit_size(
        self,
        line_edit: QLineEdit,
        *,
        min_width: int,
        max_width: int | None = None,
        height: int = CONSOLE_BUTTON_HEIGHT,
        expanding: bool = False,
    ) -> None:
        line_edit.setMinimumSize(
            scaled_int(min_width, self.ui_scale),
            scaled_int(height, self.ui_scale),
        )
        line_edit.setMaximumSize(
            scaled_int(max_width, self.ui_scale) if max_width else QT_MAX_SIZE,
            scaled_int(height, self.ui_scale),
        )
        line_edit.setSizePolicy(
            QSizePolicy.Expanding if expanding else QSizePolicy.Minimum,
            QSizePolicy.Fixed,
        )

    def _set_scaled_combo_size(
        self,
        combo: QComboBox,
        *,
        min_width: int,
        height: int = CONSOLE_WS_CONTROL_HEIGHT,
        fit_contents: bool = False,
    ) -> None:
        scaled_min_width = scaled_int(min_width, self.ui_scale)
        if fit_contents:
            scaled_min_width = max(scaled_min_width, combo.sizeHint().width())
        combo.setMinimumSize(
            scaled_min_width,
            scaled_int(height, self.ui_scale),
        )
        combo.setMaximumSize(QT_MAX_SIZE, scaled_int(height, self.ui_scale))
        combo.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

    def _apply_ui_scale(self, scale: float) -> None:
        self.ui_scale = normalize_ui_scale(scale)
        self.setStyleSheet(scaled_console_style(self.ui_scale))
        window_min_width = scaled_int(self.console_min_width, self.ui_scale)
        window_min_height = scaled_int(self.console_min_height, self.ui_scale)
        if self.is_director_mode:
            self.setMinimumSize(window_min_width, window_min_height)
            self.setMaximumSize(QT_MAX_SIZE, QT_MAX_SIZE)
            self.resize(
                max(self.width(), window_min_width),
                max(self.height(), window_min_height),
            )
        else:
            self.setMinimumSize(window_min_width, window_min_height)
            self.setMaximumSize(QT_MAX_SIZE, QT_MAX_SIZE)
            self.resize(
                max(self.width(), window_min_width),
                max(self.height(), window_min_height),
            )

        if hasattr(self, "console_content"):
            self.console_content.setMinimumWidth(
                scaled_int(self.console_min_width, self.ui_scale)
            )

        for title in getattr(self, "section_title_labels", []):
            title.setStyleSheet(scaled_section_title_style(self.ui_scale))
        for section in getattr(self, "collapsible_sections", []):
            section.apply_ui_scale(self.ui_scale)

        self._set_scaled_line_edit_size(
            self.ws_host_input,
            min_width=CONSOLE_WS_HOST_INPUT_WIDTH,
            max_width=360,
            height=CONSOLE_WS_CONTROL_HEIGHT,
            expanding=True,
        )
        self._set_scaled_line_edit_size(
            self.ws_port_input,
            min_width=CONSOLE_WS_PORT_INPUT_WIDTH,
            max_width=CONSOLE_WS_PORT_INPUT_WIDTH,
            height=CONSOLE_WS_CONTROL_HEIGHT,
        )
        self._set_scaled_line_edit_size(
            self.link_human_name_input,
            min_width=LINK_HUMAN_NAME_INPUT_WIDTH,
            max_width=220,
            height=CONSOLE_BUTTON_HEIGHT,
        )
        self._set_scaled_button_size(
            self.link_human_name_auto_button,
            width=LINK_HUMAN_NAME_AUTO_BUTTON_WIDTH,
        )
        self._set_scaled_line_edit_size(
            self.anchor_text_input,
            min_width=ANCHOR_TEXT_INPUT_WIDTH,
            height=CONSOLE_BUTTON_HEIGHT,
            expanding=True,
        )

        for button in (self.reconnect_button, self.default_ws_button):
            self._set_scaled_button_size(
                button,
                height=CONSOLE_WS_CONTROL_HEIGHT,
            )
        self._set_scaled_button_size(
            self.connection_button,
            height=CONSOLE_WS_CONTROL_HEIGHT,
        )
        for button in (
            self.microphone_button,
            self.link_microphone_button,
            self.live2d_button,
            self.sleep_button,
            self.voice_button,
            self.voice_cutoff_button,
            self.game_vision_button,
            self.paint_button,
            self.mode_button,
            self.gift_thanks_button,
            self.live_streaming_agent_subtitle_button,
            self.barrage_subtitle_button,
            self.image_mode_button,
            self.project_config_button,
            self.performance_button,
        ):
            self._set_scaled_button_size(button)
        self._set_scaled_button_size(
            self.anchor_text_send_button,
            width=ANCHOR_TEXT_SEND_BUTTON_WIDTH,
        )

        self._set_scaled_combo_size(
            self.reply_probability_combo,
            min_width=CONSOLE_REPLY_PERCENT_COMBO_WIDTH,
            fit_contents=True,
        )
        self._set_scaled_combo_size(
            self.cold_time_combo,
            min_width=CONSOLE_COLD_TIME_COMBO_MIN_WIDTH,
        )
        self._set_scaled_combo_size(
            self.ui_scale_combo,
            min_width=64,
            fit_contents=True,
        )

        self.microphone_volume_indicator.set_ui_scale(self.ui_scale)
        self.link_microphone_volume_indicator.set_ui_scale(self.ui_scale)

        if self.director_metric_list:
            self.director_metric_list.setMinimumSize(
                scaled_int(240, self.ui_scale),
                scaled_int(150, self.ui_scale),
            )
            self.director_metric_list.setMaximumSize(
                scaled_int(DIRECTOR_METRIC_PANEL_WIDTH, self.ui_scale),
                scaled_int(150, self.ui_scale),
            )
        for widgets in self.director_metric_widgets.values():
            row = widgets.get("row")
            if isinstance(row, DirectorMetricRow):
                row.apply_ui_scale(self.ui_scale)

        if hasattr(self, "story_panel"):
            self.story_panel.setMinimumSize(
                scaled_int(DIRECTOR_STORY_PANEL_WIDTH, self.ui_scale)
                if self.is_director_mode
                else scaled_int(420, self.ui_scale),
                scaled_int(DIRECTOR_STORY_PANEL_MIN_HEIGHT, self.ui_scale),
            )
            self.story_panel.setMaximumWidth(QT_MAX_SIZE)
        if hasattr(self, "story_title"):
            self.story_title.setMinimumWidth(
                scaled_int(DIRECTOR_STORY_PANEL_WIDTH, self.ui_scale)
            )
            self.story_title.setMaximumWidth(QT_MAX_SIZE)
        if hasattr(self, "story_content"):
            self.story_content.setMinimumHeight(
                scaled_int(DIRECTOR_STORY_PANEL_MIN_HEIGHT - 28, self.ui_scale)
            )
        for row in self.story_rows:
            row.setMinimumHeight(
                scaled_int(DIRECTOR_STORY_ROW_MIN_HEIGHT, self.ui_scale)
            )

        self.updateGeometry()

    def handle_ui_scale_changed(self, *_args: Any) -> None:
        scale = normalize_ui_scale(self.ui_scale_combo.currentData())
        if abs(scale - self.ui_scale) < 0.001:
            return
        self._apply_ui_scale(scale)
        save_ui_scale(scale)

    def eventFilter(self, source: Any, event: Any) -> bool:
        if event.type() == QEvent.MouseButtonPress:
            self._commit_link_human_name_on_external_click(source)
        return super().eventFilter(source, event)

    def _commit_link_human_name_on_external_click(self, source: Any) -> None:
        if not self.link_human_name_input.hasFocus():
            return
        if not isinstance(source, QWidget) or source.window() is not self:
            return
        if (
            source is self.link_human_name_input
            or self.link_human_name_input.isAncestorOf(source)
        ):
            return

        self.handle_link_human_name_changed()
        self.link_human_name_input.clearFocus()
        self.setFocus(Qt.MouseFocusReason)

    def _create_vision_image_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("visionImagePanel")
        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(10, 10, 10, 8)
        panel_layout.setSpacing(6)
        panel.setLayout(panel_layout)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        title = QLabel("视觉识别图片")
        title.setStyleSheet(SECTION_TITLE_STYLE)

        self.vision_model_combo = QComboBox()
        self.vision_model_combo.addItem("Doubao Vision", "doubao_vision_llm")
        self.vision_model_combo.addItem("Qwen3-VL", "qwen3_vl_llm")
        self.vision_model_combo.addItem("GLM-5V-Turbo", "glm_5v_turbo_llm")
        self.vision_model_combo.setFixedSize(170, 34)

        self.vision_image_select_button = QPushButton("选择图片")
        self.vision_image_clear_button = QPushButton("清除")
        for button in (
            self.vision_image_select_button,
            self.vision_image_clear_button,
        ):
            button.setFixedHeight(34)
            button.setCursor(Qt.PointingHandCursor)

        toolbar.addWidget(title)
        toolbar.addStretch(1)
        toolbar.addWidget(self.vision_model_combo)
        toolbar.addWidget(self.vision_image_select_button)
        toolbar.addWidget(self.vision_image_clear_button)

        self.vision_image_preview = VisionImageDropLabel()
        self.vision_image_status = QLabel(
            "先放入图片，再通过主播输入框或麦克风提出问题（视觉多轮）"
        )
        self.vision_image_status.setObjectName("visionImageStatus")
        self.vision_image_status.setTextFormat(Qt.PlainText)

        panel_layout.addLayout(toolbar)
        panel_layout.addWidget(self.vision_image_preview)
        panel_layout.addWidget(self.vision_image_status)

        self.vision_image_select_button.clicked.connect(
            lambda: self._select_vision_image()
        )
        self.vision_image_clear_button.clicked.connect(
            lambda _checked=False: self._clear_vision_image()
        )
        self.vision_image_preview.file_dropped.connect(self._select_vision_image)
        return panel

    def _set_vision_image_panel_enabled(self, enabled: bool) -> None:
        active = bool(enabled) and self.is_streamer_mode
        if self.vision_image_panel:
            self.vision_image_panel.setProperty("imageModeActive", active)
            self.vision_image_panel.style().unpolish(self.vision_image_panel)
            self.vision_image_panel.style().polish(self.vision_image_panel)
            self.vision_image_panel.update()
        for widget in (
            self.vision_model_combo,
            self.vision_image_select_button,
            self.vision_image_clear_button,
            self.vision_image_preview,
        ):
            if widget:
                widget.setEnabled(active)
        if self.vision_image_preview:
            self.vision_image_preview.setAcceptDrops(active)

    def _select_vision_image(self, path: str = "") -> None:
        if not self.image_mode_enabled:
            self._set_vision_image_status(
                "图片模式未开启：点击“图片模式”后再选择或拖入图片",
                error=True,
            )
            return
        selected_path = path
        if not selected_path:
            selected_path, _filter = QFileDialog.getOpenFileName(
                self,
                "选择待识别图片",
                "",
                "图片文件 (*.jpg *.jpeg *.png *.webp)",
            )
        if not selected_path:
            return
        self._load_vision_image(Path(selected_path))

    def _set_vision_image_status(self, text: str, *, error: bool = False) -> None:
        if not self.vision_image_status:
            return
        self.vision_image_status.setText(text)
        self.vision_image_status.setStyleSheet(
            "color: #b42318; font-size: 12px; font-weight: 600;"
            if error
            else ""
        )

    def _load_vision_image(self, path: Path) -> None:
        mime_type = VISION_IMAGE_MIME_TYPES.get(path.suffix.lower())
        if not mime_type:
            self._set_vision_image_status("仅支持 JPG、PNG、WebP 图片", error=True)
            return

        try:
            image_bytes = path.read_bytes()
        except OSError as exc:
            self._set_vision_image_status(f"图片读取失败：{exc}", error=True)
            return

        if len(image_bytes) > VISION_IMAGE_MAX_BYTES:
            self._set_vision_image_status("图片不能超过 10 MB", error=True)
            return

        pixmap = QPixmap()
        if not pixmap.loadFromData(image_bytes):
            self._set_vision_image_status("图片格式无效或无法解码", error=True)
            return

        encoded = base64.b64encode(image_bytes).decode("ascii")
        self.pending_vision_image = {
            "source": "upload",
            "data": f"data:{mime_type};base64,{encoded}",
            "mime_type": mime_type,
            "name": path.name,
        }
        if self.vision_image_preview:
            preview = pixmap.scaled(
                QSize(680, VISION_IMAGE_PREVIEW_HEIGHT),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.vision_image_preview.setPixmap(preview)
        self._set_vision_image_status(f"待识别：{path.name}")
        logger.info(
            "Loaded pending visual image: name={} mime_type={} bytes={}",
            path.name,
            mime_type,
            len(image_bytes),
        )

    def _clear_vision_image(self, *, notify_backend: bool = True) -> None:
        self.pending_vision_image = None
        if self.vision_image_preview:
            self.vision_image_preview.clear()
            self.vision_image_preview.setText("暂无图片\n点击选择或拖入图片")
        if notify_backend:
            self.visual_image_context_active = False
            self.visual_image_reply_pending = False
            self.send_console_message("clear-vision-context")
            self._set_vision_image_status("已清除待识别图片和后端图片上下文")
        else:
            self._set_vision_image_status(
                "先放入图片，再通过主播输入框或麦克风提出问题（视觉多轮）"
            )

    def _attach_pending_vision_image(self, payload: dict[str, Any]) -> bool:
        if (
            not self.image_mode_enabled
            or not self.pending_vision_image
            or not self.vision_model_combo
        ):
            return False
        payload["images"] = [
            {
                "source": self.pending_vision_image["source"],
                "data": self.pending_vision_image["data"],
                "mime_type": self.pending_vision_image["mime_type"],
            }
        ]
        payload["metadata"] = {
            **(payload.get("metadata") or {}),
            "vision_model_provider": self.vision_model_combo.currentData(),
            "vision_context_mode": self._current_vision_context_mode(),
            "visual_image_attached": True,
            "vision_image_name": self.pending_vision_image.get("name"),
        }
        return True

    def _current_vision_model_provider(self) -> str | None:
        if not self.vision_model_combo:
            return None
        provider = self.vision_model_combo.currentData()
        return str(provider) if provider else None

    def _current_vision_model_label(self) -> str:
        if not self.vision_model_combo:
            return "default"
        label = str(self.vision_model_combo.currentText() or "").strip()
        return label or "default"

    def _current_vision_context_mode(self) -> str:
        return VISION_CONTEXT_MODE_PERSISTENT

    def update_image_mode_button_state(self, enabled: bool) -> None:
        self.image_mode_button.setText(
            IMAGE_MODE_BUTTON_ACTIVE_TEXT if enabled else IMAGE_MODE_BUTTON_TEXT
        )
        self.image_mode_button.setStyleSheet(
            CONNECTED_BUTTON_STYLE if enabled else INACTIVE_BUTTON_STYLE
        )

    def toggle_image_mode(self) -> None:
        self._set_image_mode_enabled(not self.image_mode_enabled)

    def _set_image_mode_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled) and self.is_streamer_mode
        if self.image_mode_enabled == enabled:
            return

        self.image_mode_enabled = enabled
        if self.vision_image_panel:
            self.vision_image_panel.setVisible(self.is_streamer_mode)
            self._set_vision_image_panel_enabled(enabled)
        if enabled:
            self._set_vision_image_status(
                "图片模式已开启：先放入图片，再通过主播输入框或麦克风提问（视觉多轮）"
            )
        else:
            self._clear_vision_image(notify_backend=True)
            self._set_vision_image_status(
                "图片模式已关闭：视觉识别图片输入栏已禁用"
            )
        self.update_image_mode_button_state(enabled)
        self.updateGeometry()

    def _finish_visual_image_reply(self, reason: str) -> None:
        if self.visual_image_reply_pending:
            logger.debug("Visual image reply finished by {}", reason)
        self.visual_image_reply_pending = False

    def _handle_visual_image_reply_error(self, data: dict[str, Any]) -> None:
        if not self.visual_image_reply_pending:
            return
        self.visual_image_reply_pending = False
        message = str(data.get("message") or "").strip()
        self._set_vision_image_status(
            f"识图失败：{message}" if message else "识图失败",
            error=True,
        )

    def update_game_vision_button_state(self, enabled: bool) -> None:
        self.game_vision_button.setStyleSheet(
            CONNECTED_BUTTON_STYLE if enabled else INACTIVE_BUTTON_STYLE
        )
        self._apply_game_window_binding_ui()

    def _game_window_binding_label(self) -> str:
        if self.is_director_mode:
            return "由主播端绑定"
        return self.game_vision_window_title or "未绑定游戏窗口"

    def _apply_game_window_binding_ui(self) -> None:
        """把当前绑定状态反映到按钮 tooltip, 让主播知道点击可配置。"""
        state_text = "已开启" if self.game_vision_enabled else "未开启"
        self.game_vision_button.setToolTip(
            "点击：打开游戏识图菜单（开关 / 绑定窗口）\n"
            f"当前状态：{state_text}\n"
            f"当前截取范围：{self._game_window_binding_label()}"
        )

    def _show_game_vision_button_menu(self) -> None:
        self._show_game_window_bind_menu(self.game_vision_button.rect().bottomLeft())

    def _show_game_window_bind_menu(self, pos) -> None:
        menu = QMenu(self)
        state_text = "已开启" if self.game_vision_enabled else "未开启"
        state_action = menu.addAction(f"当前状态：{state_text}")
        state_action.setEnabled(False)
        header = menu.addAction(f"当前截取：{self._game_window_binding_label()}")
        header.setEnabled(False)
        menu.addSeparator()
        toggle_action = menu.addAction(
            (
                "关闭游戏识图"
                if self.game_vision_enabled
                else (
                    "开启游戏识图"
                    if self.is_director_mode or self.game_vision_window_title
                    else "绑定并开启游戏识图"
                )
            )
        )
        toggle_action.setEnabled(True)
        menu.addSeparator()
        bind_action = menu.addAction("绑定/更换游戏窗口…")
        unbind_action = menu.addAction("解除绑定（停止识图截图）")
        bind_action.setEnabled(not self.is_director_mode)
        unbind_action.setEnabled(
            not self.is_director_mode and bool(self.game_vision_window_title)
        )
        chosen = menu.exec_(self.game_vision_button.mapToGlobal(pos))
        if chosen is toggle_action:
            self.toggle_game_vision()
        elif chosen is bind_action:
            self._prompt_bind_game_window()
        elif chosen is unbind_action:
            self._unbind_game_window()

    def _prompt_bind_game_window(self) -> bool:
        if sys.platform != "win32":
            self._set_vision_image_status("窗口绑定仅在 Windows 可用", error=True)
            return False
        windows = _win_enumerate_windows()
        if not windows:
            self._set_vision_image_status("未找到可绑定的窗口", error=True)
            return False
        titles: list[str] = []
        seen: set[str] = set()
        for _, wtitle in windows:
            if wtitle not in seen:
                seen.add(wtitle)
                titles.append(wtitle)
        current = self.game_vision_window_title
        start_index = titles.index(current) if current in titles else 0
        title, ok = QInputDialog.getItem(
            self,
            "绑定游戏窗口",
            "选择游戏识图只截取的窗口：",
            titles,
            start_index,
            False,
        )
        if not ok or not title:
            return False
        self.game_vision_window_title = title
        save_game_window_binding(title)
        self._apply_game_window_binding_ui()
        self._set_vision_image_status(f"游戏识图已绑定窗口：{title}")
        logger.info("Bound game vision window: {}", title)
        return True

    def _unbind_game_window(self) -> None:
        was_enabled = self.game_vision_enabled
        self.game_vision_window_title = None
        save_game_window_binding(None)
        if was_enabled:
            self.game_vision_enabled = False
            self.game_vision_request_id = None
            self.game_vision_cold_reply_pending = False
            self._cancel_game_vision_cold_timer("game-window-unbound")
            self.update_game_vision_button_state(False)
            self.send_console_message(
                "game-vision-mode",
                enabled=False,
                cold_idle_seconds=GAME_VISION_COLD_IDLE_SECONDS,
            )
        else:
            self._apply_game_window_binding_ui()
        self._set_vision_image_status(
            "游戏识图已解除绑定，已停止截图；请重新绑定游戏窗口后再开启"
        )
        logger.info("Unbound game vision window")

    def update_paint_button_state(self, enabled: bool) -> None:
        self.paint_button.setText(
            PAINT_BUTTON_ACTIVE_TEXT if enabled else PAINT_BUTTON_INACTIVE_TEXT
        )
        self.paint_button.setStyleSheet(
            CONNECTED_BUTTON_STYLE if enabled else INACTIVE_BUTTON_STYLE
        )

    def toggle_paint(self) -> None:
        self._set_paint_global_enabled(not self.paint_enabled)
        self.send_console_message("paint-mode", enabled=self.paint_enabled)

    def _ensure_paint_window(self) -> PaintWindow:
        if self.paint_window is None:
            self.paint_window = PaintWindow()
            self.paint_window.closed.connect(self._on_paint_window_closed)
        return self.paint_window

    def _open_paint_window(self) -> PaintWindow:
        window = self._ensure_paint_window()
        window.show()
        window.raise_()
        return window

    def _close_paint_window(self) -> None:
        if self.paint_window is None:
            return
        try:
            self.paint_window.closed.disconnect(self._on_paint_window_closed)
        except (TypeError, RuntimeError):
            pass
        self.paint_window.close()
        self.paint_window = None

    def _on_paint_window_closed(self) -> None:
        self.paint_window = None
        self.paint_enabled = False
        self.update_paint_button_state(False)
        if not self._closing_console:
            self.send_console_message("paint-mode", enabled=False)

    def _set_paint_global_enabled(self, enabled: bool) -> None:
        self.paint_enabled = bool(enabled)
        if self.paint_enabled:
            self._open_paint_window()
        else:
            self._close_paint_window()
        self.update_paint_button_state(self.paint_enabled)

    def toggle_game_vision(self) -> None:
        if self.is_director_mode:
            self.send_console_message(
                "game-vision-mode",
                enabled=not self.game_vision_enabled,
                cold_idle_seconds=GAME_VISION_COLD_IDLE_SECONDS,
            )
            return

        target_enabled = not self.game_vision_enabled
        if target_enabled and not self.game_vision_window_title:
            self._set_vision_image_status(
                "游戏识图需要先绑定游戏窗口；为避免直播泄露，不会截取整个屏幕",
                error=True,
            )
            if not self._prompt_bind_game_window():
                self.game_vision_enabled = False
                self.update_game_vision_button_state(False)
                self._cancel_game_vision_cold_timer("game-vision-unbound")
                return

        self.game_vision_enabled = target_enabled
        if not self.game_vision_enabled:
            self.game_vision_request_id = None
            self.game_vision_cold_reply_pending = False
            self._cancel_game_vision_cold_timer("game-vision-disabled")
        else:
            self._schedule_game_vision_cold_timer("game-vision-enabled")
        self.update_game_vision_button_state(self.game_vision_enabled)
        self.send_console_message(
            "game-vision-mode",
            enabled=self.game_vision_enabled,
            cold_idle_seconds=GAME_VISION_COLD_IDLE_SECONDS,
        )
        state_text = (
            f"游戏识图已开启：只截取绑定窗口「{self.game_vision_window_title}」，"
            f"主播开口时会立刻识别，{GAME_VISION_COLD_IDLE_SECONDS}秒没说话会自动冷场识图"
            if self.game_vision_enabled
            else "\u6e38\u620f\u8bc6\u56fe\u5df2\u5173\u95ed"
        )
        self._set_vision_image_status(state_text)
        logger.info("Game vision capture {}", "enabled" if self.game_vision_enabled else "disabled")

    def _game_vision_cold_delay_ms(self) -> int:
        return int(max(1, GAME_VISION_COLD_IDLE_SECONDS) * 1000)

    def _game_vision_cold_available(self) -> bool:
        return (
            self.game_vision_enabled
            and bool(self.game_vision_window_title)
            and not self.is_director_mode
            and self._microphone_input_allowed()
        )

    def _cancel_game_vision_cold_timer(self, reason: str) -> None:
        if self.game_vision_cold_timer.isActive():
            self.game_vision_cold_timer.stop()
            logger.debug("Game vision cold timer cancelled by {}", reason)

    def _schedule_game_vision_cold_timer(self, reason: str) -> None:
        if self._closing_console:
            self._cancel_game_vision_cold_timer(f"{reason}:closing")
            return
        if not self.game_vision_enabled or self.is_director_mode:
            self._cancel_game_vision_cold_timer(reason)
            return
        if not self.game_vision_window_title:
            self._cancel_game_vision_cold_timer(f"{reason}:unbound")
            return
        if self.game_vision_cold_reply_pending:
            logger.debug(
                "Game vision cold timer not scheduled by {} because reply is pending",
                reason,
            )
            return
        if not self._microphone_input_allowed():
            self._cancel_game_vision_cold_timer(f"{reason}:input-disabled")
            return
        self.game_vision_cold_timer.start(self._game_vision_cold_delay_ms())
        logger.debug(
            "Game vision cold timer scheduled by {} ({}s)",
            reason,
            GAME_VISION_COLD_IDLE_SECONDS,
        )

    def _has_game_vision_cold_blocker(self) -> bool:
        if self.game_vision_request_id or self.game_vision_cold_reply_pending:
            return True
        if self.microphone_paused_playback or self.link_microphone_paused_playback:
            return True
        live2d_window = self.live2d_window
        if live2d_window is None:
            return False
        if getattr(live2d_window, "active_turn_id", None):
            return True
        is_playing_audio = getattr(live2d_window, "is_playing_audio", None)
        if callable(is_playing_audio):
            try:
                return bool(is_playing_audio())
            except Exception as exc:
                logger.debug("Could not inspect Live2D audio state: {}", exc)
        return False

    def _finish_game_vision_cold_reply(self, reason: str) -> None:
        if self.game_vision_cold_reply_pending:
            logger.debug("Game vision cold reply finished by {}", reason)
        self.game_vision_cold_reply_pending = False
        self._schedule_game_vision_cold_timer(reason)

    def _handle_game_vision_cold_timeout(self) -> None:
        if not self.game_vision_enabled or self.is_director_mode:
            return
        if not self._microphone_input_allowed():
            self._schedule_game_vision_cold_timer("cold-timeout-input-disabled")
            return
        if self._has_game_vision_cold_blocker():
            self._schedule_game_vision_cold_timer("cold-timeout-busy")
            return
        if not self._send_game_vision_cold_input():
            self._schedule_game_vision_cold_timer("cold-timeout-send-failed")

    def _send_game_vision_cold_input(self) -> bool:
        if not self._game_vision_cold_available():
            return False

        image_payload = self._capture_game_vision_screenshot_payload()
        if not image_payload:
            return False

        request_id = uuid.uuid4().hex
        provider = self._current_vision_model_provider()
        metadata: dict[str, Any] = {
            "input_source": "game_vision_cold",
            "human_name": "\u6e38\u620f\u753b\u9762",
            "display_label": "\u6e38\u620f\u8bc6\u56fe\u51b7\u573a",
            "capture_source": "game_vision_cold",
            "game_vision_request_id": request_id,
            "game_vision_cold_idle_seconds": GAME_VISION_COLD_IDLE_SECONDS,
            "game_vision_reply_mode": "vision_model",
            "skip_history": True,
            "skip_memory": True,
            "skip_story_match": True,
            "proactive_speak": True,
        }
        if provider:
            metadata["vision_model_provider"] = provider

        payload = {
            "type": "text-input",
            "request_id": request_id,
            "text": GAME_VISION_COLD_PROMPT,
            "images": [image_payload],
            "metadata": metadata,
        }
        self.backend_client.send_json(payload)
        self.game_vision_cold_reply_pending = True
        self._set_vision_image_status(
            f"\u6e38\u620f\u8bc6\u56fe\u51b7\u573a\u5df2\u622a\u5c4f\uff0c\u6b63\u5728\u8ba9\u89c6\u89c9\u6a21\u578b\u56de\u590d\uff08{provider or 'default'}\uff09"
        )
        logger.info(
            "Sent game vision cold visual input: request_id={} provider={} bytes={}",
            request_id,
            provider,
            len(image_payload["data"]),
        )
        return True

    def _encode_game_vision_screenshot(self, pixmap: QPixmap) -> bytes | None:
        if pixmap.isNull():
            return None

        max_edges = (
            GAME_VISION_SCREENSHOT_MAX_EDGE,
            1280,
            960,
            720,
        )
        for max_edge in max_edges:
            working = pixmap
            if max(pixmap.width(), pixmap.height()) > max_edge:
                working = pixmap.scaled(
                    QSize(max_edge, max_edge),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )

            byte_array = QByteArray()
            buffer = QBuffer(byte_array)
            if not buffer.open(QIODevice.WriteOnly):
                continue
            try:
                if not working.save(
                    buffer,
                    "JPG",
                    GAME_VISION_SCREENSHOT_JPEG_QUALITY,
                ):
                    continue
            finally:
                buffer.close()

            image_bytes = bytes(byte_array)
            if image_bytes and len(image_bytes) <= VISION_IMAGE_MAX_BYTES:
                return image_bytes

        return None

    def _capture_game_vision_screenshot_payload(self) -> dict[str, str] | None:
        bound_title = self.game_vision_window_title
        if not bound_title:
            self._set_vision_image_status(
                "识图失败：未绑定游戏窗口；为避免直播泄露，不会截取整个屏幕",
                error=True,
            )
            logger.warning("Game vision capture skipped: no bound game window")
            return None

        # 已绑定窗口: 只截该窗口，屏幕其它内容 (浏览器/桌面/通知) 不入画。
        # 找不到或截不到时宁可跳过，也绝不回退整屏，避免直播泄露。
        hwnd = _win_find_window_by_title(bound_title)
        if not hwnd:
            self._set_vision_image_status(
                f"识图失败：未找到绑定窗口「{bound_title}」，已跳过截图",
                error=True,
            )
            logger.warning(
                "Game vision: bound window not found: {}", bound_title
            )
            return None
        pixmap = _win_grab_window_pixmap(hwnd)
        if pixmap is None or pixmap.isNull():
            self._set_vision_image_status(
                f"识图失败：无法截取窗口「{bound_title}」，已跳过截图",
                error=True,
            )
            logger.warning(
                "Game vision: failed to grab bound window: {}", bound_title
            )
            return None
        capture_label = f"window:{bound_title}"

        image_bytes = self._encode_game_vision_screenshot(pixmap)
        if not image_bytes:
            self._set_vision_image_status(
                "识图失败：截图过大或无法编码",
                error=True,
            )
            logger.warning("Game vision capture failed: screenshot encoding failed")
            return None

        encoded = base64.b64encode(image_bytes).decode("ascii")
        logger.info(
            "Game vision capture ok: source={} bytes={}",
            capture_label,
            len(image_bytes),
        )
        return {
            "source": "screen",
            "data": f"data:image/jpeg;base64,{encoded}",
            "mime_type": "image/jpeg",
            "name": f"game-screen-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jpg",
        }

    def _send_game_vision_capture(self) -> None:
        if not self.game_vision_enabled or self.is_director_mode:
            return

        image_payload = self._capture_game_vision_screenshot_payload()
        if not image_payload:
            return

        request_id = uuid.uuid4().hex
        self.game_vision_request_id = request_id
        provider = self._current_vision_model_provider()
        metadata: dict[str, Any] = {
            "capture_source": "game_vision",
            "game_vision_request_id": request_id,
        }
        if provider:
            metadata["vision_model_provider"] = provider

        self.backend_client.send_json(
            {
                "type": "game-vision-capture",
                "mic_source": "local",
                "request_id": request_id,
                "images": [image_payload],
                "metadata": metadata,
            }
        )
        self._set_vision_image_status(
            f"\u6e38\u620f\u8bc6\u56fe\u5df2\u622a\u5c4f\uff0c\u6b63\u5728\u8bc6\u522b\uff08{provider or 'default'}\uff09"
        )
        logger.info(
            "Sent game vision screenshot: request_id={} provider={} bytes={}",
            request_id,
            provider,
            len(image_payload["data"]),
        )

    def _handle_game_vision_state(self, data: dict[str, Any]) -> None:
        state = str(data.get("state") or "")
        provider = data.get("provider")
        message = str(data.get("message") or "")
        if state == "completed":
            text = (
                f"\u6e38\u620f\u8bc6\u56fe\u5b8c\u6210\uff08{provider}\uff09"
                if provider
                else "\u6e38\u620f\u8bc6\u56fe\u5b8c\u6210"
            )
            self._set_vision_image_status(text)
        elif state == "started":
            self._set_vision_image_status("\u6e38\u620f\u8bc6\u56fe\u5df2\u53d1\u9001\u540e\u7aef")
        elif state == "timeout":
            self._set_vision_image_status(
                "\u8bc6\u56fe\u5931\u8d25\uff1a\u6e38\u620f\u8bc6\u56fe\u8d85\u65f6\uff0c\u672c\u8f6e\u5c06\u4ec5\u6839\u636e\u8bed\u97f3\u56de\u590d",
                error=True,
            )
        elif state == "error":
            self._set_vision_image_status(
                f"\u8bc6\u56fe\u5931\u8d25\uff1a{message}"
                if message
                else "\u8bc6\u56fe\u5931\u8d25",
                error=True,
            )

    def _handle_paint_state(self, data: dict[str, Any]) -> None:
        state = str(data.get("state") or "")
        prompt = str(data.get("prompt") or "")
        if state == "started":
            self._set_paint_global_enabled(True)
            self._open_paint_window().set_loading(prompt)
            if self.image_mode_enabled:
                self._set_vision_image_status(
                    f"画图中：{prompt}" if prompt else "画图中"
                )
            return
        if state == "completed":
            self._set_paint_global_enabled(True)
            try:
                self._open_paint_window().set_image(str(data.get("image") or ""), prompt)
                if self.image_mode_enabled:
                    self._set_vision_image_status(
                        f"画图完成：{prompt}" if prompt else "画图完成"
                    )
            except Exception as exc:
                logger.warning("Failed to display paint image: {}", exc)
                message = f"图片显示失败：{exc}"
                self._open_paint_window().set_error(message)
                self._set_vision_image_status(f"画图失败：{message}", error=True)
            return
        if state == "error":
            self._set_paint_global_enabled(True)
            message = str(data.get("message") or "画图失败")
            self._open_paint_window().set_error(message)
            self._set_vision_image_status(
                f"画图失败：{message}" if message else "画图失败",
                error=True,
            )
            return

    def _create_director_metric_list(self) -> QWidget:
        metric_list = QWidget()
        metric_list.setFixedWidth(DIRECTOR_METRIC_PANEL_WIDTH)
        metric_list.setFixedHeight(150)
        self.director_metric_layout = QVBoxLayout()
        self.director_metric_layout.setContentsMargins(0, 0, 0, 0)
        self.director_metric_layout.setSpacing(6)
        metric_list.setLayout(self.director_metric_layout)

        for key, title in DIRECTOR_METRIC_FIELDS:
            row = DirectorMetricRow(
                key=key,
                title=title,
                on_changed=self.send_director_metrics,
                on_drag_release=self._move_director_metric_by_drop,
            )
            self.director_metric_widgets[key] = {
                "title": title,
                "checkbox": row.checkbox,
                "input": row.value_input,
                "row": row,
            }
            self.director_metric_order.append(key)
            self.director_metric_layout.addWidget(row)
        return metric_list

    def _rebuild_director_metric_rows(self) -> None:
        if not self.director_metric_layout:
            return

        while self.director_metric_layout.count():
            item = self.director_metric_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        for key in self.director_metric_order:
            row = self.director_metric_widgets.get(key, {}).get("row")
            if row:
                self.director_metric_layout.addWidget(row)

    def _move_director_metric_by_drop(self, key: str, drop_global_y: int) -> None:
        if key not in self.director_metric_order:
            return

        next_order = [item_key for item_key in self.director_metric_order if item_key != key]
        target_index = 0
        for item_key in next_order:
            row = self.director_metric_widgets.get(item_key, {}).get("row")
            if not row:
                continue
            if drop_global_y > row.mapToGlobal(row.rect().center()).y():
                target_index += 1

        next_order.insert(target_index, key)
        if next_order == self.director_metric_order:
            self._rebuild_director_metric_rows()
            return

        self.director_metric_order = next_order
        self._rebuild_director_metric_rows()
        self.send_director_metrics()

    def _director_metric_value(self, text: str) -> int | float | str:
        text = text.strip()
        if not text:
            return 0
        try:
            number = float(text)
        except ValueError:
            return text
        if number.is_integer():
            return int(number)
        return number

    def _director_metric_payload(self) -> tuple[list[dict[str, Any]], list[str]]:
        if not self.director_metric_order:
            return [], []

        metrics = []
        order = []
        for row_index, key in enumerate(self.director_metric_order):
            widgets = self.director_metric_widgets.get(key)
            if not widgets:
                continue

            value_input = widgets["input"]
            checkbox = widgets["checkbox"]
            raw_value = str(value_input.text()).strip()
            order.append(key)
            metrics.append(
                {
                    "key": key,
                    "title": widgets["title"],
                    "order": row_index,
                    "enabled": bool(checkbox.isChecked()),
                    "value": self._director_metric_value(raw_value),
                    "raw_value": raw_value,
                }
            )

        return metrics, order

    def send_director_metrics(self, *_args: Any) -> None:
        if not self.is_director_mode:
            return

        metrics, order = self._director_metric_payload()
        self.send_console_message(
            "director-metrics",
            metrics=metrics,
            order=order,
        )

    def _validated_backend_url_from_inputs(self) -> str | None:
        host = self.ws_host_input.text().strip() or self.default_ws_host
        port = self.ws_port_input.text().strip() or self.default_ws_port
        if "://" in host:
            parsed = urlparse(host)
            host = parsed.hostname or self.default_ws_host
            if parsed.port:
                port = str(parsed.port)

        try:
            port_number = int(port)
        except ValueError:
            self.handle_backend_error(f"Invalid WebSocket port: {port}")
            return None
        if not 1 <= port_number <= 65535:
            self.handle_backend_error(f"WebSocket port out of range: {port}")
            return None

        self.ws_host_input.setText(host)
        self.ws_port_input.setText(str(port_number))
        return self.build_backend_url(host, str(port_number))

    def reconnect_backend_from_inputs(self) -> None:
        url = self._validated_backend_url_from_inputs()
        if not url:
            return
        self.url = url
        self.backend_client.reconnect(url)
        self.update_backend_state(False)
        logger.info("Reconnect requested from console controls: {}", url)

    def reconnect_backend_with_default(self) -> None:
        self.ws_host_input.setText(self.default_ws_host)
        self.ws_port_input.setText(self.default_ws_port)
        self.reconnect_backend_from_inputs()

    def _remember_successful_backend_url(self) -> None:
        backend_client = getattr(self, "backend_client", None)
        url = backend_client.current_url() if backend_client else self.url
        try:
            host, port = self.split_backend_url(url)
            normalized_url = self.build_backend_url(host, port)
        except Exception as exc:
            logger.warning("Cannot remember invalid backend WebSocket URL: {}", exc)
            return

        self.url = normalized_url
        self.default_ws_host = host
        self.default_ws_port = port
        self.ws_host_input.setText(host)
        self.ws_port_input.setText(port)
        save_backend_ws_url(normalized_url)

    def send_console_message(self, action: str, **payload: Any) -> None:
        message = {
            "type": "console-message",
            "action": action,
            **payload,
        }
        self.backend_client.send_json(message)
        logger.info("Sent console-message: {}", message)

    def send_anchor_text_input(self) -> None:
        text = self.anchor_text_input.text().strip()
        if not text:
            return

        message = {
            "type": "text-input",
            "text": text,
            "metadata": {
                "input_source": "anchor_text",
                "human_name": ANCHOR_HUMAN_NAME,
                "display_label": "\u4e3b\u64ad\u53d1\u8a00",
            },
        }
        image_attached = self._attach_pending_vision_image(message)
        vision_context_reused = (
            self.image_mode_enabled
            and self.visual_image_context_active
            and not image_attached
        )
        backend_connected = self.backend_client.is_connected()
        self.backend_client.send_json(message)
        if image_attached and backend_connected:
            provider = self._current_vision_model_label()
            image_name = (self.pending_vision_image or {}).get("name")
            self.visual_image_context_active = True
            self.visual_image_reply_pending = True
            self._clear_vision_image(notify_backend=False)
            self._set_vision_image_status(
                f"图片已随主播发言发送（{provider} / 视觉多轮）"
            )
            logger.info(
                "Sent pending visual image with anchor text: name={} provider={} mode={}",
                image_name,
                (message.get("metadata") or {}).get("vision_model_provider"),
                VISION_CONTEXT_MODE_PERSISTENT,
            )
        elif vision_context_reused and backend_connected:
            self.visual_image_reply_pending = True
            self._set_vision_image_status(
                f"已用上一张图片随主播发言继续识图（{self._current_vision_model_label()} / 视觉多轮）"
            )
        self.anchor_text_input.clear()
        self.anchor_text_input.setFocus()
        logger.info("Sent anchor text-input: {}", truncate_data(message))

    def _report_link_microphone_fault(self, faulted: bool, reason: str) -> None:
        if self.is_director_mode:
            return
        if self._reported_link_microphone_faulted == faulted:
            return
        self._reported_link_microphone_faulted = faulted
        self.send_console_message(
            "link-microphone-fault",
            faulted=faulted,
            reason=reason,
        )

    def _message_turn_id(self, data: dict[str, Any]) -> str | None:
        return data.get("turn_id") or data.get("request_id")

    def show_project_config(self) -> None:
        if self.project_config_dialog is None:
            self.project_config_dialog = ProjectConfigDialog(self)
            self.project_config_dialog.save_requested.connect(
                self._save_project_config
            )
            self.project_config_dialog.test_requested.connect(
                self._test_project_config
            )
        self.backend_client.send_json({"type": "project-config-request"})
        self.project_config_dialog.show()
        self.project_config_dialog.raise_()
        self.project_config_dialog.activateWindow()

    def _save_project_config(self, payload: dict[str, Any]) -> None:
        if not self.backend_client.is_connected():
            if self.project_config_dialog is not None:
                self.project_config_dialog.show_config_error("后端未连接")
            return
        self.backend_client.send_json(
            {"type": "project-config-update", **payload}
        )

    def _test_project_config(self, payload: dict[str, Any]) -> None:
        if not self.backend_client.is_connected():
            if self.project_config_dialog is not None:
                self.project_config_dialog.show_config_error("后端未连接")
            return
        self.backend_client.send_json(
            {"type": "project-config-test", **payload}
        )

    def show_performance_monitor(self) -> None:
        if self.performance_monitor is None:
            self.performance_monitor = PerformanceMonitorDialog(self)
            self.performance_monitor.average_reset_requested.connect(
                self._handle_performance_average_reset_requested
            )
        self._sync_performance_monitor_status()
        self.performance_monitor.show()
        self.performance_monitor.raise_()
        self.performance_monitor.activateWindow()

    def _start_performance_speech(self, source: str) -> None:
        if not self.is_streamer_mode:
            return
        if source in self._performance_speech_started_at:
            return
        self._performance_speech_started_at[source] = time.monotonic()
        source_name = "连线主播" if source == "link" else "主播"
        self._append_performance_log(f"检测到{source_name}人声，正在接收语音…")

    def _cancel_performance_speech(self, source: str) -> None:
        started_at = self._performance_speech_started_at.pop(source, None)
        if started_at is not None:
            source_name = "连线主播" if source == "link" else "主播"
            self._append_performance_log(
                f"{source_name}语音过短，本次输入已忽略。"
            )

    def _finish_performance_speech(self, source: str) -> str | None:
        if not self.is_streamer_mode:
            return None
        ended_at = time.monotonic()
        started_at = self._performance_speech_started_at.pop(source, None)
        performance_id = uuid.uuid4().hex
        metrics: dict[str, float] = {}
        if started_at is not None:
            metrics["user_speech_seconds"] = max(0.0, ended_at - started_at)
        input_source = "link_microphone" if source == "link" else "mic"
        self._performance_pending[performance_id] = {
            "source": input_source,
            "speech_end_monotonic": ended_at,
            "metrics": metrics,
        }
        while len(self._performance_pending) > 20:
            self._performance_pending.popitem(last=False)
        source_name = "连线主播" if source == "link" else "主播"
        self._append_performance_log(
            f"{source_name}说话结束，正在汇总 ASR 识别结果…"
        )
        return performance_id

    def _bind_performance_turn(self, data: dict[str, Any]) -> None:
        if not self.is_streamer_mode:
            return
        turn_id = str(self._message_turn_id(data) or "").strip()
        performance_id = str(data.get("performance_id") or "").strip()
        if not turn_id or not performance_id:
            return
        state = self._performance_pending.pop(performance_id, None)
        if not state:
            logger.debug(
                "Performance transcription has no pending frontend timing: turn_id={} performance_id={}",
                turn_id,
                performance_id,
            )
            return
        self._performance_turn_state[turn_id] = state
        self._trim_performance_turn_state()
        self._set_performance_state("thinking")
        self._update_performance_monitor(turn_id)
        source_name = (
            "连线主播"
            if state.get("source") == "link_microphone"
            else "主播"
        )
        self._append_performance_log(
            f"{source_name} ASR 识别完成。"
        )

    def _handle_performance_stage(self, data: dict[str, Any]) -> None:
        if not self.is_streamer_mode:
            return
        turn_id = str(self._message_turn_id(data) or "").strip()
        stage = str(data.get("stage") or "").strip()
        if not turn_id or not stage:
            return
        if stage in {
            "knowledge-start",
            "web-search-start",
            "llm-start",
        }:
            self._set_performance_state("thinking")
        state = self._performance_turn_state.get(turn_id)
        if not state:
            return
        seen_stages = state.setdefault("performance_stages_seen", set())
        if stage in seen_stages:
            return
        seen_stages.add(stage)
        message = {
            "knowledge-start": "开始检索知识库…",
            "knowledge-complete": "知识库检索完成。",
            "web-search-start": "开始联网搜索…",
            "web-search-complete": "联网搜索完成。",
            "llm-start": "开始调用大模型，等待首字…",
            "llm-first-token": "大模型已返回首字，等待首句…",
            "llm-first-sentence": "大模型已返回首句。",
            "llm-complete": "大模型完整输出完成。",
            "llm-failed": "大模型调用失败。",
            "tts-start": "开始调用 TTS，等待首音…",
            "tts-first-audio": "TTS 已返回首音，准备播放…",
            "tts-complete": "TTS 全部生成完成，等待播放结束…",
        }.get(stage)
        if message:
            self._append_performance_log(message)

    def _handle_backend_performance_metrics(self, data: dict[str, Any]) -> None:
        if not self.is_streamer_mode:
            return
        turn_id = str(self._message_turn_id(data) or "").strip()
        source = str(data.get("input_source") or "").strip()
        if not turn_id or source not in {"mic", "link_microphone"}:
            return
        state = self._performance_turn_state.setdefault(
            turn_id,
            {"source": source, "metrics": {}},
        )
        state["source"] = source
        backend_metrics = data.get("metrics")
        if isinstance(backend_metrics, dict):
            state.setdefault("metrics", {}).update(backend_metrics)
        self._trim_performance_turn_state()
        self._update_performance_monitor(turn_id)

    def _handle_performance_audio_started(self, data: dict[str, Any]) -> None:
        if not self.is_streamer_mode:
            return
        turn_id = str(self._message_turn_id(data) or "").strip()
        if turn_id:
            self._set_performance_state("speaking")
        state = self._performance_turn_state.get(turn_id)
        if not turn_id or not state or state.get("audio_started_monotonic") is not None:
            return
        started_at = time.monotonic()
        state["audio_started_monotonic"] = started_at
        speech_end = state.get("speech_end_monotonic")
        if speech_end is not None:
            state.setdefault("metrics", {})[
                "speech_end_to_audio_start_seconds"
            ] = max(0.0, started_at - float(speech_end))
        self._append_performance_log(
            "Live Streaming Agent 回复开始播放。"
        )
        self._update_performance_monitor(turn_id)

    def _complete_performance_playback(
        self,
        turn_id: str | None,
        *,
        skipped: bool = False,
    ) -> None:
        if not self.is_streamer_mode or not turn_id or skipped:
            return
        state = self._performance_turn_state.get(str(turn_id))
        if not state or state.get("completed"):
            return
        audio_started = state.get("audio_started_monotonic")
        if audio_started is None:
            return
        completed_at = time.monotonic()
        metrics = state.setdefault("metrics", {})
        metrics["ai_playback_seconds"] = max(
            0.0,
            completed_at - float(audio_started),
        )
        speech_end = state.get("speech_end_monotonic")
        if speech_end is not None:
            metrics["speech_end_to_playback_complete_seconds"] = max(
                0.0,
                completed_at - float(speech_end),
            )
        state["completed"] = True
        self._append_performance_log("本轮回复已完整播放。")
        self._update_performance_monitor(str(turn_id))

    def _trim_performance_turn_state(self) -> None:
        while len(self._performance_turn_state) > PERFORMANCE_MONITOR_MAX_TURNS:
            oldest_turn_id = next(iter(self._performance_turn_state))
            self._performance_turn_state.pop(oldest_turn_id, None)

    def _update_performance_monitor(self, turn_id: str) -> None:
        if not self.performance_monitor:
            return
        state = self._performance_turn_state.get(turn_id)
        if not state:
            return
        self.performance_monitor.update_turn(
            turn_id,
            state.get("metrics") or {},
            source=state.get("source"),
            completed=bool(state.get("completed")),
        )
        if self.is_streamer_mode:
            self.backend_client.send_json(
                {
                    "type": "performance-monitor-sync",
                    "kind": "turn",
                    "turn_id": turn_id,
                    "metrics": dict(state.get("metrics") or {}),
                    "source": state.get("source"),
                    "completed": bool(state.get("completed")),
                }
            )

    def _append_performance_log(self, message: str) -> None:
        if not self.is_streamer_mode or self.performance_monitor is None:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.performance_monitor.append_log(message, timestamp)
        self.backend_client.send_json(
            {
                "type": "performance-monitor-sync",
                "kind": "log",
                "timestamp": timestamp,
                "message": message,
            }
        )

    def _send_performance_monitor_snapshot(self) -> None:
        if not self.is_streamer_mode or self.performance_monitor is None:
            return
        self.backend_client.send_json(
            {
                "type": "performance-monitor-sync",
                "kind": "snapshot",
                "snapshot": self.performance_monitor.export_snapshot(),
            }
        )

    def _set_performance_state(self, state: str, *, sync: bool = True) -> None:
        normalized = str(state or "").strip().lower()
        if normalized not in PERFORMANCE_STATE_LABELS:
            normalized = "idle"
        if normalized == self.performance_state:
            self._sync_performance_monitor_status()
            return
        self.performance_state = normalized
        self._sync_performance_monitor_status()
        if sync and self.is_streamer_mode:
            self.backend_client.send_json(
                {
                    "type": "performance-monitor-sync",
                    "kind": "status",
                    "status": normalized,
                }
            )

    def _handle_performance_average_reset_requested(self) -> None:
        self.backend_client.send_json(
            {
                "type": "performance-monitor-sync",
                "kind": "reset",
            }
        )

    def _handle_performance_monitor_sync(self, data: dict[str, Any]) -> None:
        if self.performance_monitor is None:
            return
        kind = str(data.get("kind") or "").strip()
        if kind == "reset":
            self.performance_monitor.reset_averages()
            return
        if not self.is_director_mode:
            return
        if kind == "turn":
            self.performance_monitor.update_turn(
                str(self._message_turn_id(data) or ""),
                data.get("metrics") or {},
                source=data.get("source"),
                completed=bool(data.get("completed")),
            )
        elif kind == "log":
            self.performance_monitor.append_log(
                str(data.get("message") or ""),
                str(data.get("timestamp") or "") or None,
            )
        elif kind == "status":
            self._set_performance_state(
                str(data.get("status") or "idle"),
                sync=False,
            )
        elif kind == "snapshot" and isinstance(data.get("snapshot"), dict):
            snapshot = data["snapshot"]
            self.performance_monitor.load_snapshot(snapshot)
            self._set_performance_state(
                str(snapshot.get("status") or "idle"),
                sync=False,
            )

    def _send_frontend_playback_complete(
        self,
        turn_id: str | dict[str, Any] | None = None,
    ) -> None:
        if self.is_director_mode:
            logger.debug(
                "Director mode does not acknowledge frontend playback completion: {}",
                turn_id,
            )
            return

        payload = {"type": "frontend-playback-complete"}
        if isinstance(turn_id, dict):
            payload.update({key: value for key, value in turn_id.items() if value is not None})
        elif turn_id:
            payload["turn_id"] = turn_id
        self._complete_performance_playback(
            str(payload.get("turn_id") or "") or None,
            skipped=bool(payload.get("skipped")),
        )
        if not payload.get("skipped"):
            self._set_performance_state("idle")
        self.backend_client.send_json(payload)

    def toggle_microphone(self) -> None:
        self.send_console_message("microphone-toggle")

    def toggle_link_microphone(self) -> None:
        self.send_console_message("link-microphone-toggle")

    def schedule_link_human_name_autosave(self, *_args: Any) -> None:
        self._cancel_link_human_name_detection("manual-edit")
        self.link_human_name_save_timer.start()

    def handle_link_human_name_autosave(self) -> None:
        self._commit_link_human_name(normalize_empty=False)

    def handle_link_human_name_changed(self) -> None:
        self.link_human_name_save_timer.stop()
        self._commit_link_human_name(normalize_empty=True)

    def _commit_link_human_name(self, *, normalize_empty: bool) -> None:
        raw_name = self.link_human_name_input.text()
        name = raw_name.strip()
        if not name:
            if not normalize_empty:
                return
            name = DEFAULT_LINK_HUMAN_NAME
        if normalize_empty and raw_name != name:
            self.link_human_name_input.setText(name)
        if name == self.link_human_name:
            return
        self.link_human_name = name
        self.send_console_message("link-microphone-name", name=name)

    def _apply_link_human_name_auto_tooltip(self) -> None:
        self.link_human_name_auto_button.setToolTip(
            "\u4ec5\u4f7f\u7528\u672c\u673a DouyinBarrage WebSocket \u6293\u5305\u8bc6\u522b\u8fde\u7ebf\u4e3b\u64ad\n"
            f"\u672c\u673a\u6293\u5305\u5730\u5740\uff1a{self._local_douyin_barrage_ws_url()}\n"
            "\u53f3\u952e\uff1a\u67e5\u770b\u4e0a\u6b21\u6293\u5305\u65e5\u5fd7"
        )

    def _show_link_human_name_auto_menu(self, pos) -> None:
        menu = QMenu(self)
        ws_action = menu.addAction(
            f"\u672c\u673a\u6293\u5305\uff1a{self._local_douyin_barrage_ws_url()}"
        )
        ws_action.setEnabled(False)
        menu.addSeparator()
        detect_action = menu.addAction("\u5f00\u59cb\u81ea\u52a8\u8bc6\u522b")
        debug_action = menu.addAction("\u67e5\u770b\u4e0a\u6b21\u672c\u673a\u6293\u5305\u65e5\u5fd7")
        detect_action.setEnabled(not self.link_human_name_detect_pending)
        chosen = menu.exec_(self.link_human_name_auto_button.mapToGlobal(pos))
        if chosen is detect_action:
            self.handle_link_human_name_auto_clicked()
        elif chosen is debug_action:
            self._show_link_anchor_probe_debug()

    def _show_link_anchor_probe_debug(self) -> None:
        text = self.link_human_name_last_probe_text.strip()
        path_text = (
            str(self.link_human_name_last_probe_debug_path)
            if self.link_human_name_last_probe_debug_path
            else "\u672a\u4fdd\u5b58"
        )
        if not text:
            text = "\u8fd8\u6ca1\u6709\u7b2c\u4e00\u65b9\u6848\u6293\u53d6\u6587\u672c"
        box = QMessageBox(self)
        box.setWindowTitle("\u7b2c\u4e00\u65b9\u6848\u6293\u53d6\u6587\u672c")
        box.setText(f"\u4fdd\u5b58\u8def\u5f84\uff1a{path_text}")
        box.setDetailedText(text[:12000])
        box.exec_()

    def _prompt_link_name_roi(self) -> None:
        current = _link_name_target_roi()
        value, ok = QInputDialog.getText(
            self,
            "\u8bbe\u7f6e\u6635\u79f0\u622a\u56fe\u533a\u57df ROI",
            (
                "\u8f93\u5165 left,top,right,bottom\uff080~1\uff09\uff0c"
                "\u4f8b\u5982\uff1a0.58,0.70,0.73,0.90"
            ),
            QLineEdit.Normal,
            _format_link_name_roi(current),
        )
        if not ok:
            return
        ratio = _parse_link_name_roi(value)
        if not ratio:
            QMessageBox.warning(
                self,
                "\u65e0\u6548 ROI",
                "\u8bf7\u8f93\u5165 4 \u4e2a 0~1 \u7684\u6570\uff0c\u4e14 left < right\u3001top < bottom\u3002",
            )
            return
        save_link_name_roi(ratio)
        self._apply_link_human_name_auto_tooltip()
        QMessageBox.information(
            self,
            "\u5df2\u4fdd\u5b58 ROI",
            f"\u5f53\u524d\u6635\u79f0\u622a\u56fe\u533a\u57df\uff1a{_format_link_name_roi(ratio)}",
        )

    def handle_link_human_name_auto_clicked(self) -> None:
        self.link_human_name_save_timer.stop()
        self.link_human_name_detect_pending = True
        self.link_human_name_detect_request_id = uuid.uuid4().hex
        self.link_human_name_vision_requested = False
        self.link_human_name_local_ws_stop = threading.Event()
        self.link_human_name_local_ws_started_at = time.monotonic()
        self.link_human_name_backend_detect_started_at = time.monotonic()
        self.link_human_name_backend_detect_attempts = 0
        self.link_human_name_auto_button.setEnabled(False)
        self.link_human_name_auto_button.setText("\u8bc6\u522b\u4e2d")
        self.link_human_name_auto_button.setToolTip(
            "\u6b63\u5728\u7b49\u5f85\u672c\u673a DouyinBarrage WebSocket \u539f\u59cb PK/\u8fde\u7ebf\u6570\u636e\uff1b"
            "\u4e0d\u4f7f\u7528 OCR/\u89c6\u89c9\u515c\u5e95\uff0c\u8d85\u8fc7 60 \u79d2\u4f1a\u81ea\u52a8\u5931\u8d25"
        )
        self.link_human_name_detect_timer.start()
        self._start_local_douyin_barrage_link_probe_thread(
            self.link_human_name_detect_request_id
        )

    def _local_douyin_barrage_ws_url(self) -> str:
        return (
            os.environ.get("LOCAL_DOUYIN_BARRAGE_WS_URL")
            or DEFAULT_LOCAL_BARRAGE_WS_URL
        )

    def _start_local_douyin_barrage_link_probe_thread(
        self,
        request_id: str | None,
    ) -> None:
        stop_event = self.link_human_name_local_ws_stop or threading.Event()
        duration = max(
            1.0,
            min(55.0, (LINK_HUMAN_NAME_DETECT_TIMEOUT_MS - 1_000) / 1000),
        )
        ws_url = self._local_douyin_barrage_ws_url()
        search_roots = [
            PROJECT_ROOT,
            LIVE_FRONTEND_BUNDLE_ROOT,
            PROJECT_ROOT.parent,
        ]

        def run_local_probe_worker() -> None:
            payload: dict[str, Any]
            try:
                payload = detect_local_link_anchor_candidate(
                    ws_url=ws_url,
                    duration_seconds=duration,
                    request_id=request_id,
                    stop_event=stop_event,
                    search_roots=search_roots,
                    autostart=True,
                )
            except Exception as exc:
                payload = {
                    "request_id": request_id,
                    "found": False,
                    "candidate": None,
                    "source": "local_barrage_raw_protobuf",
                    "ws_url": ws_url,
                    "errors": [str(exc)],
                    "sources": [],
                    "done": True,
                }
                logger.exception("Local DouyinBarrage link probe failed")
            self.link_human_name_local_ws_finished.emit(payload)

        threading.Thread(
            target=run_local_probe_worker,
            name="local-douyin-barrage-link-probe",
            daemon=True,
        ).start()

    def _request_link_human_name_backend_detection(self, reason: str) -> bool:
        logger.debug(
            "Link human name backend detection disabled; only local WebSocket capture is used: reason={} attempts={}",
            reason,
            self.link_human_name_backend_detect_attempts,
        )
        return False

    def _schedule_link_human_name_backend_retry(self, data: dict[str, Any]) -> bool:
        logger.debug(
            "Link human name backend retry disabled; only local WebSocket capture is used: {}",
            truncate_data(data, 120),
        )
        return False

    def _handle_link_human_name_backend_retry(self) -> None:
        if not self.link_human_name_detect_pending:
            return
        self._finish_link_human_name_detection(
            "\u8bc6\u522b\u5931\u8d25\uff1a\u5f53\u524d\u7248\u672c\u53ea\u4f7f\u7528\u672c\u673a WebSocket \u6293\u5305\uff0c\u4e0d\u4f7f\u7528\u540e\u7aef\u91cd\u8bd5/OCR/\u89c6\u89c9\u515c\u5e95\u3002",
            error=True,
        )

    def _cancel_link_human_name_detection(self, reason: str) -> None:
        if not self.link_human_name_detect_pending:
            return
        self.link_human_name_detect_pending = False
        self.link_human_name_detect_request_id = None
        self.link_human_name_vision_requested = False
        if self.link_human_name_local_ws_stop:
            self.link_human_name_local_ws_stop.set()
        self.link_human_name_local_ws_stop = None
        self.link_human_name_local_ws_started_at = 0.0
        self.link_human_name_backend_retry_timer.stop()
        self.link_human_name_backend_detect_started_at = 0.0
        self.link_human_name_backend_detect_attempts = 0
        self.link_human_name_detect_timer.stop()
        self.link_human_name_auto_button.setEnabled(True)
        self.link_human_name_auto_button.setText(LINK_HUMAN_NAME_AUTO_BUTTON_TEXT)
        self._apply_link_human_name_auto_tooltip()
        logger.info("Link human name detection cancelled: {}", reason)

    def _finish_link_human_name_detection(
        self,
        message: str,
        *,
        error: bool = False,
    ) -> None:
        self.link_human_name_detect_pending = False
        self.link_human_name_detect_request_id = None
        self.link_human_name_vision_requested = False
        if self.link_human_name_local_ws_stop:
            self.link_human_name_local_ws_stop.set()
        self.link_human_name_local_ws_stop = None
        self.link_human_name_local_ws_started_at = 0.0
        self.link_human_name_backend_retry_timer.stop()
        self.link_human_name_backend_detect_started_at = 0.0
        self.link_human_name_backend_detect_attempts = 0
        self.link_human_name_detect_timer.stop()
        self.link_human_name_auto_button.setEnabled(True)
        self.link_human_name_auto_button.setText(
            LINK_HUMAN_NAME_FAILED_BUTTON_TEXT if error else LINK_HUMAN_NAME_AUTO_BUTTON_TEXT
        )
        self.link_human_name_auto_button.setToolTip(message)
        log = logger.warning if error else logger.info
        log("Link human name detection finished: {}", message)

    def _handle_link_human_name_detect_timeout(self) -> None:
        if not self.link_human_name_detect_pending:
            return
        self._finish_link_human_name_detection(
            "\u8bc6\u522b\u5931\u8d25\uff1a\u8d85\u8fc7 60 \u79d2\u672a\u62ff\u5230\u8fde\u7ebf\u4e3b\u64ad\u540d\uff0c\u53ef\u4ee5\u91cd\u65b0\u70b9\u51fb\u81ea\u52a8\u8bc6\u522b",
            error=True,
        )

    def _link_anchor_probe_duration(self) -> float:
        raw_value = os.environ.get("LINK_ANCHOR_PROBE_DURATION", "")
        try:
            value = float(raw_value) if raw_value else LINK_HUMAN_NAME_PROBE_DURATION_SECONDS
        except ValueError:
            value = LINK_HUMAN_NAME_PROBE_DURATION_SECONDS
        return max(1.0, min(value, 55.0))

    def _link_anchor_probe_headless(self) -> bool:
        value = os.environ.get("LINK_ANCHOR_PROBE_HEADLESS", "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _start_link_anchor_probe_thread(
        self,
        request_id: str | None,
        window_title: str | None,
        window_hwnd: int | None = None,
    ) -> None:
        web_url = None
        cdp_url = None
        duration = self._link_anchor_probe_duration()
        headless = self._link_anchor_probe_headless()

        def run_probe_worker() -> None:
            payload: dict[str, Any] = {
                "request_id": request_id,
                "found": False,
                "candidate": None,
                "sources": [],
                "errors": [],
            }
            try:
                try:
                    from open_llm_vtuber.link_anchor_probe import run_probe
                except ModuleNotFoundError:
                    src_root = PROJECT_ROOT / "src"
                    if str(src_root) not in sys.path:
                        sys.path.insert(0, str(src_root))
                    from open_llm_vtuber.link_anchor_probe import run_probe

                result = run_probe(
                    url=web_url,
                    cdp_url=cdp_url,
                    live_companion_window=window_title,
                    live_companion_hwnd=window_hwnd,
                    duration_seconds=duration,
                    headless=headless,
                )
                candidate_payload = None
                if result.candidate:
                    candidate_payload = {
                        "nickname": result.candidate.nickname,
                        "display_id": result.candidate.display_id,
                        "sec_uid": result.candidate.sec_uid,
                        "room_id": result.candidate.room_id,
                        "source": result.candidate.source,
                        "confidence": result.candidate.confidence,
                        "path": result.candidate.path,
                        "raw_text": result.candidate.raw_text,
                    }
                payload.update(
                    {
                        "found": result.found,
                        "candidate": candidate_payload,
                        "sources": result.sources,
                        "errors": result.errors,
                        "elapsed_seconds": result.elapsed_seconds,
                        "done": True,
                    }
                )
            except Exception as exc:
                payload["errors"] = [str(exc)]
                payload["done"] = True
                logger.exception("Link anchor probe failed")
            self.link_human_name_probe_finished.emit(payload)

        threading.Thread(
            target=run_probe_worker,
            name="link-anchor-probe",
            daemon=True,
        ).start()

    def _save_link_anchor_probe_debug(self, data: dict[str, Any]) -> None:
        try:
            LINK_NAME_PROBE_DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
            request_id = str(data.get("request_id") or "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            json_path = LINK_NAME_PROBE_DEBUG_ROOT / (
                f"link_anchor_probe_{timestamp}_{request_id[:8]}.json"
            )
            latest_json_path = LINK_NAME_PROBE_DEBUG_ROOT / "link_anchor_probe_latest.json"
            text_path = LINK_NAME_PROBE_DEBUG_ROOT / "link_anchor_probe_latest.txt"

            with json_path.open("w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
            with latest_json_path.open("w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)

            sections: list[str] = []
            for index, source in enumerate(data.get("sources") or [], start=1):
                if not isinstance(source, dict):
                    continue
                text = str(source.get("text_full") or source.get("text_sample") or "")
                sections.append(
                    "\n".join(
                        [
                            f"source #{index}: {source.get('source')}",
                            f"status: {source.get('status')}",
                            f"window_title: {source.get('window_title')}",
                            f"window_hwnd: {source.get('window_hwnd')}",
                            f"candidate_count: {source.get('candidate_count')}",
                            "text:",
                            text,
                        ]
                    )
                )
            debug_text = "\n\n---\n\n".join(sections).strip()
            if not debug_text:
                debug_text = json.dumps(data, ensure_ascii=False, indent=2)
            with text_path.open("w", encoding="utf-8") as file:
                file.write(debug_text)

            self.link_human_name_last_probe_text = debug_text
            self.link_human_name_last_probe_debug_path = text_path
            logger.info(
                "Saved link anchor probe debug text: path={} request_id={}",
                text_path,
                request_id,
            )
        except Exception as exc:
            logger.warning("Failed to save link anchor probe debug text: {}", exc)

    def _handle_link_anchor_probe_finished(self, data: dict[str, Any]) -> None:
        if not self.link_human_name_detect_pending:
            logger.debug("Ignoring link anchor probe result without pending request")
            return
        request_id = data.get("request_id")
        if (
            self.link_human_name_detect_request_id
            and request_id
            and request_id != self.link_human_name_detect_request_id
        ):
            logger.debug("Ignoring stale link anchor probe result: {}", request_id)
            return

        self._save_link_anchor_probe_debug(data)
        candidate = data.get("candidate")
        if isinstance(candidate, dict):
            name = normalize_link_human_name_candidate(
                candidate.get("nickname")
                or candidate.get("display_id")
                or candidate.get("sec_uid")
            )
            if name:
                if self._is_bound_live_room_owner_candidate(name):
                    logger.info(
                        "Link anchor probe candidate rejected because it is the bound live room owner: name={} title={}",
                        name,
                        self.link_name_window_title,
                    )
                else:
                    logger.info(
                        "Link anchor probe candidate accepted: name={} source={} confidence={} raw={}",
                        name,
                        candidate.get("source"),
                        candidate.get("confidence"),
                        truncate_data(candidate, 120),
                    )
                    self._apply_auto_link_human_name(
                        name,
                        str(candidate.get("source") or "link_anchor_probe"),
                        candidate.get("confidence"),
                    )
                    return

        logger.info(
            "Link anchor probe result ignored; only local WebSocket capture is used: {}",
            truncate_data(data, 120),
        )
        self._finish_link_human_name_detection(
            "\u8bc6\u522b\u5931\u8d25\uff1a\u5f53\u524d\u7248\u672c\u53ea\u63a5\u53d7\u672c\u673a DouyinBarrage WebSocket \u6293\u5305\u7ed3\u679c\uff0c\u4e0d\u4f7f\u7528 Web/OCR/\u89c6\u89c9\u515c\u5e95\u3002",
            error=True,
        )

    def _handle_local_douyin_barrage_link_probe_finished(
        self,
        data: dict[str, Any],
    ) -> None:
        if not self.link_human_name_detect_pending:
            logger.debug("Ignoring local DouyinBarrage probe result without pending request")
            return
        request_id = data.get("request_id")
        if (
            self.link_human_name_detect_request_id
            and request_id
            and request_id != self.link_human_name_detect_request_id
        ):
            logger.debug("Ignoring stale local DouyinBarrage probe result: {}", request_id)
            return

        self.link_human_name_local_ws_stop = None
        self.link_human_name_local_ws_started_at = 0.0
        self._save_link_anchor_probe_debug(data)

        candidate = data.get("candidate")
        if isinstance(candidate, dict):
            name = normalize_link_human_name_candidate(
                candidate.get("name")
                or candidate.get("nickname")
                or candidate.get("display_id")
                or candidate.get("sec_uid")
            )
            if name:
                logger.info(
                    "Local DouyinBarrage candidate accepted: name={} source={} confidence={} raw={}",
                    name,
                    candidate.get("source"),
                    candidate.get("confidence"),
                    truncate_data(candidate, 120),
                )
                self._apply_auto_link_human_name(
                    name,
                    str(candidate.get("source") or "local_barrage_raw_protobuf"),
                    candidate.get("confidence"),
                )
                return

        logger.info(
            "Local DouyinBarrage probe found nothing; finishing without OCR/vision fallback: {}",
            truncate_data(data, 160),
        )
        errors = [str(item) for item in (data.get("errors") or []) if item]
        messages = int(data.get("messages") or 0)
        raw_messages = int(data.get("raw_payload_messages") or 0)
        status = ""
        for source in data.get("sources") or []:
            if isinstance(source, dict) and source.get("status"):
                status = str(source.get("status") or "")
                break

        if status == "connection_failed" or messages <= 0:
            detail = "\uff1b".join(errors[:2])
            if detail:
                detail = f"\n{detail}"
            self._finish_link_human_name_detection(
                "\u8bc6\u522b\u5931\u8d25\uff1a\u672c\u673a DouyinBarrage WebSocket "
                f"{data.get('ws_url') or DEFAULT_LOCAL_BARRAGE_WS_URL} \u672a\u8fde\u4e0a\u6216\u6ca1\u6709\u6d88\u606f\u3002"
                "\u5982\u679c\u5f39\u51fa UAC \u7ba1\u7406\u5458\u6388\u6743\uff0c\u8bf7\u5141\u8bb8\uff1b"
                "\u6216\u624b\u52a8\u4ee5\u7ba1\u7406\u5458\u8fd0\u884c\u5305\u5185 _internal\\barrage_grab\\WssBarrageServer.exe \u540e\u91cd\u8bd5\u3002"
                f"{detail}",
                error=True,
            )
            return

        self._finish_link_human_name_detection(
            "\u8bc6\u522b\u5931\u8d25\uff1a\u5df2\u8fde\u4e0a\u672c\u673a DouyinBarrage WebSocket\uff0c"
            f"\u4f46\u5728 {messages} \u6761\u6d88\u606f\u4e2d\u6ca1\u6709\u89e3\u51fa\u8fde\u7ebf\u4e3b\u64ad\u540d"
            f"\uff08raw payload \u6d88\u606f {raw_messages} \u6761\uff09\u3002"
            "\u8bf7\u786e\u8ba4\u5df2\u8fdb\u5165 PK/\u8fde\u7ebf\uff0c\u4e14\u672c\u673a\u6293\u5305\u5de5\u5177\u6536\u5230 Type=201/PayloadBase64\u3002",
            error=True,
        )

    def _handle_link_human_name_detect_response(self, data: dict[str, Any]) -> None:
        if not self.link_human_name_detect_pending:
            logger.debug("Ignoring link human name detect response without pending request")
            return
        request_id = data.get("request_id")
        if (
            self.link_human_name_detect_request_id
            and request_id
            and request_id != self.link_human_name_detect_request_id
        ):
            logger.debug(
                "Ignoring stale link human name detect response: {}",
                request_id,
            )
            return

        candidate = normalize_link_human_name_candidate(data.get("candidate"))
        if candidate and self._is_bound_live_room_owner_candidate(candidate):
            logger.info(
                "Link human name backend candidate rejected because it is the bound live room owner: candidate={} title={}",
                candidate,
                self.link_name_window_title,
            )
            candidate = None
        if candidate:
            self._apply_auto_link_human_name(
                candidate,
                str(data.get("source") or "barrage_structured"),
                data.get("confidence"),
            )
            return

        logger.info(
            "Link human name backend response ignored; only local WebSocket capture is used: {}",
            truncate_data(data, 160),
        )
        self._finish_link_human_name_detection(
            "\u8bc6\u522b\u5931\u8d25\uff1a\u5f53\u524d\u7248\u672c\u53ea\u4f7f\u7528\u672c\u673a WebSocket \u6293\u5305\u8bc6\u522b\u8fde\u7ebf\u4e3b\u64ad\u540d\uff0c\u5df2\u5ffd\u7565\u540e\u7aef/OCR/\u89c6\u89c9\u8fd4\u56de\u3002",
            error=True,
        )

    def _apply_auto_link_human_name(
        self,
        name: str,
        source: str,
        confidence: Any = None,
    ) -> None:
        candidate = normalize_link_human_name_candidate(name)
        if not candidate:
            self._finish_link_human_name_detection(
                "\u8bc6\u522b\u5931\u8d25\uff1a\u672c\u673a WebSocket \u8fd4\u56de\u7684\u8fde\u7ebf\u4e3b\u64ad\u5019\u9009\u65e0\u6548\uff0c\u5df2\u5ffd\u7565\u3002",
                error=True,
            )
            return
        if (
            not str(source or "").startswith("local_barrage_raw_protobuf")
            and self._is_bound_live_room_owner_candidate(candidate)
        ):
            logger.info(
                "Rejected auto link human name because it is the bound live room owner: candidate={} title={}",
                candidate,
                self.link_name_window_title,
            )
            self._finish_link_human_name_detection(
                "\u8bc6\u522b\u5230\u7684\u662f\u5f53\u524d\u76f4\u64ad\u95f4\u4e3b\u64ad\uff0c\u5df2\u5ffd\u7565\uff1b"
                "\u8bf7\u786e\u8ba4\u672c\u673a\u6293\u5305\u5de5\u5177\u6536\u5230\u7684\u662f\u5f53\u524d PK/\u8fde\u7ebf\u6570\u636e\uff0c\u6216\u624b\u52a8\u8f93\u5165\u3002",
                error=True,
            )
            return

        old_state = self.link_human_name_input.blockSignals(True)
        try:
            self.link_human_name_input.setText(candidate)
        finally:
            self.link_human_name_input.blockSignals(old_state)
        self._commit_link_human_name(normalize_empty=True)
        confidence_text = ""
        if confidence is not None:
            confidence_text = f" confidence={confidence}"
        self._finish_link_human_name_detection(
            f"\u5df2\u81ea\u52a8\u8bc6\u522b\u8fde\u7ebf\u4e3b\u64ad\uff1a{candidate} "
            f"({source}{confidence_text})"
        )

    def _link_name_window_binding_label(self) -> str:
        if not self.link_name_window_title:
            return "\u672a\u7ed1\u5b9a\u8fde\u7ebf\u7a97\u53e3"
        if self.link_name_window_hwnd:
            return f"{self.link_name_window_title} (hwnd={self.link_name_window_hwnd})"
        return self.link_name_window_title

    def _bound_live_room_owner_name(self) -> str | None:
        title = self.link_name_window_title or ""
        match = re.search(r"(.+?)\u7684\u6296\u97f3\u76f4\u64ad\u95f4", title)
        if not match:
            match = re.search(r"(.+?)\u7684\u76f4\u64ad\u95f4", title)
        if not match:
            return None
        owner = match.group(1).strip()
        return owner or None

    def _is_bound_live_room_owner_candidate(self, candidate: str | None) -> bool:
        candidate = str(candidate or "").strip()
        owner = self._bound_live_room_owner_name()
        if not candidate or not owner:
            return False
        candidate_folded = candidate.casefold()
        owner_folded = owner.casefold()
        owner_without_suffix = re.sub(r"[\d\u2070-\u209f]+", "", owner).casefold()
        return (
            candidate_folded in owner_folded
            or owner_folded in candidate_folded
            or (
                bool(owner_without_suffix)
                and (
                    candidate_folded in owner_without_suffix
                    or owner_without_suffix in candidate_folded
                )
            )
        )

    def _resolve_link_name_window_hwnd(self) -> int | None:
        title = self.link_name_window_title or ""
        hwnd = self.link_name_window_hwnd
        if hwnd and _win_window_is_usable(hwnd, title):
            actual_title = _win_get_window_title(hwnd)
            if actual_title:
                self.link_name_window_title = actual_title
            return hwnd
        if hwnd:
            logger.warning(
                "Bound link name hwnd is unavailable; refusing ambiguous title fallback: title={} hwnd={}",
                title,
                hwnd,
            )
            return None
        if title:
            matches = [
                candidate_hwnd
                for candidate_hwnd, wtitle in _win_enumerate_windows()
                if wtitle == title or title in wtitle or wtitle in title
            ]
            if len(matches) == 1:
                self.link_name_window_hwnd = matches[0]
                return matches[0]
            if matches:
                logger.warning(
                    "Bound link name title is ambiguous; please bind exact hwnd: title={} matches={}",
                    title,
                    matches,
                )
        return None

    def _iter_link_name_window_hwnd_candidates(self) -> list[int]:
        title = (self.link_name_window_title or "").strip()
        seen: set[int] = set()
        candidates: list[int] = []

        def add(hwnd: int | None) -> None:
            if not hwnd or hwnd in seen:
                return
            if not _win_window_is_usable(hwnd, title):
                return
            seen.add(hwnd)
            candidates.append(hwnd)

        resolved = self._resolve_link_name_window_hwnd()
        if resolved:
            add(resolved)
            return candidates
        if title and not self.link_name_window_hwnd:
            for hwnd, wtitle in _win_enumerate_windows():
                if (
                    wtitle == title
                    or title in wtitle
                    or wtitle in title
                ):
                    add(hwnd)
        return candidates

    def _capture_link_name_roi_pixmap(
        self,
        purpose: str,
    ) -> tuple[int, QPixmap, dict[str, Any]] | None:
        title = self.link_name_window_title
        if not title:
            return None
        hwnd_candidates = self._iter_link_name_window_hwnd_candidates()
        if not hwnd_candidates:
            logger.warning(
                "Link name {} capture skipped: bound window not found: title={} hwnd={}",
                purpose,
                title,
                self.link_name_window_hwnd,
            )
            return None

        roi = _link_name_target_roi()
        blank_hwnds: list[int] = []
        for hwnd in hwnd_candidates:
            for method, use_print_window in (
                ("print_window", True),
                ("screen", False),
            ):
                pixmap = _win_grab_window_pixmap(
                    hwnd,
                    use_print_window=use_print_window,
                )
                if pixmap is None or pixmap.isNull():
                    logger.debug(
                        "Link name {} capture candidate failed: title={} hwnd={} method={}",
                        purpose,
                        title,
                        hwnd,
                        method,
                    )
                    continue

                cropped, roi_info = _crop_pixmap_by_ratio(pixmap, roi)
                stats = _pixmap_content_stats(cropped)
                roi_info = {
                    **roi_info,
                    "hwnd": hwnd,
                    "capture_method": method,
                    "content": stats,
                }
                logger.info(
                    "Link name {} capture candidate: title={} hwnd={} method={} roi={} stats={}",
                    purpose,
                    title,
                    hwnd,
                    method,
                    roi_info.get("ratio"),
                    stats,
                )
                if stats.get("is_blank"):
                    blank_hwnds.append(hwnd)
                    continue

                return hwnd, cropped, roi_info

        logger.warning(
            "Link name {} capture skipped: all ROI candidates were blank or failed: title={} hwnds={} blank_hwnds={}",
            purpose,
            title,
            hwnd_candidates,
            sorted(set(blank_hwnds)),
        )
        return None

    def _prompt_bind_link_name_window(self) -> bool:
        if sys.platform != "win32":
            self._finish_link_human_name_detection(
                "\u8fde\u7ebf\u7a97\u53e3\u7ed1\u5b9a\u4ec5\u5728 Windows \u53ef\u7528",
                error=True,
            )
            return False
        all_windows = _win_enumerate_windows()
        windows = [
            (hwnd, title)
            for hwnd, title in all_windows
            if "\u76f4\u64ad\u4f34\u4fa3" in title
        ]
        if not windows:
            self._finish_link_human_name_detection(
                "\u672a\u627e\u5230\u53ef\u7ed1\u5b9a\u7684\u76f4\u64ad\u4f34\u4fa3\u7a97\u53e3",
                error=True,
            )
            return False

        labels: list[str] = []
        label_to_window: dict[str, tuple[int, str, tuple[int, int, int, int] | None]] = {}
        for index, (hwnd, wtitle) in enumerate(windows, start=1):
            rect = _win_window_rect(hwnd)
            size_text = ""
            if rect:
                left, top, right, bottom = rect
                size_text = f", {max(0, right - left)}x{max(0, bottom - top)}"
            label = f"{index}. {wtitle} [hwnd={hwnd}{size_text}]"
            labels.append(label)
            label_to_window[label] = (hwnd, wtitle, rect)
        current = self.link_name_window_title or self.game_vision_window_title
        start_index = 0
        exact_index: int | None = None
        title_index: int | None = None
        for index, label in enumerate(labels):
            hwnd, wtitle, _rect = label_to_window[label]
            if hwnd == self.link_name_window_hwnd:
                exact_index = index
                break
            if title_index is None and current and wtitle == current:
                title_index = index
        if exact_index is not None:
            start_index = exact_index
        elif title_index is not None:
            start_index = title_index
        label, ok = QInputDialog.getItem(
            self,
            "\u7ed1\u5b9a\u8fde\u7ebf\u4e3b\u64ad\u7a97\u53e3",
            "\u9009\u62e9\u542b\u6709\u8fde\u7ebf\u4e3b\u64ad\u540d\u5b57\u7684\u76f4\u64ad\u4f34\u4fa3\u7a97\u53e3\uff1a",
            labels,
            start_index,
            False,
        )
        if not ok or not label:
            return False
        hwnd, title, rect = label_to_window[label]
        self.link_name_window_title = title
        self.link_name_window_hwnd = hwnd
        rect_payload = None
        if rect:
            left, top, right, bottom = rect
            rect_payload = {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "width": max(0, right - left),
                "height": max(0, bottom - top),
            }
        save_link_name_window_binding(title, hwnd, {"rect": rect_payload} if rect_payload else None)
        self._apply_link_human_name_auto_tooltip()
        logger.info("Bound link name window: title={} hwnd={} rect={}", title, hwnd, rect_payload)
        return True

    def _try_local_link_human_name_detection(self, reason: str) -> None:
        logger.info(
            "Skipping link human name OCR/vision fallback; only local WebSocket capture is used: {}",
            reason,
        )
        self._finish_link_human_name_detection(
            "\u8bc6\u522b\u5931\u8d25\uff1a\u5f53\u524d\u7248\u672c\u53ea\u4f7f\u7528\u672c\u673a DouyinBarrage WebSocket \u6293\u5305\uff0c\u4e0d\u4f7f\u7528 OCR/\u89c6\u89c9\u515c\u5e95\u3002",
            error=True,
        )

    def _request_link_human_name_vision_detection(self, reason: str) -> bool:
        logger.info(
            "Link human name vision detection disabled; only local WebSocket capture is used: {}",
            reason,
        )
        return False

    def _capture_link_name_screenshot_payload(self) -> dict[str, str] | None:
        title = self.link_name_window_title
        if not title:
            return None
        capture = self._capture_link_name_roi_pixmap("vision")
        if not capture:
            return None
        hwnd, pixmap, roi_info = capture
        try:
            LINK_NAME_VISION_CAPTURE_ROOT.mkdir(parents=True, exist_ok=True)
            image_path = (
                LINK_NAME_VISION_CAPTURE_ROOT
                / f"link_name_vision_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
            )
            if pixmap.save(str(image_path), "PNG"):
                logger.info(
                    "Saved link name vision capture: path={} title={} hwnd={} roi={}",
                    image_path,
                    title,
                    hwnd,
                    roi_info,
                )
        except Exception as exc:
            logger.debug("Failed to save link name vision capture: {}", exc)
        image_bytes = self._encode_game_vision_screenshot(pixmap)
        if not image_bytes:
            logger.warning("Link name screenshot skipped: encode failed")
            return None
        data_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")
        return {
            "source": "screen",
            "data": data_url,
            "mime_type": "image/jpeg",
        }

    def _ocr_link_name_bound_window(self) -> str:
        title = self.link_name_window_title
        if not title:
            return ""
        capture = self._capture_link_name_roi_pixmap("ocr")
        if not capture:
            return ""
        hwnd, pixmap, roi_info = capture

        LINK_NAME_OCR_ROOT.mkdir(parents=True, exist_ok=True)
        image_path = (
            LINK_NAME_OCR_ROOT
            / f"link_name_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
        )
        if not pixmap.save(str(image_path), "PNG"):
            logger.warning("Link name OCR skipped: failed to save screenshot")
            return ""
        logger.info(
            "Saved link name OCR capture: path={} title={} hwnd={} roi={}",
            image_path,
            title,
            hwnd,
            roi_info,
        )

        text = self._ocr_image_with_pytesseract(image_path)
        if text:
            return text
        return self._ocr_image_with_tesseract_cli(image_path)

    def _accessibility_text_link_name_bound_window(self) -> str:
        title = self.link_name_window_title
        if not title:
            return ""
        hwnd = self._resolve_link_name_window_hwnd()
        if not hwnd:
            logger.warning(
                "Link name UIA skipped: bound window not found: title={} hwnd={}",
                title,
                self.link_name_window_hwnd,
            )
            return ""

        text = self._accessibility_text_with_uiautomation(hwnd)
        if text:
            return text
        return self._accessibility_text_with_pywinauto(hwnd)

    def _accessibility_text_with_uiautomation(self, hwnd: int) -> str:
        try:
            import uiautomation as auto  # type: ignore
        except Exception:
            return ""
        try:
            root = auto.ControlFromHandle(hwnd)
        except Exception as exc:
            logger.debug("uiautomation root lookup failed: {}", exc)
            return ""

        values: list[str] = []
        seen: set[str] = set()

        def collect(control: Any, depth: int = 0) -> None:
            if depth > 4 or len(values) >= 80:
                return
            try:
                name = str(getattr(control, "Name", "") or "").strip()
            except Exception:
                name = ""
            if name and name not in seen:
                seen.add(name)
                values.append(name)
            try:
                children = control.GetChildren()
            except Exception:
                children = []
            for child in children[:30]:
                collect(child, depth + 1)

        collect(root)
        return "\n".join(values)

    def _accessibility_text_with_pywinauto(self, hwnd: int) -> str:
        try:
            from pywinauto import Desktop  # type: ignore
        except Exception:
            return ""
        try:
            window = Desktop(backend="uia").window(handle=hwnd)
            controls = window.descendants()
        except Exception as exc:
            logger.debug("pywinauto UIA lookup failed: {}", exc)
            return ""

        values: list[str] = []
        seen: set[str] = set()
        for control in controls[:120]:
            try:
                texts = control.texts()
            except Exception:
                texts = []
            for text in texts:
                value = str(text or "").strip()
                if value and value not in seen:
                    seen.add(value)
                    values.append(value)
        return "\n".join(values)

    def _ocr_image_with_pytesseract(self, image_path: Path) -> str:
        try:
            import pytesseract  # type: ignore
        except Exception:
            return ""
        for lang in ("chi_sim+eng", "eng"):
            try:
                text = pytesseract.image_to_string(str(image_path), lang=lang)
            except Exception as exc:
                logger.debug("pytesseract OCR failed lang={} error={}", lang, exc)
                continue
            if text.strip():
                return text
        return ""

    def _ocr_image_with_tesseract_cli(self, image_path: Path) -> str:
        executable = shutil.which("tesseract")
        if not executable:
            return ""
        for lang in ("chi_sim+eng", "eng"):
            try:
                result = subprocess.run(
                    [executable, str(image_path), "stdout", "-l", lang],
                    capture_output=True,
                    text=True,
                    timeout=8,
                    check=False,
                )
            except Exception as exc:
                logger.debug("tesseract OCR failed lang={} error={}", lang, exc)
                continue
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
        return ""

    def _candidate_from_link_name_text(self, text: str) -> str | None:
        if not text:
            return None
        lines = [
            line.strip()
            for line in re.split(r"[\r\n]+", text)
            if line and line.strip()
        ]
        relation_pattern = re.compile(
            r"(?:^|[\s\r\n])(?:\u4e0e|\u548c)\s*"
            r"(?P<name>[\w\u4e00-\u9fff\u00b7\u2022._-]{1,24})\s*"
            r"(?:PK\u4e2d|pk\u4e2d|PK|pk|\u8fde\u7ebf\u4e2d|\u8fde\u9ea6\u4e2d)"
        )
        for match in relation_pattern.finditer(text):
            candidate = normalize_link_human_name_candidate(match.group("name"))
            if candidate and not self._is_bound_live_room_owner_candidate(candidate):
                return candidate
        label_pattern = re.compile(
            r"(?:\u5bf9\u65b9|\u5609\u5bbe|\u4e3b\u64ad|\u7528\u6237|\u6635\u79f0)[:：\s]+(.+)"
        )
        for line in lines[:40]:
            label_match = label_pattern.search(line)
            value = label_match.group(1) if label_match else line
            for part in re.split(r"\s{2,}|[|｜]", value):
                candidate = normalize_link_human_name_candidate(part)
                if candidate and not any(
                    word.lower() == candidate.lower()
                    for word in LINK_NAME_REJECT_WORDS
                ) and not self._is_bound_live_room_owner_candidate(candidate):
                    return candidate
        return None

    def start_microphone(self) -> None:
        self.microphone_requested = True
        if self.is_director_mode:
            self.update_microphone_button_state(True)
            logger.info(
                "Director mode received microphone-on sync; local microphone stays disabled"
            )
            return

        if self.microphone_enabled:
            return

        try:
            device_info = QAudioDeviceInfo.defaultInputDevice()
            audio_format = self._create_microphone_format(device_info)
            self.mic_worker = MicrophoneVadWorker(
                on_speech_candidate_start=(
                    self.microphone_speech_candidate_started.emit
                ),
                on_speech_start=self.microphone_speech_started.emit,
                on_speech_cancelled=self.microphone_speech_cancelled.emit,
                on_audio_detected=self.microphone_audio_detected.emit,
                on_audio_confirmed=self.microphone_audio_confirmed.emit,
                on_error=self.microphone_error.emit,
            )
            self.mic_worker.start()
            self.mic_audio_input = QAudioInput(device_info, audio_format, self)
            self.mic_audio_input.stateChanged.connect(
                self.handle_microphone_audio_state_changed
            )
            self.mic_device = self.mic_audio_input.start()
            if not self.mic_device:
                raise RuntimeError("Failed to start microphone input device")
            self.mic_device.readyRead.connect(self.handle_microphone_ready_read)
            self.mic_format = audio_format
            self.mic_byte_buffer.clear()
            self.mic_started_at = time.monotonic()
            self.mic_last_data_at = self.mic_started_at
            self.microphone_enabled = True
            self.microphone_faulted = False
            self.update_microphone_button_state(True)
            logger.info(
                "Display microphone started: device={} sample_rate={} channels={} sample_size={}",
                device_info.deviceName(),
                audio_format.sampleRate(),
                audio_format.channelCount(),
                audio_format.sampleSize(),
            )
        except Exception as exc:
            self.microphone_faulted = True
            self.stop_microphone(keep_requested=True)
            logger.error(
                "Display microphone start failed; watchdog will retry while requested: {}",
                exc,
            )

    def stop_microphone(self, *, keep_requested: bool = False) -> None:
        if not keep_requested:
            self.microphone_requested = False
            self.microphone_faulted = False
        if self.mic_device:
            try:
                self.mic_device.readyRead.disconnect(self.handle_microphone_ready_read)
            except (TypeError, RuntimeError):
                pass
            self.mic_device = None

        if self.mic_audio_input:
            try:
                self.mic_audio_input.stateChanged.disconnect(
                    self.handle_microphone_audio_state_changed
                )
            except (TypeError, RuntimeError):
                pass
            self.mic_audio_input.stop()
            self.mic_audio_input = None

        if self.mic_worker:
            self.mic_worker.stop()
            self.mic_worker.join(timeout=1.0)
            self.mic_worker = None

        self.mic_format = None
        self.mic_byte_buffer.clear()
        self.handle_microphone_speech_cancelled()
        self.mic_started_at = 0.0
        self.mic_last_data_at = 0.0
        self.microphone_enabled = False
        self.update_microphone_button_state(self.microphone_requested)
        logger.info("Display microphone stopped")

    def start_link_microphone(self) -> None:
        self.link_microphone_requested = True
        if self.is_director_mode:
            self.link_microphone_pending = True
            self.update_link_microphone_button_state(True)
            logger.info(
                "Director mode received link microphone-on sync; local capture stays disabled"
            )
            return

        if self.link_microphone_enabled:
            self.link_microphone_pending = False
            self.update_link_microphone_button_state(True)
            return

        self.link_microphone_pending = True
        try:
            device_info = self._find_voicemeeter_b1_input_device()
            audio_format = self._create_microphone_format(device_info)
            self.link_mic_worker = MicrophoneVadWorker(
                on_speech_candidate_start=(
                    self.link_microphone_speech_candidate_started.emit
                ),
                on_speech_start=self.link_microphone_speech_started.emit,
                on_speech_cancelled=self.link_microphone_speech_cancelled.emit,
                on_audio_detected=self.link_microphone_audio_detected.emit,
                on_audio_confirmed=self.link_microphone_audio_confirmed.emit,
                on_error=self.link_microphone_error.emit,
            )
            self.link_mic_worker.start()
            self.link_mic_audio_input = QAudioInput(device_info, audio_format, self)
            self.link_mic_audio_input.stateChanged.connect(
                self.handle_link_microphone_audio_state_changed
            )
            self.link_mic_device = self.link_mic_audio_input.start()
            if not self.link_mic_device:
                raise RuntimeError("Failed to start link microphone input device")
            self.link_mic_device.readyRead.connect(
                self.handle_link_microphone_ready_read
            )
            self.link_mic_format = audio_format
            self.link_mic_byte_buffer.clear()
            self.link_mic_started_at = time.monotonic()
            self.link_mic_last_data_at = self.link_mic_started_at
            self.link_microphone_enabled = True
            self.link_microphone_faulted = False
            self.link_microphone_pending = False
            self.link_microphone_confirmed = True
            self._report_link_microphone_fault(False, "started")
            self.update_link_microphone_button_state(True)
            logger.info(
                "Display link microphone started: device={} sample_rate={} channels={} sample_size={}",
                device_info.deviceName(),
                audio_format.sampleRate(),
                audio_format.channelCount(),
                audio_format.sampleSize(),
            )
        except Exception as exc:
            self.link_microphone_faulted = True
            self.link_microphone_pending = False
            self.link_microphone_confirmed = False
            self.stop_link_microphone(keep_requested=True)
            self._report_link_microphone_fault(True, str(exc))
            logger.error(
                "Display link microphone start failed; watchdog will retry while requested: {}",
                exc,
            )

    def stop_link_microphone(self, *, keep_requested: bool = False) -> None:
        if not keep_requested:
            self.link_microphone_requested = False
            self.link_microphone_faulted = False
            self.link_microphone_pending = False
            self.link_microphone_confirmed = False
        if self.link_mic_device:
            try:
                self.link_mic_device.readyRead.disconnect(
                    self.handle_link_microphone_ready_read
                )
            except (TypeError, RuntimeError):
                pass
            self.link_mic_device = None

        if self.link_mic_audio_input:
            try:
                self.link_mic_audio_input.stateChanged.disconnect(
                    self.handle_link_microphone_audio_state_changed
                )
            except (TypeError, RuntimeError):
                pass
            self.link_mic_audio_input.stop()
            self.link_mic_audio_input = None

        if self.link_mic_worker:
            self.link_mic_worker.stop()
            self.link_mic_worker.join(timeout=1.0)
            self.link_mic_worker = None

        self.link_mic_format = None
        self.link_mic_byte_buffer.clear()
        self.handle_link_microphone_speech_cancelled()
        self.link_mic_started_at = 0.0
        self.link_mic_last_data_at = 0.0
        self.link_microphone_enabled = False
        if not keep_requested:
            self.link_microphone_confirmed = False
        self.update_link_microphone_button_state(self.link_microphone_requested)
        logger.info("Display link microphone stopped")

    def _find_voicemeeter_b1_input_device(self) -> QAudioDeviceInfo:
        devices = QAudioDeviceInfo.availableDevices(QAudio.AudioInput)
        if not devices:
            raise RuntimeError("No microphone input device is available")

        names = [device.deviceName() for device in devices]
        for device in devices:
            name = device.deviceName().lower()
            if (
                "voicemeeter output" in name
                and "aux" not in name
                and "vaio3" not in name
            ):
                return device

        for device in devices:
            name = device.deviceName().lower()
            if "voicemeeter" in name and "b1" in name:
                return device

        raise RuntimeError(
            "VoiceMeeter B1 input device was not found. Available inputs: "
            + ", ".join(names)
        )

    def _create_microphone_format(
        self,
        device_info: QAudioDeviceInfo | None = None,
    ) -> QAudioFormat:
        if device_info is None:
            device_info = QAudioDeviceInfo.defaultInputDevice()
        if device_info.isNull():
            raise RuntimeError("No microphone input device is available")

        audio_format = QAudioFormat()
        audio_format.setCodec("audio/pcm")
        audio_format.setSampleRate(MIC_SAMPLE_RATE)
        audio_format.setChannelCount(MIC_CHANNELS)
        audio_format.setSampleSize(MIC_SAMPLE_SIZE_BITS)
        audio_format.setSampleType(QAudioFormat.SignedInt)
        audio_format.setByteOrder(QAudioFormat.LittleEndian)

        if not device_info.isFormatSupported(audio_format):
            nearest = device_info.nearestFormat(audio_format)
            logger.warning(
                "Microphone format 16k/mono/s16 is not supported for {}; using nearest format: "
                "sample_rate={} channels={} sample_size={} sample_type={}",
                device_info.deviceName(),
                nearest.sampleRate(),
                nearest.channelCount(),
                nearest.sampleSize(),
                nearest.sampleType(),
            )
            audio_format = nearest

        if (
            audio_format.sampleSize() != MIC_SAMPLE_SIZE_BITS
            or audio_format.sampleType() != QAudioFormat.SignedInt
        ):
            raise RuntimeError(
                "Unsupported microphone format; need signed 16-bit PCM input"
            )

        return audio_format

    def _mark_microphone_data_received(self, source: str, byte_count: int) -> None:
        now = time.monotonic()
        if source == "local":
            self.mic_last_data_at = now
        else:
            self.link_mic_last_data_at = now
        if byte_count <= 0:
            return

    def _drain_microphone_device(
        self,
        source: str,
        device: Any,
        byte_buffer: bytearray,
    ) -> None:
        data = bytes(device.readAll())
        if data:
            self._mark_microphone_data_received(source, len(data))
        byte_buffer.clear()

    def _read_microphone_samples(
        self,
        source: str,
        device: Any,
        audio_format: QAudioFormat,
        byte_buffer: bytearray,
    ) -> np.ndarray | None:
        data = bytes(device.readAll())
        if not data:
            return None

        self._mark_microphone_data_received(source, len(data))
        byte_buffer.extend(data)
        channel_count = max(audio_format.channelCount(), 1)
        bytes_per_frame = (audio_format.sampleSize() // 8) * channel_count
        usable_bytes = (len(byte_buffer) // bytes_per_frame) * bytes_per_frame
        if usable_bytes <= 0:
            return None

        raw = bytes(byte_buffer[:usable_bytes])
        del byte_buffer[:usable_bytes]
        dtype = np.dtype(
            "<i2" if audio_format.byteOrder() == QAudioFormat.LittleEndian else ">i2"
        )
        samples = np.frombuffer(raw, dtype=dtype).astype(np.float32) / 32768.0
        if channel_count > 1:
            samples = samples.reshape(-1, channel_count).mean(axis=1)

        source_rate = audio_format.sampleRate()
        if source_rate != MIC_SAMPLE_RATE:
            samples = self._resample_microphone_samples(samples, source_rate)

        return samples

    def _microphone_level_from_samples(self, samples: np.ndarray) -> float:
        if samples.size == 0:
            return 0.0
        samples = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
        rms = math.sqrt(float(np.mean(np.square(samples))))
        peak = float(np.max(np.abs(samples)))
        return max(min(rms * 12.0, 1.0), min(peak * 0.75, 1.0))

    def _update_microphone_volume_indicator(
        self,
        source: str,
        samples: np.ndarray,
    ) -> None:
        if not self.is_streamer_mode:
            return
        level = self._microphone_level_from_samples(samples)
        if source == "local":
            self.microphone_volume_indicator.set_level(level)
        else:
            self.link_microphone_volume_indicator.set_level(level)

    def handle_microphone_ready_read(self) -> None:
        if not self.mic_device or not self.mic_format or not self.mic_worker:
            return

        if not self._microphone_input_allowed():
            self._drain_microphone_device(
                "local",
                self.mic_device,
                self.mic_byte_buffer,
            )
            self.mic_byte_buffer.clear()
            return

        samples = self._read_microphone_samples(
            "local",
            self.mic_device,
            self.mic_format,
            self.mic_byte_buffer,
        )
        if samples is None:
            return

        self._update_microphone_volume_indicator("local", samples)
        self.mic_worker.submit(samples)

    def handle_link_microphone_ready_read(self) -> None:
        if (
            not self.link_mic_device
            or not self.link_mic_format
            or not self.link_mic_worker
        ):
            return

        if not self._microphone_input_allowed():
            self._drain_microphone_device(
                "link",
                self.link_mic_device,
                self.link_mic_byte_buffer,
            )
            self.link_mic_byte_buffer.clear()
            return

        samples = self._read_microphone_samples(
            "link",
            self.link_mic_device,
            self.link_mic_format,
            self.link_mic_byte_buffer,
        )
        if samples is None:
            return

        self._update_microphone_volume_indicator("link", samples)
        self.link_mic_worker.submit(samples)

    def handle_microphone_audio_state_changed(self, state: Any) -> None:
        self._handle_microphone_audio_state_changed("local", state)

    def handle_link_microphone_audio_state_changed(self, state: Any) -> None:
        self._handle_microphone_audio_state_changed("link", state)

    def _qaudio_state_name(self, state: Any) -> str:
        for name in ("ActiveState", "IdleState", "SuspendedState", "StoppedState"):
            if state == getattr(QAudio, name):
                return name
        return str(state)

    def _qaudio_error_name(self, error: Any) -> str:
        for name in ("NoError", "OpenError", "IOError", "UnderrunError", "FatalError"):
            if error == getattr(QAudio, name):
                return name
        return str(error)

    def _microphone_health_snapshot(self, source: str) -> dict[str, Any]:
        if source == "local":
            return {
                "label": "Display microphone",
                "requested": self.microphone_requested,
                "enabled": self.microphone_enabled,
                "audio_input": self.mic_audio_input,
                "device": self.mic_device,
                "worker": self.mic_worker,
                "started_at": self.mic_started_at,
                "last_data_at": self.mic_last_data_at,
                "last_restart_at": self.mic_last_restart_at,
                "restarting": self.microphone_restarting,
            }
        return {
            "label": "Display link microphone",
            "requested": self.link_microphone_requested,
            "enabled": self.link_microphone_enabled,
            "audio_input": self.link_mic_audio_input,
            "device": self.link_mic_device,
            "worker": self.link_mic_worker,
            "started_at": self.link_mic_started_at,
            "last_data_at": self.link_mic_last_data_at,
            "last_restart_at": self.link_mic_last_restart_at,
            "restarting": self.link_microphone_restarting,
        }

    def _set_microphone_last_restart_at(self, source: str, value: float) -> None:
        if source == "local":
            self.mic_last_restart_at = value
        else:
            self.link_mic_last_restart_at = value

    def _set_microphone_restarting(self, source: str, value: bool) -> None:
        if source == "local":
            self.microphone_restarting = value
        else:
            self.link_microphone_restarting = value

    def _handle_microphone_audio_state_changed(self, source: str, state: Any) -> None:
        snapshot = self._microphone_health_snapshot(source)
        label = snapshot["label"]
        audio_input = snapshot["audio_input"]
        error = audio_input.error() if audio_input else QAudio.NoError
        logger.debug(
            "{} QAudio state changed: state={} error={}",
            label,
            self._qaudio_state_name(state),
            self._qaudio_error_name(error),
        )
        if (
            not snapshot["requested"]
            or self.is_director_mode
            or self._closing_console
            or snapshot["restarting"]
        ):
            return

        if error != QAudio.NoError:
            self._restart_microphone_source(
                source,
                f"qaudio-error:{self._qaudio_error_name(error)}",
            )
            return

        if state in (QAudio.StoppedState, QAudio.SuspendedState):
            self._restart_microphone_source(
                source,
                f"qaudio-state:{self._qaudio_state_name(state)}",
            )

    def check_microphone_health(self) -> None:
        self._check_microphone_source_health("local")
        self._check_microphone_source_health("link")

    def _check_microphone_source_health(self, source: str) -> None:
        snapshot = self._microphone_health_snapshot(source)
        if (
            not snapshot["requested"]
            or self.is_director_mode
            or self._closing_console
            or snapshot["restarting"]
        ):
            return

        label = snapshot["label"]
        audio_input: QAudioInput | None = snapshot["audio_input"]
        worker: MicrophoneVadWorker | None = snapshot["worker"]
        now = time.monotonic()

        if not snapshot["enabled"] or not audio_input or not snapshot["device"] or not worker:
            self._restart_microphone_source(source, "missing-capture-components")
            return

        if not worker.is_alive():
            self._restart_microphone_source(source, "vad-worker-stopped")
            return

        error = audio_input.error()
        if error != QAudio.NoError:
            self._restart_microphone_source(
                source,
                f"qaudio-error:{self._qaudio_error_name(error)}",
            )
            return

        state = audio_input.state()
        if state in (QAudio.StoppedState, QAudio.SuspendedState):
            self._restart_microphone_source(
                source,
                f"qaudio-state:{self._qaudio_state_name(state)}",
            )
            return

        if not self._microphone_input_allowed():
            return

        started_at = float(snapshot["started_at"] or now)
        last_data_at = float(snapshot["last_data_at"] or started_at)
        if now - started_at < MIC_HEALTH_START_GRACE_SECONDS:
            return

        silence_elapsed = now - last_data_at
        if silence_elapsed >= MIC_HEALTH_NO_DATA_TIMEOUT_SECONDS:
            logger.error(
                "{} has received no input data for {:.1f}s while enabled; restarting",
                label,
                silence_elapsed,
            )
            self._restart_microphone_source(
                source,
                f"no-data:{silence_elapsed:.1f}s",
            )

    def _restart_microphone_source(self, source: str, reason: str) -> None:
        snapshot = self._microphone_health_snapshot(source)
        if (
            not snapshot["requested"]
            or self.is_director_mode
            or self._closing_console
            or snapshot["restarting"]
        ):
            return

        now = time.monotonic()
        elapsed_since_restart = now - float(snapshot["last_restart_at"] or 0.0)
        if elapsed_since_restart < MIC_HEALTH_RESTART_COOLDOWN_SECONDS:
            logger.debug(
                "{} unhealthy but restart is throttled: reason={} cooldown_left={:.1f}s",
                snapshot["label"],
                reason,
                MIC_HEALTH_RESTART_COOLDOWN_SECONDS - elapsed_since_restart,
            )
            return

        self._set_microphone_last_restart_at(source, now)
        self._set_microphone_restarting(source, True)
        logger.error("{} unhealthy; restarting capture: reason={}", snapshot["label"], reason)
        try:
            if source == "local":
                self.stop_microphone(keep_requested=True)
                self.start_microphone()
            else:
                self.stop_link_microphone(keep_requested=True)
                self.start_link_microphone()
        finally:
            self._set_microphone_restarting(source, False)

    def _resample_microphone_samples(
        self,
        samples: np.ndarray,
        source_rate: int,
    ) -> np.ndarray:
        if samples.size == 0 or source_rate <= 0:
            return samples.astype(np.float32)

        target_count = max(1, int(round(samples.size * MIC_SAMPLE_RATE / source_rate)))
        if target_count == samples.size:
            return samples.astype(np.float32)

        source_positions = np.linspace(0.0, 1.0, samples.size, endpoint=False)
        target_positions = np.linspace(0.0, 1.0, target_count, endpoint=False)
        return np.interp(target_positions, source_positions, samples).astype(np.float32)

    def _format_story_entry(self, entry: dict[str, Any]) -> str:
        try:
            story_index = int(entry.get("index")) + 1
            prefix = f"{story_index}. "
        except (TypeError, ValueError):
            prefix = ""

        user_text = str(entry.get("user") or "").strip()
        ai_text = str(entry.get("ai") or "").strip()
        return f"{prefix}用户：{user_text}\nAI：{ai_text}"

    def _set_story_rows(
        self,
        display_entries: list[dict[str, Any]],
        highlighted_index: int | None = None,
    ) -> None:
        for row_index, row in enumerate(self.story_rows):
            if row_index >= len(display_entries):
                row.setText(STORY_EMPTY_TEXT if row_index == 0 else "")
                row.setVisible(row_index == 0)
                row.setStyleSheet(STORY_ROW_STYLE)
                continue

            entry = display_entries[row_index]
            row.setText(self._format_story_entry(entry))
            row.setVisible(True)
            row.setStyleSheet(
                STORY_ROW_HIGHLIGHT_STYLE
                if entry.get("index") == highlighted_index
                else STORY_ROW_STYLE
            )

    def sync_story_state(self, state: Any) -> None:
        if not isinstance(state, dict) or not state.get("has_story"):
            self.story_candidates = []
            self.story_progress_index = 0
            self.story_total = 0
            self._set_story_rows([])
            return

        items = [
            item
            for item in state.get("items", [])
            if isinstance(item, dict)
        ]
        active_entry = state.get("active_entry")
        display_entries = []
        highlighted_index = None
        if isinstance(active_entry, dict):
            display_entries.append(active_entry)
            highlighted_index = active_entry.get("index")
        display_entries.extend(items)

        self.story_candidates = items[:3]
        try:
            self.story_progress_index = int(state.get("progress_index", 0))
        except (TypeError, ValueError):
            self.story_progress_index = 0
        try:
            self.story_total = int(state.get("total", 0))
        except (TypeError, ValueError):
            self.story_total = 0

        self._set_story_rows(display_entries, highlighted_index)
        logger.info(
            "Synced story state: progress={}/{} candidates={} highlighted={}",
            self.story_progress_index,
            self.story_total,
            len(self.story_candidates),
            highlighted_index,
        )

    def send_microphone_audio_segment(self, audio: np.ndarray) -> None:
        if self.is_director_mode:
            self.handle_microphone_speech_cancelled()
            logger.info("Director mode cannot send microphone audio; dropping segment")
            return

        if audio.size == 0:
            self.handle_microphone_speech_cancelled()
            return

        if not self._microphone_input_allowed():
            self.handle_microphone_speech_cancelled()
            logger.info(
                "Dropping microphone utterance while VTuber input is disabled: samples={}",
                len(audio),
            )
            return

        self._commit_microphone_playback_interrupt()
        audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
        for index in range(0, len(audio), MIC_SEND_CHUNK_SAMPLES):
            chunk = audio[index : index + MIC_SEND_CHUNK_SAMPLES]
            self.backend_client.send_json(
                {
                    "type": "mic-audio-data",
                    "mic_source": "local",
                    "audio": chunk.tolist(),
                }
            )
        self.backend_client.send_json(
            {"type": "mic-audio-segment-end", "mic_source": "local"}
        )
        logger.info("Sent microphone segment to backend: samples={}", len(audio))

    def confirm_microphone_audio(self) -> None:
        if self.is_director_mode:
            self.handle_microphone_speech_cancelled()
            logger.info(
                "Director mode cannot confirm microphone audio; dropping confirmation"
            )
            return

        if not self._microphone_input_allowed():
            self.handle_microphone_speech_cancelled()
            logger.info("Dropping microphone confirmation while VTuber input is disabled")
            return

        payload: dict[str, Any] = {
            "type": "mic-audio-end",
            "mic_source": "local",
        }
        performance_id = self._finish_performance_speech("local")
        if performance_id:
            payload["performance_id"] = performance_id
        if self.game_vision_request_id:
            payload["metadata"] = {
                **(payload.get("metadata") or {}),
                "game_vision_request_id": self.game_vision_request_id,
            }
        if self.story_candidates:
            payload["story_candidates"] = self.story_candidates
        has_game_vision_capture = bool(self.game_vision_request_id)
        image_attached = self._attach_pending_vision_image(payload)
        vision_context_reused = (
            self.image_mode_enabled
            and self.visual_image_context_active
            and not image_attached
            and not has_game_vision_capture
        )
        backend_connected = self.backend_client.is_connected()
        self.backend_client.send_json(payload)
        self.game_vision_request_id = None
        self._schedule_game_vision_cold_timer("microphone-confirmed")
        if image_attached and backend_connected:
            provider = self._current_vision_model_label()
            image_name = (self.pending_vision_image or {}).get("name")
            self.visual_image_context_active = True
            self.visual_image_reply_pending = True
            self._clear_vision_image(notify_backend=False)
            self._set_vision_image_status(
                f"图片已随本次问题发送（{provider} / 视觉多轮）"
            )
            logger.info(
                "Sent pending visual image with microphone question: name={} provider={}",
                image_name,
                (payload.get("metadata") or {}).get("vision_model_provider"),
            )
        elif vision_context_reused and backend_connected:
            self.visual_image_reply_pending = True
            self._set_vision_image_status(
                f"已用上一张图片随本次问题继续识图（{self._current_vision_model_label()} / 视觉多轮）"
            )
        self.microphone_interrupt_committed = False
        self._set_microphone_playback_start_blocked("local", False)
        logger.info("Sent microphone utterance confirmation to backend")

    def _link_microphone_metadata(self) -> dict[str, str]:
        name = self.link_human_name_input.text().strip() or self.link_human_name
        name = name.strip() or DEFAULT_LINK_HUMAN_NAME
        return {
            "human_name": name,
            "input_source": "link_microphone",
            "mic_source": "link",
        }

    def send_link_microphone_audio_segment(self, audio: np.ndarray) -> None:
        if self.is_director_mode:
            self.handle_link_microphone_speech_cancelled()
            logger.info("Director mode cannot send link microphone audio; dropping segment")
            return

        if audio.size == 0:
            self.handle_link_microphone_speech_cancelled()
            return

        if not self._microphone_input_allowed():
            self.handle_link_microphone_speech_cancelled()
            logger.info(
                "Dropping link microphone utterance while VTuber input is disabled: samples={}",
                len(audio),
            )
            return

        self._commit_link_microphone_playback_interrupt()
        audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
        for index in range(0, len(audio), MIC_SEND_CHUNK_SAMPLES):
            chunk = audio[index : index + MIC_SEND_CHUNK_SAMPLES]
            self.backend_client.send_json(
                {
                    "type": "mic-audio-data",
                    "mic_source": "link",
                    "audio": chunk.tolist(),
                }
            )
        self.backend_client.send_json(
            {"type": "mic-audio-segment-end", "mic_source": "link"}
        )
        logger.info(
            "Sent link microphone segment to backend: samples={} human_name={}",
            len(audio),
            self._link_microphone_metadata()["human_name"],
        )

    def confirm_link_microphone_audio(self) -> None:
        if self.is_director_mode:
            self.handle_link_microphone_speech_cancelled()
            logger.info(
                "Director mode cannot confirm link microphone audio; dropping confirmation"
            )
            return

        if not self._microphone_input_allowed():
            self.handle_link_microphone_speech_cancelled()
            logger.info("Dropping link microphone confirmation while VTuber input is disabled")
            return

        payload: dict[str, Any] = {
            "type": "mic-audio-end",
            "mic_source": "link",
            "metadata": self._link_microphone_metadata(),
        }
        performance_id = self._finish_performance_speech("link")
        if performance_id:
            payload["performance_id"] = performance_id
        if self.story_candidates:
            payload["story_candidates"] = self.story_candidates
        self.backend_client.send_json(payload)
        self.link_microphone_interrupt_committed = False
        self._set_microphone_playback_start_blocked("link", False)
        self._schedule_game_vision_cold_timer("link-microphone-confirmed")
        logger.info(
            "Sent link microphone utterance confirmation to backend: human_name={}",
            payload["metadata"]["human_name"],
        )

    def _set_microphone_playback_start_blocked(
        self,
        source: str,
        blocked: bool,
    ) -> None:
        if source == "local":
            self.microphone_playback_start_blocked = blocked
        else:
            self.link_microphone_playback_start_blocked = blocked

        if self.live2d_window:
            self.live2d_window.set_audio_start_blocked(
                f"{source}-microphone-candidate",
                blocked,
            )

    def handle_microphone_speech_candidate_started(self) -> None:
        if self.is_director_mode:
            return
        self._start_performance_speech("local")
        self._set_microphone_playback_start_blocked("local", True)
        logger.debug("Microphone speech candidate; blocking new audio playback")

    def handle_link_microphone_speech_candidate_started(self) -> None:
        if self.is_director_mode:
            return
        self._start_performance_speech("link")
        self._set_microphone_playback_start_blocked("link", True)
        logger.debug("Link microphone speech candidate; blocking new audio playback")

    def handle_microphone_speech_started(self) -> None:
        if self.is_director_mode:
            return

        self.microphone_interrupt_committed = False
        self.game_vision_cold_reply_pending = False
        self._cancel_game_vision_cold_timer("microphone-started")
        if self.game_vision_enabled and self._microphone_input_allowed():
            self._send_game_vision_capture()
        if self.live2d_window and self.live2d_window.is_playing_audio():
            if self.live2d_window.pause_audio_for_microphone():
                self.microphone_paused_playback = True
                self._pause_live_streaming_agent_subtitle_for_audio_pause("microphone-started")
                logger.info("Microphone speech detected; paused Live2D playback")

    def handle_microphone_speech_cancelled(self) -> None:
        self._cancel_performance_speech("local")
        self.microphone_interrupt_committed = False
        self.game_vision_request_id = None
        self._set_microphone_playback_start_blocked("local", False)
        if not self.microphone_paused_playback:
            self._schedule_game_vision_cold_timer("microphone-cancelled")
            return

        resumed = bool(
            self.live2d_window
            and self.live2d_window.resume_audio_after_microphone_cancelled()
        )
        self.microphone_paused_playback = False
        if resumed:
            self._resume_live_streaming_agent_subtitle_after_audio_pause("microphone-cancelled")
        else:
            self._stop_live_streaming_agent_subtitle_for_interrupt(
                reason="microphone-cancelled-without-resume"
            )
        self._schedule_game_vision_cold_timer("microphone-cancelled")
        logger.info(
            "Microphone speech was not sent; Live2D playback {}",
            "resumed" if resumed else "was not resumed",
        )

    def handle_link_microphone_speech_started(self) -> None:
        if self.is_director_mode:
            return

        self.link_microphone_interrupt_committed = False
        self.game_vision_cold_reply_pending = False
        self._cancel_game_vision_cold_timer("link-microphone-started")
        if self.live2d_window and self.live2d_window.is_playing_audio():
            if self.live2d_window.pause_audio_for_microphone():
                self.link_microphone_paused_playback = True
                self._pause_live_streaming_agent_subtitle_for_audio_pause("link-microphone-started")
                logger.info("Link microphone speech detected; paused Live2D playback")

    def handle_link_microphone_speech_cancelled(self) -> None:
        self._cancel_performance_speech("link")
        self.link_microphone_interrupt_committed = False
        self._set_microphone_playback_start_blocked("link", False)
        if not self.link_microphone_paused_playback:
            self._schedule_game_vision_cold_timer("link-microphone-cancelled")
            return

        resumed = bool(
            self.live2d_window
            and self.live2d_window.resume_audio_after_microphone_cancelled()
        )
        self.link_microphone_paused_playback = False
        if resumed:
            self._resume_live_streaming_agent_subtitle_after_audio_pause(
                "link-microphone-cancelled"
            )
        else:
            self._stop_live_streaming_agent_subtitle_for_interrupt(
                reason="link-microphone-cancelled-without-resume"
            )
        self._schedule_game_vision_cold_timer("link-microphone-cancelled")
        logger.info(
            "Link microphone speech was not sent; Live2D playback {}",
            "resumed" if resumed else "was not resumed",
        )

    def _commit_microphone_playback_interrupt(self) -> None:
        if self.microphone_interrupt_committed:
            return

        if not self.live2d_window:
            self.microphone_paused_playback = False
            return

        active_turn_id = self.live2d_window.active_turn_id
        was_playing_audio = self.live2d_window.is_playing_audio()
        should_interrupt_active_turn = (
            bool(active_turn_id)
            or self.microphone_paused_playback
            or was_playing_audio
        )
        if not should_interrupt_active_turn:
            return

        self.microphone_paused_playback = False

        heard_response = self.live2d_window.interrupt_from_console()
        self._stop_live_streaming_agent_subtitle_for_interrupt(
            active_turn_id,
            reason="microphone-interrupt",
        )
        if was_playing_audio:
            logger.info("Microphone utterance sent; interrupted Live2D playback")
        else:
            logger.info(
                "Microphone utterance sent; interrupted active backend turn without "
                "local audio playback"
            )
        self.send_console_message(
            "voice-cutoff",
            text=heard_response,
            turn_id=active_turn_id,
        )
        self.microphone_interrupt_committed = True

    def _commit_link_microphone_playback_interrupt(self) -> None:
        if self.link_microphone_interrupt_committed:
            return

        if not self.live2d_window:
            self.link_microphone_paused_playback = False
            return

        active_turn_id = self.live2d_window.active_turn_id
        was_playing_audio = self.live2d_window.is_playing_audio()
        should_interrupt_active_turn = (
            bool(active_turn_id)
            or self.link_microphone_paused_playback
            or was_playing_audio
        )
        if not should_interrupt_active_turn:
            return

        self.link_microphone_paused_playback = False

        heard_response = self.live2d_window.interrupt_from_console()
        self._stop_live_streaming_agent_subtitle_for_interrupt(
            active_turn_id,
            reason="link-microphone-interrupt",
        )
        if was_playing_audio:
            logger.info("Link microphone utterance sent; interrupted Live2D playback")
        else:
            logger.info(
                "Link microphone utterance sent; interrupted active backend turn without "
                "local audio playback"
            )
        self.send_console_message(
            "voice-cutoff",
            text=heard_response,
            turn_id=active_turn_id,
        )
        self.link_microphone_interrupt_committed = True

    def handle_microphone_error(self, message: str) -> None:
        logger.error("Display microphone error: {}", message)
        if self.microphone_requested:
            self._restart_microphone_source("local", f"vad-error:{message}")
        elif self.microphone_enabled:
            self.stop_microphone()

    def handle_link_microphone_error(self, message: str) -> None:
        logger.error("Display link microphone error: {}", message)
        if self.link_microphone_requested:
            self._restart_microphone_source("link", f"vad-error:{message}")
        elif self.link_microphone_enabled:
            self.stop_link_microphone()

    def _microphone_input_allowed(self) -> bool:
        return (
            self.is_streamer_mode
            and not self.sleeping
            and not self.punished
            and not self.wake_animation_pending
            and self.vtuber_mode != "idle"
        )

    def _reset_microphone_capture(self, reason: str) -> None:
        self.mic_byte_buffer.clear()
        if self.mic_worker:
            self.mic_worker.reset()
        self.handle_microphone_speech_cancelled()
        self.link_mic_byte_buffer.clear()
        if self.link_mic_worker:
            self.link_mic_worker.reset()
        self.handle_link_microphone_speech_cancelled()
        logger.debug("Display microphone capture reset by {}", reason)

    def handle_voice_cutoff_clicked(self) -> None:
        if not self.live2d_window:
            logger.info("Voice cutoff button clicked while Live2D window is closed")
            return

        if not self.live2d_window.is_playing_audio():
            logger.info("Voice cutoff button ignored because Live2D is not playing audio")
            return

        active_turn_id = self.live2d_window.active_turn_id
        heard_response = self.live2d_window.interrupt_from_console()
        self._stop_live_streaming_agent_subtitle_for_interrupt(
            active_turn_id,
            reason="voice-cutoff-button",
        )
        logger.info("Voice cutoff button interrupted Live2D playback")
        self.send_console_message(
            "voice-cutoff",
            text=heard_response,
            turn_id=active_turn_id,
        )

    def _pause_live_streaming_agent_subtitle_for_audio_pause(self, reason: str) -> None:
        """TTS 音频被暂停时, 让Agent 字幕同一时刻停止推进."""
        if self.live_streaming_agent_subtitle_window is None:
            return
        try:
            self.live_streaming_agent_subtitle_window.pause_subtitle_progress()
            logger.debug("Live Streaming Agent subtitle progress paused by {}", reason)
        except Exception as exc:
            logger.debug("Failed to pause Live Streaming Agent subtitle progress: {}", exc)

    def _resume_live_streaming_agent_subtitle_after_audio_pause(self, reason: str) -> None:
        """TTS 音频恢复时, 让Agent 字幕继续从暂停点推进."""
        if self.live_streaming_agent_subtitle_window is None:
            return
        try:
            self.live_streaming_agent_subtitle_window.resume_subtitle_progress()
            logger.debug("Live Streaming Agent subtitle progress resumed by {}", reason)
        except Exception as exc:
            logger.debug("Failed to resume Live Streaming Agent subtitle progress: {}", exc)

    def _stop_live_streaming_agent_subtitle_for_interrupt(
        self,
        turn_id: str | None = None,
        *,
        reason: str = "interrupt",
    ) -> None:
        """打断 TTS 时同步停止Agent 字幕继续显示同一轮后续文本."""
        if turn_id:
            self.live_streaming_agent_subtitle_blocked_turn_ids.add(str(turn_id))
        if self.live_streaming_agent_subtitle_window is None:
            return
        try:
            self.live_streaming_agent_subtitle_window.stop_subtitle_progress()
            logger.debug(
                "Live Streaming Agent subtitle progress stopped by {} for turn {}",
                reason,
                turn_id,
            )
        except Exception as exc:
            logger.debug("Failed to stop Live Streaming Agent subtitle progress: {}", exc)

    def handle_mode_clicked(self) -> None:
        if self._syncing_mode_buttons:
            return

        mode = "co_host" if self.interaction_mode == "barrage" else "barrage"
        self.interaction_mode = mode
        self.update_vtuber_mode_buttons()
        self.send_console_message(mode)

    def toggle_sleep_mode(self) -> None:
        if self.wake_animation_pending:
            logger.info("Ignoring sleep button click while Live2D wake animation is pending")
            return
        self.send_console_message("sleep")

    def toggle_punish_mode(self) -> None:
        self.send_console_message("punish")

    def toggle_gift_thanks(self) -> None:
        self.gift_thanks_enabled = not self.gift_thanks_enabled
        self.update_gift_thanks_button_state(self.gift_thanks_enabled)
        self.send_console_message(
            "gift-thanks",
            enabled=self.gift_thanks_enabled,
        )

    # -------------------- Agent 字幕 (LiveStreamingAgentSubtitleWindow) --------------------

    def toggle_live_streaming_agent_subtitle(self) -> None:
        """点击按钮: 打开/关闭Agent 字幕窗口."""
        self.send_console_message("live-streaming-agent-subtitle-toggle")

    def _open_live_streaming_agent_subtitle(self) -> None:
        if self.live_streaming_agent_subtitle_window is None:
            self.live_streaming_agent_subtitle_window = LiveStreamingAgentSubtitleWindow(
                icon_dir=LIVE_STREAMING_AGENT_ICON_ROOT,
            )
            self.live_streaming_agent_subtitle_window.closed.connect(
                self._on_live_streaming_agent_subtitle_closed
            )
        self.live_streaming_agent_subtitle_window.show()
        self.live_streaming_agent_subtitle_window.raise_()
        self.live_streaming_agent_subtitle_enabled = True
        self.update_live_streaming_agent_subtitle_button_state(True)

    def _close_live_streaming_agent_subtitle(self) -> None:
        if self.live_streaming_agent_subtitle_window is not None:
            # 关闭时切断 closed 信号防止重复触发 _on_live_streaming_agent_subtitle_closed
            try:
                self.live_streaming_agent_subtitle_window.closed.disconnect(
                    self._on_live_streaming_agent_subtitle_closed
                )
            except (TypeError, RuntimeError):
                pass
            self.live_streaming_agent_subtitle_window.close()
            self.live_streaming_agent_subtitle_window = None
        self.live_streaming_agent_subtitle_enabled = False
        self.update_live_streaming_agent_subtitle_button_state(False)

    def _on_live_streaming_agent_subtitle_closed(self) -> None:
        """用户手动关掉窗口 (X 按钮) 时同步按钮状态."""
        self.live_streaming_agent_subtitle_window = None
        self.live_streaming_agent_subtitle_enabled = False
        self.update_live_streaming_agent_subtitle_button_state(False)
        if not self._closing_console:
            self.send_console_message("live-streaming-agent-subtitle-stop")

    def update_live_streaming_agent_subtitle_button_state(self, enabled: bool) -> None:
        self.live_streaming_agent_subtitle_button.setStyleSheet(
            CONNECTED_BUTTON_STYLE if enabled else INACTIVE_BUTTON_STYLE
        )

    def _maybe_update_live_streaming_agent_subtitle(self, data: dict[str, Any]) -> None:
        """audio 消息流过时, 把 display_text + emotion 喂给字幕窗口.

        关键点:
        - display_text.text 里通常带着 LLM 写出的 [happy] / [neutral] 等表情
          标签, 必须先剥掉再喂给字幕. 否则用户会看到一行 "[happy]" 而不是真话.
        - 静音 emotion payload 把字面 "[happy]" 当 text 送过来, 剥完是空 ->
          只更新头像 emotion, 不动正在显示的字幕.
        - GPT-SoVITS 流式 TTS 后续 chunk 不带 text, 后端已 fallback 到外层
          display_text, 所以同一句话每个 chunk 都会重复送相同 text; 用
          set_subtitle 的 "同文本不重启" 机制去重.
        """
        if self.live_streaming_agent_subtitle_window is None:
            return
        display_text = data.get("display_text")

        emotion: str | None = None
        actions = data.get("actions") or {}
        emotions = actions.get("emotions") if isinstance(actions, dict) else None
        if isinstance(emotions, list) and emotions:
            emotion = str(emotions[0]).strip() or None

        # 没带 display_text -> 纯音频 continuation, 不动文字, 也不动 emotion
        if not isinstance(display_text, dict):
            return

        raw_text = display_text.get("text") or ""

        # 剥掉 [emotion_keyword] / [neutral] 等表情标签, 只保留人话.
        # 用宽容字符类 [^\[\]]{1,20}: 中文 emotion 名 (例如 [开心]) 也能命中.
        # 多个标签和标签后多余空格一起清掉.
        cleaned_text = re.sub(r"\[[^\[\]]{1,20}\]", "", raw_text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        # 剥完什么都不剩 (纯标签段, 例如 silent emotion payload "[happy]")
        # -> 保留当前字幕不动, 只更新头像 emotion
        text_arg: str | None = cleaned_text if cleaned_text else None

        # 用 volumes (每 slice_length ms 一个采样) 推算总时长 → 打字机节奏 = 总时长/字数
        # 这样字幕展开速度就匹配 TTS 播放速度 (近似于"念到哪个字, 字幕到哪个字")
        duration_ms: int | None = None
        volumes = data.get("volumes")
        slice_length = data.get("slice_length")
        if (
            isinstance(volumes, list)
            and isinstance(slice_length, (int, float))
            and slice_length > 0
            and len(volumes) > 0
        ):
            duration_ms = int(len(volumes) * float(slice_length))

        # turn_id 用来区分轮次: 同一轮内所有 segment 字幕累积显示, 下一轮才清空
        raw_turn_id = data.get("turn_id")
        turn_id: str | None = (
            str(raw_turn_id) if raw_turn_id not in (None, "") else None
        )
        if turn_id and turn_id in self.live_streaming_agent_subtitle_blocked_turn_ids:
            logger.debug(
                "Skipping Live Streaming Agent subtitle update for interrupted turn {}",
                turn_id,
            )
            return

        try:
            self.live_streaming_agent_subtitle_window.set_subtitle(
                text_arg,
                emotion=emotion,
                duration_ms=duration_ms,
                turn_id=turn_id,
            )
        except Exception as exc:
            logger.debug("Failed to update live streaming agent subtitle: {}", exc)

    # -------------------- 弹幕字幕 --------------------

    def toggle_barrage_subtitle(self) -> None:
        """点击按钮: 打开/关闭弹幕字幕窗口."""
        self.send_console_message("barrage-subtitle-toggle")

    def _open_barrage_subtitle(self) -> None:
        if self.barrage_subtitle_window is None:
            self.barrage_subtitle_window = BarrageSubtitleWindow()
            self.barrage_subtitle_window.closed.connect(
                self._on_barrage_subtitle_closed
            )
        self.barrage_subtitle_window.show()
        self.barrage_subtitle_window.raise_()
        self.barrage_subtitle_enabled = True
        self.update_barrage_subtitle_button_state(True)

    def _close_barrage_subtitle(self) -> None:
        if self.barrage_subtitle_window is not None:
            try:
                self.barrage_subtitle_window.closed.disconnect(
                    self._on_barrage_subtitle_closed
                )
            except (TypeError, RuntimeError):
                pass
            self.barrage_subtitle_window.close()
            self.barrage_subtitle_window = None
        self.barrage_subtitle_enabled = False
        self.update_barrage_subtitle_button_state(False)

    def _on_barrage_subtitle_closed(self) -> None:
        """用户手动关掉窗口 (X 按钮) 时同步按钮状态."""
        self.barrage_subtitle_window = None
        self.barrage_subtitle_enabled = False
        self.update_barrage_subtitle_button_state(False)
        if not self._closing_console:
            self.send_console_message("barrage-subtitle-stop")

    def update_barrage_subtitle_button_state(self, enabled: bool) -> None:
        self.barrage_subtitle_button.setStyleSheet(
            CONNECTED_BUTTON_STYLE if enabled else INACTIVE_BUTTON_STYLE
        )

    def _maybe_update_barrage_subtitle(self, data: dict[str, Any]) -> None:
        """barrage-display 消息流过时, 把被回复的弹幕喂给弹幕字幕窗口."""
        if self.barrage_subtitle_window is None:
            return
        nickname = str(data.get("nickname") or "").strip()
        content = str(data.get("content") or "").strip()
        if not nickname and not content:
            return
        try:
            self.barrage_subtitle_window.set_barrage(
                nickname, content
            )
        except Exception as exc:
            logger.debug("Failed to update barrage subtitle: {}", exc)

    def closeEvent(self, event: Any) -> None:
        self._closing_console = True
        self._cancel_game_vision_cold_timer("console-closing")
        app = QApplication.instance()
        if app:
            app.removeEventFilter(self)
        self.link_human_name_save_timer.stop()
        self._commit_link_human_name(normalize_empty=True)
        self.microphone_health_timer.stop()
        self.stop_microphone()
        self.stop_link_microphone()
        self.backend_client.close()
        if self.live2d_window:
            self.live2d_window.close()
        if self.live_streaming_agent_subtitle_window:
            try:
                self.live_streaming_agent_subtitle_window.closed.disconnect(
                    self._on_live_streaming_agent_subtitle_closed
                )
            except (TypeError, RuntimeError):
                pass
            self.live_streaming_agent_subtitle_window.close()
            self.live_streaming_agent_subtitle_window = None
        if self.barrage_subtitle_window:
            try:
                self.barrage_subtitle_window.closed.disconnect(
                    self._on_barrage_subtitle_closed
                )
            except (TypeError, RuntimeError):
                pass
            self.barrage_subtitle_window.close()
            self.barrage_subtitle_window = None
        if self.paint_window:
            try:
                self.paint_window.closed.disconnect(self._on_paint_window_closed)
            except (TypeError, RuntimeError):
                pass
            self.paint_window.close()
            self.paint_window = None
        try:
            pygame.mixer.quit()
        finally:
            super().closeEvent(event)

    def toggle_live2d_window(self) -> None:
        self.send_console_message("live2d-toggle")

    def handle_backend_message(self, data: dict[str, Any]) -> None:
        if data["type"] not in ("heartbeat-ack", ):
            logger.info('received backend message {}'.format(truncate_data(data)))
        msg_type = data.get("type")
        if msg_type == "user-input-transcription":
            self._bind_performance_turn(data)
        elif msg_type == "performance-stage":
            self._handle_performance_stage(data)
        elif msg_type == "performance-metrics":
            self._handle_backend_performance_metrics(data)
        elif msg_type == "performance-monitor-request":
            self._send_performance_monitor_snapshot()
        elif msg_type == "performance-monitor-sync":
            self._handle_performance_monitor_sync(data)
        elif msg_type == "project-config-state":
            if self.project_config_dialog is not None:
                self.project_config_dialog.apply_state(data)
        elif msg_type == "project-config-test-result":
            if self.project_config_dialog is not None:
                self.project_config_dialog.show_test_result(data)
        elif msg_type == "project-config-error":
            if self.project_config_dialog is not None:
                self.project_config_dialog.show_config_error(
                    str(data.get("message") or "未知错误")
                )
        if msg_type == "backend-synth-complete":
            self._finish_game_vision_cold_reply("backend-synth-complete")
            self._finish_visual_image_reply("backend-synth-complete")
        elif msg_type == "error":
            self._finish_game_vision_cold_reply("backend-error")
            self._handle_visual_image_reply_error(data)
            self._set_performance_state("idle")
            error_text = str(data.get("message") or data.get("text") or "未知错误")
            self._append_performance_log(f"后端处理失败：{error_text}")
        elif msg_type == "control":
            control_text_for_cold = str(data.get("text") or "")
            if control_text_for_cold in {
                "conversation-chain-end",
                "interrupt",
            }:
                self._finish_game_vision_cold_reply(
                    f"backend-control:{control_text_for_cold}"
                )
                self._finish_visual_image_reply(
                    f"backend-control:{control_text_for_cold}"
                )
        if msg_type == "interrupt-signal" or (
            msg_type == "control" and str(data.get("text") or "") == "interrupt"
        ):
            self._set_performance_state("interrupting")
            turn_id = self._message_turn_id(data)
            active_turn_id = (
                self.live2d_window.active_turn_id if self.live2d_window else None
            )
            if not turn_id or not active_turn_id or turn_id == active_turn_id:
                self._stop_live_streaming_agent_subtitle_for_interrupt(
                    turn_id,
                    reason=f"backend-{msg_type}",
                )
            elif turn_id:
                self.live_streaming_agent_subtitle_blocked_turn_ids.add(str(turn_id))
        # Agent 字幕旁路: 不管 live2d_window 是否存在都尝试更新字幕窗口
        if msg_type == "audio" and (not self.live2d_window or not data.get("audio")):
            self._maybe_update_live_streaming_agent_subtitle(data)
        # 弹幕字幕旁路: 被回复的弹幕推送, 直接消费, 不往 live2d 转发
        if msg_type == "barrage-display":
            self._maybe_update_barrage_subtitle(data)
            return
        if msg_type == "set-model-and-conf":
            self._handle_model_info(data)
        elif msg_type == "story-state":
            self.sync_story_state(data.get("story_state"))
        elif msg_type == "game-vision-state":
            self._handle_game_vision_state(data)
        elif msg_type == "paint-state":
            self._handle_paint_state(data)
        elif msg_type == "link-microphone-name-detect":
            self._handle_link_human_name_detect_response(data)
        elif msg_type == "mode-changed":
            self._handle_mode_changed(data)
        elif msg_type in {
            "performance-stage",
            "performance-metrics",
            "performance-monitor-request",
            "performance-monitor-sync",
            "project-config-state",
            "project-config-test-result",
            "project-config-error",
        }:
            pass
        elif msg_type == "heartbeat-ack":
            # logger.debug("Received backend heartbeat ack")
            pass
        elif msg_type == "control" and self._handle_display_control(
            str(data.get("text") or ""),
            data.get("display_state"),
        ):
            pass
        elif self.live2d_window:
            self.live2d_window.handle_backend_message(data)
        elif self.is_director_mode and msg_type == "backend-synth-complete":
            logger.debug(
                "Director mode ignored backend synth completion for turn {}",
                self._message_turn_id(data),
            )
        elif self.is_director_mode and msg_type == "audio":
            logger.debug(
                "Director mode ignored audio payload for turn {}",
                self._message_turn_id(data),
            )
        elif msg_type == "backend-synth-complete":
            self._send_frontend_playback_complete(self._message_turn_id(data))
            logger.debug("No Live2D window; acknowledged backend synth completion immediately")
        elif msg_type == "audio":
            logger.warning("Dropping audio payload because Live2D window is closed")
        elif msg_type == "error":
            logger.error("Backend error: {}", data.get("message"))
        elif msg_type == "full-text" and data.get("text"):
            logger.info("Backend text: {}", data["text"])
        elif msg_type == "control":
            logger.info("Backend control: {}", data.get("text"))

    def _handle_display_control(
        self,
        control_text: str,
        display_state: Any = None,
    ) -> bool:
        if control_text == "link-mic-fault":
            self._apply_link_microphone_fault_state(display_state)
            return True

        if display_state is not None:
            self._apply_display_state(display_state)

        if control_text == "open-live2d":
            self._set_live2d_global_enabled(True)
            return True
        if control_text == "close-live2d":
            self._set_live2d_global_enabled(False)
            return True
        if control_text == "start-mic":
            self._set_microphone_global_enabled(True)
            return True
        if control_text == "stop-mic":
            self._set_microphone_global_enabled(False)
            return True
        if control_text == "start-link-mic":
            self._set_link_microphone_global_enabled(True)
            return True
        if control_text == "stop-link-mic":
            self._set_link_microphone_global_enabled(False)
            return True
        if control_text == "link-mic-name":
            return True
        if control_text == "gift-thanks-on":
            self._set_gift_thanks_global_enabled(True)
            return True
        if control_text == "gift-thanks-off":
            self._set_gift_thanks_global_enabled(False)
            return True
        if control_text == "open-live-streaming-agent-subtitle":
            self._set_live_streaming_agent_subtitle_global_enabled(True)
            return True
        if control_text == "close-live-streaming-agent-subtitle":
            self._set_live_streaming_agent_subtitle_global_enabled(False)
            return True
        if control_text == "open-barrage-subtitle":
            self._set_barrage_subtitle_global_enabled(True)
            return True
        if control_text == "close-barrage-subtitle":
            self._set_barrage_subtitle_global_enabled(False)
            return True
        if control_text == "game-vision-on":
            if display_state is None:
                self._apply_display_state({"game_vision_enabled": True})
            return True
        if control_text == "game-vision-off":
            if display_state is None:
                self._apply_display_state({"game_vision_enabled": False})
            return True
        if control_text == "paint-on":
            self._set_paint_global_enabled(True)
            return True
        if control_text == "paint-off":
            self._set_paint_global_enabled(False)
            return True
        return False

    def _apply_link_microphone_fault_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            return
        if "link_microphone_enabled" in state:
            self.link_microphone_requested = bool(
                state.get("link_microphone_enabled")
            )
        if "link_microphone_faulted" in state:
            self.link_microphone_faulted = bool(
                state.get("link_microphone_faulted")
            )
            self.link_microphone_pending = bool(
                state.get("link_microphone_pending", False)
            )
            self.link_microphone_confirmed = bool(
                state.get("link_microphone_confirmed", False)
            )
            if self.is_streamer_mode:
                self._reported_link_microphone_faulted = self.link_microphone_faulted
        self.update_link_microphone_button_state(self.link_microphone_requested)

    def _apply_display_state(
        self,
        state: Any,
        *,
        apply_live2d: bool = True,
    ) -> None:
        if not isinstance(state, dict):
            return

        if "live2d_open" in state:
            live2d_open = bool(state.get("live2d_open"))
            if apply_live2d:
                self._set_live2d_global_enabled(live2d_open)
            else:
                self.live2d_global_enabled = live2d_open
                self.update_live2d_button_state(live2d_open)

        if "microphone_enabled" in state:
            self._set_microphone_global_enabled(
                bool(state.get("microphone_enabled"))
            )

        if "link_human_name" in state:
            name = str(state.get("link_human_name") or "").strip()
            self.link_human_name = name or DEFAULT_LINK_HUMAN_NAME
            if self.link_human_name_input.text() != self.link_human_name:
                old_state = self.link_human_name_input.blockSignals(True)
                try:
                    self.link_human_name_input.setText(self.link_human_name)
                finally:
                    self.link_human_name_input.blockSignals(old_state)

        if "link_microphone_faulted" in state:
            self.link_microphone_faulted = bool(state.get("link_microphone_faulted"))
            if self.is_streamer_mode and not bool(
                state.get("link_microphone_pending", False)
            ):
                self._reported_link_microphone_faulted = self.link_microphone_faulted
            self.update_link_microphone_button_state(self.link_microphone_requested)

        if "link_microphone_pending" in state:
            self.link_microphone_pending = bool(state.get("link_microphone_pending"))
            self.update_link_microphone_button_state(self.link_microphone_requested)

        if "link_microphone_confirmed" in state:
            self.link_microphone_confirmed = bool(
                state.get("link_microphone_confirmed")
            )
            self.update_link_microphone_button_state(self.link_microphone_requested)

        if "link_microphone_enabled" in state:
            self._set_link_microphone_global_enabled(
                bool(state.get("link_microphone_enabled"))
            )

        if "gift_thanks_enabled" in state:
            self._set_gift_thanks_global_enabled(
                bool(state.get("gift_thanks_enabled"))
            )

        if "live_streaming_agent_subtitle_enabled" in state:
            self._set_live_streaming_agent_subtitle_global_enabled(
                bool(state.get("live_streaming_agent_subtitle_enabled"))
            )

        if "barrage_subtitle_enabled" in state:
            self._set_barrage_subtitle_global_enabled(
                bool(state.get("barrage_subtitle_enabled"))
            )

        if "game_vision_enabled" in state:
            enabled = bool(state.get("game_vision_enabled"))
            if self.is_director_mode:
                self.game_vision_enabled = enabled
                self.game_vision_request_id = None
                self.game_vision_cold_reply_pending = False
                self._cancel_game_vision_cold_timer("director-mode-state-sync")
                self.update_game_vision_button_state(enabled)

            if (
                self.is_streamer_mode
                and enabled
                and not self.game_vision_window_title
            ):
                enabled = False
                self._set_vision_image_status(
                    "游戏识图未开启：请先绑定游戏窗口；不会截取整个屏幕",
                    error=True,
                )
                self.send_console_message(
                    "game-vision-mode",
                    enabled=False,
                    cold_idle_seconds=GAME_VISION_COLD_IDLE_SECONDS,
                )
            self.game_vision_enabled = enabled
            if not enabled:
                self.game_vision_request_id = None
                self.game_vision_cold_reply_pending = False
                self._cancel_game_vision_cold_timer("display-state-disabled")
            else:
                self._schedule_game_vision_cold_timer("display-state-enabled")
            self.update_game_vision_button_state(enabled)

        if "paint_enabled" in state:
            self._set_paint_global_enabled(bool(state.get("paint_enabled")))

    def _set_live2d_global_enabled(self, enabled: bool) -> None:
        self.live2d_global_enabled = enabled
        if self.is_director_mode:
            if self.live2d_window:
                self.live2d_window.close()
            self.update_live2d_button_state(enabled)
            logger.info(
                "Director mode synced Live2D window control: {}",
                "open" if enabled else "closed",
            )
            return

        if enabled:
            self.user_closed_live2d = False
            if not self.model_config:
                self.model_config = self._load_fallback_model_config()
                logger.warning(
                    "Using fallback Live2D model before backend model info is available"
                )
            if not self.live2d_window:
                self.open_live2d_window()
            else:
                self.update_live2d_button_state(True)
            return

        self.user_closed_live2d = True
        if self.live2d_window:
            self.live2d_window.close()
        else:
            self.update_live2d_button_state(False)

    def _set_microphone_global_enabled(self, enabled: bool) -> None:
        self.microphone_requested = enabled
        if self.is_director_mode:
            if self.microphone_enabled:
                self.stop_microphone()
            else:
                self.update_microphone_button_state(enabled)
            logger.info(
                "Director mode synced microphone control: {}",
                "enabled" if enabled else "disabled",
            )
            return

        if enabled:
            self.start_microphone()
        else:
            self.stop_microphone()

    def _set_link_microphone_global_enabled(self, enabled: bool) -> None:
        self.link_microphone_requested = enabled
        if self.is_director_mode:
            if not enabled:
                self.link_microphone_pending = False
                self.link_microphone_confirmed = False
                self.link_microphone_faulted = False
            self.update_link_microphone_button_state(enabled)
            logger.info(
                "Director mode synced link microphone control: {}",
                "enabled" if enabled else "disabled",
            )
            return

        if enabled:
            self._reported_link_microphone_faulted = None
            self.start_link_microphone()
        else:
            self._reported_link_microphone_faulted = False
            self.link_microphone_pending = False
            self.link_microphone_confirmed = False
            self.stop_link_microphone()

    def _set_gift_thanks_global_enabled(self, enabled: bool) -> None:
        self.gift_thanks_enabled = enabled
        self.update_gift_thanks_button_state(enabled)

    def _set_live_streaming_agent_subtitle_global_enabled(self, enabled: bool) -> None:
        if self.is_director_mode:
            if self.live_streaming_agent_subtitle_window is not None:
                self._close_live_streaming_agent_subtitle()
            self.live_streaming_agent_subtitle_enabled = bool(enabled)
            self.update_live_streaming_agent_subtitle_button_state(self.live_streaming_agent_subtitle_enabled)
            return

        if enabled:
            self._open_live_streaming_agent_subtitle()
        else:
            self._close_live_streaming_agent_subtitle()

    def _set_barrage_subtitle_global_enabled(self, enabled: bool) -> None:
        if self.is_director_mode:
            if self.barrage_subtitle_window is not None:
                self._close_barrage_subtitle()
            self.barrage_subtitle_enabled = bool(enabled)
            self.update_barrage_subtitle_button_state(
                self.barrage_subtitle_enabled
            )
            return

        if enabled:
            self._open_barrage_subtitle()
        else:
            self._close_barrage_subtitle()

    def update_backend_state(self, connected: bool) -> None:
        if connected:
            self._remember_successful_backend_url()
            self.connection_button.setText("已连接")
            self.connection_button.setStyleSheet(CONNECTED_BUTTON_STYLE)
            self.send_console_message("display-client-mode", mode=self.display_mode)
            return

        self.connection_button.setText("未连接")
        self.connection_button.setStyleSheet(WARNING_BUTTON_STYLE)

    def update_live2d_button_state(self, opened: bool) -> None:
        self.live2d_button.setText("关闭Live2D" if opened else "打开Live2D")
        self.live2d_button.setStyleSheet(
            CONNECTED_BUTTON_STYLE if opened else WARNING_BUTTON_STYLE
        )

    def update_vtuber_mode_buttons(self) -> None:
        sleeping = self.sleeping
        punished = self.punished
        barrage_mode = self.interaction_mode == "barrage"
        awake = not sleeping

        if self.wake_animation_pending:
            self.sleep_button.setText(SLEEP_WAKING_TEXT)
            self.sleep_button.setStyleSheet(WAKE_PENDING_BUTTON_STYLE)
            self.sleep_button.setCursor(Qt.ArrowCursor)
        else:
            self.sleep_button.setText(SLEEP_AWAKE_TEXT if awake else SLEEP_SLEEPING_TEXT)
            self.sleep_button.setStyleSheet(
                CONNECTED_BUTTON_STYLE if awake else WARNING_BUTTON_STYLE
            )
            self.sleep_button.setCursor(Qt.PointingHandCursor)

        self.voice_button.setText(
            PUNISH_ACTIVE_TEXT if punished else PUNISH_INACTIVE_TEXT
        )
        self.voice_button.setStyleSheet(
            WARNING_BUTTON_STYLE if punished else CONNECTED_BUTTON_STYLE
        )

        self.mode_button.setText(
            BARRAGE_REPLY_TEXT if barrage_mode else BARRAGE_IGNORE_TEXT
        )
        self.mode_button.setStyleSheet(
            CONNECTED_BUTTON_STYLE if barrage_mode else INACTIVE_BUTTON_STYLE
        )
        self._sync_performance_monitor_status()

        logger.debug(
            "Synced VTuber mode controls: mode={} sub_mode={} interaction_mode={} sleeping={} punished={}",
            self.vtuber_mode,
            self.vtuber_sub_mode,
            self.interaction_mode,
            sleeping,
            punished,
        )

    def _sync_performance_monitor_status(self) -> None:
        if self.performance_monitor is None:
            return
        self.performance_monitor.set_status(self.performance_state)

    def update_microphone_button_state(self, enabled: bool) -> None:
        faulted = (
            self.microphone_faulted
            and self.microphone_requested
            and not self.microphone_enabled
            and self.is_streamer_mode
        )
        if faulted:
            self.microphone_button.setText(MIC_ERROR_TEXT)
            self.microphone_button.setStyleSheet(WARNING_BUTTON_STYLE)
            self.microphone_button.setToolTip(MIC_ERROR_TOOLTIP)
            self.microphone_volume_indicator.set_active(False)
            return

        self.microphone_button.setText(MIC_ON_TEXT if enabled else MIC_OFF_TEXT)
        self.microphone_button.setStyleSheet(
            CONNECTED_BUTTON_STYLE if enabled else WARNING_BUTTON_STYLE
        )
        self.microphone_button.setToolTip("")
        self.microphone_volume_indicator.set_active(enabled and self.is_streamer_mode)

    def update_link_microphone_button_state(self, enabled: bool) -> None:
        requested = self.link_microphone_requested or enabled
        confirmed = self.link_microphone_enabled or self.link_microphone_confirmed
        faulted = (
            self.link_microphone_faulted
            and requested
            and not confirmed
        )
        if faulted:
            self.link_microphone_button.setText(LINK_MIC_ERROR_TEXT)
            self.link_microphone_button.setStyleSheet(WARNING_BUTTON_STYLE)
            self.link_microphone_button.setToolTip(LINK_MIC_ERROR_TOOLTIP)
            self.link_microphone_volume_indicator.set_active(False)
            return

        pending = (
            requested
            and not confirmed
            and (self.link_microphone_pending or self.is_director_mode)
        )
        if pending:
            self.link_microphone_button.setText(LINK_MIC_PENDING_TEXT)
            self.link_microphone_button.setStyleSheet(WAKE_PENDING_BUTTON_STYLE)
            self.link_microphone_button.setToolTip("")
            self.link_microphone_volume_indicator.set_active(False)
            return

        self.link_microphone_button.setText(
            LINK_MIC_ON_TEXT if requested and confirmed else LINK_MIC_OFF_TEXT
        )
        self.link_microphone_button.setStyleSheet(
            CONNECTED_BUTTON_STYLE
            if requested and confirmed
            else WARNING_BUTTON_STYLE
        )
        self.link_microphone_button.setToolTip("")
        self.link_microphone_volume_indicator.set_active(
            requested and confirmed and self.is_streamer_mode
        )

    def update_gift_thanks_button_state(self, enabled: bool) -> None:
        self.gift_thanks_button.setText(
            GIFT_THANKS_ACTIVE_TEXT if enabled else GIFT_THANKS_INACTIVE_TEXT
        )
        self.gift_thanks_button.setStyleSheet(
            CONNECTED_BUTTON_STYLE if enabled else INACTIVE_BUTTON_STYLE
        )

    def sync_cold_time_from_backend(self, seconds: Any) -> None:
        try:
            value = int(round(float(seconds)))
        except (TypeError, ValueError):
            return

        if value <= 0:
            return

        index = self.cold_time_combo.findData(value)
        if index < 0:
            self.cold_time_combo.addItem(f"{value}s", value)
            self.cold_time_combo.updateGeometry()
            index = self.cold_time_combo.findData(value)

        if index < 0 or index == self.cold_time_combo.currentIndex():
            return

        old_state = self.cold_time_combo.blockSignals(True)
        try:
            self.cold_time_combo.setCurrentIndex(index)
        finally:
            self.cold_time_combo.blockSignals(old_state)

        logger.info("Synced cold time from backend: {}s", value)

    def _apply_vtuber_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            return

        was_input_allowed = self._microphone_input_allowed()
        mode = str(state.get("mode") or self.vtuber_mode)
        sub_mode = state.get("sub_mode")
        if "sleeping" in state:
            self.sleeping = bool(state.get("sleeping"))
        else:
            self.sleeping = sub_mode == "sleep"
        if "punished" in state:
            self.punished = bool(state.get("punished"))
        else:
            self.punished = sub_mode == "punish"
        if "wake_animation_pending" in state:
            self.wake_animation_pending = bool(state.get("wake_animation_pending"))
        interaction_mode = state.get("interaction_mode")
        if interaction_mode in {"co_host", "barrage"}:
            self.interaction_mode = str(interaction_mode)
        elif mode in {"co_host", "barrage"}:
            self.interaction_mode = mode
        self.vtuber_sub_mode = str(sub_mode or "")
        self.vtuber_mode = mode
        self.update_vtuber_mode_buttons()
        self._sync_live2d_sleep_motion()
        if was_input_allowed and not self._microphone_input_allowed():
            self._reset_microphone_capture("vtuber-input-disabled")
        if self.game_vision_enabled:
            if self._microphone_input_allowed():
                self._schedule_game_vision_cold_timer("vtuber-input-enabled")
            else:
                self._cancel_game_vision_cold_timer("vtuber-input-disabled")

    def _handle_mode_changed(self, data: dict[str, Any]) -> None:
        detail = data.get("detail") if isinstance(data.get("detail"), dict) else {}
        self._apply_vtuber_state(
            {
                "mode": data.get("mode") or detail.get("new_mode"),
                "sub_mode": data.get("sub_mode") or detail.get("sub_mode"),
                "interaction_mode": data.get("interaction_mode")
                or detail.get("interaction_mode"),
                "sleeping": data.get("sleeping", detail.get("sleeping", False)),
                "punished": data.get("punished", detail.get("punished", False)),
                "wake_animation_pending": data.get(
                    "wake_animation_pending",
                    detail.get("wake_animation_pending", False),
                ),
            }
        )
        logger.info(
            "Backend VTuber mode changed: mode={} sub_mode={}",
            self.vtuber_mode,
            self.vtuber_sub_mode,
        )

    def handle_backend_error(self, message: str) -> None:
        logger.warning("Backend WebSocket error: {}", message)

    def open_live2d_window(self) -> None:
        if self.is_director_mode:
            self.update_live2d_button_state(self.live2d_global_enabled)
            logger.info("Director mode does not create a local Live2D window")
            return

        if self.live2d_window or not self.model_config:
            return

        self.live2d_window = Live2DWindow(
            model_config=self.model_config,
            width=self.live2d_width,
            height=self.live2d_height,
        )
        self.live2d_window.playback_complete.connect(
            self._send_frontend_playback_complete
        )
        self.live2d_window.window_closed.connect(self._handle_live2d_closed)
        self.live2d_window.wake_animation_state_changed.connect(
            self.handle_wake_animation_state_changed
        )
        self.live2d_window.audio_started.connect(self._maybe_update_live_streaming_agent_subtitle)
        self.live2d_window.audio_started.connect(
            self._handle_performance_audio_started
        )
        self.live2d_window.set_output_muted(self.output_muted)
        self.live2d_window.set_audio_start_blocked(
            "local-microphone-candidate",
            self.microphone_playback_start_blocked,
        )
        self.live2d_window.set_audio_start_blocked(
            "link-microphone-candidate",
            self.link_microphone_playback_start_blocked,
        )
        self.live2d_window.show()
        self._sync_live2d_sleep_motion()
        self.update_live2d_button_state(True)
        logger.info("Live2D window opened for model {}", self.model_config.name)

    def _sync_live2d_sleep_motion(self) -> None:
        if self.live2d_window:
            self.live2d_window.set_sleeping(self.sleeping)

    def handle_wake_animation_state_changed(self, active: bool) -> None:
        was_input_allowed = self._microphone_input_allowed()
        self.wake_animation_pending = active
        self.update_vtuber_mode_buttons()
        if was_input_allowed and not self._microphone_input_allowed():
            self._reset_microphone_capture("wake-animation-pending")
        self.send_console_message(
            "wake-animation-start" if active else "wake-animation-complete"
        )

    def _handle_live2d_closed(self) -> None:
        if self.live2d_window:
            self.live2d_width = self.live2d_window.width()
            self.live2d_height = self.live2d_window.height()
            save_live2d_window_size(self.live2d_width, self.live2d_height)
        self.live2d_window = None
        self.wake_animation_pending = False
        self.update_live2d_button_state(False)
        self.update_vtuber_mode_buttons()
        if not self.recreating_live2d:
            self.user_closed_live2d = True
            if self.live2d_global_enabled:
                self.live2d_global_enabled = False
                if not self._closing_console:
                    self.send_console_message("live2d-close")
        logger.info("Live2D window closed")

    def _handle_model_info(self, data: dict[str, Any]) -> None:
        self._apply_vtuber_state(data.get("vtuber_state"))
        self.sync_cold_time_from_backend(data.get("proactive_idle_seconds"))
        self.sync_story_state(data.get("story_state"))
        self._apply_display_state(data.get("display_state"), apply_live2d=False)
        try:
            model_config = model_config_from_backend(data)
        except Exception as exc:
            logger.exception("Failed to load backend Live2D model info: {}", exc)
            model_config = self._load_fallback_model_config()

        logger.info(
            "Backend Live2D model ready: {} ({})",
            model_config.name,
            model_config.model_path,
        )

        if self.live2d_window and self.model_config == model_config:
            self.model_config = model_config
            self.live2d_window.model_config = model_config
            if not self.live2d_global_enabled:
                self.user_closed_live2d = True
                self.live2d_window.close()
                return
            logger.debug("Live2D model unchanged; keeping current window")
            return

        self.model_config = model_config
        if self.live2d_window:
            self.recreating_live2d = True
            self.live2d_window.close()
            self.recreating_live2d = False
            self.user_closed_live2d = False
        if self.live2d_global_enabled and not self.user_closed_live2d:
            self.open_live2d_window()
        else:
            self.update_live2d_button_state(False)

    def _load_fallback_model_config(self) -> ModelConfig:
        return load_model_config(self.fallback_model_name, self.fallback_model_path)


def model_config_from_backend(data: dict[str, Any]) -> ModelConfig:
    logger.debug(f"data: {data}")
    model_info = data.get("model_info") or {}
    if not isinstance(model_info, dict):
        raise ValueError("model_info must be an object")

    url = model_info.get("url") or model_info.get("model_path")
    if not url:
        raise ValueError("model_info does not contain url/model_path")

    name = str(model_info.get("name") or data.get("conf_name") or Path(str(url)).stem)
    frontend_model_info = lookup_frontend_model_info(
        model_name=name,
        model_path=str(url),
    )
    mapping_source = frontend_model_info or model_info

    return ModelConfig(
        name=name,
        model_path=model_path_from_backend_url(str(url)),
        emotion_map=normalize_emotion_map(mapping_source.get("emotionMap")),
        motion_map=normalize_emotion_map(mapping_source.get("motionMap")),
    )


def model_path_from_backend_url(url: str) -> Path:
    parsed = urlparse(url)
    path_text = parsed.path if parsed.scheme else url
    path_text = path_text.replace("\\", "/").lstrip("/")
    if path_text.startswith("live2d-models/"):
        model_path = (
            LIVE2D_RESOURCE_ROOT / path_text.removeprefix("live2d-models/")
        ).resolve()
        allowed_root = LIVE2D_RESOURCE_ROOT.resolve()
    else:
        model_path = (PROJECT_ROOT / path_text).resolve()
        allowed_root = PROJECT_ROOT.resolve()

    try:
        model_path.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"Backend model path escapes project root: {url}") from exc

    if not model_path.exists():
        raise FileNotFoundError(f"Live2D model file not found: {model_path}")
    return model_path


def normalize_emotion_tag(value: Any) -> str:
    tag = str(value or "").strip().strip("[]").lower()
    aliases = {
        "\u56f0\u5026": "sleep",
        "drowsy": "sleep",
        "sleepy": "sleep",
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
        "醒来": "wake",
    }
    return aliases.get(tag, tag)


def normalize_emotion_map(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        normalize_emotion_tag(key): mapped
        for key, mapped in value.items()
        if normalize_emotion_tag(key)
    }


def load_frontend_model_dict() -> list[dict[str, Any]]:
    try:
        with MODEL_DICT_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as exc:
        logger.warning("Failed to read frontend model_dict {}: {}", MODEL_DICT_PATH, exc)
        return []

    if not isinstance(data, list):
        logger.warning("Ignoring invalid frontend model_dict: {}", MODEL_DICT_PATH)
        return []

    return [item for item in data if isinstance(item, dict)]


def model_dict_url_matches(model_info: dict[str, Any], model_path: str | None) -> bool:
    if not model_path:
        return False

    model_url = str(model_info.get("url") or model_info.get("model_path") or "")
    if not model_url:
        return False

    try:
        left = model_path_from_backend_url(model_url)
        right = model_path_from_backend_url(model_path)
    except Exception:
        return False
    return left == right


def lookup_frontend_model_info(
    *,
    model_name: str | None,
    model_path: str | None,
) -> dict[str, Any]:
    models = load_frontend_model_dict()

    if model_name:
        matched_model = next(
            (item for item in models if str(item.get("name") or "") == model_name),
            None,
        )
        if matched_model:
            return matched_model

    matched_model = next(
        (item for item in models if model_dict_url_matches(item, model_path)),
        None,
    )
    return matched_model or {}


def load_model_config(model_name: str, model_path: str | None) -> ModelConfig:
    model_info = lookup_frontend_model_info(model_name=model_name, model_path=model_path)
    if model_path:
        return ModelConfig(
            name=Path(model_path).stem,
            model_path=Path(model_path).resolve(),
            emotion_map=normalize_emotion_map(model_info.get("emotionMap")),
            motion_map=normalize_emotion_map(model_info.get("motionMap")),
        )

    if not model_info:
        raise ValueError(f"Model {model_name!r} not found in {MODEL_DICT_PATH}")

    return ModelConfig(
        name=model_info["name"],
        model_path=model_path_from_backend_url(str(model_info["url"])),
        emotion_map=normalize_emotion_map(model_info.get("emotionMap")),
        motion_map=normalize_emotion_map(model_info.get("motionMap")),
    )


def main() -> int:
    init_logger()
    ensure_frontend_state_files(
        DEFAULT_WS_URL,
        DEFAULT_LIVE2D_WIDTH,
        DEFAULT_LIVE2D_HEIGHT,
    )
    display_mode = load_display_mode()
    logger.info("Live2D window initialized in {} mode", display_mode)
    app = QApplication([])
    console = ConsoleWindow(
        url=load_backend_ws_url(DEFAULT_WS_URL),
        fallback_model_name=DEFAULT_MODEL_NAME,
        fallback_model_path=None,
        live2d_width=DEFAULT_LIVE2D_WIDTH,
        live2d_height=DEFAULT_LIVE2D_HEIGHT,
        display_mode=display_mode,
    )
    console.show()

    try:
        return app.exec_()
    finally:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
