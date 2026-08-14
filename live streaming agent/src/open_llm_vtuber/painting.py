from __future__ import annotations

import ast
import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from loguru import logger
from openai import OpenAI

from .config_manager.utils import read_yaml
from .resource_paths import PROJECT_ROOT
from .utils.turn_trace import record_turn_event


WebSocketSend = Callable[[str], Awaitable[None]]

PAINT_COMMAND_RE = re.compile(
    r"\[画图(?P<nested>\[)?(?P<prompt>[^\[\]\r\n]{1,500})(?(nested)\]\]|\])"
)
PAINT_OUTPUT_ROOT = PROJECT_ROOT / "logs" / "paint"
PAINT_IMAGE_SIZE = (1024, 768)
PAINT_MODEL_TIMEOUT_SECONDS = 90.0
PAINT_CODE_TIMEOUT_SECONDS = 25.0
PAINT_CODE_GENERATION_ATTEMPTS = 3

PAINT_MODEL_CONFIGS = {
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_envs": ("ZHIPUAI_API_KEY",),
        "model": "glm-5v-turbo",
        "extra_body": {"thinking": {"type": "disabled"}},
    },
    "ark": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key_envs": ("ARK_API_KEY",),
        "model": "doubao-seed-2-1-pro-260628",
        "extra_body": {"thinking": {"type": "disabled"}},
    },
}

PAINT_CONFIG_KEY = "paint_config"
TURTLE_TERMINAL_CALLS = {"bye", "done", "exitonclick", "mainloop"}

PAINT_SYSTEM_PROMPT = f"""
你是一个 Python 画图代码生成器。
只输出可直接运行的 Python 代码，不要 Markdown，不要解释。
必须使用 Python 标准库 turtle 绘图，可以 import turtle、math、random、colorsys。
必须在画图前调用 turtle.tracer(False)。
画布大小固定为 IMAGE_WIDTH x IMAGE_HEIGHT，也就是 {PAINT_IMAGE_SIZE[0]}x{PAINT_IMAGE_SIZE[1]}。
坐标原点在画布中心，请控制图形主要内容落在画布内。
不要使用 Pillow/PIL、tkinter、matplotlib、opencv 或任何非 turtle 绘图库。
不要读取或写入任何文件，不要访问网络，不要调用系统命令。
不要调用 turtle.done()、turtle.mainloop()、turtle.exitonclick()、turtle.bye()。
如果使用 turtle.write 写文字，font 必须是 ("字体名", 字号数字, "normal" 或 "bold")，不要把多个字体名放进同一个 font 元组。
颜色请使用字符串或 RGB 三元组，例如 (255, 200, 120)，不要使用 RGBA 四元组。
不需要保存图片，系统会在代码运行后自动更新并保存 turtle 画布。
代码必须完整闭合所有括号、字符串、函数调用和代码块，控制在 100 行以内。
画面应尽量贴近用户描述，构图清楚，色彩丰富，可以使用简单几何图形、文字和装饰。
""".strip()


@dataclass
class PaintCommandExtraction:
    text: str
    prompts: list[str]


def extract_paint_commands(text: str) -> PaintCommandExtraction:
    prompts: list[str] = []

    def replace(match: re.Match[str]) -> str:
        prompt = match.group("prompt").strip()
        if prompt:
            prompts.append(prompt)
        return ""

    cleaned = PAINT_COMMAND_RE.sub(replace, str(text or ""))
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return PaintCommandExtraction(text=cleaned, prompts=prompts)


def paint_capability_prompt() -> str:
    return (
        "你具备画图能力。当用户明确要求你画图、画一张图、生成插画、画面或草图时，"
        "请在自然回复中额外输出一个画图指令，格式必须严格为："
        "[画图[具体画面描述]]。"
        "具体画面描述要包含主体、环境、颜色、风格和关键细节，50字以内。"
        "不要把画图指令用于普通闲聊。你的画图能力很厉害，随便什么都能画。"
    )


def _strip_code_fence(text: str) -> str:
    code = str(text or "").strip()
    fenced = re.search(r"```(?:python)?\s*(.*?)```", code, re.S | re.I)
    if fenced:
        return fenced.group(1).strip()
    opening_fence = re.match(r"^```(?:python)?\s*(.*)$", code, re.S | re.I)
    if opening_fence:
        code = opening_fence.group(1).strip()
        code = re.sub(r"\n```\s*$", "", code).strip()
    return code


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _is_turtle_terminal_call(node: ast.Call) -> bool:
    return _call_name(node).lower() in TURTLE_TERMINAL_CALLS


class _PaintCodeSanitizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.removed_calls: list[str] = []

    def visit_Expr(self, node: ast.Expr) -> ast.AST | None:
        if isinstance(node.value, ast.Call) and _is_turtle_terminal_call(node.value):
            self.removed_calls.append(_call_name(node.value))
            return None
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if _is_turtle_terminal_call(node):
            self.removed_calls.append(_call_name(node))
            return ast.Constant(value=None)
        return self.generic_visit(node)


def _sanitize_generated_code(code: str) -> str:
    tree = ast.parse(code)
    sanitizer = _PaintCodeSanitizer()
    sanitized_tree = sanitizer.visit(tree)
    ast.fix_missing_locations(sanitized_tree)
    if sanitizer.removed_calls:
        logger.info(
            "Removed blocking turtle calls from generated paint code: {}",
            sanitizer.removed_calls,
        )
    sanitized = ast.unparse(sanitized_tree).strip()
    return sanitized


def _validate_generated_code(code: str) -> None:
    tree = ast.parse(code)
    allowed_import_roots = {"turtle", "math", "random", "colorsys"}
    banned_call_names = {
        "__import__",
        "compile",
        "done",
        "eval",
        "exec",
        "exitonclick",
        "input",
        "mainloop",
        "open",
    }
    banned_import_roots = {
        "ctypes",
        "cv2",
        "matplotlib",
        "os",
        "pathlib",
        "PIL",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "tkinter",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in banned_import_roots or root not in allowed_import_roots:
                    raise ValueError(f"Unsupported import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in banned_import_roots or root not in allowed_import_roots:
                raise ValueError(f"Unsupported import: {node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in banned_call_names:
                raise ValueError(f"Unsupported call: {node.func.id}")
            if isinstance(node.func, ast.Attribute):
                attr = node.func.attr.lower()
                if attr in {
                    "bye",
                    "done",
                    "exitonclick",
                    "mainloop",
                    "mkdir",
                    "popen",
                    "remove",
                    "rmdir",
                    "system",
                    "unlink",
                }:
                    raise ValueError(f"Unsupported attribute call: {attr}")


def _load_paint_config() -> dict[str, Any]:
    try:
        config_data = read_yaml(PROJECT_ROOT / "conf.yaml") or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning("Failed to read paint_config from conf.yaml: {}", exc)
        return {}

    paint_config = config_data.get(PAINT_CONFIG_KEY) or {}
    if not isinstance(paint_config, dict):
        logger.warning("Ignoring invalid paint_config in conf.yaml: {}", type(paint_config))
        return {}
    return paint_config


def _config_text(config: Mapping[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _config_float(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid paint_config.{} value {!r}; using default {}",
            key,
            value,
            default,
        )
        return default


def _model_config(paint_config: Mapping[str, Any]) -> dict[str, Any]:
    provider = (
        _config_text(paint_config, "provider")
        or os.getenv("PAINT_PROVIDER")
        or "glm"
    ).strip().lower()
    config = PAINT_MODEL_CONFIGS.get(provider)
    if not config:
        raise RuntimeError(f"Unsupported paint_config.provider: {provider}")
    return {**config, "provider": provider}


def _api_key(config: dict[str, Any], paint_config: Mapping[str, Any]) -> str:
    api_key = _config_text(paint_config, "api_key")
    if api_key:
        return api_key

    env_names = tuple(config.get("api_key_envs") or ())
    api_key = next((os.getenv(name) for name in env_names if os.getenv(name)), None)
    if not api_key:
        raise RuntimeError(
            "Missing paint model API key: set paint_config.api_key in conf.yaml"
            + (f" or one of {', '.join(env_names)}" if env_names else "")
        )
    return api_key


def _save_paint_generation_attempt(
    work_dir: Path | None,
    *,
    attempt: int,
    raw_code: str,
    stripped_code: str | None = None,
    error: BaseException | None = None,
) -> None:
    if work_dir is None:
        return
    try:
        (work_dir / f"paint_raw_attempt_{attempt}.txt").write_text(
            raw_code,
            encoding="utf-8",
        )
        if stripped_code is not None:
            (work_dir / f"paint_code_attempt_{attempt}.py").write_text(
                stripped_code,
                encoding="utf-8",
            )
        if error is not None:
            (work_dir / f"paint_error_attempt_{attempt}.txt").write_text(
                f"{type(error).__name__}: {error}",
                encoding="utf-8",
            )
    except Exception as exc:
        logger.warning("Failed to save paint generation attempt {}: {}", attempt, exc)


def _collect_paint_code(client: OpenAI, request_params: dict[str, Any]) -> str:
    if not request_params.get("stream"):
        response = client.chat.completions.create(**request_params)
        return response.choices[0].message.content or ""

    chunks: list[str] = []
    stream = client.chat.completions.create(**request_params)
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            chunks.append(content)
    return "".join(chunks)


def _paint_retry_messages(
    *,
    prompt: str,
    failed_code: str,
    error: BaseException,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PAINT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"原始画面需求：{prompt}\n\n"
                "上一次生成的 turtle Python 代码无法解析或无法通过安全校验。\n"
                f"错误：{type(error).__name__}: {error}\n\n"
                "请从零重新生成一份完整、语法正确、可直接运行的 Python turtle 代码。"
                "只输出代码，不要 Markdown，不要解释。"
                "不要复用失败代码里的错误写法。"
                "所有函数调用、括号、字符串和缩进必须完整闭合。"
                "请控制在 100 行以内，优先使用简单几何图形完成画面。"
            ),
        },
    ]


def _fallback_paint_code(prompt: str) -> str:
    prompt_text = str(prompt or "").strip()[:80] or "Live Streaming Agent 的画"
    prompt_literal = repr(prompt_text)
    return f"""
import turtle
import math

turtle.tracer(False)
turtle.colormode(255)

pen = turtle.Turtle()
pen.hideturtle()
pen.speed(0)

def goto(x, y):
    pen.penup()
    pen.goto(x, y)
    pen.pendown()

def filled_circle(x, y, radius, color, outline=(70, 55, 45)):
    goto(x, y - radius)
    pen.color(outline, color)
    pen.begin_fill()
    pen.circle(radius)
    pen.end_fill()

def filled_rect(x, y, width, height, color, outline=(70, 55, 45)):
    goto(x, y)
    pen.color(outline, color)
    pen.begin_fill()
    for _ in range(2):
        pen.forward(width)
        pen.right(90)
        pen.forward(height)
        pen.right(90)
    pen.end_fill()

def line(x1, y1, x2, y2, color=(45, 40, 36), width=4):
    pen.color(color)
    pen.width(width)
    goto(x1, y1)
    pen.goto(x2, y2)
    pen.width(1)

# 复古纸张背景
filled_rect(-500, 360, 1000, 720, (246, 230, 187), (246, 230, 187))
for i in range(24):
    x = -480 + (i * 41) % 960
    y = -330 + (i * 73) % 660
    filled_circle(x, y, 3 + i % 5, (220, 196, 150), (220, 196, 150))

# 旧海报边框
pen.color(80, 58, 45)
pen.width(8)
goto(-455, 315)
for _ in range(2):
    pen.forward(910)
    pen.right(90)
    pen.forward(630)
    pen.right(90)
pen.width(1)

# 太阳和装饰
filled_circle(350, 235, 62, (255, 213, 91))
for angle in range(0, 360, 30):
    rad = math.radians(angle)
    line(350 + math.cos(rad) * 78, 235 + math.sin(rad) * 78, 350 + math.cos(rad) * 105, 235 + math.sin(rad) * 105, (192, 117, 45), 3)

# 主体：复古小战士/猫式简化角色
filled_circle(0, 110, 72, (240, 214, 172))
filled_circle(-25, 125, 9, (20, 20, 20))
filled_circle(25, 125, 9, (20, 20, 20))
line(-24, 88, 24, 88, (80, 45, 38), 5)
filled_rect(-80, 40, 160, 150, (92, 120, 76))
filled_rect(-105, -110, 62, 130, (86, 62, 48))
filled_rect(43, -110, 62, 130, (86, 62, 48))
line(-72, 25, -210, 120, (40, 40, 42), 7)
line(72, 25, 210, 120, (40, 40, 42), 7)
line(-210, 120, -285, 165, (190, 190, 185), 8)
line(210, 120, 285, 165, (190, 190, 185), 8)

# 杯子人
filled_rect(-365, -120, 95, 120, (245, 245, 230))
filled_circle(-318, -10, 45, (245, 245, 230))
filled_circle(-335, 0, 5, (30, 30, 30))
filled_circle(-300, 0, 5, (30, 30, 30))
line(-342, -35, -292, -35, (80, 50, 42), 4)
line(-275, -55, -205, -25, (80, 50, 42), 5)

# 标题和说明
goto(-390, 260)
pen.color(74, 51, 39)
pen.write("LIVE STREAMING AGENT PAINT TEST", font=("Arial", 30, "bold"))
goto(-390, -285)
pen.write({prompt_literal}, font=("Arial", 18, "normal"))

turtle.update()
""".strip()


def _generate_paint_code(prompt: str, work_dir: Path | None = None) -> str:
    paint_config = _load_paint_config()
    config = _model_config(paint_config)
    timeout_seconds = _config_float(
        paint_config,
        "timeout_seconds",
        PAINT_MODEL_TIMEOUT_SECONDS,
    )
    client = OpenAI(
        base_url=_config_text(paint_config, "base_url")
        or os.getenv("PAINT_BASE_URL")
        or config["base_url"],
        api_key=_api_key(config, paint_config),
        timeout=timeout_seconds,
    )
    messages = [
        {"role": "system", "content": PAINT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    request_params: dict[str, Any] = {
        "model": _config_text(paint_config, "model")
        or os.getenv("PAINT_MODEL_NAME")
        or config["model"],
        "messages": messages,
        "max_tokens": int(_config_float(paint_config, "max_tokens", 8192)),
        "temperature": 0,
        "stream": False,
    }
    if config.get("extra_body"):
        request_params["extra_body"] = config["extra_body"]

    last_error: BaseException | None = None
    for attempt in range(1, PAINT_CODE_GENERATION_ATTEMPTS + 1):
        request_params["messages"] = messages
        raw_code = _collect_paint_code(client, request_params)
        stripped_code = _strip_code_fence(raw_code)
        try:
            code = _sanitize_generated_code(stripped_code)
            _validate_generated_code(code)
            _save_paint_generation_attempt(
                work_dir,
                attempt=attempt,
                raw_code=raw_code,
                stripped_code=stripped_code,
            )
            return code
        except (SyntaxError, ValueError) as exc:
            last_error = exc
            _save_paint_generation_attempt(
                work_dir,
                attempt=attempt,
                raw_code=raw_code,
                stripped_code=stripped_code,
                error=exc,
            )
            if attempt >= PAINT_CODE_GENERATION_ATTEMPTS:
                raise
            logger.warning(
                "Paint code generation attempt {} failed validation: {}; retrying.",
                attempt,
                exc,
            )
            messages = _paint_retry_messages(
                prompt=prompt,
                failed_code=stripped_code,
                error=exc,
            )

    if last_error is not None:
        logger.warning(
            "Paint code generation failed after {} attempts; using local fallback: {}",
            PAINT_CODE_GENERATION_ATTEMPTS,
            last_error,
        )
        fallback_code = _fallback_paint_code(prompt)
        _save_paint_generation_attempt(
            work_dir,
            attempt=PAINT_CODE_GENERATION_ATTEMPTS + 1,
            raw_code=fallback_code,
            stripped_code=fallback_code,
            error=last_error,
        )
        return fallback_code
    raise RuntimeError("Paint code generation returned no code.")


def _run_paint_code(code: str, output_path: Path, work_dir: Path) -> Path:
    script_path = work_dir / "paint_generated.py"
    eps_path = output_path.with_suffix(".eps")
    script = (
        "import turtle\n"
        "\n"
        f"OUTPUT_PATH = {str(output_path)!r}\n"
        f"EPS_PATH = {str(eps_path)!r}\n"
        f"IMAGE_WIDTH = {PAINT_IMAGE_SIZE[0]}\n"
        f"IMAGE_HEIGHT = {PAINT_IMAGE_SIZE[1]}\n\n"
        "def _configure_paint_screen(bg=None):\n"
        "    screen = turtle.Screen()\n"
        "    screen.setup(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)\n"
        "    screen.screensize(IMAGE_WIDTH, IMAGE_HEIGHT)\n"
        "    if bg is not None:\n"
        "        screen.bgcolor(bg)\n"
        "    outer_canvas = screen.getcanvas()\n"
        "    tk_canvas = getattr(outer_canvas, '_canvas', outer_canvas)\n"
        "    for scrollbar_name in ('hscroll', 'vscroll'):\n"
        "        scrollbar = getattr(outer_canvas, scrollbar_name, None)\n"
        "        if scrollbar is not None:\n"
        "            try:\n"
        "                scrollbar.grid_forget()\n"
        "            except Exception:\n"
        "                pass\n"
        "    tk_canvas.configure(\n"
        "        width=IMAGE_WIDTH,\n"
        "        height=IMAGE_HEIGHT,\n"
        "        borderwidth=0,\n"
        "        highlightthickness=0,\n"
        "        relief='flat',\n"
        "    )\n"
        "    return screen, tk_canvas\n\n"
        "def _normalize_turtle_font(font):\n"
        "    if not isinstance(font, (tuple, list)):\n"
        "        return font\n"
        "    values = list(font)\n"
        "    size = next((item for item in values if isinstance(item, int)), None)\n"
        "    if size is None:\n"
        "        return font\n"
        "    style_values = {'normal', 'bold', 'italic', 'underline', 'overstrike'}\n"
        "    style = next(\n"
        "        (\n"
        "            item.lower()\n"
        "            for item in values\n"
        "            if isinstance(item, str) and item.lower() in style_values\n"
        "        ),\n"
        "        'normal',\n"
        "    )\n"
        "    family = next(\n"
        "        (\n"
        "            item\n"
        "            for item in values\n"
        "            if isinstance(item, str) and item.lower() not in style_values\n"
        "        ),\n"
        "        'Arial',\n"
        "    )\n"
        "    return (family, size, style)\n\n"
        "_ORIGINAL_TURTLE_WRITE = turtle.RawTurtle.write\n\n"
        "def _safe_turtle_write(\n"
        "    self,\n"
        "    arg,\n"
        "    move=False,\n"
        "    align='left',\n"
        "    font=('Arial', 8, 'normal'),\n"
        "):\n"
        "    return _ORIGINAL_TURTLE_WRITE(\n"
        "        self,\n"
        "        arg,\n"
        "        move,\n"
        "        align,\n"
        "        _normalize_turtle_font(font),\n"
        "    )\n\n"
        "turtle.RawTurtle.write = _safe_turtle_write\n"
        "turtle.Turtle.write = _safe_turtle_write\n\n"
        "def _normalize_turtle_color_arg(value, screen=None):\n"
        "    if isinstance(value, (tuple, list)) and len(value) >= 3:\n"
        "        values = list(value)\n"
        "        rgb = values[:3]\n"
        "        if all(isinstance(item, (int, float)) for item in rgb):\n"
        "            try:\n"
        "                current_colormode = screen.colormode() if screen is not None else None\n"
        "            except Exception:\n"
        "                current_colormode = None\n"
        "            use_255 = (\n"
        "                current_colormode == 255\n"
        "                or any(abs(item) > 1 for item in values if isinstance(item, (int, float)))\n"
        "            )\n"
        "            if use_255:\n"
        "                if screen is not None:\n"
        "                    try:\n"
        "                        screen.colormode(255)\n"
        "                    except Exception:\n"
        "                        pass\n"
        "                return tuple(max(0, min(255, int(round(item)))) for item in rgb)\n"
        "            return tuple(max(0.0, min(1.0, float(item))) for item in rgb)\n"
        "    return value\n\n"
        "def _normalize_turtle_color_args(screen, args):\n"
        "    if len(args) >= 3 and all(isinstance(arg, (int, float)) for arg in args[:3]):\n"
        "        return (_normalize_turtle_color_arg(args, screen),)\n"
        "    return tuple(_normalize_turtle_color_arg(arg, screen) for arg in args)\n\n"
        "_ORIGINAL_TURTLE_PENCOLOR = turtle.RawTurtle.pencolor\n"
        "_ORIGINAL_TURTLE_FILLCOLOR = turtle.RawTurtle.fillcolor\n"
        "_ORIGINAL_TURTLE_COLOR = turtle.RawTurtle.color\n\n"
        "def _safe_turtle_pencolor(self, *args):\n"
        "    return _ORIGINAL_TURTLE_PENCOLOR(\n"
        "        self,\n"
        "        *_normalize_turtle_color_args(self.screen, args),\n"
        "    )\n\n"
        "def _safe_turtle_fillcolor(self, *args):\n"
        "    return _ORIGINAL_TURTLE_FILLCOLOR(\n"
        "        self,\n"
        "        *_normalize_turtle_color_args(self.screen, args),\n"
        "    )\n\n"
        "def _safe_turtle_color(self, *args):\n"
        "    return _ORIGINAL_TURTLE_COLOR(\n"
        "        self,\n"
        "        *_normalize_turtle_color_args(self.screen, args),\n"
        "    )\n\n"
        "turtle.RawTurtle.pencolor = _safe_turtle_pencolor\n"
        "turtle.RawTurtle.fillcolor = _safe_turtle_fillcolor\n"
        "turtle.RawTurtle.color = _safe_turtle_color\n"
        "turtle.Turtle.pencolor = _safe_turtle_pencolor\n"
        "turtle.Turtle.fillcolor = _safe_turtle_fillcolor\n"
        "turtle.Turtle.color = _safe_turtle_color\n\n"
        "screen, canvas = _configure_paint_screen('white')\n"
        "turtle.tracer(False)\n\n"
        f"{code}\n"
        "\n"
        "screen, canvas = _configure_paint_screen()\n"
        "turtle.update()\n"
        "canvas.update()\n"
        "canvas.postscript(file=EPS_PATH, colormode='color')\n"
        "\n"
        "def _save_turtle_png():\n"
        "    try:\n"
        "        from PIL import Image\n"
        "        with Image.open(EPS_PATH) as image:\n"
        "            image.load()\n"
        "            image.convert('RGB').save(OUTPUT_PATH)\n"
        "            return\n"
        "    except Exception as eps_exc:\n"
        "        last_error = eps_exc\n"
        "    try:\n"
        "        from PIL import ImageGrab\n"
        "        root = canvas.winfo_toplevel()\n"
        "        root.lift()\n"
        "        root.attributes('-topmost', True)\n"
        "        root.update_idletasks()\n"
        "        root.update()\n"
        "        root.attributes('-topmost', False)\n"
        "        x0 = canvas.winfo_rootx()\n"
        "        y0 = canvas.winfo_rooty()\n"
        "        x1 = x0 + canvas.winfo_width()\n"
        "        y1 = y0 + canvas.winfo_height()\n"
        "        ImageGrab.grab((x0, y0, x1, y1)).save(OUTPUT_PATH)\n"
        "        return\n"
        "    except Exception as grab_exc:\n"
        "        raise RuntimeError(\n"
        "            f'Failed to save turtle canvas as PNG: '\n"
        "            f'eps={last_error!r}; grab={grab_exc!r}'\n"
        "        ) from grab_exc\n"
        "\n"
        "_save_turtle_png()\n"
        "try:\n"
        "    turtle.bye()\n"
        "except Exception:\n"
        "    pass\n"
    )
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(work_dir),
        text=True,
        capture_output=True,
        timeout=PAINT_CODE_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Paint code failed: "
            f"stdout={result.stdout[-1000:]!r} stderr={result.stderr[-1000:]!r}"
        )
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError("Paint code did not create an output image.")
    return eps_path


def _create_paint_image(prompt: str, request_id: str) -> dict[str, Any]:
    day_dir = PAINT_OUTPUT_ROOT / datetime.now().strftime("%Y-%m-%d")
    work_dir = day_dir / request_id
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = work_dir / "paint.png"

    started = time.perf_counter()
    code = _generate_paint_code(prompt, work_dir=work_dir)
    code_path = work_dir / "paint_code.py"
    code_path.write_text(code, encoding="utf-8")
    eps_path = _run_paint_code(code, output_path, work_dir)
    image_bytes = output_path.read_bytes()
    metadata = {
        "request_id": request_id,
        "prompt": prompt,
        "image_file": str(output_path),
        "eps_file": str(eps_path),
        "code_file": str(code_path),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (work_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        **metadata,
        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
        "mime_type": "image/png",
    }


class PaintManager:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.enabled = False

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    async def request_paint(
        self,
        *,
        prompt: str,
        websocket_send: WebSocketSend,
        turn_id: str | None = None,
    ) -> bool:
        prompt = str(prompt or "").strip()
        if not prompt:
            return False
        if not self.enabled:
            logger.info("Ignoring paint request because paint mode is disabled.")
            record_turn_event(
                turn_id,
                "paint_manager",
                "paint_request_ignored_disabled",
                prompt_len=len(prompt),
                prompt_preview=prompt[:120],
            )
            return False
        if self._lock.locked():
            logger.info("Ignoring paint request while another image is still running.")
            record_turn_event(
                turn_id,
                "paint_manager",
                "paint_request_ignored_busy",
                prompt_len=len(prompt),
                prompt_preview=prompt[:120],
            )
            return False

        request_id = uuid.uuid4().hex
        task = asyncio.create_task(
            self._run_request(
                prompt=prompt,
                websocket_send=websocket_send,
                request_id=request_id,
                turn_id=turn_id,
            )
        )
        task.add_done_callback(self._consume_task_result)
        return True

    async def _run_request(
        self,
        *,
        prompt: str,
        websocket_send: WebSocketSend,
        request_id: str,
        turn_id: str | None,
    ) -> None:
        async with self._lock:
            await websocket_send(
                json.dumps(
                    {
                        "type": "paint-state",
                        "state": "started",
                        "request_id": request_id,
                        "turn_id": turn_id,
                        "prompt": prompt,
                    },
                    ensure_ascii=False,
                )
            )
            record_turn_event(
                turn_id,
                "paint_manager",
                "paint_started",
                request_id=request_id,
                prompt_len=len(prompt),
                prompt_preview=prompt[:120],
            )
            try:
                result = await asyncio.to_thread(
                    _create_paint_image,
                    prompt,
                    request_id,
                )
            except Exception as exc:
                logger.exception("Paint generation failed: {}", exc)
                if not self.enabled:
                    record_turn_event(
                        turn_id,
                        "paint_manager",
                        "paint_error_dropped_disabled",
                        request_id=request_id,
                        error=str(exc),
                    )
                    return
                await websocket_send(
                    json.dumps(
                        {
                            "type": "paint-state",
                            "state": "error",
                            "request_id": request_id,
                            "turn_id": turn_id,
                            "prompt": prompt,
                            "message": str(exc),
                        },
                        ensure_ascii=False,
                    )
                )
                record_turn_event(
                    turn_id,
                    "paint_manager",
                    "paint_failed",
                    request_id=request_id,
                    error=str(exc),
                )
                return

            if not self.enabled:
                record_turn_event(
                    turn_id,
                    "paint_manager",
                    "paint_completed_dropped_disabled",
                    request_id=request_id,
                    image_file=result["image_file"],
                )
                return

            await websocket_send(
                json.dumps(
                    {
                        "type": "paint-state",
                        "state": "completed",
                        "request_id": request_id,
                        "turn_id": turn_id,
                        "prompt": prompt,
                        "image": result["image_base64"],
                        "mime_type": result["mime_type"],
                        "image_file": result["image_file"],
                        "elapsed_seconds": result["elapsed_seconds"],
                    },
                    ensure_ascii=False,
                )
            )
            record_turn_event(
                turn_id,
                "paint_manager",
                "paint_completed",
                request_id=request_id,
                image_file=result["image_file"],
                elapsed_seconds=result["elapsed_seconds"],
            )

    @staticmethod
    def _consume_task_result(task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Unexpected paint task error: {}", exc)


_paint_manager = PaintManager()


def get_paint_manager() -> PaintManager:
    return _paint_manager
