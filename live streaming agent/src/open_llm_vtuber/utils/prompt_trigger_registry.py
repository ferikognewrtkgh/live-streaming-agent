from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRIGGER_PROMPT_DIR = PROJECT_ROOT / "resource" / "prompt" / "triggers"


@dataclass(frozen=True)
class TriggerPrompt:
    text: str
    expression: str | None = None
    raw: dict[str, Any] | None = None


class PromptTriggerRegistry:
    """Load and serve reusable trigger prompts from resource/prompt/triggers."""

    def __init__(self, prompt_dir: Path = DEFAULT_TRIGGER_PROMPT_DIR) -> None:
        self.prompt_dir = prompt_dir
        self._prompts: dict[str, list[TriggerPrompt]] = {}
        self._queues: dict[str, list[TriggerPrompt]] = {}
        self._last_prompt: dict[str, TriggerPrompt] = {}
        self._loaded = False

    def load_all(self, force: bool = False) -> None:
        if self._loaded and not force:
            return

        prompts: dict[str, list[TriggerPrompt]] = {}
        if not self.prompt_dir.exists():
            logger.warning("Trigger prompt directory does not exist: {}", self.prompt_dir)
            self._prompts = prompts
            self._queues = {}
            self._last_prompt = {}
            self._loaded = True
            return

        for path in sorted(self.prompt_dir.glob("*.json")):
            prompts[path.stem] = self._load_prompt_file(path)

        self._prompts = prompts
        self._queues = {}
        self._last_prompt = {}
        self._loaded = True
        logger.info(
            "Loaded trigger prompt groups: {}",
            {name: len(items) for name, items in prompts.items()},
        )

    def get_next(self, trigger_name: str) -> TriggerPrompt | None:
        self.load_all()
        prompts = self._prompts.get(trigger_name, [])
        if not prompts:
            logger.warning("No trigger prompts found for {!r}", trigger_name)
            return None

        queue = self._queues.get(trigger_name)
        if not queue:
            queue = list(prompts)
            random.shuffle(queue)
            last_prompt = self._last_prompt.get(trigger_name)
            if len(queue) > 1 and last_prompt is not None and queue[0] == last_prompt:
                queue.append(queue.pop(0))
            self._queues[trigger_name] = queue

        prompt = queue.pop(0)
        self._last_prompt[trigger_name] = prompt
        return prompt

    def get_random(self, trigger_name: str) -> TriggerPrompt | None:
        return self.get_next(trigger_name)

    def _load_prompt_file(self, path: Path) -> list[TriggerPrompt]:
        try:
            with path.open("r", encoding="utf-8") as prompt_file:
                data = json.load(prompt_file)
        except Exception:
            logger.exception("Failed to load trigger prompt file: {}", path)
            return []

        if isinstance(data, dict):
            data = data.get("prompts") or data.get("items") or data.get("triggers") or []
        if not isinstance(data, list):
            logger.warning("Trigger prompt file must contain a list: {}", path)
            return []

        prompts = []
        for index, item in enumerate(data):
            prompt = self._parse_prompt_item(item)
            if not prompt:
                logger.warning("Skipping invalid trigger prompt {}[{}]: {}", path, index, item)
                continue
            prompts.append(prompt)
        return prompts

    def _parse_prompt_item(self, item: Any) -> TriggerPrompt | None:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return None
            return TriggerPrompt(text=text)

        if not isinstance(item, dict):
            return None

        text = str(item.get("text") or item.get("content") or "").strip()
        if not text:
            return None

        expression = item.get("expression")
        return TriggerPrompt(
            text=text,
            expression=str(expression).strip() if expression else None,
            raw=item,
        )


_REGISTRY = PromptTriggerRegistry()


def get_prompt_trigger_registry() -> PromptTriggerRegistry:
    return _REGISTRY
