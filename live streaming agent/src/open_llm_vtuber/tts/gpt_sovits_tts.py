import asyncio
import io
import json
import os
import re
import struct
from typing import Any, AsyncIterator, Iterator
from urllib.parse import urljoin

import aiohttp
import requests
from loguru import logger
from pydub import AudioSegment

from ..emotion_tags import normalize_tts_emotion_tag
from .tts_interface import TTSInterface, TTSStreamChunk


class TTSEngine(TTSInterface):
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:9880/tts",
        text_lang: str = "zh",
        ref_audio_path: str = "",
        prompt_lang: str = "zh",
        prompt_text: str = "",
        text_split_method: str = "cut5",
        batch_size: str = "1",
        media_type: str = "wav",
        streaming_mode: str = "false",
        stream_api_url: str | None = None,
        default_emotion: str = "normal",
        speed: float = 1.0,
        first: bool = True,
        request_timeout: float = 360.0,
        stream_read_timeout: float = 30.0,
    ):
        self.api_url = api_url or "http://127.0.0.1:9880/tts"
        self.stream_api_url = self._resolve_stream_api_url(
            self.api_url, stream_api_url
        )
        self.text_lang = text_lang or "zh"
        self.ref_audio_path = os.path.abspath(ref_audio_path) if ref_audio_path else ""
        self.prompt_lang = prompt_lang or "zh"
        self.prompt_text = prompt_text or ""
        self.text_split_method = text_split_method or "cut5"
        self.batch_size = batch_size or "1"
        self.media_type = media_type or "wav"
        self.streaming_mode = streaming_mode or "false"
        self.default_emotion = default_emotion or "normal"
        self.speed = float(speed) if speed is not None else 1.0
        self.first = True if first is None else bool(first)
        self.request_timeout = (
            float(request_timeout) if request_timeout is not None else 360.0
        )
        self.stream_read_timeout = (
            float(stream_read_timeout) if stream_read_timeout is not None else 30.0
        )

        logger.info(
            "TTS Engine Config initialized: {}",
            {
                "api_url": self.api_url,
                "stream_api_url": self.stream_api_url,
                "text_lang": self.text_lang,
                "ref_audio_path": self.ref_audio_path,
                "prompt_lang": self.prompt_lang,
                "prompt_text": self.prompt_text,
                "text_split_method": self.text_split_method,
                "batch_size": self.batch_size,
                "media_type": self.media_type,
                "streaming_mode": self.streaming_mode,
                "default_emotion": self.default_emotion,
                "speed": self.speed,
                "first": self.first,
                "request_timeout": self.request_timeout,
                "stream_read_timeout": self.stream_read_timeout,
            },
        )

    @staticmethod
    def _resolve_stream_api_url(api_url: str, stream_api_url: str | None) -> str | None:
        if stream_api_url:
            return stream_api_url
        if api_url and api_url.rstrip("/").endswith("/tts/stream"):
            return api_url
        return None

    def supports_streaming_audio(self) -> bool:
        return bool(self.stream_api_url)

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\[.*?\]", "", text).strip()

    def _stream_payload(
        self,
        text: str,
        mood: str | None,
        is_first: bool | None = None,
    ) -> dict[str, Any]:
        requested_emotion = mood or self.default_emotion
        emotion = normalize_tts_emotion_tag(requested_emotion)
        if not emotion and requested_emotion != self.default_emotion:
            emotion = normalize_tts_emotion_tag(self.default_emotion)
        if not emotion:
            emotion = "default"
        return {
            "emotion": emotion,
            "text": self._clean_text(text),
            "speed": self.speed,
            "is_first": self.first if is_first is None else bool(is_first),
        }

    def _legacy_payload(self, text: str) -> dict[str, Any]:
        return {
            "text": self._clean_text(text),
            "text_lang": self.text_lang,
            "ref_audio_path": self.ref_audio_path,
            "prompt_lang": self.prompt_lang,
            "prompt_text": self.prompt_text,
            "text_split_method": self.text_split_method,
            "batch_size": self.batch_size,
            "media_type": self.media_type,
            "streaming_mode": self.streaming_mode,
        }

    def _make_url(self, base_url: str, path: str) -> str:
        return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))

    def _read_stream_frames_sync(
        self, response: requests.Response
    ) -> Iterator[tuple[dict[str, Any], bytes]]:
        stream = response.raw
        while True:
            header = self._read_exact_sync(stream, 4)
            if not header:
                break
            metadata_len = struct.unpack(">I", header)[0]
            metadata_bytes = self._read_exact_sync(stream, metadata_len)
            metadata = json.loads(metadata_bytes.decode("utf-8"))
            audio_len_bytes = self._read_exact_sync(stream, 4)
            audio_len = struct.unpack(">I", audio_len_bytes)[0]
            audio_bytes = self._read_exact_sync(stream, audio_len)
            yield metadata, audio_bytes
            if metadata.get("type") == "error":
                break

    def _read_exact_sync(self, stream: Any, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                if remaining == size:
                    return b""
                raise EOFError(f"Unexpected EOF while reading {size} bytes")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    async def _read_stream_frames_async(
        self, content: aiohttp.StreamReader
    ) -> AsyncIterator[tuple[dict[str, Any], bytes]]:
        while True:
            try:
                header = await content.readexactly(4)
            except asyncio.IncompleteReadError as exc:
                if not exc.partial:
                    break
                raise EOFError("Unexpected EOF while reading metadata length") from exc
            metadata_len = struct.unpack(">I", header)[0]
            try:
                metadata_bytes = await content.readexactly(metadata_len)
                audio_len_bytes = await content.readexactly(4)
                audio_len = struct.unpack(">I", audio_len_bytes)[0]
                audio_bytes = await content.readexactly(audio_len)
            except asyncio.IncompleteReadError as exc:
                raise EOFError("Unexpected EOF while reading TTS stream frame") from exc
            metadata = json.loads(metadata_bytes.decode("utf-8"))
            yield metadata, audio_bytes
            if metadata.get("type") == "error":
                break

    def generate_audio(self, text: str, file_name_no_ext=None) -> str | None:
        if self.supports_streaming_audio():
            return self._generate_audio_from_stream(text, file_name_no_ext)

        file_name = self.generate_cache_file_name(file_name_no_ext, self.media_type)
        data = self._legacy_payload(text)
        logger.info("start to get legacy GPT-SoVITS TTS: {}", data)

        response = requests.get(self.api_url, params=data, timeout=120)

        if response.status_code == 200:
            with open(file_name, "wb") as audio_file:
                audio_file.write(response.content)
            return file_name

        logger.critical(
            "Error: Failed to generate audio. Status code: {}, body: {}",
            response.status_code,
            response.text[:500],
        )
        return None

    def _generate_audio_from_stream(
        self, text: str, file_name_no_ext=None, mood: str | None = None
    ) -> str | None:
        if not self.stream_api_url:
            return None

        file_name = self.generate_cache_file_name(file_name_no_ext, "wav")
        payload = self._stream_payload(text, mood)
        logger.info("start to get streaming GPT-SoVITS TTS: {}", payload)
        response = requests.post(
            self.stream_api_url,
            json=payload,
            stream=True,
            timeout=(10.0, self.stream_read_timeout),
        )
        if response.status_code != 200:
            logger.critical(
                "Error: Failed to generate streaming audio. Status code: {}, body: {}",
                response.status_code,
                response.text[:500],
            )
            return None

        combined = AudioSegment.empty()
        for metadata, audio_bytes in self._read_stream_frames_sync(response):
            frame_type = metadata.get("type")
            if frame_type == "error":
                raise RuntimeError(metadata.get("message", "Unknown TTS stream error"))
            if frame_type != "chunk" or not audio_bytes:
                continue
            combined += AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
            if metadata.get("is_final_chunk"):
                break

        if len(combined) == 0:
            logger.error("Streaming GPT-SoVITS returned no audio chunks.")
            return None

        combined.export(file_name, format="wav")
        return file_name

    async def async_generate_audio_stream(
        self,
        text: str,
        file_name_no_ext=None,
        mood: str | None = None,
        is_first: bool = False,
    ) -> AsyncIterator[TTSStreamChunk]:
        if not self.stream_api_url:
            audio_path = await self.async_generate_audio(text, file_name_no_ext)
            if audio_path:
                yield TTSStreamChunk(audio_path=audio_path, text=text)
            return

        payload = self._stream_payload(text, mood, is_first=is_first)
        logger.info("start to get streaming GPT-SoVITS TTS: {}", payload)
        timeout = aiohttp.ClientTimeout(
            total=self.request_timeout,
            connect=10.0,
            sock_connect=10.0,
            sock_read=self.stream_read_timeout,
        )
        response: aiohttp.ClientResponse | None = None
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                response = await session.post(self.stream_api_url, json=payload)
                try:
                    if response.status != 200:
                        body = await response.text()
                        raise RuntimeError(
                            "Failed to generate streaming audio. "
                            f"Status code: {response.status}, body: {body[:500]}"
                        )

                    async for metadata, audio_bytes in self._read_stream_frames_async(
                        response.content
                    ):
                        frame_type = metadata.get("type")
                        if frame_type == "error":
                            raise RuntimeError(
                                metadata.get("message", "Unknown TTS stream error")
                            )
                        if frame_type != "chunk" or not audio_bytes:
                            continue
                        yield TTSStreamChunk(
                            audio_bytes=audio_bytes,
                            text=str(metadata.get("text") or ""),
                            metadata=metadata,
                        )
                        if metadata.get("is_final_chunk"):
                            break
                finally:
                    response.close()
        except asyncio.CancelledError:
            if response is not None and not response.closed:
                response.close()
            logger.debug("GPT-SoVITS stream request cancelled.")
            raise
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                "Timed out while waiting for GPT-SoVITS stream data "
                f"after {self.stream_read_timeout:.1f}s"
            ) from exc
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"GPT-SoVITS stream request failed: {exc!r}") from exc
