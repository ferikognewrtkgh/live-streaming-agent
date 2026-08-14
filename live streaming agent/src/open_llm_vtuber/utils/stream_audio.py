import base64
import io
from typing import Any

from loguru import logger
from pydub import AudioSegment
from pydub.utils import make_chunks
from ..agent.output_types import Actions
from ..agent.output_types import DisplayText
from .turn_trace import record_turn_event


TTS_TAIL_TRIM_MIN_OVERFLOW_MS = 1800
TTS_TAIL_TRIM_MIN_CAP_MS = 2200
TTS_TAIL_TRIM_BASE_MS = 1200
TTS_TAIL_TRIM_PER_CHAR_MS = 260
TTS_TAIL_TRIM_PUNCTUATION_MS = 180
TTS_TAIL_TRIM_FADE_OUT_MS = 80
TTS_TAIL_TRIM_PUNCTUATION = set("，。！？,.!?、；;：:")


def _display_text_to_dict(display_text: DisplayText | dict | None) -> dict | None:
    if isinstance(display_text, DisplayText):
        return display_text.to_dict()
    return display_text


def _actions_to_dict(actions: Actions | dict | None) -> dict | None:
    if actions is None:
        return None

    if isinstance(actions, Actions):
        actions_dict = actions.to_dict()
    elif isinstance(actions, dict):
        actions_dict = actions
    else:
        return None

    if actions_dict:
        return actions_dict
    return {"emotions": []}


def _get_volume_by_chunks(audio: AudioSegment, chunk_length_ms: int) -> list:
    """
    Calculate the normalized volume (RMS) for each chunk of the audio.

    Parameters:
        audio (AudioSegment): The audio segment to process.
        chunk_length_ms (int): The length of each audio chunk in milliseconds.

    Returns:
        list: Normalized volumes for each chunk.
    """
    chunks = make_chunks(audio, chunk_length_ms)
    volumes = [chunk.rms for chunk in chunks]
    if not volumes:
        return []
    max_volume = max(volumes)
    if max_volume == 0:
        return [0.0 for _ in volumes]
    return [volume / max_volume for volume in volumes]


def _text_for_duration_estimate(text: str | None) -> str:
    if not text:
        return ""
    return "".join(char for char in str(text).strip() if not char.isspace())


def _estimate_tts_max_duration_ms(text: str | None) -> int | None:
    normalized = _text_for_duration_estimate(text)
    if not normalized:
        return None

    punctuation_count = sum(1 for char in normalized if char in TTS_TAIL_TRIM_PUNCTUATION)
    content_count = sum(1 for char in normalized if char not in TTS_TAIL_TRIM_PUNCTUATION)
    if content_count <= 0:
        return None

    return max(
        TTS_TAIL_TRIM_MIN_CAP_MS,
        int(
            TTS_TAIL_TRIM_BASE_MS
            + content_count * TTS_TAIL_TRIM_PER_CHAR_MS
            + punctuation_count * TTS_TAIL_TRIM_PUNCTUATION_MS
        ),
    )


def _trim_tts_long_tail(
    audio: AudioSegment,
    *,
    text: str | None,
    turn_id: str | None,
) -> tuple[AudioSegment, dict[str, Any] | None]:
    """Trim obvious GPT-SoVITS hallucinated tails such as long humming after text."""
    max_duration_ms = _estimate_tts_max_duration_ms(text)
    if max_duration_ms is None:
        return audio, None

    original_duration_ms = len(audio)
    if original_duration_ms <= max_duration_ms + TTS_TAIL_TRIM_MIN_OVERFLOW_MS:
        return audio, None

    trimmed = audio[:max_duration_ms]
    fade_out_ms = min(TTS_TAIL_TRIM_FADE_OUT_MS, max(len(trimmed) // 3, 0))
    if fade_out_ms > 0:
        trimmed = trimmed.fade_out(fade_out_ms)

    trim_info = {
        "original_duration_ms": original_duration_ms,
        "trimmed_duration_ms": len(trimmed),
        "max_duration_ms": max_duration_ms,
        "text_len": len(_text_for_duration_estimate(text)),
    }
    logger.warning(
        "Trimmed abnormal TTS long tail: text={!r} original_ms={} trimmed_ms={} "
        "max_ms={} turn_id={}",
        text,
        original_duration_ms,
        len(trimmed),
        max_duration_ms,
        turn_id,
    )
    return trimmed, trim_info


def prepare_audio_payload(
    audio_path: str | None,
    chunk_length_ms: int = 20,
    display_text: DisplayText = None,
    actions: Actions = None,
    forwarded: bool = False,
    turn_id: str | None = None,
    trim_long_tts_tail: bool = False,
    tts_text: str | None = None,
) -> dict[str, Any]:
    """
    Prepares the audio payload for sending to a broadcast endpoint.
    If audio_path is None, returns a payload with audio=None for silent display.

    Parameters:
        audio_path (str | None): The path to the audio file to be processed, or None for silent display
        chunk_length_ms (int): The length of each audio chunk in milliseconds
        display_text (DisplayText, optional): Text to be displayed with the audio
        actions (Actions, optional): Actions associated with the audio

    Returns:
        dict: The audio payload to be sent
    """
    display_text = _display_text_to_dict(display_text)

    if not audio_path:
        # Return payload for silent display
        payload = {
            "type": "audio",
            "audio": None,
            "volumes": [],
            "slice_length": chunk_length_ms,
            "display_text": display_text,
            "actions": _actions_to_dict(actions),
            "forwarded": forwarded,
        }
        if turn_id:
            payload["turn_id"] = turn_id
        record_turn_event(
            turn_id,
            "stream_audio",
            "silent_payload_prepared",
            display_text=display_text,
            actions=payload["actions"],
            forwarded=forwarded,
        )
        return payload

    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        raise ValueError(
            f"Error loading generated audio file '{audio_path}': {e}"
        ) from e
    return prepare_audio_payload_from_segment(
        audio=audio,
        chunk_length_ms=chunk_length_ms,
        display_text=display_text,
        actions=actions,
        forwarded=forwarded,
        turn_id=turn_id,
        trim_long_tts_tail=trim_long_tts_tail,
        tts_text=tts_text,
    )


def prepare_audio_payload_from_bytes(
    audio_bytes: bytes,
    chunk_length_ms: int = 20,
    display_text: DisplayText = None,
    actions: Actions = None,
    forwarded: bool = False,
    audio_format: str = "wav",
    turn_id: str | None = None,
    trim_long_tts_tail: bool = False,
    tts_text: str | None = None,
) -> dict[str, Any]:
    """Prepare an audio websocket payload from in-memory audio bytes."""
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
    except Exception as e:
        raise ValueError(f"Error loading generated audio bytes: {e}") from e
    return prepare_audio_payload_from_segment(
        audio=audio,
        chunk_length_ms=chunk_length_ms,
        display_text=display_text,
        actions=actions,
        forwarded=forwarded,
        turn_id=turn_id,
        trim_long_tts_tail=trim_long_tts_tail,
        tts_text=tts_text,
    )


def prepare_audio_payload_from_segment(
    audio: AudioSegment,
    chunk_length_ms: int = 20,
    display_text: DisplayText = None,
    actions: Actions = None,
    forwarded: bool = False,
    turn_id: str | None = None,
    trim_long_tts_tail: bool = False,
    tts_text: str | None = None,
) -> dict[str, Any]:
    """Prepare an audio websocket payload from a pydub audio segment."""
    display_text = _display_text_to_dict(display_text)
    trim_info = None
    if trim_long_tts_tail:
        text_for_trim = tts_text
        if not text_for_trim and isinstance(display_text, dict):
            text_for_trim = str(display_text.get("text") or "")
        audio, trim_info = _trim_tts_long_tail(
            audio,
            text=text_for_trim,
            turn_id=turn_id,
        )

    audio_bytes = audio.export(format="wav").read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    volumes = _get_volume_by_chunks(audio, chunk_length_ms)

    payload = {
        "type": "audio",
        "audio": audio_base64,
        "volumes": volumes,
        "slice_length": chunk_length_ms,
        "display_text": display_text,
        "actions": _actions_to_dict(actions),
        "forwarded": forwarded,
    }
    if turn_id:
        payload["turn_id"] = turn_id
    record_turn_event(
        turn_id,
        "stream_audio",
        "audio_payload_prepared",
        duration_ms=len(audio),
        audio_bytes=len(audio_bytes),
        base64_chars=len(audio_base64),
        volumes_count=len(volumes),
        chunk_length_ms=chunk_length_ms,
        trim_info=trim_info,
        display_text=display_text,
        actions=payload["actions"],
        forwarded=forwarded,
    )
    return payload


# Example usage:
# payload, duration = prepare_audio_payload("path/to/audio.mp3", display_text="Hello", expression_list=[0,1,2])
