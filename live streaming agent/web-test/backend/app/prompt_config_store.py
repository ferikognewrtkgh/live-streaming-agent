import re
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException, status

from .config import BACKEND_DIR, Settings
from .schemas import (
    PersonaPromptVersion,
    PromptConfigResponse,
    PromptConfigUpdate,
    SpeakerPromptVersion,
)

NO_SPEAKER_VERSION = "__none__"


def ensure_persona_prompt_identity(
    content: str,
    speaker_identity: str,
) -> str:
    normalized_identity = speaker_identity.strip()
    if not content.strip() or not normalized_identity:
        return content
    if normalized_identity in content:
        return content
    return f"{normalized_identity}是：\n{content}"


class PromptConfigStore:
    def __init__(self, settings: Settings) -> None:
        resources_path = settings.prompt_resources_path
        self.resources_path = (
            resources_path
            if resources_path.is_absolute()
            else BACKEND_DIR / resources_path
        )

    def get_config(
        self,
        document: dict[str, Any] | None = None,
    ) -> PromptConfigResponse:
        document = document or {}
        persona_versions = self._load_persona_versions()
        active_version = str(document.get("active_version") or "")
        if active_version not in {
            version.version for version in persona_versions
        }:
            active_version = self._default_persona_version(persona_versions)

        speaker_versions = self._load_speaker_versions(document)
        active_speaker_version = str(
            document.get("active_speaker_version") or NO_SPEAKER_VERSION
        )
        if active_speaker_version not in {
            version.version for version in speaker_versions
        } | {NO_SPEAKER_VERSION}:
            active_speaker_version = NO_SPEAKER_VERSION

        persona_prompt = self._find_version(
            persona_versions,
            active_version,
            "persona",
        ).content
        active_speaker = next(
            (
                version
                for version in speaker_versions
                if version.version == active_speaker_version
            ),
            None,
        )
        speaker_prompt = active_speaker.content if active_speaker else ""
        speaker_identity = (
            active_speaker.speaker_identity if active_speaker else ""
        )
        persona_prompt = ensure_persona_prompt_identity(
            persona_prompt,
            speaker_identity,
        )
        return PromptConfigResponse(
            active_version=active_version,
            persona_prompt=persona_prompt,
            speaker_prompt=speaker_prompt,
            speaker_identity=speaker_identity,
            versions=persona_versions,
            active_speaker_version=active_speaker_version,
            speaker_versions=speaker_versions,
        )

    def save_config(
        self,
        payload: PromptConfigUpdate,
        document: dict[str, Any] | None = None,
    ) -> tuple[PromptConfigResponse, dict[str, Any]]:
        current = self.get_config(document)
        active_version = payload.active_version or current.active_version
        self._find_version(current.versions, active_version, "persona")

        speaker_versions = list(current.speaker_versions)
        active_speaker_version = (
            payload.active_speaker_version or current.active_speaker_version
        )
        if active_speaker_version != NO_SPEAKER_VERSION:
            self._find_version(
                speaker_versions,
                active_speaker_version,
                "speaker",
            )
        if payload.create_speaker_version:
            if not payload.speaker_identity:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="speaker identity is required",
                )
            active_speaker_version = self._next_speaker_version(
                speaker_versions
            )
            speaker_versions.append(
                SpeakerPromptVersion(
                    version=active_speaker_version,
                    title=self._next_speaker_title(
                        speaker_versions,
                        payload.speaker_identity,
                    ),
                    content=payload.speaker_prompt or "",
                    speaker_identity=payload.speaker_identity,
                )
            )
        if payload.update_speaker_version is not None:
            updated_version = self._find_version(
                speaker_versions,
                payload.update_speaker_version,
                "speaker",
            )
            if not payload.speaker_identity:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="speaker identity is required",
                )
            speaker_versions = [
                (
                    version.model_copy(
                        update={
                            "content": payload.speaker_prompt or "",
                            "speaker_identity": payload.speaker_identity,
                        }
                    )
                    if version.version == updated_version.version
                    else version
                )
                for version in speaker_versions
            ]
        if payload.delete_speaker_version is not None:
            self._find_version(
                speaker_versions,
                payload.delete_speaker_version,
                "speaker",
            )
            speaker_versions = [
                version
                for version in speaker_versions
                if version.version != payload.delete_speaker_version
            ]
            if active_speaker_version == payload.delete_speaker_version:
                active_speaker_version = NO_SPEAKER_VERSION
        if payload.rename_speaker_version is not None:
            renamed_version = self._find_version(
                speaker_versions,
                payload.rename_speaker_version,
                "speaker",
            )
            if payload.speaker_version_title is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="speaker version title is required",
                )
            speaker_versions = [
                (
                    version.model_copy(
                        update={"title": payload.speaker_version_title}
                    )
                    if version.version == renamed_version.version
                    else version
                )
                for version in speaker_versions
            ]

        updated_document = {
            "active_version": active_version,
            "active_speaker_version": active_speaker_version,
            "speaker_versions": [
                version.model_dump(mode="json") for version in speaker_versions
            ],
            "updated_at": datetime.now(UTC).isoformat(),
        }
        return self.get_config(updated_document), updated_document

    def _load_persona_versions(self) -> list[PersonaPromptVersion]:
        try:
            prompt_files = sorted(
                (
                    path
                    for path in self.resources_path.iterdir()
                    if path.is_file() and not path.name.startswith(".")
                ),
                key=lambda path: path.name.casefold(),
            )
        except OSError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"prompt resources are unavailable: {error}",
            ) from error
        versions: list[PersonaPromptVersion] = []
        for prompt_file in prompt_files:
            try:
                content = prompt_file.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"failed to read prompt file: {prompt_file.name}",
                ) from error
            versions.append(
                PersonaPromptVersion(
                    version=prompt_file.name,
                    title=prompt_file.stem,
                    content=content,
                )
            )
        if not versions:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"no prompt files found in {self.resources_path}",
            )
        return versions

    @staticmethod
    def _default_persona_version(
        versions: list[PersonaPromptVersion],
    ) -> str:
        preferred = next(
            (version for version in versions if version.version == "Live Streaming Agent.txt"),
            None,
        )
        return (preferred or versions[0]).version

    @staticmethod
    def _load_speaker_versions(
        document: dict[str, Any],
    ) -> list[SpeakerPromptVersion]:
        raw_versions = document.get("speaker_versions")
        versions: list[SpeakerPromptVersion] = []
        if isinstance(raw_versions, list):
            for item in raw_versions:
                try:
                    version = SpeakerPromptVersion.model_validate(item)
                    if version.content or version.speaker_identity:
                        versions.append(version)
                except (TypeError, ValueError):
                    continue
        if versions:
            return versions

        legacy_prompt = str(document.get("speaker_prompt") or "")
        if not legacy_prompt:
            return []
        return [
            SpeakerPromptVersion(
                version="v1.0",
                title="v1.0",
                content=legacy_prompt,
            )
        ]

    @staticmethod
    def _next_speaker_version(
        versions: list[SpeakerPromptVersion],
    ) -> str:
        numbers = [
            int(match.group(1))
            for version in versions
            if (match := re.fullmatch(r"v1\.(\d+)", version.version))
        ]
        return f"v1.{max(numbers, default=-1) + 1}"

    @staticmethod
    def _next_speaker_title(
        versions: list[SpeakerPromptVersion],
        speaker_identity: str,
    ) -> str:
        pattern = re.compile(rf"{re.escape(speaker_identity)}(\d+)")
        identity_exists = any(
            version.title == speaker_identity for version in versions
        )
        numbers = [
            int(match.group(1))
            for version in versions
            if (match := pattern.fullmatch(version.title))
        ]
        if not identity_exists and not numbers:
            return speaker_identity
        return f"{speaker_identity}{max(numbers, default=0) + 1}"

    @staticmethod
    def _find_version(
        versions: list[PersonaPromptVersion],
        version: str,
        prompt_kind: str,
    ) -> PersonaPromptVersion:
        for prompt_version in versions:
            if prompt_version.version == version:
                return prompt_version
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"unknown {prompt_kind} prompt version: {version}",
        )
