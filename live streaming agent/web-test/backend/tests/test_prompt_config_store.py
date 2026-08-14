import pytest
from backend.app.config import Settings
from backend.app.prompt_config_store import (
    PromptConfigStore,
    ensure_persona_prompt_identity,
)
from backend.app.schemas import PromptConfigUpdate
from fastapi import HTTPException


def build_store(tmp_path) -> PromptConfigStore:
    resources_path = tmp_path / "resource" / "prompt"
    resources_path.mkdir(parents=True)
    (resources_path / "Live Streaming Agent.txt").write_text(
        "你是从文件读取的 Live Streaming Agent。",
        encoding="utf-8",
    )
    (resources_path / "Live Streaming Agent_1.txt").write_text(
        "你是第二版 Live Streaming Agent。",
        encoding="utf-8",
    )
    return PromptConfigStore(
        Settings(
            prompt_resources_path=resources_path,
        )
    )


def test_prompt_config_store_reads_personas_from_resource_files(tmp_path) -> None:
    store = build_store(tmp_path)

    config = store.get_config()

    assert config.active_version == "Live Streaming Agent.txt"
    assert config.persona_prompt == "你是从文件读取的 Live Streaming Agent。"
    assert [version.title for version in config.versions] == [
        "Live Streaming Agent",
        "Live Streaming Agent_1",
    ]
    assert config.speaker_prompt == ""
    assert config.speaker_identity == ""
    assert config.active_speaker_version == "__none__"
    assert config.speaker_versions == []


def test_prompt_config_store_versions_speaker_prompts_independently(
    tmp_path,
) -> None:
    store = build_store(tmp_path)

    saved, document = store.save_config(
        PromptConfigUpdate(
            active_version="Live Streaming Agent_1.txt",
            speaker_prompt="莱叔，成熟稳重，喜欢和 Live Streaming Agent 轻松互怼。",
            speaker_identity="莱叔",
            create_speaker_version=True,
        )
    )

    assert saved.active_version == "Live Streaming Agent_1.txt"
    assert saved.persona_prompt == "莱叔是：\n你是第二版 Live Streaming Agent。"
    assert saved.active_speaker_version == "v1.0"
    assert saved.speaker_identity == "莱叔"
    assert saved.speaker_prompt == "莱叔，成熟稳重，喜欢和 Live Streaming Agent 轻松互怼。"
    assert [version.version for version in saved.speaker_versions] == [
        "v1.0",
    ]
    assert saved.speaker_versions[0].title == "莱叔"

    switched, _ = store.save_config(
        PromptConfigUpdate(active_speaker_version="__none__"),
        document,
    )
    assert switched.speaker_prompt == ""
    assert switched.speaker_identity == ""
    assert switched.persona_prompt == "你是第二版 Live Streaming Agent。"


def test_persona_prompt_identity_is_not_added_when_name_exists() -> None:
    prompt = "你正在和莱叔交流。\n保持自然交流。"

    assert ensure_persona_prompt_identity(prompt, "莱叔") == prompt


def test_persona_prompt_identity_is_added_when_name_is_missing() -> None:
    prompt = "保持自然交流。"

    assert (
        ensure_persona_prompt_identity(prompt, "莱叔")
        == "莱叔是：\n保持自然交流。"
    )


def test_persona_prompt_identity_is_not_added_without_a_speaker() -> None:
    prompt = "保持自然交流。"

    assert ensure_persona_prompt_identity(prompt, "") == prompt


def test_prompt_config_store_creates_a_new_speaker_version(
    tmp_path,
) -> None:
    store = build_store(tmp_path)

    first, document = store.save_config(
        PromptConfigUpdate(
            speaker_prompt="测试",
            speaker_identity="甲",
            create_speaker_version=True,
        )
    )
    second, _ = store.save_config(
        PromptConfigUpdate(
            speaker_prompt="测试",
            speaker_identity="乙",
            create_speaker_version=True,
        ),
        document,
    )

    assert first.active_speaker_version == "v1.0"
    assert second.active_speaker_version == "v1.1"
    assert len(second.speaker_versions) == 2
    assert second.speaker_identity == "乙"
    assert [version.title for version in second.speaker_versions] == [
        "甲",
        "乙",
    ]


def test_prompt_config_store_requires_identity_for_new_speaker_version(
    tmp_path,
) -> None:
    store = build_store(tmp_path)

    with pytest.raises(HTTPException):
        store.save_config(
            PromptConfigUpdate(
                speaker_prompt="测试",
                create_speaker_version=True,
            )
        )


def test_prompt_config_store_deletes_speaker_version_and_selects_none(
    tmp_path,
) -> None:
    store = build_store(tmp_path)
    created, document = store.save_config(
        PromptConfigUpdate(
            speaker_prompt="测试",
            speaker_identity="甲",
            create_speaker_version=True,
        )
    )

    deleted, updated_document = store.save_config(
        PromptConfigUpdate(
            delete_speaker_version=created.active_speaker_version,
        ),
        document,
    )

    assert deleted.active_speaker_version == "__none__"
    assert deleted.speaker_identity == ""
    assert deleted.speaker_prompt == ""
    assert deleted.speaker_versions == []
    assert updated_document["speaker_versions"] == []


def test_prompt_config_store_numbers_and_renames_speaker_versions(
    tmp_path,
) -> None:
    store = build_store(tmp_path)
    first, document = store.save_config(
        PromptConfigUpdate(
            speaker_prompt="第一版",
            speaker_identity="莱叔",
            create_speaker_version=True,
        )
    )
    second, document = store.save_config(
        PromptConfigUpdate(
            speaker_prompt="第二版",
            speaker_identity="莱叔",
            create_speaker_version=True,
        ),
        document,
    )

    renamed, _ = store.save_config(
        PromptConfigUpdate(
            rename_speaker_version=second.active_speaker_version,
            speaker_version_title="店长",
        ),
        document,
    )

    assert [version.title for version in first.speaker_versions] == ["莱叔"]
    assert [version.title for version in second.speaker_versions] == [
        "莱叔",
        "莱叔1",
    ]
    assert [version.title for version in renamed.speaker_versions] == [
        "莱叔",
        "店长",
    ]
    assert renamed.active_speaker_version == second.active_speaker_version


def test_prompt_config_store_updates_existing_speaker_version(
    tmp_path,
) -> None:
    store = build_store(tmp_path)
    created, document = store.save_config(
        PromptConfigUpdate(
            speaker_prompt="旧描述",
            speaker_identity="莱叔",
            create_speaker_version=True,
        )
    )

    updated, _ = store.save_config(
        PromptConfigUpdate(
            update_speaker_version=created.active_speaker_version,
            speaker_prompt="新描述",
            speaker_identity="店长",
        ),
        document,
    )

    assert len(updated.speaker_versions) == 1
    assert updated.speaker_versions[0].version == created.active_speaker_version
    assert updated.speaker_versions[0].title == "莱叔"
    assert updated.speaker_versions[0].content == "新描述"
    assert updated.speaker_versions[0].speaker_identity == "店长"
    assert updated.speaker_prompt == "新描述"
    assert updated.speaker_identity == "店长"


def test_prompt_config_store_rejects_unknown_version(tmp_path) -> None:
    store = build_store(tmp_path)

    with pytest.raises(HTTPException):
        store.save_config(PromptConfigUpdate(active_version="missing.txt"))
