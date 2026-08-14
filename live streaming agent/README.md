# Live Streaming Agent

Live Streaming Agent is a non-commercial portfolio project for interactive AI
live streaming. It combines real-time barrage ingestion, LLM-driven dialogue,
streaming text-to-speech, and a desktop control interface.

## Highlights

- Receives Douyin live-room events through a local WebSocket adapter compatible
  with DouyinBarrageGrab.
- Streams synthesized speech through a GPT-SoVITS-compatible service.
- Supports multiple LLM providers and configurable dialogue personas.
- Provides a PyQt desktop interface for live interaction and monitoring.
- Includes an independent `web-test/` console for model, prompt, knowledge-base,
  conversation-history, and live-room integration testing.

## Repository boundaries

This repository intentionally does **not** include:

- Open-LLM-VTuber-Web or its prebuilt frontend artifacts;
- voice reference recordings, cloned voices, or model weights;
- API keys, credentials, private configuration, logs, or chat history;
- the DouyinBarrageGrab executable or source distribution.

You must provide your own authorized voice assets and external services when
running the project. Do not use another person's voice, likeness, account, or
live-room data without permission.

## Upstream and third-party software

The backend originated from Open-LLM-VTuber 1.2.1 and is distributed with its
MIT copyright notice preserved. This repository contains substantial original
integration and application work but is a derivative project, not an official
Open-LLM-VTuber release.

GPT-SoVITS and DouyinBarrageGrab are external projects and are not bundled here.
Their licenses, usage notices, and applicable platform rules remain independent.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Intended use

This repository is published as a non-commercial engineering portfolio and
learning project. Users are responsible for complying with software licenses,
platform terms, privacy requirements, and local law.

## Configuration

1. Copy `conf.yaml.example` to `conf.yaml`.
2. Configure your own LLM and TTS endpoints without committing credentials.
3. Supply only voice reference audio and models that you are authorized to use.
4. If using DouyinBarrageGrab, run it separately and configure the local
   WebSocket address described in `抖音弹幕接入指南.txt`.

`conf.yaml`, `.env`, voice reference audio, model files, logs, and local editor
settings are excluded by `.gitignore`.
