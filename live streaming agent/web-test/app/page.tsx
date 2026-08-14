"use client";

import {
  Fragment,
  FormEvent,
  type ReactNode,
  type KeyboardEvent as ReactKeyboardEvent,
  type DragEvent as ReactDragEvent,
  type PointerEvent as ReactPointerEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  ApiError,
  archiveConversation,
  checkServiceHealth,
  createConversation,
  finishDouyinLogin,
  getDouyinLoginStatus,
  loadConversation,
  loadConversationMessages,
  loadModelConfig,
  loadPromptConfig,
  loadShortTermMemories,
  loadUserPerformanceMessages,
  loadUserWorkspace,
  releaseLiveCapture,
  reportChatLatency,
  renameConversation,
  reorderConversations,
  rewindConversation,
  savePromptConfig,
  saveUserModelConfig,
  searchUserConversations,
  startDouyinLogin,
  startLiveCapture,
  stopLiveCapture,
  streamChatMessage,
  streamLiveEvents,
  testModelConnection,
  type ConversationSummary,
  type ConversationSearchResult,
  type ChatStreamPhase,
  type ChatPerformanceMetrics,
  type MessageRecord,
  type LiveRoomEvent,
  type LiveLoginStatus,
  type PersonaPromptVersion,
  type SpeakerPromptVersion,
  type SavedProviderConfig,
  type ShortTermMemory,
  type UserWebSearchConfig,
  type WebSearchSource,
} from "../lib/api";
import { createClientId } from "../lib/client-id";

type ProviderModel = {
  id: string;
  label: string;
};

type ProviderOption = {
  id: string;
  name: string;
  defaultModel: string;
  models: ProviderModel[];
};

type ReplyableLiveEvent = Extract<
  LiveRoomEvent,
  { type: "chat" | "gift" }
>;

function liveGiftValue(event: Extract<LiveRoomEvent, { type: "gift" }>) {
  return Math.max(1, event.diamond_count ?? 1) * Math.max(1, event.gift_count);
}

function liveEventMessage(event: ReplyableLiveEvent) {
  if (event.type === "chat") {
    return {
      content: event.content.trim(),
      attributedContent: `${event.nickname}：“${event.content.trim()}”`,
    };
  }
  const value = liveGiftValue(event);
  const giftText = `送出 ${event.gift_name} ×${event.gift_count}${
    event.diamond_count ? `，价值 ${value} 抖币` : ""
  }`;
  return {
    content: giftText,
    attributedContent: `${event.nickname}：“${giftText}”`,
  };
}

type SpeakerPromptTransition =
  | { kind: "select"; version: string }
  | { kind: "create" };

type ConversationSearchTarget = {
  conversationId: string;
  messageId: string;
  phrase: string;
};

type PerformanceSample = ChatPerformanceMetrics & {
  messageId: string;
  createdAt: string;
  provider: string;
  model: string;
};

type ModelComparisonMetric =
  | "model_first_token_ms"
  | "model_first_sentence_ms";

const performanceSeries: Array<{
  key: keyof ChatPerformanceMetrics;
  label: string;
  color: string;
}> = [
  {
    key: "knowledge_duration_ms",
    label: "知识库用时",
    color: "#14a66f",
  },
  {
    key: "web_search_duration_ms",
    label: "网络搜索用时",
    color: "#f38b18",
  },
  {
    key: "model_first_token_ms",
    label: "模型首字延迟",
    color: "#0869f7",
  },
  {
    key: "model_first_sentence_ms",
    label: "首句延迟",
    color: "#8f3fff",
  },
];

const modelProviders: ProviderOption[] = [
  {
    id: "deepseek",
    name: "DeepSeek",
    defaultModel: "deepseek-v4-pro",
    models: [
      { id: "deepseek-v4-pro", label: "DeepSeek V4 Pro" },
      { id: "deepseek-v4-flash", label: "DeepSeek V4 Flash" },
    ],
  },
  {
    id: "doubao",
    name: "豆包 / 火山方舟",
    defaultModel: "doubao-seed-2-0-lite-260215",
    models: [
      {
        id: "doubao-seed-evolving",
        label: "Doubao Seed Evolving",
      },
      { id: "doubao-seed-2-1-turbo-260628", label: "Doubao Seed 2.1 Turbo" },
      { id: "doubao-seed-2-1-pro-260628", label: "Doubao Seed 2.1 Pro" },
      { id: "doubao-seed-2-0-mini-260428", label: "Doubao Seed 2.0 Mini" },
      { id: "doubao-seed-2-0-pro-260215", label: "Doubao Seed 2.0 Pro" },
      { id: "doubao-seed-2-0-lite-260215", label: "Doubao Seed 2.0 Lite" },
    ],
  },
  {
    id: "qwen",
    name: "通义千问 / 百炼",
    defaultModel: "qwen3.6-flash",
    models: [
      { id: "qwen3.7-max", label: "Qwen3.7 Max" },
      { id: "qwen3.7-plus", label: "Qwen3.7 Plus" },
      { id: "qwen3.6-max-preview", label: "Qwen3.6 Max Preview" },
      { id: "qwen3.6-plus", label: "Qwen3.6 Plus" },
      { id: "qwen3.6-flash", label: "Qwen3.6 Flash" },
      { id: "qwen3.5-plus", label: "Qwen3.5 Plus" },
      { id: "qwen3.5-flash", label: "Qwen3.5 Flash" },
    ],
  },
  {
    id: "kimi",
    name: "Kimi / Moonshot",
    defaultModel: "kimi-k2.6",
    models: [
      { id: "kimi-k3", label: "Kimi K3" },
      { id: "kimi-k2.6", label: "Kimi K2.6" },
    ],
  },
  {
    id: "zhipu",
    name: "GLM 智谱",
    defaultModel: "glm-5.2",
    models: [
      { id: "glm-5.2", label: "GLM-5.2" },
      { id: "glm-5.1", label: "GLM-5.1" },
      { id: "glm-5", label: "GLM-5" },
      { id: "glm-5-turbo", label: "GLM-5-Turbo" },
      { id: "glm-4.7", label: "GLM-4.7" },
      { id: "glm-4.7-flashx", label: "GLM-4.7-FlashX" },
      { id: "glm-4.7-flash", label: "GLM-4.7-Flash" },
      { id: "glm-4.6", label: "GLM-4.6" },
      { id: "glm-4.5-air", label: "GLM-4.5-Air" },
      { id: "glm-4.5-airx", label: "GLM-4.5-AirX" },
    ],
  },
  {
    id: "tencent-yuanbao",
    name: "腾讯元宝 / TokenHub",
    defaultModel: "hy3",
    models: [
      { id: "hy3", label: "Hy3" },
      { id: "hy3-preview", label: "Hy3 Preview" },
      { id: "hy-mt2-pro", label: "Hy-MT2 Pro" },
      { id: "hy-mt2-plus", label: "Hy-MT2 Plus" },
      { id: "hy-mt2-lite", label: "Hy-MT2 Lite" },
      { id: "hunyuan-role-latest", label: "Hy-Role Latest" },
      { id: "hy-role", label: "Hy-Role" },
    ],
  },
];

function providerInfo(providerId: string) {
  return (
    modelProviders.find((candidate) => candidate.id === providerId) ??
    modelProviders[0]
  );
}

function savedProviderConfig(
  providers: Record<string, SavedProviderConfig>,
  providerId: string,
) {
  return providers[providerId];
}

function providerWebSearchConfig(
  configs: Record<string, UserWebSearchConfig>,
  providerId: string,
): UserWebSearchConfig {
  return configs[providerId] ?? {
    enabled: false,
    forced: false,
    max_tool_calls: 1,
    result_limit: 3,
  };
}

function providerTemperature(
  temperatures: Record<string, number>,
  providerId: string,
) {
  const value = temperatures[providerId];
  return typeof value === "number" && Number.isFinite(value) ? value : 0.8;
}

function nextVisibleConversationTitle(
  conversations: ConversationSummary[],
) {
  const highestNumber = conversations.reduce((highest, conversation) => {
    const match = /^新对话\s*(\d+)$/.exec(conversation.title.trim());
    return match ? Math.max(highest, Number(match[1])) : highest;
  }, 0);
  return `新对话 ${highestNumber + 1}`;
}

const legacyModelAliases: Record<string, string> = {
  "doubao-seed-2-0-pro-250528": "doubao-seed-2-0-pro-260215",
  "doubao-seed-2-0-mini-250528": "doubao-seed-2-0-mini-260428",
  "doubao-seed-2-1-pro": "doubao-seed-2-1-pro-260628",
  "doubao-seed-2-1-turbo": "doubao-seed-2-1-turbo-260628",
  "qwen3.7-flash": "qwen3.6-flash",
  "kimi-k2.7": "kimi-k2.6",
  "kimi-k2.7-code": "kimi-k2.6",
  "kimi-latest": "kimi-k2.6",
  "kimi-k2-0711-preview": "kimi-k2.6",
  "kimi-k2-turbo-preview": "kimi-k2.6",
  "hunyuan-turbos-latest": "hy3",
  "hunyuan-a13b-instruct": "hy3",
  "hunyuan-large-longcontext": "hy3",
  "hunyuan-large-role": "hunyuan-role-latest",
  "hunyuan-translation": "hy-mt2-plus",
  "hunyuan-translation-lite": "hy-mt2-lite",
};

function providerSupportsWebSearch(providerId: string) {
  return providerId === "doubao" || providerId === "qwen";
}

function catalogModelId(
  provider: ProviderOption,
  candidate?: string,
) {
  const normalizedCandidate = candidate
    ? (legacyModelAliases[candidate] ?? candidate)
    : candidate;
  return provider.models.some((model) => model.id === normalizedCandidate)
    ? normalizedCandidate!
    : provider.defaultModel;
}

function timeOf(iso: string) {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(new Date(iso));
}

function shortDate(iso: string) {
  const date = new Date(iso);
  const now = new Date();
  if (date.toDateString() === now.toDateString()) {
    return timeOf(iso).slice(0, 5);
  }
  return `${date.getMonth() + 1}/${date.getDate()}`;
}

function messageDisplay(message: MessageRecord) {
  if (message.role === "assistant") {
    return { speaker: "Live Streaming Agent", content: message.content };
  }

  if (message.role === "user") {
    const separator = "：“";
    const separatorIndex = message.content.indexOf(separator);
    if (separatorIndex > 0 && message.content.endsWith("”")) {
      const identity = message.content.slice(0, separatorIndex).trim();
      if (identity && identity.length <= 40) {
        return {
          speaker: identity,
          content: message.content.slice(
            separatorIndex + separator.length,
            -1,
          ),
        };
      }
    }
  }

  return { speaker: "用户", content: message.content };
}

function exportDateTime(iso: string) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const parts = [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, "0"),
    String(date.getDate()).padStart(2, "0"),
  ];
  const time = [
    String(date.getHours()).padStart(2, "0"),
    String(date.getMinutes()).padStart(2, "0"),
    String(date.getSeconds()).padStart(2, "0"),
  ];
  return `${parts.join("-")} ${time.join(":")}`;
}

function safeExportFilename(value: string) {
  return value.replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_").trim() || "对话";
}

function highlightedSearchText(text: string, phrase: string) {
  const normalizedPhrase = phrase.trim();
  if (!normalizedPhrase) return text;
  const textForSearch = text.toLocaleLowerCase();
  const phraseForSearch = normalizedPhrase.toLocaleLowerCase();
  const parts: ReactNode[] = [];
  let cursor = 0;
  let matchIndex = textForSearch.indexOf(phraseForSearch);
  while (matchIndex >= 0) {
    if (matchIndex > cursor) {
      parts.push(text.slice(cursor, matchIndex));
    }
    const matchEnd = matchIndex + normalizedPhrase.length;
    parts.push(
      <mark key={`${matchIndex}-${matchEnd}`}>
        {text.slice(matchIndex, matchEnd)}
      </mark>,
    );
    cursor = matchEnd;
    matchIndex = textForSearch.indexOf(phraseForSearch, cursor);
  }
  if (!parts.length) return text;
  if (cursor < text.length) parts.push(text.slice(cursor));
  return parts;
}

function includesSearchPhrase(text: string, phrase: string) {
  const normalizedPhrase = phrase.trim().toLocaleLowerCase();
  return Boolean(
    normalizedPhrase && text.toLocaleLowerCase().includes(normalizedPhrase),
  );
}

function conversationSearchSourceLabel(
  source: ConversationSearchResult["matches"][number]["source"],
) {
  if (source === "knowledge") return "知识库";
  if (source === "web_search") return "联网搜索";
  return "对话";
}

function messageWebSearchSources(message: MessageRecord): WebSearchSource[] {
  const rawSources = message.metadata?.web_search_sources;
  if (!Array.isArray(rawSources)) return [];
  return rawSources.flatMap((source) => {
    if (!source || typeof source !== "object") return [];
    const candidate = source as Record<string, unknown>;
    const url = typeof candidate.url === "string" ? candidate.url.trim() : "";
    if (!/^https?:\/\//i.test(url)) return [];
    return [{
      title:
        typeof candidate.title === "string" && candidate.title.trim()
          ? candidate.title.trim()
          : url,
      url,
      snippet:
        typeof candidate.snippet === "string" ? candidate.snippet.trim() : "",
    }];
  }).slice(0, 20);
}

function RobotMark({ small = false }: { small?: boolean }) {
  return (
    <span className={small ? "robot-mark robot-mark-small" : "robot-mark"} aria-hidden="true">
      <span className="robot-antenna" />
      <span className="robot-face">
        <i />
        <i />
      </span>
    </span>
  );
}

function durationParts(milliseconds: number | null) {
  if (milliseconds === null) return { value: "—", unit: "" };
  if (milliseconds >= 1000) {
    return {
      value: (milliseconds / 1000).toFixed(2),
      unit: "s",
    };
  }
  return {
    value: Math.round(milliseconds).toString(),
    unit: "ms",
  };
}

function localDateKey(date = new Date()) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function ensurePersonaPromptIdentity(
  prompt: string,
  speakerIdentity: string,
) {
  const normalized = prompt.trim();
  const normalizedIdentity = speakerIdentity.trim();
  if (
    !normalized ||
    !normalizedIdentity ||
    normalized.includes(normalizedIdentity)
  ) {
    return normalized;
  }
  return `${normalizedIdentity}是：\n${normalized}`;
}

function messagePerformanceSample(
  message: MessageRecord,
): PerformanceSample | null {
  if (message.role !== "assistant") return null;
  const rawMetrics = message.metadata?.performance_metrics;
  if (!rawMetrics || typeof rawMetrics !== "object") return null;
  const metrics = rawMetrics as Record<string, unknown>;
  const requiredValues = [
    metrics.knowledge_duration_ms,
    metrics.model_first_token_ms,
    metrics.model_first_sentence_ms,
  ];
  if (
    requiredValues.some(
      (value) =>
        typeof value !== "number" || !Number.isFinite(value) || value < 0,
    )
  ) {
    return null;
  }
  return {
    messageId: message.message_id,
    createdAt: message.created_at,
    provider:
      typeof message.metadata?.provider === "string"
        ? message.metadata.provider
        : "",
    model:
      typeof message.metadata?.model === "string" && message.metadata.model
        ? message.metadata.model
        : "未记录模型",
    knowledge_duration_ms: requiredValues[0] as number,
    web_search_duration_ms:
      typeof metrics.web_search_duration_ms === "number" &&
      Number.isFinite(metrics.web_search_duration_ms) &&
      metrics.web_search_duration_ms >= 0
        ? metrics.web_search_duration_ms
        : 0,
    model_first_token_ms: requiredValues[1] as number,
    model_first_sentence_ms: requiredValues[2] as number,
  };
}

function performanceMetricsFromSamples(samples: PerformanceSample[]) {
  const sample = samples[samples.length - 1];
  if (!sample) {
    return {
      knowledge_duration_ms: null,
      web_search_duration_ms: null,
      model_first_token_ms: null,
      model_first_sentence_ms: null,
    };
  }
  return {
    knowledge_duration_ms: sample.knowledge_duration_ms,
    web_search_duration_ms: sample.web_search_duration_ms,
    model_first_token_ms: sample.model_first_token_ms,
    model_first_sentence_ms: sample.model_first_sentence_ms,
  };
}

function PerformanceLineChart({
  samples,
}: {
  samples: PerformanceSample[];
}) {
  if (!samples.length) {
    return (
      <div className="performance-chart-empty">
        完成一次模型回复后显示趋势
      </div>
    );
  }

  const chartWidth = 640;
  const chartHeight = 184;
  const left = 54;
  const right = 14;
  const top = 14;
  const bottom = 30;
  const plotWidth = chartWidth - left - right;
  const plotHeight = chartHeight - top - bottom;
  const maximum = Math.max(
    1,
    ...samples.flatMap((sample) =>
      performanceSeries.map(({ key }) => sample[key]),
    ),
  );
  const axisMaximum = Math.ceil(maximum * 1.1);
  const xOf = (index: number) =>
    left +
    (samples.length === 1 ? plotWidth / 2 : (index / (samples.length - 1)) * plotWidth);
  const yOf = (value: number) =>
    top + plotHeight - (value / axisMaximum) * plotHeight;
  const axisLabels = [axisMaximum, axisMaximum / 2, 0];

  return (
    <div className="performance-chart">
      <div className="performance-chart-legend">
        {performanceSeries.map((series) => (
          <span key={series.key}>
            <i style={{ backgroundColor: series.color }} />
            {series.label}
          </span>
        ))}
      </div>
      <svg
        viewBox={`0 0 ${chartWidth} ${chartHeight}`}
        role="img"
        aria-label="最近对话性能指标折线图"
      >
        {axisLabels.map((value, index) => {
          const y = top + (index / 2) * plotHeight;
          const display = durationParts(value);
          return (
            <g key={`axis-${index}`}>
              <line
                className="performance-grid-line"
                x1={left}
                x2={chartWidth - right}
                y1={y}
                y2={y}
              />
              <text
                className="performance-axis-label"
                x={left - 8}
                y={y + 4}
                textAnchor="end"
              >
                {display.value}{display.unit}
              </text>
            </g>
          );
        })}
        {performanceSeries.map((series) => {
          const points = samples
            .map(
              (sample, index) =>
                `${xOf(index)},${yOf(sample[series.key])}`,
            )
            .join(" ");
          return (
            <g key={series.key}>
              <polyline
                points={points}
                fill="none"
                stroke={series.color}
                strokeWidth="2.5"
                strokeLinejoin="round"
                strokeLinecap="round"
                vectorEffect="non-scaling-stroke"
              />
              {samples.map((sample, index) => (
                <circle
                  key={`${series.key}-${sample.messageId}`}
                  cx={xOf(index)}
                  cy={yOf(sample[series.key])}
                  r="3.5"
                  fill={series.color}
                >
                  <title>
                    {series.label}：{Math.round(sample[series.key])} ms
                  </title>
                </circle>
              ))}
            </g>
          );
        })}
        {samples.map((sample, index) => (
          <text
            className="performance-time-label"
            key={sample.messageId}
            x={xOf(index)}
            y={chartHeight - 8}
            textAnchor="middle"
          >
            {timeOf(sample.createdAt).slice(0, 5)}
          </text>
        ))}
      </svg>
    </div>
  );
}

const modelComparisonColors = [
  "#0869f7",
  "#8f3fff",
  "#f38b18",
  "#14a66f",
  "#e84a8a",
  "#00a6b8",
  "#cf5b24",
  "#5367d9",
];

function percentile(values: number[], quantile: number) {
  if (!values.length) return null;
  const ordered = [...values].sort((left, right) => left - right);
  const position = (ordered.length - 1) * Math.min(1, Math.max(0, quantile));
  const lowerIndex = Math.floor(position);
  const upperIndex = Math.ceil(position);
  if (lowerIndex === upperIndex) return ordered[lowerIndex];
  const weight = position - lowerIndex;
  return (
    ordered[lowerIndex] * (1 - weight) + ordered[upperIndex] * weight
  );
}

function durationText(milliseconds: number | null) {
  const display = durationParts(milliseconds);
  return display.unit ? `${display.value} ${display.unit}` : display.value;
}

function ModelComparisonPercentiles({
  samples,
}: {
  samples: PerformanceSample[];
}) {
  if (!samples.length) return null;

  const grouped = new Map<string, PerformanceSample[]>();
  samples.forEach((sample) => {
    const key = `${sample.provider || "unknown"}:${sample.model}`;
    grouped.set(key, [...(grouped.get(key) ?? []), sample]);
  });
  const rows = [...grouped.entries()]
    .map(([key, values]) => ({
      key,
      label: values[0].model,
      count: values.length,
      firstTokenP50: percentile(
        values.map((sample) => sample.model_first_token_ms),
        0.5,
      ),
      firstTokenP90: percentile(
        values.map((sample) => sample.model_first_token_ms),
        0.9,
      ),
      firstSentenceP50: percentile(
        values.map((sample) => sample.model_first_sentence_ms),
        0.5,
      ),
      firstSentenceP90: percentile(
        values.map((sample) => sample.model_first_sentence_ms),
        0.9,
      ),
    }))
    .sort((left, right) => left.label.localeCompare(right.label));

  return (
    <section className="model-percentile-summary">
      <header>
        <h3>各模型 P50 / P90</h3>
        <small>当前用户当日成功调用</small>
      </header>
      <div className="model-percentile-table-wrap">
        <table>
          <thead>
            <tr>
              <th>模型</th>
              <th>调用</th>
              <th>首字 P50</th>
              <th>首字 P90</th>
              <th>首句 P50</th>
              <th>首句 P90</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.key}>
                <td title={row.label}>{row.label}</td>
                <td>{row.count}</td>
                <td>{durationText(row.firstTokenP50)}</td>
                <td>{durationText(row.firstTokenP90)}</td>
                <td>{durationText(row.firstSentenceP50)}</td>
                <td>{durationText(row.firstSentenceP90)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ModelComparisonDotChart({
  samples,
  metric,
  title,
}: {
  samples: PerformanceSample[];
  metric: ModelComparisonMetric;
  title: string;
}) {
  if (!samples.length) {
    return (
      <div className="performance-chart-empty model-comparison-empty">
        当前用户今天还没有模型调用记录
      </div>
    );
  }

  const orderedSamples = [...samples].sort(
    (left, right) =>
      new Date(left.createdAt).getTime() -
      new Date(right.createdAt).getTime(),
  );
  const grouped = new Map<string, PerformanceSample[]>();
  orderedSamples.forEach((sample) => {
    const key = `${sample.provider || "unknown"}:${sample.model}`;
    grouped.set(key, [...(grouped.get(key) ?? []), sample]);
  });
  const series = [...grouped.entries()].map(([key, values], index) => ({
    key,
    label: values[0].model,
    color: modelComparisonColors[index % modelComparisonColors.length],
    values,
  }));

  const chartWidth = 640;
  const chartHeight = 184;
  const left = 54;
  const right = 14;
  const top = 14;
  const bottom = 30;
  const plotWidth = chartWidth - left - right;
  const plotHeight = chartHeight - top - bottom;
  const timestamps = orderedSamples.map((sample) =>
    new Date(sample.createdAt).getTime(),
  );
  const minimumTime = Math.min(...timestamps);
  const maximumTime = Math.max(...timestamps);
  const timeSpan = maximumTime - minimumTime;
  const maximumValue = Math.max(
    1,
    ...orderedSamples.map((sample) => sample[metric]),
  );
  const axisMaximum = Math.ceil(maximumValue * 1.1);
  const xOf = (createdAt: string) =>
    left +
    (timeSpan === 0
      ? plotWidth / 2
      : ((new Date(createdAt).getTime() - minimumTime) / timeSpan) *
        plotWidth);
  const yOf = (value: number) =>
    top + plotHeight - (value / axisMaximum) * plotHeight;
  const axisLabels = [axisMaximum, axisMaximum / 2, 0];
  const tickCount = timeSpan === 0 ? 1 : 5;
  const timeTicks = Array.from({ length: tickCount }, (_, index) =>
    tickCount === 1
      ? minimumTime
      : minimumTime + (index / (tickCount - 1)) * timeSpan,
  );

  return (
    <section className="model-comparison-chart">
      <h3>{title}</h3>
      <div className="performance-chart">
        <div className="performance-chart-legend model-comparison-legend">
          {series.map((item) => (
            <span key={item.key}>
              <i style={{ backgroundColor: item.color }} />
              {item.label}
              <small>{item.values.length} 次</small>
            </span>
          ))}
        </div>
        <svg
          viewBox={`0 0 ${chartWidth} ${chartHeight}`}
          role="img"
          aria-label={`${title}折线图`}
        >
          {axisLabels.map((value, index) => {
            const y = top + (index / 2) * plotHeight;
            const display = durationParts(value);
            return (
              <g key={`axis-${index}`}>
                <line
                  className="performance-grid-line"
                  x1={left}
                  x2={chartWidth - right}
                  y1={y}
                  y2={y}
                />
                <text
                  className="performance-axis-label"
                  x={left - 8}
                  y={y + 4}
                  textAnchor="end"
                >
                  {display.value}{display.unit}
                </text>
              </g>
            );
          })}
          {series.map((item) => (
            <g key={item.key}>
              {item.values.map((sample) => (
                <circle
                  key={`${metric}-${sample.messageId}`}
                  cx={xOf(sample.createdAt)}
                  cy={yOf(sample[metric])}
                  r="3.5"
                  fill={item.color}
                >
                  <title>
                    {item.label} · {timeOf(sample.createdAt)} ·{" "}
                    {Math.round(sample[metric])} ms
                  </title>
                </circle>
              ))}
            </g>
          ))}
          {timeTicks.map((timestamp, index) => (
            <text
              className="performance-time-label"
              key={`${timestamp}-${index}`}
              x={
                tickCount === 1
                  ? left + plotWidth / 2
                  : left + (index / (tickCount - 1)) * plotWidth
              }
              y={chartHeight - 8}
              textAnchor="middle"
            >
              {timeOf(new Date(timestamp).toISOString()).slice(0, 5)}
            </text>
          ))}
        </svg>
      </div>
    </section>
  );
}

export default function Home() {
  const [knowledgeEnabled, setKnowledgeEnabled] = useState(true);
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);
  const [webSearchForced, setWebSearchForced] = useState(false);
  const [webSearchMaxToolCalls, setWebSearchMaxToolCalls] = useState(1);
  const [webSearchResultLimit, setWebSearchResultLimit] = useState(3);
  const [connectionState, setConnectionState] = useState<
    "idle" | "testing" | "success" | "error"
  >("idle");
  const [provider, setProvider] = useState(modelProviders[0].id);
  const [model, setModel] = useState(modelProviders[0].defaultModel);
  const [temperature, setTemperature] = useState(0.8);
  const [userProviderModels, setUserProviderModels] = useState<
    Record<string, string>
  >({});
  const [userProviderTemperatures, setUserProviderTemperatures] = useState<
    Record<string, number>
  >({});
  const [
    userProviderWebSearchConfigs,
    setUserProviderWebSearchConfigs,
  ] = useState<Record<string, UserWebSearchConfig>>({});
  const [savedProviderConfigs, setSavedProviderConfigs] = useState<
    Record<string, SavedProviderConfig>
  >({});
  const [modelConfigNotice, setModelConfigNotice] =
    useState("正在读取模型配置...");
  const [testingModelConfig, setTestingModelConfig] = useState(false);
  const [promptVersions, setPromptVersions] = useState<PersonaPromptVersion[]>(
    [],
  );
  const [activePromptVersion, setActivePromptVersion] = useState("v1.0");
  const [personaPrompt, setPersonaPrompt] = useState("正在读取人设提示词...");
  const [speakerPromptVersions, setSpeakerPromptVersions] = useState<
    SpeakerPromptVersion[]
  >([]);
  const [activeSpeakerPromptVersion, setActiveSpeakerPromptVersion] =
    useState("__none__");
  const [creatingSpeakerPrompt, setCreatingSpeakerPrompt] = useState(false);
  const [pendingSpeakerPromptDelete, setPendingSpeakerPromptDelete] =
    useState<SpeakerPromptVersion | null>(null);
  const [deletingSpeakerPromptVersion, setDeletingSpeakerPromptVersion] =
    useState("");
  const [editingSpeakerPromptVersion, setEditingSpeakerPromptVersion] =
    useState("");
  const [speakerPromptTitleInput, setSpeakerPromptTitleInput] = useState("");
  const [renamingSpeakerPromptVersion, setRenamingSpeakerPromptVersion] =
    useState("");
  const [speakerPrompt, setSpeakerPrompt] = useState("");
  const [savedSpeakerPromptIdentity, setSavedSpeakerPromptIdentity] =
    useState("");
  const [savedSpeakerPrompt, setSavedSpeakerPrompt] = useState("");
  const [pendingSpeakerPromptTransition, setPendingSpeakerPromptTransition] =
    useState<SpeakerPromptTransition | null>(null);
  const [promptConfigNotice, setPromptConfigNotice] =
    useState("正在读取提示词配置...");
  const [savingPromptConfig, setSavingPromptConfig] = useState(false);
  const [usernameInput, setUsernameInput] = useState("");
  const [username, setUsername] = useState("");
  const [usernameDialogOpen, setUsernameDialogOpen] = useState(true);
  const [usernameDialogError, setUsernameDialogError] = useState("");
  const [serviceState, setServiceState] = useState<
    "checking" | "running" | "database-error" | "backend-error"
  >("checking");
  const [conversations, setConversations] =
    useState<ConversationSummary[]>([]);
  const [conversationSearchQuery, setConversationSearchQuery] = useState("");
  const [conversationSearchResults, setConversationSearchResults] = useState<
    ConversationSearchResult[]
  >([]);
  const [searchingConversations, setSearchingConversations] = useState(false);
  const [conversationSearchError, setConversationSearchError] = useState("");
  const [conversationSearchTarget, setConversationSearchTarget] =
    useState<ConversationSearchTarget | null>(null);
  const [activeConversationId, setActiveConversationId] = useState("");
  const [messages, setMessages] = useState<MessageRecord[]>([]);
  const [speakerIdentity, setSpeakerIdentity] = useState("");
  const [savedSpeakerIdentity, setSavedSpeakerIdentity] = useState("莱叔");
  const [messageDraft, setMessageDraft] = useState("");
  const [messageError, setMessageError] = useState("");
  const [liveRoomId, setLiveRoomId] = useState("");
  const [liveCaptureState, setLiveCaptureState] = useState<
    "idle" | "starting" | "running" | "stopping" | "error"
  >("idle");
  const [liveCaptureMessage, setLiveCaptureMessage] =
    useState("输入房间号后开始抓取");
  const [livePanelCollapsed, setLivePanelCollapsed] = useState(true);
  const [replyLiveChats, setReplyLiveChats] = useState(false);
  const [replyLiveGifts, setReplyLiveGifts] = useState(false);
  const [liveGiftReplyWakeTick, setLiveGiftReplyWakeTick] = useState(0);
  const [douyinLoginDialogOpen, setDouyinLoginDialogOpen] = useState(false);
  const [douyinLoginBusy, setDouyinLoginBusy] = useState(false);
  const [douyinLoginChecking, setDouyinLoginChecking] = useState(true);
  const [douyinLoginStatus, setDouyinLoginStatus] =
    useState<LiveLoginStatus>({
      status: "idle",
      message: "尚未开始抖音扫码登录",
      qr_image: null,
    });
  const [liveEvents, setLiveEvents] = useState<LiveRoomEvent[]>([]);
  const [workspaceNotice, setWorkspaceNotice] =
    useState("请输入用户名载入历史对话");
  const [loadingWorkspace, setLoadingWorkspace] = useState(false);
  const [creatingConversation, setCreatingConversation] = useState(false);
  const [exportingConversation, setExportingConversation] = useState(false);
  const [archivingConversationId, setArchivingConversationId] = useState("");
  const [pendingConversationDelete, setPendingConversationDelete] =
    useState<ConversationSummary | null>(null);
  const [editingConversationId, setEditingConversationId] = useState("");
  const [conversationTitleInput, setConversationTitleInput] = useState("");
  const [renamingConversationId, setRenamingConversationId] = useState("");
  const [draggingConversationId, setDraggingConversationId] = useState("");
  const [isReplying, setIsReplying] = useState(false);
  const [performanceMetrics, setPerformanceMetrics] = useState<
    Record<keyof ChatPerformanceMetrics, number | null>
  >({
    knowledge_duration_ms: null,
    web_search_duration_ms: null,
    model_first_token_ms: null,
    model_first_sentence_ms: null,
  });
  const [performanceHistory, setPerformanceHistory] = useState<
    PerformanceSample[]
  >([]);
  const [performanceView, setPerformanceView] = useState<
    "overview" | "model-comparison"
  >("overview");
  const [dailyModelPerformance, setDailyModelPerformance] = useState<
    PerformanceSample[]
  >([]);
  const [dailyPerformanceDay, setDailyPerformanceDay] = useState(
    localDateKey(),
  );
  const [isRewinding, setIsRewinding] = useState(false);
  const [shortTermMemories, setShortTermMemories] = useState<
    ShortTermMemory[]
  >([]);
  const [chatContentView, setChatContentView] = useState<
    "conversation" | "memory"
  >("conversation");
  const [loadingMemories, setLoadingMemories] = useState(false);
  const [conversationPanePercent, setConversationPanePercent] = useState(50);
  const [resizingMemoryPane, setResizingMemoryPane] = useState(false);
  const [workspaceHeight, setWorkspaceHeight] = useState(600);
  const [dashboardHeight, setDashboardHeight] = useState<number | null>(null);
  const [resizingWorkspace, setResizingWorkspace] = useState(false);
  const messageCacheRef = useRef(new Map<string, MessageRecord[]>());
  const conversationLoadSequenceRef = useRef(0);
  const dashboardGridRef = useRef<HTMLElement>(null);
  const chatContentRef = useRef<HTMLDivElement>(null);
  const messageListRef = useRef<HTMLDivElement>(null);
  const messageInputRef = useRef<HTMLInputElement>(null);
  const selectedModelRef = useRef({
    provider: modelProviders[0].id,
    model: modelProviders[0].defaultModel,
  });
  const liveEventListRef = useRef<HTMLDivElement>(null);
  const liveStreamAbortRef = useRef<AbortController | null>(null);
  const liveSequenceRef = useRef(0);
  const liveReplyChatStartSequenceRef = useRef(0);
  const liveReplyGiftStartSequenceRef = useRef(0);
  const handledLiveReplySequencesRef = useRef(new Set<number>());
  const liveReplyInFlightRef = useRef(false);
  const lastRepliedGiftRef = useRef<{
    value: number;
    repliedAt: number;
  } | null>(null);
  const liveGiftReplyWakeTimerRef = useRef<number | null>(null);
  const sendMessageContentRef = useRef<
    | ((
        content: string,
        options?: {
          attributedContent?: string;
          clickStartedAt?: number;
          clickTimestamp?: string;
          clearDraft?: boolean;
        },
      ) => Promise<void>)
    | null
  >(null);
  const douyinLoginInitialCheckRef = useRef(false);
  const resizingMemoryPaneRef = useRef(false);
  const resizingWorkspaceRef = useRef(false);
  const speakerPromptRenameInFlightRef = useRef(false);
  const speakerPromptRenameCancelledRef = useRef(false);
  const conversationOrderBeforeDragRef = useRef<ConversationSummary[]>([]);
  const conversationOrderDuringDragRef = useRef<ConversationSummary[]>([]);
  const conversationDropCommittedRef = useRef(false);

  const activeConversation = useMemo(
    () =>
      conversations.find(
        (conversation) =>
          conversation.conversation_id === activeConversationId,
      ),
    [activeConversationId, conversations],
  );
  const performanceDisplay = useMemo(
    () => ({
      knowledge: durationParts(performanceMetrics.knowledge_duration_ms),
      webSearch: durationParts(
        performanceMetrics.web_search_duration_ms &&
          performanceMetrics.web_search_duration_ms > 0
          ? performanceMetrics.web_search_duration_ms
          : null,
      ),
      firstToken: durationParts(performanceMetrics.model_first_token_ms),
      firstSentence: durationParts(
        performanceMetrics.model_first_sentence_ms,
      ),
    }),
    [performanceMetrics],
  );
  const dailyModelCount = useMemo(
    () =>
      new Set(
        dailyModelPerformance.map(
          (sample) => `${sample.provider || "unknown"}:${sample.model}`,
        ),
      ).size,
    [dailyModelPerformance],
  );
  const speakerPromptHasUnsavedChanges =
    (creatingSpeakerPrompt ||
      activeSpeakerPromptVersion !== "__none__") &&
    (speakerIdentity.trim() !== savedSpeakerPromptIdentity ||
      speakerPrompt !== savedSpeakerPrompt);
  const messageRoundNumbers = useMemo(() => {
    const rounds = new Map<string, number>();
    let currentRound = 0;
    for (const message of messages) {
      const storedRound = message.metadata?.round_number;
      if (message.role === "user") {
        currentRound =
          typeof storedRound === "number" ? storedRound : currentRound + 1;
      } else if (typeof storedRound === "number") {
        currentRound = storedRound;
      }
      if (currentRound > 0) rounds.set(message.message_id, currentRound);
    }
    return rounds;
  }, [messages]);
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const pollServiceHealth = async () => {
      try {
        const health = await checkServiceHealth();
        if (!cancelled) {
          setServiceState(
            health.elasticsearch === "connected"
              ? "running"
              : "database-error",
          );
        }
      } catch {
        if (!cancelled) setServiceState("backend-error");
      } finally {
        if (!cancelled) timer = setTimeout(pollServiceHealth, 10_000);
      }
    };

    void pollServiceHealth();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    if (
      douyinLoginInitialCheckRef.current ||
      serviceState === "checking" ||
      serviceState === "backend-error"
    ) {
      return;
    }
    douyinLoginInitialCheckRef.current = true;
    let cancelled = false;
    setDouyinLoginChecking(true);
    getDouyinLoginStatus()
      .then((status) => {
        if (cancelled) return;
        setDouyinLoginStatus(status);
        if (status.status === "ready") {
          setLiveCaptureMessage(status.message);
        }
      })
      .catch((error) => {
        if (cancelled) return;
        setDouyinLoginStatus({
          status: "error",
          message:
            error instanceof ApiError
              ? error.message
              : "读取抖音登录状态失败",
          qr_image: null,
        });
      })
      .finally(() => {
        if (!cancelled) setDouyinLoginChecking(false);
      });
    return () => {
      cancelled = true;
    };
  }, [serviceState]);

  useEffect(() => {
    const messageList = messageListRef.current;
    if (!messageList || conversationSearchTarget) return;
    messageList.scrollTop = messageList.scrollHeight;
  }, [activeConversationId, conversationSearchTarget, messages]);

  useEffect(() => {
    if (!conversationSearchTarget) return;
    const messageList = messageListRef.current;
    if (!messageList) return;
    const frame = window.requestAnimationFrame(() => {
      const target = [...messageList.querySelectorAll<HTMLElement>(
        "[data-message-id]",
      )].find(
        (element) =>
          element.dataset.messageId === conversationSearchTarget.messageId,
      );
      target?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [conversationSearchTarget, messages]);

  useEffect(() => {
    const eventList = liveEventListRef.current;
    if (!eventList) return;
    eventList.scrollTop = eventList.scrollHeight;
  }, [liveEvents, livePanelCollapsed]);

  useEffect(() => {
    liveStreamAbortRef.current?.abort();
    liveStreamAbortRef.current = null;
    liveSequenceRef.current = 0;
    liveReplyChatStartSequenceRef.current = 0;
    liveReplyGiftStartSequenceRef.current = 0;
    handledLiveReplySequencesRef.current.clear();
    liveReplyInFlightRef.current = false;
    lastRepliedGiftRef.current = null;
    if (liveGiftReplyWakeTimerRef.current !== null) {
      window.clearTimeout(liveGiftReplyWakeTimerRef.current);
      liveGiftReplyWakeTimerRef.current = null;
    }
    return () => {
      liveStreamAbortRef.current?.abort();
      if (liveGiftReplyWakeTimerRef.current !== null) {
        window.clearTimeout(liveGiftReplyWakeTimerRef.current);
        liveGiftReplyWakeTimerRef.current = null;
      }
    };
  }, [username]);

  useEffect(() => {
    if (
      !douyinLoginDialogOpen ||
      douyinLoginStatus.status !== "waiting_scan"
    ) {
      return;
    }
    let cancelled = false;
    const timer = window.setInterval(() => {
      getDouyinLoginStatus()
        .then((status) => {
          if (cancelled) return;
          setDouyinLoginStatus(status);
          if (status.status === "ready") {
            setLiveCaptureState("idle");
            setLiveCaptureMessage(status.message);
          }
        })
        .catch((error) => {
          if (cancelled) return;
          setDouyinLoginStatus({
            status: "error",
            message:
              error instanceof ApiError
                ? error.message
                : "读取抖音登录状态失败",
            qr_image: null,
          });
        });
    }, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [douyinLoginDialogOpen, douyinLoginStatus.status]);

  useEffect(() => {
    if (!username) return;
    let releaseRequested = false;
    const releaseCapture = () => {
      if (releaseRequested) return;
      releaseRequested = true;
      releaseLiveCapture(username);
    };
    window.addEventListener("pagehide", releaseCapture);
    window.addEventListener("beforeunload", releaseCapture);
    return () => {
      window.removeEventListener("pagehide", releaseCapture);
      window.removeEventListener("beforeunload", releaseCapture);
    };
  }, [username]);

  useEffect(() => {
    if (activeConversationId && !usernameDialogOpen) {
      messageInputRef.current?.focus();
    }
  }, [activeConversationId, usernameDialogOpen]);

  useEffect(() => {
    if (
      !activeConversationId ||
      activeConversation?.memory_status !== "compressing"
    ) {
      return;
    }

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const pollMemoryStatus = async () => {
      try {
        const updated = await loadConversation(activeConversationId);
        if (cancelled) return;
        if (updated.memory_status === "compressing") {
          timer = setTimeout(pollMemoryStatus, 1000);
        } else if (chatContentView === "memory") {
          const memories = await loadShortTermMemories(activeConversationId);
          if (cancelled) return;
          const latestMemoryIsVisible =
            updated.memory_through_round === 0 ||
            memories.some(
              (memory) =>
                memory.through_round === updated.memory_through_round &&
                memory.summary === updated.short_term_memory,
            );
          if (!latestMemoryIsVisible) {
            timer = setTimeout(pollMemoryStatus, 500);
            return;
          }
          setShortTermMemories(memories);
        }
        setConversations((current) =>
          current.map((conversation) =>
            conversation.conversation_id === updated.conversation_id
              ? updated
              : conversation,
          ),
        );
      } catch {
        if (!cancelled) timer = setTimeout(pollMemoryStatus, 2000);
      }
    };

    timer = setTimeout(pollMemoryStatus, 1000);
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [
    activeConversation?.memory_status,
    activeConversationId,
    chatContentView,
  ]);

  const currentProvider = useMemo(() => providerInfo(provider), [provider]);

  useEffect(() => {
    const phrase = conversationSearchQuery.trim();
    if (!username || !phrase) return;

    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      searchUserConversations(username, phrase, controller.signal)
        .then((results) => {
          setConversationSearchResults(results);
        })
        .catch((error) => {
          if (error instanceof DOMException && error.name === "AbortError") {
            return;
          }
          setConversationSearchResults([]);
          setConversationSearchError("搜索失败，请检查后端或数据库连接");
        })
        .finally(() => {
          if (!controller.signal.aborted) {
            setSearchingConversations(false);
          }
        });
    }, 300);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [conversationSearchQuery, username]);

  useEffect(() => {
    selectedModelRef.current = { provider, model };
  }, [model, provider]);

  const systemPrompt = useMemo(() => {
    const rawPersonaPrompt =
      promptVersions.find(
        (version) => version.version === activePromptVersion,
      )?.content ?? personaPrompt;
    const parts = [
      ensurePersonaPromptIdentity(rawPersonaPrompt, speakerIdentity),
    ].filter(Boolean);
    const normalizedSpeakerPrompt = speakerPrompt.trim();
    if (normalizedSpeakerPrompt) {
      parts.push(`对话人提示词：\n${normalizedSpeakerPrompt}`);
    }
    return parts.join("\n\n");
  }, [
    activePromptVersion,
    personaPrompt,
    promptVersions,
    speakerIdentity,
    speakerPrompt,
  ]);

  useEffect(() => {
    let mounted = true;
    loadModelConfig()
      .then((config) => {
        if (!mounted) return;
        const loadedProvider = providerInfo(config.provider);
        const savedConfig = savedProviderConfig(
          config.providers,
          loadedProvider.id,
        );
        const loadedModel = catalogModelId(
          loadedProvider,
          config.provider === loadedProvider.id
            ? config.model
            : savedConfig?.model,
        );
        const hasApiKey =
          savedConfig?.has_api_key ??
          (config.provider === loadedProvider.id && config.has_api_key);
        setSavedProviderConfigs(config.providers);
        selectedModelRef.current = {
          provider: loadedProvider.id,
          model: loadedModel,
        };
        setProvider(loadedProvider.id);
        setModel(loadedModel);
        setModelConfigNotice(
          hasApiKey
            ? `已载入 ${loadedProvider.name}，服务端凭据已配置`
            : `已载入 ${loadedProvider.name}，服务端尚未配置 API Key`,
        );
      })
      .catch(() => {
        if (!mounted) return;
        setModelConfigNotice("后端暂未连接，可先编辑配置，连接后再测试或保存");
      });
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (!username) return;
    let mounted = true;
    loadPromptConfig(username)
      .then((config) => {
        if (!mounted) return;
        setPromptVersions(config.versions);
        setActivePromptVersion(config.active_version);
        setPersonaPrompt(config.persona_prompt);
        setSpeakerPromptVersions(config.speaker_versions);
        setActiveSpeakerPromptVersion(config.active_speaker_version);
        setSpeakerPrompt(config.speaker_prompt);
        setSpeakerIdentity(config.speaker_identity);
        setSavedSpeakerPrompt(config.speaker_prompt);
        setSavedSpeakerPromptIdentity(config.speaker_identity);
        setPromptConfigNotice("提示词配置已载入");
      })
      .catch(() => {
        if (!mounted) return;
        setPromptConfigNotice("后端暂未连接，无法读取提示词版本");
      });
    return () => {
      mounted = false;
    };
  }, [username]);

  useEffect(() => {
    if (!username || !model.trim()) return;
    let cancelled = false;
    saveUserModelConfig(username, {
      provider,
      model: model.trim(),
      web_search_enabled: webSearchEnabled,
      web_search_forced: webSearchForced,
      web_search_max_tool_calls: webSearchMaxToolCalls,
      web_search_result_limit: webSearchResultLimit,
      temperature,
    }).catch(() => {
      if (cancelled) return;
      setConnectionState("error");
      setModelConfigNotice("模型配置保存失败，请检查后端和数据库连接");
    });
    return () => {
      cancelled = true;
    };
  }, [
    model,
    provider,
    temperature,
    username,
    webSearchEnabled,
    webSearchForced,
    webSearchMaxToolCalls,
    webSearchResultLimit,
  ]);

  function handleProviderChange(nextProviderId: string) {
    const nextProvider = providerInfo(nextProviderId);
    const savedConfig = savedProviderConfig(savedProviderConfigs, nextProvider.id);
    const nextModel = catalogModelId(
      nextProvider,
      userProviderModels[nextProvider.id] ?? savedConfig?.model,
    );
    const nextWebSearchConfig = providerWebSearchConfig(
      userProviderWebSearchConfigs,
      nextProvider.id,
    );
    setProvider(nextProvider.id);
    setModel(nextModel);
    selectedModelRef.current = {
      provider: nextProvider.id,
      model: nextModel,
    };
    setTemperature(
      providerTemperature(userProviderTemperatures, nextProvider.id),
    );
    setWebSearchEnabled(nextWebSearchConfig.enabled);
    setWebSearchForced(nextWebSearchConfig.forced);
    setWebSearchMaxToolCalls(nextWebSearchConfig.max_tool_calls);
    setWebSearchResultLimit(nextWebSearchConfig.result_limit);
    setUserProviderModels((current) => ({
      ...current,
      [nextProvider.id]: nextModel,
    }));
    setConnectionState("idle");
    setModelConfigNotice(
      savedConfig?.has_api_key
        ? `已切换到 ${nextProvider.name}，服务端凭据已配置`
        : `已切换到 ${nextProvider.name}，服务端尚未配置 API Key`,
    );
  }

  function modelConfigPayload() {
    return {
      provider: selectedModelRef.current.provider,
      model: selectedModelRef.current.model.trim(),
    };
  }

  async function handleTestModelConfig() {
    const payload = modelConfigPayload();
    if (!payload.model) {
      setConnectionState("error");
      setModelConfigNotice("请先填写模型 ID");
      return;
    }

    setTestingModelConfig(true);
    setConnectionState("testing");
    setModelConfigNotice("正在测试模型连接...");
    try {
      const result = await testModelConnection(payload);
      setConnectionState(result.ok ? "success" : "error");
      setModelConfigNotice(
        result.ok
          ? `连接成功，耗时 ${Math.round(result.latency_ms ?? 0)} ms`
          : result.message,
      );
    } catch (error) {
      setConnectionState("error");
      setModelConfigNotice(
        error instanceof ApiError
          ? `测试失败：${error.message}`
          : "测试失败：后端暂未连接",
      );
    } finally {
      setTestingModelConfig(false);
    }
  }

  function handleLiveEvent(event: LiveRoomEvent) {
    liveSequenceRef.current = Math.max(
      liveSequenceRef.current,
      event.sequence,
    );
    if (event.type === "status") {
      setLiveCaptureState(
        event.status === "stopped" ? "idle" : event.status,
      );
      setLiveCaptureMessage(event.message);
    }
    setLiveEvents((current) => [...current, event].slice(-300));
  }

  async function connectLiveEventStream(
    activeUsername: string,
    controller: AbortController,
  ) {
    try {
      await streamLiveEvents(
        activeUsername,
        liveSequenceRef.current,
        handleLiveEvent,
        controller.signal,
      );
    } catch (error) {
      if (controller.signal.aborted) return;
      setLiveCaptureState("error");
      setLiveCaptureMessage(
        error instanceof ApiError ? error.message : "直播间事件流连接失败",
      );
    }
  }

  async function handleStartLiveCapture(event: FormEvent) {
    event.preventDefault();
    const normalizedRoomId = liveRoomId.trim();
    if (!username) {
      setLiveCaptureState("error");
      setLiveCaptureMessage("请先输入用户名");
      return;
    }
    if (!normalizedRoomId) {
      setLiveCaptureState("error");
      setLiveCaptureMessage("请输入直播间房间号");
      return;
    }
    liveStreamAbortRef.current?.abort();
    const controller = new AbortController();
    liveStreamAbortRef.current = controller;
    liveSequenceRef.current = 0;
    liveReplyChatStartSequenceRef.current = 0;
    liveReplyGiftStartSequenceRef.current = 0;
    handledLiveReplySequencesRef.current.clear();
    liveReplyInFlightRef.current = false;
    lastRepliedGiftRef.current = null;
    if (liveGiftReplyWakeTimerRef.current !== null) {
      window.clearTimeout(liveGiftReplyWakeTimerRef.current);
      liveGiftReplyWakeTimerRef.current = null;
    }
    setLiveEvents([]);
    setLiveCaptureState("starting");
    setLiveCaptureMessage("正在连接直播间");
    try {
      const result = await startLiveCapture(username, normalizedRoomId);
      setLiveRoomId(result.room_id);
      setLiveCaptureMessage(result.message);
      void connectLiveEventStream(username, controller);
    } catch (error) {
      if (controller.signal.aborted) return;
      setLiveCaptureState("error");
      setLiveCaptureMessage(
        error instanceof ApiError ? error.message : "启动直播间抓取失败",
      );
    }
  }

  async function handleStopLiveCapture() {
    if (!username) return;
    setLiveCaptureState("stopping");
    setLiveCaptureMessage("正在停止抓取");
    try {
      const result = await stopLiveCapture(username);
      setLiveCaptureState("idle");
      setLiveCaptureMessage(result.message);
    } catch (error) {
      setLiveCaptureState("error");
      setLiveCaptureMessage(
        error instanceof ApiError ? error.message : "停止直播间抓取失败",
      );
    } finally {
      liveStreamAbortRef.current?.abort();
      liveStreamAbortRef.current = null;
    }
  }

  async function handleDouyinLogin() {
    if (!username || douyinLoginBusy || douyinLoginChecking) return;
    setDouyinLoginBusy(true);
    setDouyinLoginDialogOpen(true);
    setDouyinLoginStatus({
      status: "idle",
      message: "正在生成抖音登录二维码…",
      qr_image: null,
    });
    try {
      liveStreamAbortRef.current?.abort();
      liveStreamAbortRef.current = null;
      const result = await startDouyinLogin();
      setLiveCaptureState("idle");
      setDouyinLoginStatus(result);
      setLiveCaptureMessage(result.message);
    } catch (error) {
      const message =
        error instanceof ApiError ? error.message : "生成抖音登录二维码失败";
      setDouyinLoginStatus({
        status: "error",
        message,
        qr_image: null,
      });
      setLiveCaptureState("error");
      setLiveCaptureMessage(message);
    } finally {
      setDouyinLoginBusy(false);
    }
  }

  async function handleCloseDouyinLogin() {
    setDouyinLoginBusy(true);
    try {
      const result = await finishDouyinLogin();
      setDouyinLoginStatus(result);
      if (result.status === "ready") {
        setLiveCaptureState("idle");
        setLiveCaptureMessage(result.message);
      }
    } catch {
      // Closing the dialog should remain available if the backend went away.
    } finally {
      setDouyinLoginBusy(false);
      setDouyinLoginDialogOpen(false);
    }
  }

  async function handleLoadWorkspace(event: FormEvent) {
    event.preventDefault();
    const normalizedUsername = usernameInput.trim();
    if (!normalizedUsername) {
      setWorkspaceNotice("请先填写用户名");
      return;
    }

    if (username && username !== normalizedUsername) {
      releaseLiveCapture(username);
    }
    setLoadingWorkspace(true);
    setUsernameDialogError("");
    try {
      const workspace = await loadUserWorkspace(normalizedUsername);
      const firstConversationId =
        workspace.conversations[0]?.conversation_id ?? "";
      const performanceDay = localDateKey();
      const [historyResult, dailyResult] = await Promise.allSettled([
        loadUserPerformanceMessages(normalizedUsername),
        loadUserPerformanceMessages(normalizedUsername, {
          limit: 500,
          day: performanceDay,
        }),
      ]);
      const initialPerformanceMessages =
        historyResult.status === "fulfilled"
          ? historyResult.value
          : workspace.messages;
      const initialDailyPerformanceMessages =
        dailyResult.status === "fulfilled"
          ? dailyResult.value
          : workspace.messages.filter(
              (message) =>
                localDateKey(new Date(message.created_at)) === performanceDay,
            );
      const initialPerformanceHistory = initialPerformanceMessages
        .map(messagePerformanceSample)
        .filter((sample): sample is PerformanceSample => sample !== null)
        .slice(-20);
      const initialDailyModelPerformance =
        initialDailyPerformanceMessages
          .map(messagePerformanceSample)
          .filter(
            (sample): sample is PerformanceSample => sample !== null,
          );
      const workspaceProvider = providerInfo(workspace.llm_config.provider);
      const workspaceModel = catalogModelId(
        workspaceProvider,
        workspace.llm_config.model,
      );
      const workspaceProviderModels = {
        ...(workspace.provider_models ?? {}),
        [workspaceProvider.id]: workspaceModel,
      };
      const workspaceProviderTemperatures = {
        ...(workspace.provider_temperatures ?? {}),
        [workspaceProvider.id]: workspace.llm_config.temperature,
      };
      const workspaceProviderWebSearchConfigs = {
        ...(workspace.provider_web_search_configs ?? {}),
        [workspaceProvider.id]: {
          enabled: workspace.llm_config.web_search_enabled,
          forced: workspace.llm_config.web_search_forced,
          max_tool_calls:
            workspace.llm_config.web_search_max_tool_calls,
          result_limit: workspace.llm_config.web_search_result_limit,
        },
      };
      setSpeakerPromptVersions([]);
      setActiveSpeakerPromptVersion("__none__");
      setSpeakerPrompt("");
      setSpeakerIdentity("");
      setSavedSpeakerPrompt("");
      setSavedSpeakerPromptIdentity("");
      setCreatingSpeakerPrompt(false);
      setPromptConfigNotice("正在读取当前用户的提示词配置...");
      liveStreamAbortRef.current?.abort();
      liveStreamAbortRef.current = null;
      liveSequenceRef.current = 0;
      setLiveEvents([]);
      setLiveCaptureState("idle");
      setLiveCaptureMessage("输入房间号后开始抓取");
      setUsername(normalizedUsername);
      setSavedSpeakerIdentity(workspace.speaker_identity);
      selectedModelRef.current = {
        provider: workspaceProvider.id,
        model: workspaceModel,
      };
      setProvider(workspaceProvider.id);
      setModel(workspaceModel);
      setUserProviderModels(workspaceProviderModels);
      setTemperature(workspace.llm_config.temperature);
      setUserProviderTemperatures(workspaceProviderTemperatures);
      setUserProviderWebSearchConfigs(
        workspaceProviderWebSearchConfigs,
      );
      setWebSearchEnabled(workspace.llm_config.web_search_enabled);
      setWebSearchForced(workspace.llm_config.web_search_forced);
      setWebSearchMaxToolCalls(
        workspace.llm_config.web_search_max_tool_calls,
      );
      setWebSearchResultLimit(
        workspace.llm_config.web_search_result_limit,
      );
      setConnectionState("idle");
      setPerformanceView("overview");
      setConversationSearchQuery("");
      setConversationSearchResults([]);
      setConversationSearchError("");
      setConversationSearchTarget(null);
      setPerformanceHistory(initialPerformanceHistory);
      setDailyModelPerformance(initialDailyModelPerformance);
      setDailyPerformanceDay(performanceDay);
      setConversations(workspace.conversations);
      setMessages(workspace.messages);
      setPerformanceMetrics(
        performanceMetricsFromSamples(initialPerformanceHistory),
      );
      messageCacheRef.current = new Map();
      if (firstConversationId) {
        messageCacheRef.current.set(firstConversationId, workspace.messages);
      }
      setChatContentView("conversation");
      setShortTermMemories([]);
      setMessageError("");
      setActiveConversationId(firstConversationId);
      setWorkspaceNotice(`已载入 ${workspace.conversations.length} 个历史对话`);
      setServiceState("running");
      setUsernameDialogOpen(false);
    } catch (error) {
      if (error instanceof ApiError && error.status === 503) {
        setServiceState("database-error");
        setUsernameDialogError(
          "数据库连接失败，请确认 Elasticsearch 已启动后重试。",
        );
      } else {
        setServiceState("backend-error");
        setUsernameDialogError("后端服务连接失败，请确认后端已启动后重试。");
      }
    } finally {
      setLoadingWorkspace(false);
    }
  }

  function handleOpenUsernameDialog() {
    setUsernameInput(username);
    setUsernameDialogError("");
    setUsernameDialogOpen(true);
  }

  async function handleNewConversation() {
    if (!username || creatingConversation) return;
    const fallbackTitle = nextVisibleConversationTitle(conversations);
    setCreatingConversation(true);
    setWorkspaceNotice("正在新建对话…");
    try {
      const conversation = await createConversation(username);
      setConversations((current) => [conversation, ...current]);
      setChatContentView("conversation");
      setShortTermMemories([]);
      setMessageError("");
      setActiveConversationId(conversation.conversation_id);
      setMessages([]);
      messageCacheRef.current.set(conversation.conversation_id, []);
      setWorkspaceNotice("新对话已保存到 Elasticsearch");
    } catch {
      const now = new Date().toISOString();
      const localConversation: ConversationSummary = {
        conversation_id: `local-${createClientId()}`,
        username,
        title: fallbackTitle,
        status: "active",
        sort_order: -Date.parse(now),
        completed_rounds: 0,
        effective_char_count: 0,
        memory_compression_count: 0,
        memory_through_round: 0,
        short_term_memory: "",
        memory_status: "idle",
        memory_target_round: 0,
        created_at: now,
        updated_at: now,
      };
      setConversations((current) => [localConversation, ...current]);
      setChatContentView("conversation");
      setShortTermMemories([]);
      setMessageError("");
      setActiveConversationId(localConversation.conversation_id);
      setMessages([]);
      messageCacheRef.current.set(localConversation.conversation_id, []);
      setWorkspaceNotice("已建立本地预览对话，连接后端后可持久化");
    } finally {
      setCreatingConversation(false);
    }
  }

  async function handleSwitchConversation(
    conversationId: string,
    options: {
      preserveSearchTarget?: boolean;
      forceReload?: boolean;
    } = {},
  ) {
    if (!options.preserveSearchTarget) {
      setConversationSearchTarget(null);
    }
    if (conversationId === activeConversationId && !options.forceReload) return;

    const loadSequence = ++conversationLoadSequenceRef.current;
    if (activeConversationId) {
      messageCacheRef.current.set(activeConversationId, messages);
    }
    setChatContentView("conversation");
    setShortTermMemories([]);
    setMessageError("");
    setActiveConversationId(conversationId);
    const cachedMessages = messageCacheRef.current.get(conversationId);
    if (cachedMessages) {
      setMessages(cachedMessages);
    } else {
      setMessages([]);
    }

    setWorkspaceNotice("正在载入历史消息…");
    try {
      const loadedMessages = await loadConversationMessages(conversationId);
      messageCacheRef.current.set(conversationId, loadedMessages);
      if (conversationLoadSequenceRef.current === loadSequence) {
        setMessages(loadedMessages);
        setWorkspaceNotice(`已载入 ${loadedMessages.length} 条历史消息`);
      }
    } catch {
      if (conversationLoadSequenceRef.current === loadSequence) {
        setWorkspaceNotice("历史消息载入失败，请检查后端连接");
      }
    }
  }

  function handleOpenConversationSearchMatch(
    conversationId: string,
    messageId: string,
  ) {
    const phrase = conversationSearchQuery.trim();
    setConversationSearchTarget({ conversationId, messageId, phrase });
    setChatContentView("conversation");
    const targetIsLoaded =
      conversationId === activeConversationId &&
      messages.some((message) => message.message_id === messageId);
    void handleSwitchConversation(conversationId, {
      preserveSearchTarget: true,
      forceReload: !targetIsLoaded,
    });
  }

  function handleConversationDragStart(
    event: ReactDragEvent<HTMLElement>,
    conversationId: string,
  ) {
    conversationOrderBeforeDragRef.current = [...conversations];
    conversationOrderDuringDragRef.current = [...conversations];
    conversationDropCommittedRef.current = false;
    setDraggingConversationId(conversationId);
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", conversationId);
  }

  function handleConversationDragOver(
    event: ReactDragEvent<HTMLDivElement>,
    targetConversationId: string,
  ) {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    if (
      !draggingConversationId ||
      draggingConversationId === targetConversationId
    ) {
      return;
    }
    setConversations((current) => {
      const fromIndex = current.findIndex(
        (conversation) =>
          conversation.conversation_id === draggingConversationId,
      );
      const toIndex = current.findIndex(
        (conversation) =>
          conversation.conversation_id === targetConversationId,
      );
      if (fromIndex < 0 || toIndex < 0 || fromIndex === toIndex) {
        return current;
      }
      const reordered = [...current];
      const [dragged] = reordered.splice(fromIndex, 1);
      reordered.splice(toIndex, 0, dragged);
      conversationOrderDuringDragRef.current = reordered;
      return reordered;
    });
  }

  async function handleConversationDrop(
    event: ReactDragEvent<HTMLDivElement>,
  ) {
    event.preventDefault();
    if (!draggingConversationId) return;
    conversationDropCommittedRef.current = true;
    const reordered = conversationOrderDuringDragRef.current;
    setDraggingConversationId("");
    try {
      const saved = await reorderConversations(
        username,
        reordered.map((conversation) => conversation.conversation_id),
      );
      setConversations(saved);
      conversationOrderDuringDragRef.current = saved;
      setWorkspaceNotice("对话顺序已保存");
    } catch {
      const original = conversationOrderBeforeDragRef.current;
      setConversations(original);
      conversationOrderDuringDragRef.current = original;
      setWorkspaceNotice("对话排序保存失败，已恢复原顺序");
    }
  }

  function handleConversationDragEnd() {
    if (!conversationDropCommittedRef.current) {
      const original = conversationOrderBeforeDragRef.current;
      setConversations(original);
      conversationOrderDuringDragRef.current = original;
    }
    conversationDropCommittedRef.current = false;
    setDraggingConversationId("");
  }

  function handleStartRename(conversation: ConversationSummary) {
    setEditingConversationId(conversation.conversation_id);
    setConversationTitleInput(conversation.title);
  }

  function handleCancelRename() {
    setEditingConversationId("");
    setConversationTitleInput("");
  }

  async function handleRenameConversation(event: FormEvent) {
    event.preventDefault();
    const title = conversationTitleInput.trim();
    const conversationId = editingConversationId;
    if (!title || !conversationId || renamingConversationId) return;

    setRenamingConversationId(conversationId);
    try {
      const updatedConversation = await renameConversation(
        conversationId,
        title,
      );
      setConversations((current) =>
        current.map((conversation) =>
          conversation.conversation_id === conversationId
            ? updatedConversation
            : conversation,
        ),
      );
      setWorkspaceNotice("对话名称已更新");
      handleCancelRename();
    } catch {
      setWorkspaceNotice("重命名失败，请检查后端连接后重试");
    } finally {
      setRenamingConversationId("");
    }
  }

  async function handleArchiveConversation() {
    if (!pendingConversationDelete) return;
    const conversationId = pendingConversationDelete.conversation_id;
    setArchivingConversationId(conversationId);
    try {
      await archiveConversation(conversationId);
      messageCacheRef.current.delete(conversationId);
      setConversationSearchResults((current) =>
        current.filter(
          (result) => result.conversation_id !== conversationId,
        ),
      );
      const remainingConversations = conversations.filter(
        (conversation) => conversation.conversation_id !== conversationId,
      );
      setConversations(remainingConversations);
      if (activeConversationId === conversationId) {
        setChatContentView("conversation");
        setShortTermMemories([]);
        setMessageError("");
        setActiveConversationId(
          remainingConversations[0]?.conversation_id ?? "",
        );
        setMessages([]);
      }
      setWorkspaceNotice("对话已删除");
    } catch {
      setWorkspaceNotice("删除对话失败，请检查后端连接后重试");
    } finally {
      setArchivingConversationId("");
      setPendingConversationDelete(null);
    }
  }

  async function handleToggleMemoryView() {
    if (chatContentView === "memory") {
      setChatContentView("conversation");
      return;
    }
    if (!activeConversationId) return;
    setChatContentView("memory");
    if (loadingMemories) return;
    setLoadingMemories(true);
    try {
      const memories = await loadShortTermMemories(activeConversationId);
      setShortTermMemories(memories);
    } catch {
      setShortTermMemories([]);
      setWorkspaceNotice("短期记忆载入失败，请检查后端连接");
    } finally {
      setLoadingMemories(false);
    }
  }

  function handleMemoryDividerStart(
    event: ReactPointerEvent<HTMLDivElement>,
  ) {
    event.preventDefault();
    const divider = event.currentTarget;
    const ownerWindow = divider.ownerDocument.defaultView;
    if (!ownerWindow) return;

    divider.setPointerCapture(event.pointerId);
    resizingMemoryPaneRef.current = true;
    setResizingMemoryPane(true);

    const handlePointerMove = (moveEvent: PointerEvent) => {
      if (!resizingMemoryPaneRef.current || !chatContentRef.current) return;
      const bounds = chatContentRef.current.getBoundingClientRect();
      const nextPercent =
        ((moveEvent.clientX - bounds.left) / bounds.width) * 100;
      setConversationPanePercent(Math.min(72, Math.max(28, nextPercent)));
    };
    const handlePointerEnd = () => {
      resizingMemoryPaneRef.current = false;
      setResizingMemoryPane(false);
      ownerWindow.removeEventListener("pointermove", handlePointerMove);
      ownerWindow.removeEventListener("pointerup", handlePointerEnd);
      ownerWindow.removeEventListener("pointercancel", handlePointerEnd);
      if (divider.hasPointerCapture(event.pointerId)) {
        divider.releasePointerCapture(event.pointerId);
      }
    };

    ownerWindow.addEventListener("pointermove", handlePointerMove);
    ownerWindow.addEventListener("pointerup", handlePointerEnd);
    ownerWindow.addEventListener("pointercancel", handlePointerEnd);
  }

  function handleMemoryDividerKeyDown(
    event: ReactKeyboardEvent<HTMLDivElement>,
  ) {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
      return;
    }
    event.preventDefault();
    setConversationPanePercent((current) =>
      Math.min(
        72,
        Math.max(28, current + (event.key === "ArrowLeft" ? -2 : 2)),
      ),
    );
  }

  function resizeWorkspaceBy(delta: number) {
    const dashboard =
      dashboardHeight ?? dashboardGridRef.current?.getBoundingClientRect().height;
    if (!dashboard) return;
    const appliedDelta = Math.min(
      dashboard - 260,
      Math.max(400 - workspaceHeight, delta),
    );
    setDashboardHeight(dashboard - appliedDelta);
    setWorkspaceHeight(workspaceHeight + appliedDelta);
  }

  function handleWorkspaceDividerStart(
    event: ReactPointerEvent<HTMLDivElement>,
  ) {
    event.preventDefault();
    const divider = event.currentTarget;
    const ownerWindow = divider.ownerDocument.defaultView;
    const dashboard = dashboardGridRef.current;
    if (!ownerWindow || !dashboard || ownerWindow.innerWidth <= 860) return;

    const startY = event.clientY;
    const startDashboardHeight = dashboard.getBoundingClientRect().height;
    const startWorkspaceHeight = workspaceHeight;
    divider.setPointerCapture(event.pointerId);
    resizingWorkspaceRef.current = true;
    setResizingWorkspace(true);

    const handlePointerMove = (moveEvent: PointerEvent) => {
      if (!resizingWorkspaceRef.current) return;
      const requestedDelta = startY - moveEvent.clientY;
      const appliedDelta = Math.min(
        startDashboardHeight - 260,
        Math.max(400 - startWorkspaceHeight, requestedDelta),
      );
      setDashboardHeight(startDashboardHeight - appliedDelta);
      setWorkspaceHeight(startWorkspaceHeight + appliedDelta);
    };
    const handlePointerEnd = () => {
      resizingWorkspaceRef.current = false;
      setResizingWorkspace(false);
      ownerWindow.removeEventListener("pointermove", handlePointerMove);
      ownerWindow.removeEventListener("pointerup", handlePointerEnd);
      ownerWindow.removeEventListener("pointercancel", handlePointerEnd);
      if (divider.hasPointerCapture(event.pointerId)) {
        divider.releasePointerCapture(event.pointerId);
      }
    };

    ownerWindow.addEventListener("pointermove", handlePointerMove);
    ownerWindow.addEventListener("pointerup", handlePointerEnd);
    ownerWindow.addEventListener("pointercancel", handlePointerEnd);
  }

  function handleWorkspaceDividerKeyDown(
    event: ReactKeyboardEvent<HTMLDivElement>,
  ) {
    if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
    event.preventDefault();
    resizeWorkspaceBy(event.key === "ArrowUp" ? 24 : -24);
  }

  async function sendMessageContent(
    rawContent: string,
    options: {
      attributedContent?: string;
      clickStartedAt?: number;
      clickTimestamp?: string;
      clearDraft?: boolean;
    } = {},
  ) {
    const clickStartedAt = options.clickStartedAt ?? 0;
    const clickTimestamp = options.clickTimestamp ?? "";
    const traceId = createClientId();
    const phaseTimes: Partial<Record<ChatStreamPhase, number>> = {};
    let firstPaintScheduled = false;
    let visibleWebSearchSources: WebSearchSource[] = [];
    const content = rawContent.trim();
    if (!content || !activeConversationId || isReplying) return;
    setConversationSearchTarget(null);
    const normalizedIdentity = speakerIdentity.trim();
    const attributedContent =
      options.attributedContent ??
      (normalizedIdentity ? `${normalizedIdentity}：“${content}”` : content);

    const optimisticMessage: MessageRecord = {
      message_id: `optimistic-${createClientId()}`,
      conversation_id: activeConversationId,
      username,
      role: "user",
      content: attributedContent,
      created_at: new Date().toISOString(),
    };
    const streamingMessage: MessageRecord = {
      message_id: `streaming-${createClientId()}`,
      conversation_id: activeConversationId,
      username,
      role: "assistant",
      content: "",
      created_at: new Date().toISOString(),
      metadata: { streaming: true },
    };
    setMessages((current) => {
      const nextMessages = [
        ...current.filter(
          (message) => message.metadata?.failed_attempt !== true,
        ),
        optimisticMessage,
        streamingMessage,
      ];
      messageCacheRef.current.set(activeConversationId, nextMessages);
      return nextMessages;
    });
    if (options.clearDraft) setMessageDraft("");
    messageInputRef.current?.focus();
    setMessageError("");
    setIsReplying(true);
    setPerformanceMetrics({
      knowledge_duration_ms: null,
      web_search_duration_ms: null,
      model_first_token_ms: null,
      model_first_sentence_ms: null,
    });

    try {
      const selectedModel = selectedModelRef.current;
      const response = await streamChatMessage(
        activeConversationId,
        username,
        attributedContent,
        content,
        systemPrompt,
        knowledgeEnabled,
        normalizedIdentity,
        normalizedIdentity !== savedSpeakerIdentity,
        {
          provider: selectedModel.provider,
          model: selectedModel.model,
          web_search_enabled:
            providerSupportsWebSearch(selectedModel.provider) &&
            webSearchEnabled,
          web_search_forced:
            providerSupportsWebSearch(selectedModel.provider) &&
            webSearchEnabled &&
            webSearchForced,
          web_search_max_tool_calls: webSearchMaxToolCalls,
          web_search_result_limit: webSearchResultLimit,
          temperature,
        },
        traceId,
        (delta) => {
          setMessages((current) =>
            current.map((message) =>
              message.message_id === streamingMessage.message_id
                ? { ...message, content: message.content + delta }
                : message,
            ),
          );
          if (!firstPaintScheduled) {
            firstPaintScheduled = true;
            requestAnimationFrame(() => {
              requestAnimationFrame(() => {
                const firstPaintAt = performance.now();
                const requestStartAt =
                  phaseTimes.request_started ?? clickStartedAt;
                const responseHeadersAt =
                  phaseTimes.response_headers ?? firstPaintAt;
                const streamStartAt =
                  phaseTimes.stream_start ?? responseHeadersAt;
                const firstChunkAt =
                  phaseTimes.first_chunk ?? firstPaintAt;
                const elapsed = (end: number, start: number) =>
                  Math.max(0, Math.round((end - start) * 100) / 100);

                void reportChatLatency({
                  trace_id: traceId,
                  conversation_id: activeConversationId,
                  click_timestamp: clickTimestamp,
                  click_to_request_start_ms: elapsed(
                    requestStartAt,
                    clickStartedAt,
                  ),
                  click_to_response_headers_ms: elapsed(
                    responseHeadersAt,
                    clickStartedAt,
                  ),
                  click_to_stream_start_ms: elapsed(
                    streamStartAt,
                    clickStartedAt,
                  ),
                  click_to_first_chunk_ms: elapsed(
                    firstChunkAt,
                    clickStartedAt,
                  ),
                  click_to_first_paint_ms: elapsed(
                    firstPaintAt,
                    clickStartedAt,
                  ),
                  response_headers_to_first_chunk_ms: elapsed(
                    firstChunkAt,
                    responseHeadersAt,
                  ),
                  first_chunk_to_first_paint_ms: elapsed(
                    firstPaintAt,
                    firstChunkAt,
                  ),
                }).catch(() => {
                  // Latency telemetry must never interrupt the conversation.
                });
              });
            });
          }
        },
        (injectedContext, hitCount) => {
          if (!injectedContext) return;
          setMessages((current) =>
            current.map((message) =>
              message.message_id === streamingMessage.message_id
                ? {
                    ...message,
                    metadata: {
                      ...message.metadata,
                      knowledge_hit_count: hitCount,
                      knowledge_injected_context: injectedContext,
                    },
                  }
                : message,
            ),
          );
        },
        async (sources) => {
          const visibleSourceUrls = new Set(
            visibleWebSearchSources.map((source) => source.url),
          );
          const newSources = sources.filter(
            (source) => !visibleSourceUrls.has(source.url),
          );
          for (const source of newSources) {
            visibleWebSearchSources = [...visibleWebSearchSources, source];
            setMessages((current) =>
              current.map((message) =>
                message.message_id === streamingMessage.message_id
                  ? {
                      ...message,
                      metadata: {
                        ...message.metadata,
                        web_search_sources: visibleWebSearchSources,
                      },
                    }
                  : message,
              ),
            );
            await new Promise<void>((resolve) => {
              requestAnimationFrame(() => {
                window.setTimeout(resolve, 100);
              });
            });
          }
        },
        (metrics) => {
          setPerformanceMetrics((current) => ({
            ...current,
            ...metrics,
          }));
        },
        (phase, timestamp) => {
          phaseTimes[phase] = timestamp;
        },
      );
      setMessages((current) => {
        const completedMessages = [
          ...current.filter(
          (message) =>
            message.message_id !== optimisticMessage.message_id &&
            message.message_id !== streamingMessage.message_id,
          ),
          response.user_message,
          response.assistant_message,
        ];
        messageCacheRef.current.set(
          activeConversationId,
          completedMessages,
        );
        return completedMessages;
      });
      const completedPerformanceSample = messagePerformanceSample(
        response.assistant_message,
      );
      if (completedPerformanceSample) {
        setPerformanceHistory((current) => [
          ...current.filter(
            (sample) =>
              sample.messageId !== completedPerformanceSample.messageId,
          ),
          completedPerformanceSample,
        ].slice(-20));
        const completedDay = localDateKey(
          new Date(completedPerformanceSample.createdAt),
        );
        setDailyModelPerformance((current) =>
          (completedDay === dailyPerformanceDay
            ? [
                ...current.filter(
                  (sample) =>
                    sample.messageId !==
                    completedPerformanceSample.messageId,
                ),
                completedPerformanceSample,
              ]
            : [completedPerformanceSample]
          ).slice(-500),
        );
        setDailyPerformanceDay(completedDay);
      }
      setSavedSpeakerIdentity(normalizedIdentity);
      setConversations((current) =>
        current.map((conversation) => {
          if (conversation.conversation_id !== activeConversationId) {
            return conversation;
          }
          return {
            ...conversation,
            completed_rounds: response.completed_rounds,
            effective_char_count: response.effective_char_count,
            memory_status: response.memory_status,
            memory_through_round: response.memory_through_round,
            memory_target_round: response.memory_target_round,
          };
        }),
      );
      setWorkspaceNotice(
        `模型回复已保存${
          response.knowledge_hit_count
            ? `，引用 ${response.knowledge_hit_count} 条知识`
            : ""
        }`,
      );
      setMessageError("");
    } catch (error) {
      const isServerError = error instanceof ApiError;
      setMessages((current) => {
        const failedMessages = current
          .filter(
            (message) =>
              message.message_id !== streamingMessage.message_id,
          )
          .map((message) =>
            message.message_id === optimisticMessage.message_id
              ? {
                  ...message,
                  metadata: {
                    ...message.metadata,
                    failed_attempt: true,
                  },
                }
              : message,
          );
        messageCacheRef.current.set(
          activeConversationId,
          failedMessages,
        );
        return failedMessages;
      });
      setMessageError(
        isServerError
          ? `模型回复失败：${error.message}`
          : "后端尚未连接，消息发送失败",
      );
    } finally {
      setIsReplying(false);
    }
  }

  useEffect(() => {
    sendMessageContentRef.current = sendMessageContent;
  });

  useEffect(() => {
    if (
      liveCaptureState !== "running" ||
      isReplying ||
      liveReplyInFlightRef.current ||
      !activeConversationId ||
      (!replyLiveChats && !replyLiveGifts)
    ) {
      return;
    }

    if (liveGiftReplyWakeTimerRef.current !== null) {
      window.clearTimeout(liveGiftReplyWakeTimerRef.current);
      liveGiftReplyWakeTimerRef.current = null;
    }
    const handled = handledLiveReplySequencesRef.current;
    const now = Date.now();
    const giftCandidates = replyLiveGifts
      ? liveEvents.filter(
          (
            event,
          ): event is Extract<LiveRoomEvent, { type: "gift" }> =>
            event.type === "gift" &&
            event.sequence > liveReplyGiftStartSequenceRef.current &&
            !handled.has(event.sequence) &&
            now - new Date(event.timestamp).getTime() <= 60_000,
        )
      : [];
    const giftLock = lastRepliedGiftRef.current;
    const giftLockRemaining = giftLock
      ? Math.max(0, giftLock.repliedAt + 60_000 - now)
      : 0;
    const eligibleGiftCandidates = giftCandidates.filter(
      (event) =>
        giftLockRemaining === 0 ||
        !giftLock ||
        liveGiftValue(event) > giftLock.value,
    );
    if (
      giftCandidates.length > 0 &&
      eligibleGiftCandidates.length === 0 &&
      giftLockRemaining > 0
    ) {
      liveGiftReplyWakeTimerRef.current = window.setTimeout(() => {
        liveGiftReplyWakeTimerRef.current = null;
        setLiveGiftReplyWakeTick((current) => current + 1);
      }, giftLockRemaining + 10);
    }
    let selected: ReplyableLiveEvent | undefined;
    if (eligibleGiftCandidates.length) {
      selected = [...eligibleGiftCandidates].sort(
        (left, right) =>
          liveGiftValue(right) - liveGiftValue(left) ||
          right.sequence - left.sequence,
      )[0];
      giftCandidates.forEach((event) => handled.add(event.sequence));
    } else if (replyLiveChats) {
      const chatCandidates = liveEvents.filter(
        (
          event,
        ): event is Extract<LiveRoomEvent, { type: "chat" }> =>
          event.type === "chat" &&
          event.sequence > liveReplyChatStartSequenceRef.current &&
          !handled.has(event.sequence),
      );
      if (chatCandidates.length) {
        selected =
          chatCandidates[Math.floor(Math.random() * chatCandidates.length)];
        chatCandidates.forEach((event) => handled.add(event.sequence));
      }
    }
    const sendMessage = sendMessageContentRef.current;
    if (!selected || !sendMessage) return;

    const selectedMessage = liveEventMessage(selected);
    const selectedGiftValue =
      selected.type === "gift" ? liveGiftValue(selected) : null;
    liveReplyInFlightRef.current = true;
    void sendMessage(selectedMessage.content, {
      attributedContent: selectedMessage.attributedContent,
      clickStartedAt: performance.now(),
      clickTimestamp: new Date().toISOString(),
    }).finally(() => {
      if (selectedGiftValue !== null) {
        lastRepliedGiftRef.current = {
          value: selectedGiftValue,
          repliedAt: Date.now(),
        };
        setLiveGiftReplyWakeTick((current) => current + 1);
      }
      liveReplyInFlightRef.current = false;
    });
  }, [
    activeConversationId,
    isReplying,
    liveCaptureState,
    liveEvents,
    liveGiftReplyWakeTick,
    replyLiveChats,
    replyLiveGifts,
  ]);

  function handleSendMessage(event: FormEvent) {
    event.preventDefault();
    void sendMessageContent(messageDraft, {
      clickStartedAt: event.timeStamp,
      clickTimestamp: new Date().toISOString(),
      clearDraft: true,
    });
  }


  async function handleRewindLastTurn() {
    if (!activeConversationId || isReplying || isRewinding) return;

    setIsRewinding(true);
    setWorkspaceNotice("正在回溯上一轮…");
    try {
      const result = await rewindConversation(activeConversationId, username);
      if (result.deleted_count === 0) {
        setWorkspaceNotice("当前对话没有可回溯的问答");
        return;
      }

      setMessages((current) => {
        let latestUserIndex = -1;
        for (let index = current.length - 1; index >= 0; index -= 1) {
          if (current[index].role === "user") {
            latestUserIndex = index;
            break;
          }
        }
        const remainingMessages =
          latestUserIndex >= 0
            ? current.slice(0, latestUserIndex)
            : current.filter(
                (message) => !result.message_ids.includes(message.message_id),
              );
        messageCacheRef.current.set(
          activeConversationId,
          remainingMessages,
        );
        return remainingMessages;
      });
      setConversations((current) =>
        current.map((conversation) =>
          conversation.conversation_id === activeConversationId
            ? {
                ...conversation,
                completed_rounds: result.completed_rounds,
                effective_char_count: result.effective_char_count,
                memory_compression_count: result.memory_compression_count,
                memory_through_round: result.memory_through_round,
                short_term_memory: result.short_term_memory,
                memory_status: result.memory_status,
                memory_target_round: result.memory_target_round,
              }
            : conversation,
        ),
      );
      if (chatContentView === "memory") {
        setShortTermMemories((current) => {
          const latestByRound = new Map<number, ShortTermMemory>();
          for (const memory of [...current].sort(
            (left, right) =>
              new Date(right.created_at).getTime() -
              new Date(left.created_at).getTime(),
          )) {
            if (
              memory.through_round <= result.memory_through_round &&
              !latestByRound.has(memory.through_round)
            ) {
              latestByRound.set(memory.through_round, memory);
            }
          }
          return [...latestByRound.values()].sort(
            (left, right) => left.through_round - right.through_round,
          );
        });
      }
      setWorkspaceNotice("已回溯上一轮对话");
    } catch {
      setWorkspaceNotice("回溯失败，请检查后端连接后重试");
    } finally {
      setIsRewinding(false);
    }
  }

  async function handleExportConversation() {
    if (!activeConversationId || exportingConversation || isReplying) return;

    setExportingConversation(true);
    setWorkspaceNotice("正在整理当前对话…");
    try {
      const [exportedMessages, exportedMemories] = activeConversationId.startsWith(
        "local-",
      )
        ? [messages, shortTermMemories]
        : await Promise.all([
            loadConversationMessages(activeConversationId),
            loadShortTermMemories(activeConversationId),
          ]);
      const title = activeConversation?.title.trim() || "对话";
      const exportedAt = new Date().toISOString();
      let currentExportRound = 0;
      const exportedRoundNumbers = new Set<number>();
      const conversationLines = exportedMessages.flatMap((message) => {
        const display = messageDisplay(message);
        const storedRound = message.metadata?.round_number;
        if (message.role === "user") {
          currentExportRound =
            typeof storedRound === "number"
              ? storedRound
              : currentExportRound + 1;
          exportedRoundNumbers.add(currentExportRound);
        } else if (typeof storedRound === "number") {
          currentExportRound = storedRound;
        }
        const sections = [
          ...(message.role === "user" && currentExportRound > 0
            ? [`[第 ${currentExportRound} 轮]`]
            : []),
          `[${exportDateTime(message.created_at)}] ${display.speaker}：${display.content}`,
        ];
        const injectedKnowledge =
          typeof message.metadata?.knowledge_injected_context === "string"
            ? message.metadata.knowledge_injected_context.trim()
            : "";
        if (injectedKnowledge) {
          const hitCount =
            typeof message.metadata?.knowledge_hit_count === "number"
              ? message.metadata.knowledge_hit_count
              : 0;
          sections.push(
            `[知识库注入 · ${hitCount} 条召回内容]\n${injectedKnowledge}`,
          );
        }
        const webSearchSources = messageWebSearchSources(message);
        if (webSearchSources.length) {
          sections.push(
            `[联网搜索 · ${webSearchSources.length} 个网页]\n${webSearchSources
              .map(
                (source, index) =>
                  `${index + 1}. ${source.title}\n链接：${source.url}${
                    source.snippet ? `\n内容：${source.snippet}` : ""
                  }`,
              )
              .join("\n\n")}`,
          );
        }
        return sections;
      });
      const memoryLines = exportedMemories.flatMap((memory, index) => [
        index === 0 ? "短期记忆压缩记录" : "",
        `[第 ${memory.compression_number} 次压缩 · 已压缩至第 ${memory.through_round} 轮 · ${exportDateTime(memory.created_at)}]\n${memory.summary}`,
      ]).filter(Boolean);
      const lines = [
        `标题：${title}`,
        `用户名：${username}`,
        `总轮数：${exportedRoundNumbers.size}`,
        `消息数：${exportedMessages.length}`,
        `导出时间：${exportDateTime(exportedAt)}`,
        "",
        ...conversationLines,
        ...(memoryLines.length ? ["", ...memoryLines] : []),
      ];
      const blob = new Blob(["\uFEFF", lines.join("\n\n")], {
        type: "text/plain;charset=utf-8",
      });
      const downloadUrl = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = downloadUrl;
      anchor.download = `${safeExportFilename(title)}-${exportDateTime(exportedAt).replace(/[: ]/g, "-")}.txt`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 0);
      setWorkspaceNotice(
        `已导出 ${exportedRoundNumbers.size} 轮（${exportedMessages.length} 条消息）和 ${exportedMemories.length} 条短期记忆`,
      );
    } catch {
      setWorkspaceNotice("导出失败，请检查后端连接后重试");
    } finally {
      setExportingConversation(false);
    }
  }

  async function handlePromptVersion(version: string) {
    const nextPromptVersion = promptVersions.find(
      (promptVersion) => promptVersion.version === version,
    );
    if (!nextPromptVersion) return;

    setActivePromptVersion(version);
    setPersonaPrompt(nextPromptVersion.content);
    setPromptConfigNotice(`正在切换人设提示词：${nextPromptVersion.title}`);
    try {
      const config = await savePromptConfig(username, { active_version: version });
      setPromptVersions(config.versions);
      setActivePromptVersion(config.active_version);
      setPersonaPrompt(config.persona_prompt);
      setPromptConfigNotice(`已切换人设提示词：${nextPromptVersion.title}`);
    } catch (error) {
      setPromptConfigNotice(
        error instanceof ApiError
          ? `切换失败：${error.message}`
          : "切换失败：后端暂未连接",
      );
    }
  }

  async function handleSpeakerPromptVersion(version: string) {
    setCreatingSpeakerPrompt(false);
    if (version === "__none__") {
      setActiveSpeakerPromptVersion(version);
      setSpeakerIdentity("");
      setSpeakerPrompt("");
      setSavedSpeakerPromptIdentity("");
      setSavedSpeakerPrompt("");
      try {
        const config = await savePromptConfig(username, {
          active_speaker_version: version,
        });
        setSpeakerPromptVersions(config.speaker_versions);
        setActiveSpeakerPromptVersion(config.active_speaker_version);
        setSavedSpeakerPromptIdentity("");
        setSavedSpeakerPrompt("");
        setPromptConfigNotice("");
      } catch (error) {
        setPromptConfigNotice(
          error instanceof ApiError
            ? `切换失败：${error.message}`
            : "切换失败：后端暂未连接",
        );
      }
      return;
    }
    const nextPromptVersion = speakerPromptVersions.find(
      (promptVersion) => promptVersion.version === version,
    );
    if (!nextPromptVersion) return;

    setActiveSpeakerPromptVersion(version);
    setSpeakerIdentity(nextPromptVersion.speaker_identity);
    setSpeakerPrompt(nextPromptVersion.content);
    setSavedSpeakerPromptIdentity(nextPromptVersion.speaker_identity);
    setSavedSpeakerPrompt(nextPromptVersion.content);
    try {
      const config = await savePromptConfig(username, {
        active_speaker_version: version,
      });
      setSpeakerPromptVersions(config.speaker_versions);
      setActiveSpeakerPromptVersion(config.active_speaker_version);
      setSpeakerIdentity(config.speaker_identity);
      setSpeakerPrompt(config.speaker_prompt);
      setSavedSpeakerPromptIdentity(config.speaker_identity);
      setSavedSpeakerPrompt(config.speaker_prompt);
      setPromptConfigNotice("");
    } catch (error) {
      setPromptConfigNotice(
        error instanceof ApiError
          ? `切换失败：${error.message}`
          : "切换失败：后端暂未连接",
      );
    }
  }

  function handleCreateSpeakerPrompt() {
    setCreatingSpeakerPrompt(true);
    setActiveSpeakerPromptVersion("");
    setSpeakerIdentity("");
    setSpeakerPrompt("");
    setSavedSpeakerPromptIdentity("");
    setSavedSpeakerPrompt("");
    setPromptConfigNotice("填写身份和描述后保存为新版本");
  }

  function requestSpeakerPromptTransition(
    transition: SpeakerPromptTransition,
  ) {
    if (
      transition.kind === "select" &&
      !creatingSpeakerPrompt &&
      transition.version === activeSpeakerPromptVersion
    ) {
      return;
    }
    if (speakerPromptHasUnsavedChanges) {
      setPendingSpeakerPromptTransition(transition);
      return;
    }
    if (transition.kind === "create") {
      handleCreateSpeakerPrompt();
    } else {
      void handleSpeakerPromptVersion(transition.version);
    }
  }

  function handleConfirmSpeakerPromptTransition() {
    const transition = pendingSpeakerPromptTransition;
    setPendingSpeakerPromptTransition(null);
    if (!transition) return;
    if (transition.kind === "create") {
      handleCreateSpeakerPrompt();
    } else {
      void handleSpeakerPromptVersion(transition.version);
    }
  }

  async function handleSavePromptConfig() {
    const normalizedIdentity = speakerIdentity.trim();
    if (!normalizedIdentity) {
      setPromptConfigNotice("请填写对话人身份");
      return;
    }
    setSavingPromptConfig(true);
    setPromptConfigNotice(
      creatingSpeakerPrompt
        ? "正在保存对话人提示词..."
        : "正在保存对话人提示词修改...",
    );
    try {
      const config = await savePromptConfig(username, {
        speaker_prompt: speakerPrompt,
        speaker_identity: normalizedIdentity,
        ...(creatingSpeakerPrompt
          ? { create_speaker_version: true }
          : { update_speaker_version: activeSpeakerPromptVersion }),
      });
      setPromptVersions(config.versions);
      setActivePromptVersion(config.active_version);
      setPersonaPrompt(config.persona_prompt);
      setSpeakerPromptVersions(config.speaker_versions);
      setActiveSpeakerPromptVersion(config.active_speaker_version);
      setSpeakerIdentity(config.speaker_identity);
      setSpeakerPrompt(config.speaker_prompt);
      setSavedSpeakerPromptIdentity(config.speaker_identity);
      setSavedSpeakerPrompt(config.speaker_prompt);
      setCreatingSpeakerPrompt(false);
      setPromptConfigNotice(
        creatingSpeakerPrompt
          ? ""
          : "对话人提示词修改已保存",
      );
    } catch (error) {
      setPromptConfigNotice(
        error instanceof ApiError
          ? `保存失败：${error.message}`
          : "保存失败：后端暂未连接",
      );
    } finally {
      setSavingPromptConfig(false);
    }
  }

  async function handleDeleteSpeakerPrompt() {
    if (!pendingSpeakerPromptDelete || deletingSpeakerPromptVersion) return;
    const version = pendingSpeakerPromptDelete.version;
    setDeletingSpeakerPromptVersion(version);
    setPromptConfigNotice(`正在删除对话人提示词：${version}`);
    try {
      const config = await savePromptConfig(username, {
        delete_speaker_version: version,
      });
      setSpeakerPromptVersions(config.speaker_versions);
      setActiveSpeakerPromptVersion(config.active_speaker_version);
      setSpeakerIdentity(config.speaker_identity);
      setSpeakerPrompt(config.speaker_prompt);
      setSavedSpeakerPromptIdentity(config.speaker_identity);
      setSavedSpeakerPrompt(config.speaker_prompt);
      setCreatingSpeakerPrompt(false);
      setPromptConfigNotice(`已删除对话人提示词：${version}`);
    } catch (error) {
      setPromptConfigNotice(
        error instanceof ApiError
          ? `删除失败：${error.message}`
          : "删除失败：后端暂未连接",
      );
    } finally {
      setDeletingSpeakerPromptVersion("");
      setPendingSpeakerPromptDelete(null);
    }
  }

  function handleStartSpeakerPromptRename(
    promptVersion: SpeakerPromptVersion,
  ) {
    speakerPromptRenameCancelledRef.current = false;
    setEditingSpeakerPromptVersion(promptVersion.version);
    setSpeakerPromptTitleInput(promptVersion.title);
  }

  function handleCancelSpeakerPromptRename() {
    speakerPromptRenameCancelledRef.current = true;
    setEditingSpeakerPromptVersion("");
    setSpeakerPromptTitleInput("");
  }

  async function commitSpeakerPromptRename() {
    const version = editingSpeakerPromptVersion;
    const title = speakerPromptTitleInput.trim();
    if (!version || !title || speakerPromptRenameInFlightRef.current) return;

    speakerPromptRenameInFlightRef.current = true;
    setRenamingSpeakerPromptVersion(version);
    try {
      const config = await savePromptConfig(username, {
        rename_speaker_version: version,
        speaker_version_title: title,
      });
      setSpeakerPromptVersions(config.speaker_versions);
      setPromptConfigNotice(`对话人提示词已重命名为：${title}`);
      handleCancelSpeakerPromptRename();
    } catch (error) {
      setPromptConfigNotice(
        error instanceof ApiError
          ? `重命名失败：${error.message}`
          : "重命名失败：后端暂未连接",
      );
    } finally {
      speakerPromptRenameInFlightRef.current = false;
      setRenamingSpeakerPromptVersion("");
    }
  }

  function handleRenameSpeakerPrompt(event: FormEvent) {
    event.preventDefault();
    void commitSpeakerPromptRename();
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <RobotMark />
          <div>
            <h1>Live Streaming Agent 控制台</h1>
            <p>独立测试控制台</p>
          </div>
          <span
            className={`status-chip ${
              serviceState === "database-error" ||
              serviceState === "backend-error"
                ? "error"
                : serviceState === "checking"
                  ? "checking"
                  : ""
            }`}
            title={
              serviceState === "database-error"
                ? "无法连接 Elasticsearch"
                : serviceState === "backend-error"
                  ? "无法连接后端服务"
                  : undefined
            }
          >
            <i />
            {serviceState === "running"
              ? "运行中"
              : serviceState === "database-error"
                ? "数据库连接异常"
                : serviceState === "backend-error"
                  ? "后端连接异常"
                  : "检查中"}
          </span>
        </div>

        <div className="user-switcher">
          <span>用户名</span>
          <strong>{username || "未载入"}</strong>
          <button
            className="button secondary compact"
            type="button"
            onClick={handleOpenUsernameDialog}
          >
            切换用户名
          </button>
        </div>

      </header>

      <section
        ref={dashboardGridRef}
        className={`dashboard-grid${
          dashboardHeight !== null ? " vertically-resized" : ""
        }`}
        style={
          dashboardHeight !== null ? { height: dashboardHeight } : undefined
        }
      >
        <article className="panel model-panel">
          <h2><span>◇</span> 模型配置</h2>
          <div className="provider-summary">
            <strong>{currentProvider.name}</strong>
            <span>{currentProvider.models.length} 个内置模型，模型 ID 由选项确定</span>
          </div>
          <div className="form-grid model-form-grid">
            <label>
              <span>Provider</span>
              <select
                value={provider}
                onChange={(event) => handleProviderChange(event.target.value)}
              >
                {modelProviders.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Model</span>
              <select
                value={model}
                onChange={(event) => {
                  const nextModel = event.target.value;
                  selectedModelRef.current = {
                    provider,
                    model: nextModel,
                  };
                  setModel(nextModel);
                  setUserProviderModels((current) => ({
                    ...current,
                    [provider]: nextModel,
                  }));
                  setConnectionState("idle");
                }}
              >
                {currentProvider.models.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="wide-field">
              <span>模型 ID</span>
              <input
                value={model}
                readOnly
                aria-readonly="true"
              />
            </label>
            <label>
              <span>温度（0-2）</span>
              <input
                type="number"
                min={0}
                max={2}
                step={0.1}
                value={temperature}
                onChange={(event) => {
                  const nextTemperature = Math.max(
                    0,
                    Math.min(2, Number(event.target.value) || 0),
                  );
                  setTemperature(nextTemperature);
                  setUserProviderTemperatures((current) => ({
                    ...current,
                    [provider]: nextTemperature,
                  }));
                }}
              />
            </label>
          </div>
          {provider === "doubao" || provider === "qwen" ? (
            <div className="doubao-web-search-settings">
              <label className="doubao-web-search-option">
                <div>
                  <strong>联网搜索</strong>
                </div>
                <input
                  type="checkbox"
                  checked={
                    providerSupportsWebSearch(provider) &&
                    webSearchEnabled
                  }
                  onChange={(event) => {
                    const enabled = event.target.checked;
                    setWebSearchEnabled(enabled);
                    setUserProviderWebSearchConfigs((current) => ({
                      ...current,
                      [provider]: {
                        ...providerWebSearchConfig(current, provider),
                        enabled,
                      },
                    }));
                  }}
                />
                <span className="switch" />
              </label>
              {webSearchEnabled ? (
                <label className="doubao-web-search-option forced-search-option">
                  <div>
                    <strong>强制搜索</strong>
                  </div>
                  <input
                    type="checkbox"
                    checked={webSearchForced}
                    onChange={(event) => {
                      const forced = event.target.checked;
                      setWebSearchForced(forced);
                      setUserProviderWebSearchConfigs((current) => ({
                        ...current,
                        [provider]: {
                          ...providerWebSearchConfig(current, provider),
                          forced,
                        },
                      }));
                    }}
                  />
                  <span className="switch" />
                </label>
              ) : null}
              {provider === "doubao" && webSearchEnabled ? (
                <div className="doubao-web-search-parameters">
                  <label>
                    <span>最大搜索次数</span>
                    <input
                      type="number"
                      min={1}
                      max={10}
                      value={webSearchMaxToolCalls}
                      onChange={(event) => {
                        const maxToolCalls = Math.max(
                          1,
                          Math.min(10, Number(event.target.value) || 1),
                        );
                        setWebSearchMaxToolCalls(maxToolCalls);
                        setUserProviderWebSearchConfigs((current) => ({
                          ...current,
                          [provider]: {
                            ...providerWebSearchConfig(current, provider),
                            max_tool_calls: maxToolCalls,
                          },
                        }));
                      }}
                    />
                  </label>
                  <label>
                    <span>搜索网页数</span>
                    <input
                      type="number"
                      min={1}
                      max={20}
                      value={webSearchResultLimit}
                      onChange={(event) => {
                        const resultLimit = Math.max(
                          1,
                          Math.min(20, Number(event.target.value) || 1),
                        );
                        setWebSearchResultLimit(resultLimit);
                        setUserProviderWebSearchConfigs((current) => ({
                          ...current,
                          [provider]: {
                            ...providerWebSearchConfig(current, provider),
                            result_limit: resultLimit,
                          },
                        }));
                      }}
                    />
                  </label>
                </div>
              ) : null}
            </div>
          ) : null}
          <p
            className={`model-config-notice${
              connectionState === "error" ? " error" : ""
            }`}
            role={connectionState === "error" ? "alert" : undefined}
            aria-live="polite"
          >
            {modelConfigNotice}
          </p>
          <div className="panel-actions">
            <button
              className="button primary"
              type="button"
              onClick={handleTestModelConfig}
              disabled={testingModelConfig || !model.trim()}
            >
              {testingModelConfig ? "测试中" : "测试连接"}
            </button>
            <span
              className={
                connectionState === "success"
                  ? "success-text"
                  : connectionState === "error"
                    ? "error-text"
                    : "muted-text"
              }
            >
              {connectionState === "success"
                ? "● 连接成功"
                : connectionState === "error"
                  ? "● 需要检查"
                  : connectionState === "testing"
                    ? "○ 正在测试"
                    : "○ 等待测试"}
            </span>
          </div>
        </article>

        <article className="panel prompt-panel">
          <h2><span>♙</span> 人设提示词</h2>
          <div className="tab-row" role="tablist" aria-label="提示词版本">
            {promptVersions.length ? (
              promptVersions.map((promptVersion) => (
                <button
                  key={promptVersion.version}
                  role="tab"
                  aria-selected={activePromptVersion === promptVersion.version}
                  className={
                    activePromptVersion === promptVersion.version ? "active" : ""
                  }
                  onClick={() => handlePromptVersion(promptVersion.version)}
                  type="button"
                  title={promptVersion.title}
                >
                  {promptVersion.title}
                </button>
              ))
            ) : (
              <button type="button" disabled>
                读取中
              </button>
            )}
          </div>
          <textarea
            className="prompt-editor prompt-editor-readonly"
            value={personaPrompt}
            readOnly
            aria-label="人设提示词内容"
          />
          <div className="prompt-meta prompt-meta-end">
            <span>{personaPrompt.length} 字</span>
          </div>

          <h3 className="speaker-prompt-title">对话人提示词</h3>
          <div
            className="tab-row speaker-tab-row"
            role="tablist"
            aria-label="对话人提示词版本"
          >
            <button
              role="tab"
              aria-selected={
                activeSpeakerPromptVersion === "__none__" &&
                !creatingSpeakerPrompt
              }
              className={
                activeSpeakerPromptVersion === "__none__" &&
                !creatingSpeakerPrompt
                  ? "active"
                  : ""
              }
              onClick={() =>
                requestSpeakerPromptTransition({
                  kind: "select",
                  version: "__none__",
                })
              }
              type="button"
            >
              无
            </button>
            {speakerPromptVersions.map((promptVersion) => (
              <span
                className="speaker-version-tab"
                key={promptVersion.version}
              >
                {editingSpeakerPromptVersion === promptVersion.version ? (
                  <form
                    className="speaker-version-rename"
                    onSubmit={handleRenameSpeakerPrompt}
                  >
                    <input
                      value={speakerPromptTitleInput}
                      onChange={(event) =>
                        setSpeakerPromptTitleInput(event.target.value)
                      }
                      onBlur={() => {
                        if (speakerPromptRenameCancelledRef.current) {
                          speakerPromptRenameCancelledRef.current = false;
                        } else if (speakerPromptTitleInput.trim()) {
                          void commitSpeakerPromptRename();
                        } else {
                          handleCancelSpeakerPromptRename();
                        }
                      }}
                      onKeyDown={(event) => {
                        if (event.key === "Escape") {
                          event.preventDefault();
                          handleCancelSpeakerPromptRename();
                        }
                      }}
                      aria-label={`重命名对话人提示词：${promptVersion.title}`}
                      maxLength={40}
                      disabled={
                        renamingSpeakerPromptVersion === promptVersion.version
                      }
                      autoFocus
                    />
                  </form>
                ) : (
                  <button
                    role="tab"
                    aria-selected={
                      activeSpeakerPromptVersion === promptVersion.version
                    }
                    className={
                      activeSpeakerPromptVersion === promptVersion.version
                        ? "active"
                        : ""
                    }
                    onClick={() =>
                      requestSpeakerPromptTransition({
                        kind: "select",
                        version: promptVersion.version,
                      })
                    }
                    type="button"
                    title="点击标签空白处切换版本"
                  >
                    <span
                      className={`speaker-version-title${
                        activeSpeakerPromptVersion === promptVersion.version
                          ? " editable"
                          : ""
                      }`}
                      title={
                        activeSpeakerPromptVersion === promptVersion.version
                          ? "点击重命名"
                          : "点击切换"
                      }
                      role="button"
                      tabIndex={0}
                      onClick={(event) => {
                        if (
                          activeSpeakerPromptVersion === promptVersion.version
                        ) {
                          event.stopPropagation();
                          handleStartSpeakerPromptRename(promptVersion);
                        }
                      }}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          if (
                            activeSpeakerPromptVersion ===
                            promptVersion.version
                          ) {
                            event.stopPropagation();
                            handleStartSpeakerPromptRename(promptVersion);
                          } else {
                            requestSpeakerPromptTransition({
                              kind: "select",
                              version: promptVersion.version,
                            });
                          }
                        }
                      }}
                    >
                      {promptVersion.title}
                    </span>
                  </button>
                )}
                <button
                  className="speaker-version-delete"
                  type="button"
                  aria-label={`删除对话人提示词：${promptVersion.title}`}
                  title="删除对话人提示词"
                  disabled={
                    deletingSpeakerPromptVersion === promptVersion.version
                  }
                  onClick={() =>
                    setPendingSpeakerPromptDelete(promptVersion)
                  }
                >
                  ×
                </button>
              </span>
            ))}
            <button
              className={
                creatingSpeakerPrompt
                  ? "active add-prompt-tab"
                  : "add-prompt-tab"
              }
              type="button"
              aria-label="新建对话人提示词"
              aria-pressed={creatingSpeakerPrompt}
              onClick={() =>
                requestSpeakerPromptTransition({ kind: "create" })
              }
              title="新建对话人提示词"
            >
              +
            </button>
          </div>
          {creatingSpeakerPrompt ||
          activeSpeakerPromptVersion !== "__none__" ? (
            <>
              <label className="prompt-identity-field">
                <span>对话人身份</span>
                <input
                  value={speakerIdentity}
                  onChange={(event) => setSpeakerIdentity(event.target.value)}
                  placeholder="例如：莱叔"
                  maxLength={40}
                  autoFocus={creatingSpeakerPrompt}
                />
              </label>
              <label className="speaker-description-field">
                <span>描述</span>
                <textarea
                  className="speaker-prompt-editor"
                  value={speakerPrompt}
                  onChange={(event) => setSpeakerPrompt(event.target.value)}
                  aria-label="对话人提示词描述"
                  maxLength={10000}
                  placeholder="输入对话人的性格、关系和互动偏好；可留空。"
                />
              </label>
            </>
          ) : (
            <div className="speaker-prompt-empty">
              当前不使用对话人身份和对话人提示词
            </div>
          )}
          <div className="prompt-meta">
            <span>{promptConfigNotice}</span>
            {creatingSpeakerPrompt ||
            activeSpeakerPromptVersion !== "__none__" ? (
              <span>{speakerPrompt.length} / 10000</span>
            ) : null}
          </div>
          <div className="panel-actions prompt-actions">
            {creatingSpeakerPrompt ||
            activeSpeakerPromptVersion !== "__none__" ? (
              <button
                className="button primary"
                type="button"
                onClick={handleSavePromptConfig}
                disabled={savingPromptConfig}
              >
                {savingPromptConfig
                  ? "保存中"
                  : creatingSpeakerPrompt
                    ? "保存对话人提示词"
                    : "保存修改"}
              </button>
            ) : null}
          </div>
        </article>

        <article className="panel metrics-panel">
          <h2><span>▥</span> 性能监控</h2>
          <div className="performance-tabs" role="tablist">
            <button
              className={performanceView === "overview" ? "active" : ""}
              type="button"
              role="tab"
              aria-selected={performanceView === "overview"}
              onClick={() => setPerformanceView("overview")}
            >
              性能概览
            </button>
            <button
              className={
                performanceView === "model-comparison" ? "active" : ""
              }
              type="button"
              role="tab"
              aria-selected={performanceView === "model-comparison"}
              onClick={() => setPerformanceView("model-comparison")}
            >
              模型对比
            </button>
          </div>
          {performanceView === "overview" ? (
            <>
              <div className="metric-cards">
                <div>
                  <span>知识库用时</span>
                  <strong>
                    {performanceDisplay.knowledge.value}
                    <small>{performanceDisplay.knowledge.unit}</small>
                  </strong>
                </div>
                <div>
                  <span>网络搜索用时</span>
                  <strong>
                    {performanceDisplay.webSearch.value}
                    <small>{performanceDisplay.webSearch.unit}</small>
                  </strong>
                </div>
                <div>
                  <span>模型首字延迟</span>
                  <strong>
                    {performanceDisplay.firstToken.value}
                    <small>{performanceDisplay.firstToken.unit}</small>
                  </strong>
                </div>
                <div>
                  <span>首句延迟</span>
                  <strong>
                    {performanceDisplay.firstSentence.value}
                    <small>{performanceDisplay.firstSentence.unit}</small>
                  </strong>
                </div>
              </div>
              <p className="metrics-note">
                {isReplying
                  ? "正在更新本次回复耗时…"
                  : "显示最近一次模型回复耗时"}
              </p>
              <PerformanceLineChart samples={performanceHistory} />
            </>
          ) : (
            <div className="model-comparison-page">
              <p className="model-comparison-summary">
                {dailyPerformanceDay} · 已使用 {dailyModelCount} 个模型 ·{" "}
                {dailyModelPerformance.length} 次调用
              </p>
              <ModelComparisonPercentiles samples={dailyModelPerformance} />
              <ModelComparisonDotChart
                samples={dailyModelPerformance}
                metric="model_first_token_ms"
                title="首字延迟对比"
              />
              <ModelComparisonDotChart
                samples={dailyModelPerformance}
                metric="model_first_sentence_ms"
                title="首句延迟对比"
              />
            </div>
          )}
        </article>
      </section>

      <div
        className={`workspace-resize-divider${
          resizingWorkspace ? " resizing" : ""
        }`}
        role="separator"
        aria-label="调整用户会话区域高度"
        aria-orientation="horizontal"
        aria-valuemin={400}
        aria-valuenow={Math.round(workspaceHeight)}
        tabIndex={0}
        title="向上拖动扩大用户会话区域"
        onPointerDown={handleWorkspaceDividerStart}
        onKeyDown={handleWorkspaceDividerKeyDown}
      >
        <span />
      </div>

      <section
        className={`workspace${
          livePanelCollapsed ? " live-room-collapsed" : ""
        }`}
        style={{ gridTemplateRows: `${workspaceHeight}px` }}
      >
        <aside className="conversation-sidebar">
          <div className="sidebar-heading">
            <div>
              <span className="eyebrow">当前用户</span>
              <strong>{username}</strong>
            </div>
            <button
              className="icon-button"
              onClick={handleNewConversation}
              aria-label="新建对话"
              title="新建对话"
              disabled={creatingConversation}
            >
              {creatingConversation ? "…" : "＋"}
            </button>
          </div>
          <p className="workspace-notice">{workspaceNotice}</p>
          <label className="conversation-search-box">
            <span aria-hidden="true">⌕</span>
            <input
              type="search"
              value={conversationSearchQuery}
              onChange={(event) => {
                const nextQuery = event.target.value;
                setConversationSearchQuery(nextQuery);
                setConversationSearchTarget(null);
                setSearchingConversations(Boolean(nextQuery.trim()));
                setConversationSearchError("");
                if (!nextQuery.trim()) {
                  setConversationSearchResults([]);
                }
              }}
              placeholder="搜索全部对话内容"
              aria-label="搜索全部对话内容"
              maxLength={200}
            />
          </label>
          <div
            className={`conversation-list${
              conversationSearchQuery.trim() ? " search-results" : ""
            }`}
          >
            {conversationSearchQuery.trim() ? (
              searchingConversations ? (
                <div className="conversation-search-state">正在搜索…</div>
              ) : conversationSearchError ? (
                <div className="conversation-search-state error">
                  {conversationSearchError}
                </div>
              ) : conversationSearchResults.length === 0 ? (
                <div className="conversation-search-state">
                  没有对话包含这句话
                </div>
              ) : (
                conversationSearchResults.flatMap((result) => {
                  const currentTitle =
                    conversations.find(
                      (conversation) =>
                        conversation.conversation_id ===
                        result.conversation_id,
                    )?.title ?? result.title;
                  return result.matches.map((match, index) => (
                      <button
                        className={`conversation-search-result${
                          match.message_id ===
                          conversationSearchTarget?.messageId
                            ? " active"
                            : ""
                        }`}
                        type="button"
                        key={`${match.message_id}-${match.source}-${index}`}
                        onClick={() =>
                          handleOpenConversationSearchMatch(
                            result.conversation_id,
                            match.message_id,
                          )
                        }
                      >
                        <span>
                          <strong>{currentTitle}</strong>
                          <small>
                            第 {index + 1} / {result.match_count} 处
                          </small>
                        </span>
                        <p>
                          {conversationSearchSourceLabel(match.source)} · {match.role === "assistant" ? "Live Streaming Agent" : "用户"}
                          · {shortDate(match.created_at)}：
                          {highlightedSearchText(
                            match.snippet,
                            conversationSearchQuery,
                          )}
                        </p>
                      </button>
                    ));
                })
              )
            ) : conversations.length === 0 ? (
              <div className="empty-conversations">
                <span>＋</span>
                <p>还没有历史对话</p>
                <button
                  onClick={handleNewConversation}
                  disabled={creatingConversation}
                >
                  {creatingConversation ? "正在创建…" : "新建第一个对话"}
                </button>
              </div>
            ) : (
              conversations.map((conversation) => (
                <div
                  key={conversation.conversation_id}
                  className={
                    `${
                      conversation.conversation_id === activeConversationId
                        ? "conversation-item active"
                        : "conversation-item"
                    }${
                      draggingConversationId === conversation.conversation_id
                        ? " dragging"
                        : ""
                    }`
                  }
                  onDragOver={(event) =>
                    handleConversationDragOver(
                      event,
                      conversation.conversation_id,
                    )
                  }
                  onDrop={handleConversationDrop}
                >
                  <div
                    className="conversation-main"
                    role="button"
                    tabIndex={0}
                    aria-label={`打开对话：${conversation.title}`}
                    onClick={() =>
                      handleSwitchConversation(conversation.conversation_id)
                    }
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        handleSwitchConversation(
                          conversation.conversation_id,
                        );
                      }
                    }}
                  >
                    <span
                      className="conversation-open"
                      draggable={
                        editingConversationId !== conversation.conversation_id
                      }
                      aria-hidden="true"
                      title="上下拖动调整顺序"
                      onClick={(event) => event.stopPropagation()}
                      onDragStart={(event) =>
                        handleConversationDragStart(
                          event,
                          conversation.conversation_id,
                        )
                      }
                      onDragEnd={handleConversationDragEnd}
                    >
                      <span className="conversation-icon">⋮⋮</span>
                    </span>
                    <div className="conversation-details">
                      {editingConversationId ===
                      conversation.conversation_id ? (
                        <form
                          className="conversation-rename"
                          onSubmit={handleRenameConversation}
                          onClick={(event) => event.stopPropagation()}
                          onKeyDown={(event) => event.stopPropagation()}
                        >
                          <input
                            value={conversationTitleInput}
                            onChange={(event) =>
                              setConversationTitleInput(event.target.value)
                            }
                            onKeyDown={(event) => {
                              if (event.key === "Escape") handleCancelRename();
                            }}
                            aria-label="对话名称"
                            maxLength={120}
                            disabled={
                              renamingConversationId ===
                              conversation.conversation_id
                            }
                            autoFocus
                          />
                          <button
                            type="submit"
                            aria-label="保存对话名称"
                            title="保存"
                            disabled={
                              !conversationTitleInput.trim() ||
                              renamingConversationId ===
                                conversation.conversation_id
                            }
                          >
                            ✓
                          </button>
                        </form>
                      ) : (
                        <button
                          className="conversation-title"
                          type="button"
                          title="点击重命名"
                          onClick={(event) => {
                            event.stopPropagation();
                            handleStartRename(conversation);
                          }}
                          onKeyDown={(event) => event.stopPropagation()}
                        >
                          {conversation.title}
                        </button>
                      )}
                      <span className="conversation-date">
                        {conversation.completed_rounds} 轮 ·{" "}
                        {(conversation.effective_char_count ?? 0).toLocaleString(
                          "zh-CN",
                        )}{" "}
                        字 ·{" "}
                        {shortDate(conversation.updated_at)}
                      </span>
                    </div>
                  </div>
                  <button
                    className="conversation-delete"
                    type="button"
                    aria-label={`删除对话：${conversation.title}`}
                    title="删除对话"
                    disabled={
                      archivingConversationId === conversation.conversation_id
                    }
                    onClick={() => setPendingConversationDelete(conversation)}
                  >
                    ×
                  </button>
                </div>
              ))
            )}
          </div>
        </aside>

        <article
          className={`live-room-panel${
            livePanelCollapsed ? " collapsed" : ""
          }`}
        >
          {livePanelCollapsed ? (
            <button
              className="live-room-expand"
              type="button"
              onClick={() => setLivePanelCollapsed(false)}
              aria-label="展开直播间抓取框"
              title="展开直播间抓取框"
            >
              <span className="live-room-expand-arrow">›</span>
              <span className="live-room-expand-label">直播间</span>
              <i className={`live-room-collapsed-state ${liveCaptureState}`} />
            </button>
          ) : (
            <>
              <header className="live-room-header">
                <div>
                  <span className="eyebrow">抖音直播间</span>
                  <h2>直播间互动</h2>
                </div>
                <div className="live-room-header-actions">
                  <span
                    className={`live-state ${liveCaptureState}`}
                    aria-live="polite"
                  >
                    <i />
                    {liveCaptureState === "running"
                      ? "抓取中"
                      : liveCaptureState === "starting"
                        ? "连接中"
                        : liveCaptureState === "stopping"
                          ? "停止中"
                          : liveCaptureState === "error"
                            ? "异常"
                            : "未启动"}
                  </span>
                  <button
                    className="live-room-collapse"
                    type="button"
                    onClick={() => setLivePanelCollapsed(true)}
                    aria-label="收起直播间抓取框"
                    title="收起到左侧"
                  >
                    ‹
                  </button>
                </div>
              </header>
              <form
                className="live-room-controls"
                onSubmit={handleStartLiveCapture}
              >
                <button
                  className="button secondary live-login-button"
                  type="button"
                  onClick={handleDouyinLogin}
                  disabled={
                    !username ||
                    douyinLoginBusy ||
                    douyinLoginChecking ||
                    liveCaptureState === "starting" ||
                    liveCaptureState === "running" ||
                    liveCaptureState === "stopping"
                  }
                >
                  {douyinLoginChecking
                    ? "正在校验抖音账号…"
                    : douyinLoginBusy
                      ? "处理中…"
                      : douyinLoginStatus.status === "ready"
                        ? "抖音账号已登录"
                        : "登录抖音账号"}
                </button>
                <label htmlFor="douyin-room-id">房间号</label>
                <input
                  id="douyin-room-id"
                  value={liveRoomId}
                  onChange={(event) => setLiveRoomId(event.target.value)}
                  placeholder="输入抖音直播间房间号"
                  inputMode="numeric"
                  disabled={
                    liveCaptureState === "starting" ||
                    liveCaptureState === "running" ||
                    liveCaptureState === "stopping"
                  }
                />
                <div>
                  <button
                    className="button primary"
                    type="submit"
                    disabled={
                      !username ||
                      !liveRoomId.trim() ||
                      liveCaptureState === "starting" ||
                      liveCaptureState === "running" ||
                      liveCaptureState === "stopping"
                    }
                  >
                    {liveCaptureState === "starting"
                      ? "连接中…"
                      : "开始抓取"}
                  </button>
                  <button
                    className="button secondary"
                    type="button"
                    onClick={handleStopLiveCapture}
                    disabled={
                      liveCaptureState !== "starting" &&
                      liveCaptureState !== "running"
                    }
                  >
                    停止抓取
                  </button>
                </div>
                <div className="live-reply-controls">
                  <button
                    className={`live-reply-toggle${
                      replyLiveChats ? " active" : ""
                    }`}
                    type="button"
                    aria-pressed={replyLiveChats}
                    onClick={() => {
                      const nextEnabled = !replyLiveChats;
                      if (nextEnabled) {
                        liveReplyChatStartSequenceRef.current =
                          liveSequenceRef.current;
                      }
                      setReplyLiveChats(nextEnabled);
                    }}
                  >
                    回复弹幕
                  </button>
                  <button
                    className={`live-reply-toggle${
                      replyLiveGifts ? " active" : ""
                    }`}
                    type="button"
                    aria-pressed={replyLiveGifts}
                    onClick={() => {
                      const nextEnabled = !replyLiveGifts;
                      if (nextEnabled) {
                        liveReplyGiftStartSequenceRef.current =
                          liveSequenceRef.current;
                        lastRepliedGiftRef.current = null;
                        if (liveGiftReplyWakeTimerRef.current !== null) {
                          window.clearTimeout(
                            liveGiftReplyWakeTimerRef.current,
                          );
                          liveGiftReplyWakeTimerRef.current = null;
                        }
                      }
                      setReplyLiveGifts(nextEnabled);
                    }}
                  >
                    回复礼物
                  </button>
                </div>
              </form>
              <p
                className={
                  liveCaptureState === "error"
                    ? "live-room-notice error"
                    : "live-room-notice"
                }
              >
                {liveCaptureMessage}
              </p>
              <div
                className="live-event-list"
                ref={liveEventListRef}
                aria-live="polite"
                aria-label="直播间弹幕和礼物"
              >
                {liveEvents.length === 0 ? (
                  <div className="live-event-empty">
                    <span>◌</span>
                    <p>开始抓取后，弹幕和礼物会实时显示在这里</p>
                  </div>
                ) : (
                  liveEvents.map((event) => {
                    if (event.type === "status") {
                      return (
                        <div
                          className={`live-event status ${event.status}`}
                          key={event.sequence}
                        >
                          <span>系统</span>
                          <p>{event.message}</p>
                        </div>
                      );
                    }
                    const isGift = event.type === "gift";
                    return (
                      <div
                        className={
                          isGift ? "live-event gift" : "live-event chat"
                        }
                        key={event.sequence}
                      >
                        <span className="live-event-avatar">
                          {isGift ? "礼" : "弹"}
                        </span>
                        <div>
                          <header>
                            <strong>{event.nickname}</strong>
                            <time>
                              {new Date(event.timestamp).toLocaleTimeString(
                                "zh-CN",
                                {
                                  hour: "2-digit",
                                  minute: "2-digit",
                                  second: "2-digit",
                                },
                              )}
                            </time>
                          </header>
                          <p>
                            {isGift
                              ? `送出 ${event.gift_name} ×${event.gift_count}`
                              : event.content}
                          </p>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </>
          )}
        </article>

        <article className="chat-panel">
          <header className="chat-header">
            <div>
              <div className="chat-title-line">
                <h2>
                  <span>◌</span>{" "}
                  {activeConversation?.title ?? "尚未选择对话"}
                </h2>
                <label className="knowledge-control chat-knowledge-control">
                  <span>知识库：{knowledgeEnabled ? "开启" : "关闭"}</span>
                  <input
                    type="checkbox"
                    checked={knowledgeEnabled}
                    onChange={(event) =>
                      setKnowledgeEnabled(event.target.checked)
                    }
                  />
                  <span className="switch" />
                </label>
                <button
                  className={`memory-control${
                    chatContentView === "memory" ? " active" : ""
                  }`}
                  type="button"
                  onClick={handleToggleMemoryView}
                  disabled={!activeConversationId}
                >
                  {chatContentView === "memory" ? "关闭短期记忆" : "短期记忆"}
                  <span>{activeConversation?.memory_compression_count ?? 0}</span>
                </button>
              </div>
              <p className={activeConversation?.memory_status === "compressing"
                ? "memory-status compressing"
                : "memory-status"
              }>
                {activeConversation?.memory_status === "compressing"
                  ? `正在压缩至第 ${activeConversation.memory_target_round} 轮…`
                  : activeConversation?.memory_status === "failed"
                    ? "短期记忆压缩失败，继续使用上一次结果"
                    : activeConversation?.memory_through_round
                      ? `短期记忆已压缩至第 ${activeConversation.memory_through_round} 轮`
                      : "已自动保存"}
              </p>
            </div>
            <div className="chat-actions">
              <button
                className="button secondary"
                type="button"
                onClick={handleRewindLastTurn}
                disabled={
                  !activeConversationId ||
                  isReplying ||
                  isRewinding
                }
              >
                {isRewinding ? "回溯中…" : "↶ 回溯上一轮"}
              </button>
              <button
                className="button secondary"
                type="button"
                onClick={handleExportConversation}
                disabled={
                  !activeConversationId ||
                  isReplying ||
                  exportingConversation
                }
              >
                {exportingConversation ? "导出中…" : "⇩ 导出"}
              </button>
            </div>
          </header>

          <div
            ref={chatContentRef}
            className={`chat-content${
              chatContentView === "memory" ? " split" : ""
            }${resizingMemoryPane ? " resizing" : ""}`}
            style={
              chatContentView === "memory"
                ? {
                    gridTemplateColumns: `${conversationPanePercent}fr 10px ${
                      100 - conversationPanePercent
                    }fr`,
                  }
                : undefined
            }
          >
            <div
              ref={messageListRef}
              className="message-list"
              aria-live="polite"
            >
            {messages.length === 0 ? (
              <div className="empty-chat">
                <RobotMark />
                <h3>开始和 Live Streaming Agent 对话吧</h3>
                <p>发送的每一条消息都会归入当前用户名和对话。</p>
              </div>
            ) : (
              messages.map((message) => {
                const display = messageDisplay(message);
                const isStreaming = message.metadata?.streaming === true;
                const injectedKnowledge =
                  typeof message.metadata?.knowledge_injected_context ===
                  "string"
                    ? message.metadata.knowledge_injected_context
                    : "";
                const knowledgeHitCount =
                  typeof message.metadata?.knowledge_hit_count === "number"
                    ? message.metadata.knowledge_hit_count
                    : 0;
                const webSearchSources = messageWebSearchSources(message);
                const activeSearchPhrase = conversationSearchQuery.trim();
                const knowledgeContainsSearch = includesSearchPhrase(
                  injectedKnowledge,
                  activeSearchPhrase,
                );
                const webSearchContainsSearch = webSearchSources.some(
                  (source) =>
                    includesSearchPhrase(
                      `${source.title} ${source.snippet} ${source.url}`,
                      activeSearchPhrase,
                    ),
                );
                const isSearchTarget =
                  conversationSearchTarget?.messageId === message.message_id;
                return (
                  <Fragment key={message.message_id}>
                    {message.role === "assistant" && injectedKnowledge ? (
                      <details
                        className="knowledge-injection"
                        open={knowledgeContainsSearch || undefined}
                      >
                        <summary>
                          <span>知识库注入</span>
                          <small>{knowledgeHitCount} 条召回内容</small>
                        </summary>
                        <pre>
                          {activeSearchPhrase
                            ? highlightedSearchText(
                                injectedKnowledge,
                                activeSearchPhrase,
                              )
                            : injectedKnowledge}
                        </pre>
                      </details>
                    ) : null}
                    {message.role === "assistant" &&
                    webSearchSources.length ? (
                      <details
                        className="web-search-injection"
                        open={webSearchContainsSearch || undefined}
                      >
                        <summary>
                          <span>联网搜索</span>
                          <small>{webSearchSources.length} 个网页</small>
                        </summary>
                        <div className="web-search-source-list">
                          {webSearchSources.map((source) => (
                            <article key={source.url}>
                              <a
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                              >
                                {activeSearchPhrase
                                  ? highlightedSearchText(
                                      source.title,
                                      activeSearchPhrase,
                                    )
                                  : source.title}
                              </a>
                              {source.snippet ? (
                                <p>
                                  {activeSearchPhrase
                                    ? highlightedSearchText(
                                        source.snippet,
                                        activeSearchPhrase,
                                      )
                                    : source.snippet}
                                </p>
                              ) : null}
                              <small>
                                {activeSearchPhrase
                                  ? highlightedSearchText(
                                      source.url,
                                      activeSearchPhrase,
                                    )
                                  : source.url}
                              </small>
                            </article>
                          ))}
                        </div>
                      </details>
                    ) : null}
                    <div
                      className={`message-row ${message.role}${
                        isStreaming ? " streaming" : ""
                      }${isSearchTarget ? " search-target" : ""}`}
                      data-message-id={message.message_id}
                      aria-current={isSearchTarget ? "true" : undefined}
                    >
                      <span className="avatar">
                        {message.role === "assistant" ? (
                          <RobotMark small />
                        ) : (
                          "●"
                        )}
                      </span>
                      <strong>
                        {display.speaker}
                        {messageRoundNumbers.get(message.message_id) ? (
                          <small className="message-round">
                            第 {messageRoundNumbers.get(message.message_id)} 轮
                          </small>
                        ) : null}
                      </strong>
                      <p>
                        {activeSearchPhrase
                          ? highlightedSearchText(
                              display.content,
                              activeSearchPhrase,
                            )
                          : display.content}
                      </p>
                      <time>{timeOf(message.created_at)}</time>
                    </div>
                  </Fragment>
                );
              })
            )}
            </div>
            {chatContentView === "memory" ? (
              <>
                <div
                  className="memory-divider"
                  role="separator"
                  aria-label="调整对话和短期记忆宽度"
                  aria-orientation="vertical"
                  aria-valuemin={28}
                  aria-valuemax={72}
                  aria-valuenow={Math.round(conversationPanePercent)}
                  tabIndex={0}
                  title="拖动调整对话和短期记忆宽度"
                  onPointerDown={handleMemoryDividerStart}
                  onKeyDown={handleMemoryDividerKeyDown}
                >
                  <span />
                </div>
                <section
                  className="memory-view"
                  aria-label="短期记忆压缩记录"
                >
                  <header className="memory-view-header">
                    <div>
                      <span>当前对话</span>
                      <h3>短期记忆压缩记录</h3>
                    </div>
                    <small>
                      共 {activeConversation?.memory_compression_count ?? 0} 次
                    </small>
                  </header>
                  <p className="memory-view-description">
                    每完成 10 轮就提前压缩一次；压缩结果延后 10
                    轮才注入上下文，并保留最近 10 轮原始对话。
                  </p>
                  <div className="memory-history">
                    {loadingMemories ? (
                      <div className="memory-empty">正在载入短期记忆…</div>
                    ) : shortTermMemories.length === 0 ? (
                      <div className="memory-empty">
                        尚未产生压缩结果，完成第 10
                        轮后将进行第一次压缩。
                      </div>
                    ) : (
                      shortTermMemories.map((memory) => (
                        <article
                          key={memory.memory_id}
                          className="memory-record"
                        >
                          <header>
                            <strong>
                              第 {memory.compression_number} 次压缩
                            </strong>
                            <span>
                              已压缩至第 {memory.through_round} 轮
                            </span>
                          </header>
                          <p>{memory.summary}</p>
                          <time>
                            {new Date(memory.created_at).toLocaleString("zh-CN")}
                          </time>
                        </article>
                      ))
                    )}
                  </div>
                </section>
              </>
            ) : null}
          </div>

          <form
            className={`composer${messageError ? " has-error" : ""}`}
            onSubmit={handleSendMessage}
          >
            {messageError ? (
              <p className="composer-error" role="alert">
                {messageError}
              </p>
            ) : null}
            <input
              ref={messageInputRef}
              value={messageDraft}
              onChange={(event) => setMessageDraft(event.target.value)}
              placeholder="输入消息…"
              aria-label="输入消息"
              disabled={!activeConversationId}
            />
            <button
              className="button primary"
              disabled={
                !messageDraft.trim() || !activeConversationId || isReplying
              }
            >
              {isReplying ? "回复中…" : "发送"}
            </button>
          </form>
        </article>
      </section>

      {douyinLoginDialogOpen ? (
        <div className="username-dialog-backdrop">
          <section
            className="douyin-login-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="douyin-login-dialog-title"
          >
            <span className="dialog-eyebrow">全站共用账号</span>
            <h2 id="douyin-login-dialog-title">登录抖音账号</h2>
            <p className="douyin-login-description">
              使用抖音 App 扫码并确认。登录成功后，所有网页用户都会共用此账号抓取直播间。
            </p>
            <div
              className={`douyin-login-preview ${douyinLoginStatus.status}`}
              aria-live="polite"
            >
              {douyinLoginStatus.qr_image ? (
                // The image is a short-lived data URL returned by the local backend.
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={douyinLoginStatus.qr_image}
                  alt="抖音扫码登录页面"
                />
              ) : douyinLoginStatus.status === "ready" ? (
                <div className="douyin-login-result success" aria-hidden="true">
                  ✓
                </div>
              ) : douyinLoginStatus.status === "error" ? (
                <div className="douyin-login-result error" aria-hidden="true">
                  !
                </div>
              ) : (
                <div className="douyin-login-loading" aria-hidden="true">
                  <i />
                  <span>正在生成二维码</span>
                </div>
              )}
            </div>
            <p
              className={
                douyinLoginStatus.status === "error"
                  ? "douyin-login-message error"
                  : "douyin-login-message"
              }
            >
              {douyinLoginStatus.message}
            </p>
            <div className="douyin-login-actions">
              {douyinLoginStatus.status === "error" ? (
                <button
                  className="button secondary"
                  type="button"
                  onClick={handleDouyinLogin}
                  disabled={douyinLoginBusy}
                >
                  重新生成
                </button>
              ) : null}
              <button
                className="button primary"
                type="button"
                onClick={handleCloseDouyinLogin}
                disabled={douyinLoginBusy}
                autoFocus={douyinLoginStatus.status === "ready"}
              >
                {douyinLoginStatus.status === "ready" ? "完成" : "关闭"}
              </button>
            </div>
          </section>
        </div>
      ) : null}

      {usernameDialogOpen ? (
        <div className="username-dialog-backdrop">
          <section
            className="username-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="username-dialog-title"
          >
            <RobotMark />
            <span className="dialog-eyebrow">
              {username ? "切换工作区" : "欢迎使用"}
            </span>
            <h2 id="username-dialog-title">请输入用户名</h2>
            <p>填入用户名即可载入该用户之前保存的所有历史对话。</p>
            <form onSubmit={handleLoadWorkspace}>
              <label htmlFor="username-dialog-input">用户名</label>
              <input
                id="username-dialog-input"
                value={usernameInput}
                onChange={(event) => {
                  setUsernameInput(event.target.value);
                  setUsernameDialogError("");
                }}
                placeholder="请输入用户名"
                autoComplete="username"
                autoFocus
                disabled={loadingWorkspace}
              />
              {usernameDialogError ||
              serviceState === "database-error" ||
              serviceState === "backend-error" ? (
                <p className="username-dialog-error" role="alert">
                  {usernameDialogError ||
                    (serviceState === "database-error"
                      ? "数据库连接失败，请确认 Elasticsearch 已启动后重试。"
                      : "后端服务连接失败，请确认后端已启动后重试。")}
                </p>
              ) : null}
              <div className="username-dialog-actions">
                {username ? (
                  <button
                    className="button secondary"
                    type="button"
                    onClick={() => setUsernameDialogOpen(false)}
                    disabled={loadingWorkspace}
                  >
                    取消
                  </button>
                ) : null}
                <button
                  className="button primary"
                  type="submit"
                  disabled={loadingWorkspace || !usernameInput.trim()}
                >
                  {loadingWorkspace ? "载入中…" : "进入工作区"}
                </button>
              </div>
            </form>
          </section>
        </div>
      ) : null}

      {pendingConversationDelete ? (
        <div className="username-dialog-backdrop">
          <section
            className="delete-dialog"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="delete-dialog-title"
            aria-describedby="delete-dialog-description"
          >
            <span className="delete-dialog-icon" aria-hidden="true">×</span>
            <h2 id="delete-dialog-title">确认删除对话？</h2>
            <p id="delete-dialog-description">
              “{pendingConversationDelete.title}”
            </p>
            <div className="delete-dialog-actions">
              <button
                className="button secondary"
                type="button"
                onClick={() => setPendingConversationDelete(null)}
                disabled={Boolean(archivingConversationId)}
              >
                取消
              </button>
              <button
                className="button primary button-danger"
                type="button"
                onClick={handleArchiveConversation}
                disabled={Boolean(archivingConversationId)}
                autoFocus
              >
                {archivingConversationId ? "删除中…" : "确认删除"}
              </button>
            </div>
          </section>
        </div>
      ) : null}

      {pendingSpeakerPromptDelete ? (
        <div className="username-dialog-backdrop">
          <section
            className="delete-dialog"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="speaker-prompt-delete-dialog-title"
            aria-describedby="speaker-prompt-delete-dialog-description"
          >
            <span className="delete-dialog-icon" aria-hidden="true">×</span>
            <h2 id="speaker-prompt-delete-dialog-title">
              确认删除对话人提示词？
            </h2>
            <p id="speaker-prompt-delete-dialog-description">
              “{pendingSpeakerPromptDelete.title}”
            </p>
            <div className="delete-dialog-actions">
              <button
                className="button secondary"
                type="button"
                onClick={() => setPendingSpeakerPromptDelete(null)}
                disabled={Boolean(deletingSpeakerPromptVersion)}
              >
                取消
              </button>
              <button
                className="button primary button-danger"
                type="button"
                onClick={handleDeleteSpeakerPrompt}
                disabled={Boolean(deletingSpeakerPromptVersion)}
                autoFocus
              >
                {deletingSpeakerPromptVersion ? "删除中…" : "确认删除"}
              </button>
            </div>
          </section>
        </div>
      ) : null}

      {pendingSpeakerPromptTransition ? (
        <div className="username-dialog-backdrop">
          <section
            className="delete-dialog"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="unsaved-speaker-prompt-dialog-title"
            aria-describedby="unsaved-speaker-prompt-dialog-description"
          >
            <span className="delete-dialog-icon" aria-hidden="true">!</span>
            <h2 id="unsaved-speaker-prompt-dialog-title">
              对话人提示词尚未保存
            </h2>
            <p id="unsaved-speaker-prompt-dialog-description">
              是否放弃当前修改并继续切换？
            </p>
            <div className="delete-dialog-actions">
              <button
                className="button secondary"
                type="button"
                onClick={() => setPendingSpeakerPromptTransition(null)}
                autoFocus
              >
                继续编辑
              </button>
              <button
                className="button primary button-danger"
                type="button"
                onClick={handleConfirmSpeakerPromptTransition}
              >
                放弃修改并切换
              </button>
            </div>
          </section>
        </div>
      ) : null}
    </main>
  );
}
