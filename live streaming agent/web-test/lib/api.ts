export type ConversationSummary = {
  conversation_id: string;
  username: string;
  title: string;
  status: "active" | "archived";
  sort_order: number;
  completed_rounds: number;
  effective_char_count: number;
  memory_compression_count: number;
  memory_through_round: number;
  short_term_memory: string;
  memory_status: "idle" | "compressing" | "failed";
  memory_target_round: number;
  created_at: string;
  updated_at: string;
};

export type MessageRecord = {
  message_id: string;
  conversation_id: string;
  username: string;
  role: "user" | "assistant" | "system" | "live_viewer";
  content: string;
  reasoning_content?: string;
  created_at: string;
  emotion?: string | null;
  metadata?: Record<string, unknown>;
};

export type ConversationSearchResult = {
  conversation_id: string;
  title: string;
  match_count: number;
  matches: Array<{
    message_id: string;
    role: MessageRecord["role"];
    source: "message" | "knowledge" | "web_search";
    snippet: string;
    created_at: string;
  }>;
};

type WorkspaceResponse = {
  username: string;
  speaker_identity: string;
  llm_config: ChatModelConfig;
  provider_models: Record<string, string>;
  provider_temperatures: Record<string, number>;
  provider_web_search_configs: Record<string, UserWebSearchConfig>;
  conversations: ConversationSummary[];
  messages: MessageRecord[];
};

type MessagePageResponse = {
  items: MessageRecord[];
  next_before: string | null;
};

export type ChatModelConfig = {
  provider: string;
  model: string;
  web_search_enabled: boolean;
  web_search_forced: boolean;
  web_search_max_tool_calls: number;
  web_search_result_limit: number;
  temperature: number;
};

export type UserWebSearchConfig = {
  enabled: boolean;
  forced: boolean;
  max_tool_calls: number;
  result_limit: number;
};

export type SavedProviderConfig = {
  provider: string;
  model: string;
  has_api_key: boolean;
};

export type ModelConfigResponse = SavedProviderConfig & {
  providers: Record<string, SavedProviderConfig>;
};

export type ModelConfigPayload = {
  provider: string;
  model: string;
  web_search_enabled?: boolean;
  web_search_forced?: boolean;
  web_search_max_tool_calls?: number;
  web_search_result_limit?: number;
  temperature?: number;
};

export type ModelConnectionTestResponse = {
  ok: boolean;
  message: string;
  latency_ms?: number | null;
  provider: string;
  model: string;
};

export type PersonaPromptVersion = {
  version: string;
  title: string;
  content: string;
};

export type SpeakerPromptVersion = PersonaPromptVersion & {
  speaker_identity: string;
};

export type PromptConfigResponse = {
  active_version: string;
  persona_prompt: string;
  speaker_prompt: string;
  speaker_identity: string;
  versions: PersonaPromptVersion[];
  active_speaker_version: string;
  speaker_versions: SpeakerPromptVersion[];
};

export type PromptConfigUpdate = {
  active_version?: string;
  active_speaker_version?: string;
  speaker_prompt?: string;
  speaker_identity?: string;
  create_speaker_version?: boolean;
  update_speaker_version?: string;
  delete_speaker_version?: string;
  rename_speaker_version?: string;
  speaker_version_title?: string;
};

export type ChatResponse = {
  user_message: MessageRecord;
  assistant_message: MessageRecord;
  mode: "model";
  knowledge_hit_count: number;
  completed_rounds: number;
  effective_char_count: number;
  memory_status: "idle" | "compressing" | "failed";
  memory_through_round: number;
  memory_target_round: number;
};

export type ShortTermMemory = {
  memory_id: string;
  conversation_id: string;
  username: string;
  compression_number: number;
  through_round: number;
  summary: string;
  created_at: string;
};

export type ChatPerformanceMetrics = {
  knowledge_duration_ms: number;
  web_search_duration_ms: number;
  model_first_token_ms: number;
  model_first_sentence_ms: number;
};

export type WebSearchSource = {
  title: string;
  url: string;
  snippet: string;
};

type ChatStreamEvent =
  | {
      type: "start";
      trace_id: string;
      user_message: MessageRecord;
      knowledge_hit_count: number;
      knowledge_injected_context: string;
      round_number: number;
      knowledge_duration_ms: number;
    }
  | { type: "delta"; content: string }
  | {
      type: "web_search_sources";
      sources: WebSearchSource[];
    }
  | {
      type: "metric";
      metrics: Partial<ChatPerformanceMetrics>;
    }
  | ({
      type: "done";
      performance_metrics: ChatPerformanceMetrics;
    } & ChatResponse)
  | { type: "error"; status: number; message: string };

export type ChatStreamPhase =
  | "request_started"
  | "response_headers"
  | "stream_start"
  | "first_chunk";

export type FrontendLatencyReport = {
  trace_id: string;
  conversation_id: string;
  click_timestamp: string;
  click_to_request_start_ms: number;
  click_to_response_headers_ms: number;
  click_to_stream_start_ms: number;
  click_to_first_chunk_ms: number;
  click_to_first_paint_ms: number;
  response_headers_to_first_chunk_ms: number;
  first_chunk_to_first_paint_ms: number;
};

export type RewindResponse = {
  message_ids: string[];
  deleted_count: number;
  completed_rounds: number;
  effective_char_count: number;
  memory_compression_count: number;
  memory_through_round: number;
  short_term_memory: string;
  memory_status: "idle" | "compressing" | "failed";
  memory_target_round: number;
};

export type ServiceHealth = {
  status: "ok" | "degraded";
  elasticsearch: "connected" | "unavailable";
};

export type LiveCaptureStatus = {
  username: string;
  room_id: string;
  status: "starting" | "running" | "stopped" | "error";
  message: string;
};

export type LiveLoginStatus = {
  status: "idle" | "waiting_scan" | "ready" | "error";
  message: string;
  qr_image?: string | null;
};

export type LiveRoomEvent =
  | {
      sequence: number;
      type: "status";
      room_id: string;
      timestamp: string;
      status: "starting" | "running" | "stopped" | "error";
      message: string;
    }
  | {
      sequence: number;
      type: "chat";
      room_id: string;
      timestamp: string;
      nickname: string;
      content: string;
      msg_id?: number | null;
    }
  | {
      sequence: number;
      type: "gift";
      room_id: string;
      timestamp: string;
      nickname: string;
      gift_name: string;
      gift_count: number;
      diamond_count?: number | null;
      msg_id?: number | null;
    };

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const apiHostname =
  typeof window === "undefined" ? "localhost" : window.location.hostname;
const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? `http://${apiHostname}:8001/api`;
const SERVICE_BASE = API_BASE.replace(/\/api\/?$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    cache: "no-store",
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!response.ok) {
    let message = `API request failed with status ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // Keep the status-based fallback when the response is not JSON.
    }
    throw new ApiError(response.status, message);
  }

  return response.json() as Promise<T>;
}

export async function loadUserWorkspace(
  username: string,
): Promise<WorkspaceResponse> {
  return request<WorkspaceResponse>("/users/resolve", {
    method: "POST",
    body: JSON.stringify({ username }),
  });
}

export async function saveUserModelConfig(
  username: string,
  payload: ModelConfigPayload,
): Promise<ChatModelConfig> {
  return request<ChatModelConfig>("/users/model-config", {
    method: "PUT",
    body: JSON.stringify({ username, ...payload }),
  });
}

export async function loadConversationMessages(
  conversationId: string,
): Promise<MessageRecord[]> {
  const encodedConversationId = encodeURIComponent(conversationId);
  let before: string | null = null;
  let messages: MessageRecord[] = [];

  do {
    const query: string = before
      ? `?limit=200&before=${encodeURIComponent(before)}`
      : "?limit=200";
    const page: MessagePageResponse = await request<MessagePageResponse>(
      `/conversations/${encodedConversationId}/messages${query}`,
    );
    messages = [...page.items, ...messages];
    before = page.next_before;
  } while (before);

  return messages;
}

export async function searchUserConversations(
  username: string,
  phrase: string,
  signal?: AbortSignal,
): Promise<ConversationSearchResult[]> {
  const query = new URLSearchParams({ q: phrase });
  return request<ConversationSearchResult[]>(
    `/users/${encodeURIComponent(username)}/conversation-search?${query.toString()}`,
    { signal },
  );
}

export async function loadShortTermMemories(
  conversationId: string,
): Promise<ShortTermMemory[]> {
  return request<ShortTermMemory[]>(
    `/conversations/${encodeURIComponent(conversationId)}/memories`,
  );
}

export async function loadModelConfig(): Promise<ModelConfigResponse> {
  return request<ModelConfigResponse>("/model-config");
}

export async function saveModelConfig(
  payload: ModelConfigPayload,
): Promise<ModelConfigResponse> {
  return request<ModelConfigResponse>("/model-config", {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export async function testModelConnection(
  payload: ModelConfigPayload,
): Promise<ModelConnectionTestResponse> {
  return request<ModelConnectionTestResponse>("/model-config/test", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function checkServiceHealth(): Promise<ServiceHealth> {
  const response = await fetch(`${SERVICE_BASE}/health`, {
    cache: "no-store",
    signal: AbortSignal.timeout(5000),
  });
  if (!response.ok) {
    throw new ApiError(response.status, "Backend health check failed");
  }
  return response.json() as Promise<ServiceHealth>;
}

export async function startLiveCapture(
  username: string,
  roomId: string,
): Promise<LiveCaptureStatus> {
  return request<LiveCaptureStatus>("/live/start", {
    method: "POST",
    body: JSON.stringify({ username, room_id: roomId }),
  });
}

export async function startDouyinLogin(): Promise<LiveLoginStatus> {
  return request<LiveLoginStatus>("/live/login/start", {
    method: "POST",
  });
}

export async function getDouyinLoginStatus(): Promise<LiveLoginStatus> {
  return request<LiveLoginStatus>("/live/login/status");
}

export async function finishDouyinLogin(): Promise<LiveLoginStatus> {
  return request<LiveLoginStatus>("/live/login/finish", {
    method: "POST",
  });
}

export async function stopLiveCapture(
  username: string,
): Promise<LiveCaptureStatus> {
  return request<LiveCaptureStatus>("/live/stop", {
    method: "POST",
    body: JSON.stringify({ username }),
  });
}

export function releaseLiveCapture(username: string): boolean {
  const query = new URLSearchParams({ username });
  const url = `${API_BASE}/live/release?${query}`;
  if (typeof navigator !== "undefined" && navigator.sendBeacon) {
    if (navigator.sendBeacon(url)) return true;
  }
  void fetch(url, {
    method: "POST",
    keepalive: true,
    mode: "cors",
  }).catch(() => undefined);
  return true;
}

export async function streamLiveEvents(
  username: string,
  afterSequence: number,
  onEvent: (event: LiveRoomEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const query = new URLSearchParams({
    username,
    after: String(afterSequence),
  });
  const response = await fetch(`${API_BASE}/live/events?${query}`, {
    cache: "no-store",
    signal,
  });
  if (!response.ok) {
    let message = `直播间事件流连接失败（HTTP ${response.status}）`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // Keep the status-based fallback when the response is not JSON.
    }
    throw new ApiError(response.status, message);
  }
  if (!response.body) {
    throw new ApiError(502, "直播间事件流不可用");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line) as LiveRoomEvent | { type: "heartbeat" };
      if (event.type !== "heartbeat") onEvent(event);
    }
    if (done) break;
  }
  if (buffer.trim()) {
    const event = JSON.parse(buffer) as LiveRoomEvent | { type: "heartbeat" };
    if (event.type !== "heartbeat") onEvent(event);
  }
}

export async function loadPromptConfig(
  username: string,
): Promise<PromptConfigResponse> {
  return request<PromptConfigResponse>(
    `/prompt-config?username=${encodeURIComponent(username)}`,
  );
}

export async function savePromptConfig(
  username: string,
  payload: PromptConfigUpdate,
): Promise<PromptConfigResponse> {
  return request<PromptConfigResponse>(
    `/prompt-config?username=${encodeURIComponent(username)}`,
    {
      method: "PUT",
      body: JSON.stringify(payload),
    },
  );
}

export async function createConversation(
  username: string,
  title?: string,
): Promise<ConversationSummary> {
  return request<ConversationSummary>("/conversations", {
    method: "POST",
    body: JSON.stringify(title ? { username, title } : { username }),
  });
}

export async function reorderConversations(
  username: string,
  conversationIds: string[],
): Promise<ConversationSummary[]> {
  return request<ConversationSummary[]>("/conversations/order", {
    method: "PUT",
    body: JSON.stringify({
      username,
      conversation_ids: conversationIds,
    }),
  });
}

export async function loadConversation(
  conversationId: string,
): Promise<ConversationSummary> {
  return request<ConversationSummary>(
    `/conversations/${encodeURIComponent(conversationId)}`,
  );
}

export async function archiveConversation(
  conversationId: string,
): Promise<ConversationSummary> {
  return request<ConversationSummary>(
    `/conversations/${encodeURIComponent(conversationId)}`,
    {
      method: "PATCH",
      body: JSON.stringify({ status: "archived" }),
    },
  );
}

export async function renameConversation(
  conversationId: string,
  title: string,
): Promise<ConversationSummary> {
  return request<ConversationSummary>(
    `/conversations/${encodeURIComponent(conversationId)}`,
    {
      method: "PATCH",
      body: JSON.stringify({ title }),
    },
  );
}

export async function rewindConversation(
  conversationId: string,
  username: string,
): Promise<RewindResponse> {
  return request<RewindResponse>(
    `/conversations/${encodeURIComponent(conversationId)}/rewind`,
    {
      method: "POST",
      body: JSON.stringify({ username }),
    },
  );
}

export async function loadUserPerformanceMessages(
  username: string,
  options: {
    limit?: number;
    day?: string;
  } = {},
): Promise<MessageRecord[]> {
  const query = new URLSearchParams({
    limit: String(options.limit ?? 20),
  });
  if (options.day) {
    query.set("day", options.day);
  }
  return request<MessageRecord[]>(
    `/users/${encodeURIComponent(username)}/performance?${query.toString()}`,
  );
}

export async function streamChatMessage(
  conversationId: string,
  username: string,
  content: string,
  knowledgeQuery: string,
  systemPrompt: string,
  knowledgeEnabled: boolean,
  speakerIdentity: string,
  saveSpeakerIdentity: boolean,
  modelConfig: ChatModelConfig,
  traceId: string,
  onDelta: (content: string) => void,
  onKnowledgeContext: (content: string, hitCount: number) => void,
  onWebSearchSources: (
    sources: WebSearchSource[],
  ) => void | Promise<void>,
  onPerformance: (metrics: Partial<ChatPerformanceMetrics>) => void,
  onPhase: (phase: ChatStreamPhase, timestamp: number) => void,
): Promise<ChatResponse> {
  onPhase("request_started", performance.now());
  const response = await fetch(
    `${API_BASE}/conversations/${encodeURIComponent(conversationId)}/chat`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username,
        content,
        knowledge_query: knowledgeQuery,
        speaker_identity: speakerIdentity,
        save_speaker_identity: saveSpeakerIdentity,
        system_prompt: systemPrompt,
        knowledge_enabled: knowledgeEnabled,
        llm_config: modelConfig,
        trace_id: traceId,
      }),
    },
  );
  onPhase("response_headers", performance.now());

  if (!response.ok) {
    let message = `API request failed with status ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // Keep the status-based fallback when the response is not JSON.
    }
    throw new ApiError(response.status, message);
  }
  if (!response.body) {
    throw new ApiError(502, "Streaming response body is unavailable");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: ChatResponse | null = null;
  let firstChunkReceived = false;

  async function handleLine(line: string) {
    if (!line.trim()) return;

    let event: ChatStreamEvent;
    try {
      event = JSON.parse(line) as ChatStreamEvent;
    } catch {
      throw new ApiError(502, "Model returned an invalid stream");
    }

    if (event.type === "delta") {
      if (!firstChunkReceived) {
        firstChunkReceived = true;
        onPhase("first_chunk", performance.now());
      }
      onDelta(event.content);
      await new Promise<void>((resolve) => {
        window.requestAnimationFrame(() => resolve());
      });
      return;
    }
    if (event.type === "start") {
      onPhase("stream_start", performance.now());
      onPerformance({
        knowledge_duration_ms: event.knowledge_duration_ms,
      });
      onKnowledgeContext(
        event.knowledge_injected_context,
        event.knowledge_hit_count,
      );
      return;
    }
    if (event.type === "metric") {
      onPerformance(event.metrics);
      return;
    }
    if (event.type === "web_search_sources") {
      await onWebSearchSources(event.sources);
      return;
    }
    if (event.type === "error") {
      throw new ApiError(event.status, event.message);
    }
    if (event.type === "done") {
      onPerformance(event.performance_metrics);
      result = {
        user_message: event.user_message,
        assistant_message: event.assistant_message,
        mode: event.mode,
        knowledge_hit_count: event.knowledge_hit_count,
        completed_rounds: event.completed_rounds,
        effective_char_count: event.effective_char_count,
        memory_status: event.memory_status,
        memory_through_round: event.memory_through_round,
        memory_target_round: event.memory_target_round,
      };
    }
  }

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) await handleLine(line);
    if (done) break;
  }
  await handleLine(buffer);

  if (!result) {
    throw new ApiError(502, "Model stream ended before completion");
  }
  return result;
}

export async function reportChatLatency(
  report: FrontendLatencyReport,
): Promise<void> {
  const response = await fetch(`${API_BASE}/telemetry/chat-latency`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(report),
    keepalive: true,
  });
  if (!response.ok) {
    throw new ApiError(response.status, "Failed to record chat latency");
  }
}
