import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the LiveStreamingAgent live console", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Live Streaming Agent 控制台<\/title>/i);
  assert.match(html, /Live Streaming Agent 控制台/);
  assert.match(html, /检查中/);
  assert.match(html, /请输入用户名/);
  assert.match(html, /切换用户名/);
  assert.match(html, /进入工作区/);
  assert.match(html, /回溯上一轮/);
  assert.match(html, /⇩ 导出/);
  assert.match(html, /短期记忆/);
  assert.match(html, /已自动保存/);
  assert.match(html, /DeepSeek/);
  assert.match(html, /豆包 \/ 火山方舟/);
  assert.match(html, /通义千问 \/ 百炼/);
  assert.doesNotMatch(html, /OpenAI 兼容/);
  assert.doesNotMatch(html, /Ollama 本地/);
  assert.doesNotMatch(html, /自定义模型 ID/);
  assert.doesNotMatch(html, /Base URL/);
  assert.doesNotMatch(html, /API Key/);
  assert.doesNotMatch(html, /type="password"/);
  assert.doesNotMatch(html, /保存配置/);
  assert.doesNotMatch(html, /开始监听/);
  assert.doesNotMatch(html, /关闭短期记忆/);
  assert.doesNotMatch(html, /演示回复/);
  assert.doesNotMatch(html, /Doubao Seed 1\.6/);
  assert.doesNotMatch(html, /Doubao 1\.5/);
  assert.doesNotMatch(html, /QwQ Plus/);
  assert.doesNotMatch(html, /Kimi K3 Thinking/);
  assert.doesNotMatch(html, /Kimi K2\.6 Thinking/);
  assert.doesNotMatch(html, /Kimi Thinking Latest/);
  assert.doesNotMatch(html, /GLM-4\.1V-Thinking-Flash/);
  assert.doesNotMatch(html, /GLM-Z1-Air/);
  assert.doesNotMatch(html, /GLM-Z1-Flash/);
  assert.doesNotMatch(html, /Hunyuan T1 Latest/);
  assert.match(html, /知识库用时/);
  assert.match(html, /模型首字延迟/);
  assert.match(html, /首句延迟/);
  assert.match(html, /性能概览/);
  assert.match(html, /模型对比/);
  assert.match(html, /aria-label="展开直播间抓取框"/);
  assert.doesNotMatch(html, /完整回复时间/);
  assert.doesNotMatch(html, /中位延迟 \(P50\)/);
  assert.doesNotMatch(html, /P90 延迟/);
  assert.match(html, /对话人提示词/);
  assert.match(html, />无<\/button>/);
  assert.match(html, /aria-label="新建对话人提示词"/);
  assert.match(html, /当前不使用对话人身份和对话人提示词/);
  assert.doesNotMatch(html, /模型实际使用：/);
  assert.doesNotMatch(html, /直播间互动（对话）/);
  assert.doesNotMatch(html, /自动保存完整消息/);
  assert.doesNotMatch(html, /历史记录由独立后端写入/);
  assert.doesNotMatch(html, /重启对话/);
  assert.doesNotMatch(html, /切换版本/);
  assert.doesNotMatch(html, /codex-preview/);
  assert.doesNotMatch(html, /Your site is taking shape/);
});

test("exports every message in the active conversation as a text file", async () => {
  const source = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(source, /async function handleExportConversation/);
  assert.match(source, /loadConversationMessages\(activeConversationId\)/);
  assert.match(source, /loadShortTermMemories\(activeConversationId\)/);
  assert.match(source, /\[知识库注入 · \$\{hitCount\} 条召回内容\]/);
  assert.match(source, /\[联网搜索 · \$\{webSearchSources\.length\} 个网页\]/);
  assert.match(source, /\[第 \$\{currentExportRound\} 轮\]/);
  assert.match(source, /总轮数：\$\{exportedRoundNumbers\.size\}/);
  assert.match(source, /消息数：\$\{exportedMessages\.length\}/);
  assert.match(source, /短期记忆压缩记录/);
  assert.match(source, /memory\.compression_number/);
  assert.match(source, /memory\.through_round/);
  assert.match(source, /new Blob\(\["\\uFEFF", lines\.join\("\\n\\n"\)\]/);
  assert.match(source, /type: "text\/plain;charset=utf-8"/);
  assert.match(source, /anchor\.download = .*\.txt/);
  assert.match(source, /URL\.revokeObjectURL\(downloadUrl\)/);
  assert.match(source, /onClick=\{handleExportConversation\}/);
});

test("can collapse and restore the live capture panel without stopping it", async () => {
  const source = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(source, /livePanelCollapsed/);
  assert.match(
    source,
    /const \[livePanelCollapsed, setLivePanelCollapsed\] = useState\(true\)/,
  );
  assert.match(source, /aria-label="展开直播间抓取框"/);
  assert.match(source, /className="live-room-expand"/);
  assert.doesNotMatch(
    source,
    /setLivePanelCollapsed\(true\)[\s\S]{0,100}handleStopLiveCapture/,
  );
});

test("can automatically reply to live chats and highest recent gifts", async () => {
  const source = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(source, />\s*回复弹幕\s*</);
  assert.match(source, />\s*回复礼物\s*</);
  assert.match(source, /aria-pressed=\{replyLiveChats\}/);
  assert.match(source, /aria-pressed=\{replyLiveGifts\}/);
  assert.match(source, /now - new Date\(event\.timestamp\)\.getTime\(\) <= 60_000/);
  assert.match(source, /liveGiftValue\(right\) - liveGiftValue\(left\)/);
  assert.match(source, /lastRepliedGiftRef/);
  assert.match(source, /liveGiftValue\(event\) > giftLock\.value/);
  assert.match(source, /giftLock\.repliedAt \+ 60_000 - now/);
  assert.match(source, /setLiveGiftReplyWakeTick/);
  assert.match(source, /Math\.random\(\) \* chatCandidates\.length/);
  assert.match(source, /if \(eligibleGiftCandidates\.length\)/);
  assert.match(source, /attributedContent: `\$\{event\.nickname\}：“/);
  assert.match(source, /handledLiveReplySequencesRef/);
});

test("can drag the user conversation area upward for more space", async () => {
  const [pageSource, styles] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);

  assert.match(pageSource, /aria-label="调整用户会话区域高度"/);
  assert.match(pageSource, /startY - moveEvent\.clientY/);
  assert.match(pageSource, /setWorkspaceHeight\(startWorkspaceHeight \+ appliedDelta\)/);
  assert.match(pageSource, /event\.key === "ArrowUp" \? 24 : -24/);
  assert.match(styles, /\.workspace-resize-divider/);
  assert.match(styles, /cursor: row-resize/);
  assert.match(
    styles,
    /\.chat-content\s*\{[^}]*height: 100%;[^}]*overflow: hidden;/s,
  );
  assert.doesNotMatch(
    styles,
    /\.chat-content\s*\{[^}]*max-height: 440px;/s,
  );
});

test("offers provider-specific Doubao and Qwen web search", async () => {
  const source = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(
    source,
    /provider === "doubao" \|\| provider === "qwen"/,
  );
  assert.doesNotMatch(source, /最多 1 个搜索关键词，返回不超过 5 个网页/);
  assert.match(
    source,
    /providerSupportsWebSearch\(provider\) &&\s*webSearchEnabled/,
  );
  assert.match(
    source,
    /provider === "doubao" && webSearchEnabled/,
  );
  assert.doesNotMatch(
    source,
    /当前模型不支持 DashScope turbo 联网搜索/,
  );
  assert.match(source, /useState\(false\)/);
  assert.match(source, /useState\(1\)/);
  assert.match(source, /useState\(3\)/);
  assert.match(source, /最大搜索次数/);
  assert.match(source, /搜索网页数/);
  assert.match(source, /强制搜索/);
  assert.match(source, /web_search_forced:/);
  assert.match(source, /web_search_max_tool_calls: webSearchMaxToolCalls/);
  assert.match(source, /web_search_result_limit: webSearchResultLimit/);
  assert.match(source, /messageWebSearchSources\(message\)/);
  assert.match(source, /target="_blank"/);
  assert.match(source, /rel="noopener noreferrer"/);
});

test("remembers the selected model separately for every provider", async () => {
  const [pageSource, apiSource] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../lib/api.ts", import.meta.url), "utf8"),
  ]);

  assert.match(apiSource, /provider_models: Record<string, string>/);
  assert.match(pageSource, /userProviderModels\[nextProvider\.id\]/);
  assert.match(pageSource, /setUserProviderModels\(workspaceProviderModels\)/);
  assert.match(pageSource, /\[provider\]: nextModel/);
});

test("uses a newly selected model for chat without testing the connection", async () => {
  const source = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(source, /const selectedModelRef = useRef/);
  assert.match(
    source,
    /selectedModelRef\.current = \{\s*provider,\s*model: nextModel/,
  );
  assert.match(source, /const selectedModel = selectedModelRef\.current/);
  assert.match(source, /provider: selectedModel\.provider/);
  assert.match(source, /model: selectedModel\.model/);
});

test("new conversation numbering does not reuse a visible title", async () => {
  const [pageSource, apiSource] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../lib/api.ts", import.meta.url), "utf8"),
  ]);

  assert.match(pageSource, /nextVisibleConversationTitle\(conversations\)/);
  assert.match(pageSource, /createConversation\(username\)/);
  assert.doesNotMatch(pageSource, /conversations\.length \+ 1/);
  assert.match(apiSource, /title\?: string/);
});

test("searches a phrase across the current user's conversations", async () => {
  const [pageSource, apiSource] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../lib/api.ts", import.meta.url), "utf8"),
  ]);

  assert.match(pageSource, /placeholder="搜索全部对话内容"/);
  assert.match(pageSource, /searchUserConversations\(username, phrase, controller\.signal\)/);
  assert.match(pageSource, /window\.setTimeout\(\(\) => \{/);
  assert.match(pageSource, /conversationSearchResults\.flatMap/);
  assert.match(pageSource, /result\.matches\.map/);
  assert.match(pageSource, /result\.match_count\} 处/);
  assert.match(pageSource, /match\.snippet/);
  assert.match(pageSource, /handleOpenConversationSearchMatch/);
  assert.match(pageSource, /data-message-id=\{message\.message_id\}/);
  assert.match(pageSource, /scrollIntoView\(\{ behavior: "smooth", block: "center" \}\)/);
  assert.match(pageSource, /highlightedSearchText/);
  assert.match(apiSource, /export async function searchUserConversations/);
  assert.match(apiSource, /\/conversation-search\?\$\{query\.toString\(\)\}/);
});

test("supports per-provider model temperature input", async () => {
  const [pageSource, apiSource] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../lib/api.ts", import.meta.url), "utf8"),
  ]);

  assert.match(pageSource, /温度（0-2）/);
  assert.match(pageSource, /step=\{0\.1\}/);
  assert.match(pageSource, /setTemperature\(nextTemperature\)/);
  assert.match(pageSource, /\[provider\]: nextTemperature/);
  assert.match(pageSource, /temperature,/);
  assert.match(apiSource, /provider_temperatures: Record<string, number>/);
  assert.match(apiSource, /temperature: number/);
});

test("remembers web search settings separately for every provider", async () => {
  const [pageSource, apiSource] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../lib/api.ts", import.meta.url), "utf8"),
  ]);

  assert.match(
    apiSource,
    /provider_web_search_configs: Record<string, UserWebSearchConfig>/,
  );
  assert.match(
    pageSource,
    /providerWebSearchConfig\(\s*userProviderWebSearchConfigs,\s*nextProvider\.id/,
  );
  assert.match(
    pageSource,
    /setUserProviderWebSearchConfigs\(\s*workspaceProviderWebSearchConfigs/,
  );
  assert.match(pageSource, /\[provider\]: \{/);
  assert.match(pageSource, /max_tool_calls: maxToolCalls/);
  assert.match(pageSource, /result_limit: resultLimit/);
  assert.match(pageSource, /forced,/);
  assert.match(apiSource, /forced: boolean/);
});

test("replaces a failed local turn when the user sends again", async () => {
  const source = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(
    source,
    /message\.metadata\?\.failed_attempt !== true/,
  );
  assert.match(source, /failed_attempt: true/);
  assert.match(
    source,
    /message\.message_id === optimisticMessage\.message_id/,
  );
});

test("updates web search sources while the assistant is streaming", async () => {
  const [pageSource, apiSource] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../lib/api.ts", import.meta.url), "utf8"),
  ]);

  assert.match(apiSource, /type: "web_search_sources"/);
  assert.match(apiSource, /await onWebSearchSources\(event\.sources\)/);
  assert.match(
    pageSource,
    /web_search_sources: visibleWebSearchSources/,
  );
  assert.match(pageSource, /visibleWebSearchSources/);
  assert.match(pageSource, /window\.setTimeout\(resolve, 100\)/);
  assert.doesNotMatch(
    pageSource,
    /open=\{isStreaming \|\| undefined\}/,
  );
  assert.match(
    pageSource,
    /message\.message_id === streamingMessage\.message_id/,
  );
});

test("adds the current speaker name when the persona prompt omits it", async () => {
  const [source, apiSource] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../lib/api.ts", import.meta.url), "utf8"),
  ]);

  assert.match(source, /function ensurePersonaPromptIdentity/);
  assert.match(source, /normalized\.includes\(normalizedIdentity\)/);
  assert.match(source, /return `\$\{normalizedIdentity\}是：\\n\$\{normalized\}`/);
  assert.match(
    source,
    /ensurePersonaPromptIdentity\(rawPersonaPrompt, speakerIdentity\)/,
  );
  assert.match(
    source,
    /streamChatMessage\(\s*activeConversationId,\s*username,\s*attributedContent,\s*content,\s*systemPrompt,\s*knowledgeEnabled,\s*normalizedIdentity,\s*normalizedIdentity !== savedSpeakerIdentity,/,
  );
  assert.match(apiSource, /save_speaker_identity: saveSpeakerIdentity/);
});

test("compares today's first-token and first-sentence latency by model", async () => {
  const [pageSource, apiSource] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../lib/api.ts", import.meta.url), "utf8"),
  ]);

  assert.match(pageSource, /title="首字延迟对比"/);
  assert.match(pageSource, /title="首句延迟对比"/);
  assert.match(pageSource, /limit: 500/);
  assert.match(pageSource, /day: performanceDay/);
  assert.match(pageSource, /sample\.provider/);
  assert.match(pageSource, /sample\.model/);
  assert.match(pageSource, /function percentile\(values: number\[\], quantile: number\)/);
  assert.match(pageSource, /firstTokenP50: percentile/);
  assert.match(pageSource, /firstTokenP90: percentile/);
  assert.match(pageSource, /firstSentenceP50: percentile/);
  assert.match(pageSource, /firstSentenceP90: percentile/);
  assert.match(pageSource, /<th>首字 P50<\/th>/);
  assert.match(pageSource, /<th>首字 P90<\/th>/);
  assert.match(pageSource, /<th>首句 P50<\/th>/);
  assert.match(pageSource, /<th>首句 P90<\/th>/);
  assert.match(pageSource, /<ModelComparisonPercentiles samples=\{dailyModelPerformance\}/);
  const dotChartSource = pageSource.slice(
    pageSource.indexOf("function ModelComparisonDotChart"),
    pageSource.indexOf("export default function Home"),
  );
  assert.match(dotChartSource, /<circle/);
  assert.doesNotMatch(dotChartSource, /<polyline/);
  assert.match(apiSource, /query\.set\("day", options\.day\)/);
});

test("uses valid Doubao model versions and migrates legacy IDs", async () => {
  const source = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.ok(
    source.indexOf('id: "deepseek-v4-pro"') <
      source.indexOf('id: "deepseek-v4-flash"'),
  );
  assert.ok(
    source.indexOf('id: "doubao-seed-evolving"') <
      source.indexOf('id: "doubao-seed-2-1-turbo-260628"'),
  );
  assert.match(source, /id: "doubao-seed-2-0-pro-260215"/);
  assert.match(source, /id: "doubao-seed-2-0-mini-260428"/);
  assert.match(source, /id: "doubao-seed-2-1-pro-260628"/);
  assert.match(source, /id: "doubao-seed-2-1-turbo-260628"/);
  assert.match(
    source,
    /"doubao-seed-2-0-pro-250528": "doubao-seed-2-0-pro-260215"/,
  );
  assert.match(
    source,
    /"doubao-seed-2-0-mini-250528": "doubao-seed-2-0-mini-260428"/,
  );
  assert.match(
    source,
    /"doubao-seed-2-1-pro": "doubao-seed-2-1-pro-260628"/,
  );
  assert.match(
    source,
    /"doubao-seed-2-1-turbo": "doubao-seed-2-1-turbo-260628"/,
  );
  assert.doesNotMatch(source, /id: "doubao-seed-2-0-(?:pro|mini)-250528"/);
  assert.doesNotMatch(source, /id: "doubao-seed-2-1-(?:pro|turbo)"/);
});

test("uses Tencent TokenHub endpoint models and migrates legacy IDs", async () => {
  const source = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );
  const backendConfig = await readFile(
    new URL("../backend/app/model_config_store.py", import.meta.url),
    "utf8",
  );

  assert.match(source, /name: "腾讯元宝 \/ TokenHub"/);
  assert.match(source, /defaultModel: "hy3"/);
  assert.match(source, /id: "hy3-preview"/);
  assert.match(source, /id: "hy-mt2-plus"/);
  assert.match(source, /id: "hunyuan-role-latest"/);
  assert.doesNotMatch(source, /id: "hunyuan-turbos-latest"/);
  assert.match(source, /"hunyuan-turbos-latest": "hy3"/);
  assert.match(
    backendConfig,
    /"tencent-yuanbao": "https:\/\/tokenhub\.tencentmaas\.com\/v1"/,
  );
  assert.match(
    backendConfig,
    /LEGACY_PROVIDER_BASE_URLS[\s\S]*https:\/\/api\.hunyuan\.cloud\.tencent\.com\/v1[\s\S]*DEFAULT_PROVIDER_BASE_URLS\["tencent-yuanbao"\]/,
  );
});

test("uses valid GLM model identifiers", async () => {
  const source = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(source, /defaultModel: "glm-5\.2"/);
  assert.match(source, /id: "glm-4\.7"/);
  assert.match(source, /id: "glm-4\.6"/);
  assert.match(source, /id: "glm-4\.5-air"/);
  assert.doesNotMatch(source, /id: "glm-4-/);
  assert.doesNotMatch(source, /glm-5\.2-air/);
  assert.doesNotMatch(source, /glm-5\.1-air/);
});

test("keeps Qwen 3.x models but removes the Qwen 3 series", async () => {
  const source = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(source, /defaultModel: "qwen3\.6-flash"/);
  assert.match(source, /id: "qwen3\.7-max"/);
  assert.match(source, /id: "qwen3\.6-plus"/);
  assert.match(source, /id: "qwen3\.5-flash"/);
  assert.doesNotMatch(source, /id: "qwen3-(?:max|coder)/);
});

test("uses Kimi models available to the configured Moonshot account", async () => {
  const source = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(source, /defaultModel: "kimi-k2\.6"/);
  assert.match(source, /id: "kimi-k3"/);
  assert.match(source, /"kimi-k2\.7-code": "kimi-k2\.6"/);
  assert.doesNotMatch(source, /"kimi-k3": "kimi-k2\.6"/);
  assert.doesNotMatch(source, /id: "kimi-k2\.7-code"/);
  assert.doesNotMatch(source, /id: "kimi-latest"/);
});

test("does not require crypto.randomUUID on LAN HTTP clients", async () => {
  const [pageSource, clientIdSource] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../lib/client-id.ts", import.meta.url), "utf8"),
  ]);

  assert.doesNotMatch(pageSource, /crypto\.randomUUID\(\)/);
  assert.match(pageSource, /createClientId\(\)/);
  assert.match(clientIdSource, /typeof cryptoApi\?\.randomUUID/);
  assert.match(clientIdSource, /cryptoApi\?\.getRandomValues/);
});
