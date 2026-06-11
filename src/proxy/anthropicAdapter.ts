import { buildStableMemoryPack } from "../memory/stablePack";
import type { AssembledPrompt } from "../assembler/types";
import { assembledToAnthropicMessages, assembledToAnthropicSystem } from "../assembler/toAnthropic";
import type { Env, MemoryApiRecord, OpenAIChatMessage, OpenAIChatRequest, OpenAIChatResponse, TokenUsage } from "../types";
import { formatMemoryPatch } from "../memory/inject";
import { normalizeAiGatewayBaseUrl } from "./openaiAdapter";

interface AnthropicCacheControl {
  type: "ephemeral";
  ttl?: "5m" | "1h";
}

interface AnthropicTextBlock {
  type: "text";
  text: string;
  cache_control?: AnthropicCacheControl;
}

interface AnthropicThinkingBlock {
  type: "thinking";
  thinking: string;
  signature: string;
}

interface AnthropicToolUseBlock {
  type: "tool_use";
  id: string;
  name: string;
  input: unknown;
  cache_control?: AnthropicCacheControl;
}

interface AnthropicToolResultBlock {
  type: "tool_result";
  tool_use_id: string;
  content?: string | Array<{ type: "text"; text: string }>;
  is_error?: boolean;
  cache_control?: AnthropicCacheControl;
}

type AnthropicContentBlock =
  | AnthropicTextBlock
  | AnthropicThinkingBlock
  | AnthropicToolUseBlock
  | AnthropicToolResultBlock;

interface AnthropicMessage {
  role: "user" | "assistant";
  content: AnthropicContentBlock[];
}

interface AnthropicTool {
  name: string;
  description?: string;
  input_schema: Record<string, unknown>;
}

type AnthropicToolChoice =
  | { type: "auto" | "any" | "none"; disable_parallel_tool_use?: boolean }
  | { type: "tool"; name: string; disable_parallel_tool_use?: boolean };

interface AnthropicRequest {
  model: string;
  max_tokens: number;
  cache_control?: AnthropicCacheControl;
  temperature?: number;
  stream?: boolean;
  thinking?: {
    type: "enabled";
    budget_tokens: number;
    display?: "summarized" | "omitted";
  };
  tools?: AnthropicTool[];
  tool_choice?: AnthropicToolChoice;
  system: AnthropicTextBlock[];
  messages: AnthropicMessage[];
}

interface AnthropicResponse {
  id?: string;
  model?: string;
  role?: string;
  content?: Array<{
    type?: string;
    text?: string;
    thinking?: string;
    signature?: string;
    id?: string;
    name?: string;
    input?: unknown;
  }>;
  stop_reason?: string | null;
  usage?: TokenUsage;
}

// ---------------------------------------------------------------------------
// OpenAI tools → Anthropic tools 转换
// ---------------------------------------------------------------------------

interface OpenAIToolCallShape {
  id?: string;
  type?: string;
  function?: { name?: string; arguments?: string };
}

interface OpenAIThinkingBlockShape {
  type?: string;
  thinking?: string;
  signature?: string;
}

function parseToolArguments(args: unknown): unknown {
  if (typeof args !== "string" || !args.trim()) return {};
  try {
    return JSON.parse(args);
  } catch {
    return {};
  }
}

function extractToolCalls(message: OpenAIChatMessage): OpenAIToolCallShape[] {
  const calls = message.tool_calls;
  if (!Array.isArray(calls)) return [];
  return (calls as OpenAIToolCallShape[]).filter(
    (call) => call && (call.type === "function" || call.type == null) && call.function?.name
  );
}

/**
 * 从 OpenAI 消息中提取带签名的 thinking 块（LiteLLM 风格的 thinking_blocks 字段）。
 * 无签名的块一律丢弃——回放无签名 thinking 块正是 "signature: Field required" 的来源。
 */
function extractSignedThinkingBlocks(message: OpenAIChatMessage): AnthropicThinkingBlock[] {
  const blocks = (message as unknown as Record<string, unknown>).thinking_blocks;
  if (!Array.isArray(blocks)) return [];
  return (blocks as OpenAIThinkingBlockShape[])
    .filter(
      (block) =>
        block?.type === "thinking" &&
        typeof block.thinking === "string" &&
        typeof block.signature === "string" &&
        block.signature.length > 0
    )
    .map((block) => ({ type: "thinking", thinking: block.thinking!, signature: block.signature! }));
}

export function convertOpenAITools(req: OpenAIChatRequest): AnthropicTool[] | undefined {
  const tools = req.tools;
  if (!Array.isArray(tools) || tools.length === 0) return undefined;
  const converted: AnthropicTool[] = [];
  for (const tool of tools as Array<{ type?: string; function?: { name?: string; description?: string; parameters?: unknown } }>) {
    if (tool?.type !== "function" || !tool.function?.name) continue;
    converted.push({
      name: tool.function.name,
      ...(tool.function.description ? { description: tool.function.description } : {}),
      input_schema: isRecord(tool.function.parameters)
        ? (tool.function.parameters as Record<string, unknown>)
        : { type: "object", properties: {} }
    });
  }
  return converted.length > 0 ? converted : undefined;
}

/**
 * OpenAI tool_choice → Anthropic tool_choice。
 * 注意：extended thinking 开启时 Anthropic 仅接受 auto/none，
 * any/tool 会被上游 400，故传入 thinking 以便降级。
 */
export function convertOpenAIToolChoice(
  req: OpenAIChatRequest,
  thinkingEnabled: boolean
): AnthropicToolChoice | undefined {
  const choice = req.tool_choice;
  const disableParallel = req.parallel_tool_calls === false ? { disable_parallel_tool_use: true } : {};

  let converted: AnthropicToolChoice | undefined;
  if (choice === "none") converted = { type: "none" };
  else if (choice === "auto") converted = { type: "auto", ...disableParallel };
  else if (choice === "required") converted = { type: "any", ...disableParallel };
  else if (isRecord(choice) && choice.type === "function" && isRecord(choice.function) && typeof choice.function.name === "string") {
    converted = { type: "tool", name: choice.function.name, ...disableParallel };
  } else if (req.parallel_tool_calls === false && Array.isArray(req.tools) && req.tools.length > 0) {
    converted = { type: "auto", disable_parallel_tool_use: true };
  }

  if (converted && thinkingEnabled && (converted.type === "any" || converted.type === "tool")) {
    return { type: "auto" };
  }
  return converted;
}

/** Anthropic stop_reason → OpenAI finish_reason。 */
export function mapAnthropicStopReason(stopReason: string | null | undefined): string | null {
  if (!stopReason) return null;
  switch (stopReason) {
    case "end_turn":
    case "stop_sequence":
      return "stop";
    case "max_tokens":
      return "length";
    case "tool_use":
      return "tool_calls";
    default:
      return stopReason;
  }
}

function contentToText(content: OpenAIChatMessage["content"]): string {
  if (typeof content === "string") return content;
  if (content == null) return "";
  return content
    .map((part) => {
      if (typeof part === "string") return part;
      if (isRecord(part) && part.type === "text" && typeof part.text === "string") return part.text;
      return JSON.stringify(part);
    })
    .join("");
}

function stripAnthropicProviderPrefix(model: string): string {
  return model.replace(/^anthropic\//i, "");
}

function parseCustomProviderModel(model: string): { slug: string; model: string } | null {
  const match = model.match(/^custom-([a-z0-9-]+)\/(.+)$/i);
  if (!match) return null;
  return {
    slug: match[1],
    model: match[2]
  };
}

function stripAnthropicModelPrefix(model: string): string {
  return parseCustomProviderModel(model)?.model || stripAnthropicProviderPrefix(model);
}

function getCustomAnthropicMessagesPath(env: Env): string {
  return (env.CUSTOM_ANTHROPIC_MESSAGES_PATH || "messages").replace(/^\/+/, "");
}

function buildCacheControl(env: Env): AnthropicTextBlock["cache_control"] | undefined {
  if (env.ANTHROPIC_CACHE_ENABLED === "false") return undefined;
  const ttl = env.ANTHROPIC_CACHE_TTL === "1h" ? "1h" : "5m";
  return ttl === "1h" ? { type: "ephemeral", ttl } : { type: "ephemeral" };
}

function buildAutomaticCacheControl(env: Env): AnthropicRequest["cache_control"] | undefined {
  if (env.ANTHROPIC_CACHE_ENABLED === "false") return undefined;
  if (env.ANTHROPIC_AUTO_CACHE_ENABLED !== "true") return undefined;
  return buildCacheControl(env);
}

function getRollingCacheWindowSize(env: Env): number {
  const value = Number(env.ANTHROPIC_ROLLING_CACHE_WINDOW_SIZE || 20);
  if (!Number.isFinite(value)) return 20;
  return Math.max(Math.floor(value), 1);
}

export function getAnthropicCacheMode(env: Env): string | null {
  if (env.ANTHROPIC_CACHE_ENABLED === "false") return null;
  const parts = ["anthropic"];
  parts.push("explicit");
  if (env.ANTHROPIC_AUTO_CACHE_ENABLED === "true") parts.push("auto");
  if (env.ANTHROPIC_ROLLING_CACHE_ENABLED !== "false") parts.push("rolling");
  return parts.join("_");
}

function applyRollingMessageCache(messages: AnthropicMessage[], env: Env): void {
  const cacheControl = buildCacheControl(env);
  if (!cacheControl) return;
  if (env.ANTHROPIC_ROLLING_CACHE_ENABLED === "false") return;

  const isFullWindow = messages.length >= getRollingCacheWindowSize(env);
  const start = isFullWindow ? 0 : messages.length - 1;
  const end = isFullWindow ? messages.length : -1;
  const step = isFullWindow ? 1 : -1;

  for (let i = start; i !== end; i += step) {
    const message = messages[i];
    if (message.role !== "user" || message.content.length === 0) continue;
    // thinking 块不接受 cache_control，锚点落在最后一个可缓存块上
    for (let j = message.content.length - 1; j >= 0; j -= 1) {
      const block = message.content[j];
      if (block.type === "thinking") continue;
      block.cache_control = cacheControl;
      return;
    }
  }
}

function appendUncachedUserContext(messages: AnthropicMessage[], text: string | null | undefined): void {
  const trimmed = text?.trim();
  if (!trimmed) return;

  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (message.role !== "user") continue;
    message.content.push({ type: "text", text: trimmed });
    return;
  }

  messages.push({ role: "user", content: [{ type: "text", text: trimmed }] });
}

function splitDynamicMemorySystemBlock(
  assembled: AssembledPrompt
): { systemBlocks: AssembledPrompt["system_blocks"]; dynamicMemoryPatch: string | null } {
  const idx = assembled.meta.block_ids.indexOf("dynamic_memory_patch");
  if (idx < 0 || idx >= assembled.system_blocks.length) {
    return { systemBlocks: assembled.system_blocks, dynamicMemoryPatch: null };
  }

  return {
    systemBlocks: [
      ...assembled.system_blocks.slice(0, idx),
      ...assembled.system_blocks.slice(idx + 1),
    ],
    dynamicMemoryPatch: assembled.system_blocks[idx].text,
  };
}

function getMaxTokens(req: OpenAIChatRequest): number {
  const value = typeof req.max_tokens === "number" ? req.max_tokens : 1024;
  return Math.max(Math.floor(value), 1);
}

function clampThinkingBudget(value: unknown): number | null {
  const numeric = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
  if (!Number.isFinite(numeric)) return null;
  return Math.min(Math.max(Math.floor(numeric), 1024), 32000);
}

function getEnvThinkingBudget(env: Env): number {
  const value = clampThinkingBudget(env.ANTHROPIC_THINKING_BUDGET);
  return value ?? 1024;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function parseBooleanLike(value: unknown): boolean | null {
  if (typeof value === "boolean") return value;
  if (typeof value !== "string") return null;
  const normalized = value.trim().toLowerCase();
  if (["true", "1", "yes", "on", "enabled"].includes(normalized)) return true;
  if (["false", "0", "no", "off", "disabled", "none"].includes(normalized)) return false;
  return null;
}

function budgetFromReasoningEffort(value: unknown): number | null {
  if (typeof value !== "string") return null;
  const normalized = value.trim().toLowerCase();
  if (["none", "off", "disabled", "disable"].includes(normalized)) return 0;
  if (["minimal", "low"].includes(normalized)) return 1024;
  if (["medium", "auto"].includes(normalized)) return 2048;
  if (normalized === "high") return 4096;
  if (["xhigh", "extra_high"].includes(normalized)) return 8192;
  return null;
}

function readThinkingDirective(source: Record<string, unknown>): { enabled?: boolean; budget?: number } {
  const effortBudget = budgetFromReasoningEffort(source.reasoning_effort);
  if (effortBudget === 0) return { enabled: false };
  if (effortBudget && effortBudget > 0) return { enabled: true, budget: effortBudget };

  const enableThinking = parseBooleanLike(source.enable_thinking);
  if (enableThinking !== null) {
    return {
      enabled: enableThinking,
      budget: clampThinkingBudget(source.thinking_budget ?? source.reasoning_budget ?? source.budget_tokens) ?? undefined
    };
  }

  const thinking = source.thinking;
  if (parseBooleanLike(thinking) !== null) {
    const enabled = parseBooleanLike(thinking);
    return {
      enabled: enabled ?? undefined,
      budget: clampThinkingBudget(source.thinking_budget ?? source.reasoning_budget ?? source.budget_tokens) ?? undefined
    };
  }

  if (isRecord(thinking)) {
    const type = typeof thinking.type === "string" ? thinking.type.trim().toLowerCase() : "";
    if (["disabled", "off", "none"].includes(type)) return { enabled: false };
    const budget = clampThinkingBudget(thinking.budget_tokens ?? thinking.budget ?? source.thinking_budget);
    if (type === "enabled" || budget) return { enabled: true, budget: budget ?? undefined };
  }

  const reasoning = source.reasoning;
  if (parseBooleanLike(reasoning) !== null) {
    const enabled = parseBooleanLike(reasoning);
    return {
      enabled: enabled ?? undefined,
      budget: clampThinkingBudget(source.reasoning_budget ?? source.budget_tokens) ?? undefined
    };
  }

  if (isRecord(reasoning)) {
    const enabled = parseBooleanLike(reasoning.enabled);
    if (enabled === false) return { enabled: false };
    const budget =
      clampThinkingBudget(reasoning.budget_tokens ?? reasoning.budget ?? source.reasoning_budget) ??
      budgetFromReasoningEffort(reasoning.effort);
    if (enabled === true || (budget && budget > 0)) return { enabled: true, budget: budget ?? undefined };
  }

  const budget = clampThinkingBudget(source.thinking_budget ?? source.reasoning_budget ?? source.budget_tokens);
  if (budget) return { enabled: true, budget };

  return {};
}

function getRequestThinkingDirective(req: OpenAIChatRequest): { enabled?: boolean; budget?: number } {
  for (const source of [req, isRecord(req.extra_body) ? req.extra_body : null, isRecord(req.extraBody) ? req.extraBody : null]) {
    if (!source) continue;
    const directive = readThinkingDirective(source);
    if (directive.enabled !== undefined || directive.budget !== undefined) return directive;
  }

  return {};
}

function buildThinkingConfig(env: Env, req: OpenAIChatRequest): AnthropicRequest["thinking"] | undefined {
  if (env.ANTHROPIC_THINKING_ENABLED === "false") return undefined;
  const requestDirective = getRequestThinkingDirective(req);
  if (requestDirective.enabled === false) return undefined;

  if (requestDirective.enabled === true || requestDirective.budget) {
    return {
      type: "enabled",
      budget_tokens: requestDirective.budget ?? getEnvThinkingBudget(env),
      display: "summarized"
    };
  }

  if (env.ANTHROPIC_THINKING_ENABLED !== "true") return undefined;
  return {
    type: "enabled",
    budget_tokens: getEnvThinkingBudget(env),
    display: "summarized"
  };
}

function getAnthropicMaxTokens(
  req: OpenAIChatRequest,
  env: Env,
  thinking: AnthropicRequest["thinking"] | undefined
): number {
  const maxTokens = getMaxTokens(req);
  if (!thinking) return maxTokens;
  return Math.max(maxTokens, thinking.budget_tokens + Math.min(Math.max(maxTokens, 256), 4096));
}

function extractSystemBlocks(messages: OpenAIChatMessage[]): AnthropicTextBlock[] {
  return messages
    .filter((message) => message.role === "system")
    .map((message) => contentToText(message.content).trim())
    .filter(Boolean)
    .map((text) => ({ type: "text", text }));
}

/**
 * OpenAI 消息 → Anthropic 消息，保留结构化块：
 * - assistant.tool_calls   → tool_use 块（arguments JSON 解析为 input）
 * - role:"tool"            → user 消息中的 tool_result 块（tool_result 必须置于消息内容首部）
 * - assistant.thinking_blocks（带签名）→ thinking 块，置于消息首部
 * - 纯文本照旧合并相邻同角色消息
 */
export function convertMessages(messages: OpenAIChatMessage[]): AnthropicMessage[] {
  const result: AnthropicMessage[] = [];

  const pushBlocks = (role: "user" | "assistant", blocks: AnthropicContentBlock[]): void => {
    if (blocks.length === 0) return;
    const previous = result[result.length - 1];
    if (previous?.role === role) {
      // tool_result 必须先于其他块；插入到已有 tool_result 序列之后、首个非 tool_result 块之前
      const toolResults = blocks.filter((b) => b.type === "tool_result");
      const others = blocks.filter((b) => b.type !== "tool_result");
      if (toolResults.length > 0) {
        let insertAt = 0;
        while (insertAt < previous.content.length && previous.content[insertAt].type === "tool_result") {
          insertAt += 1;
        }
        previous.content.splice(insertAt, 0, ...toolResults);
      }
      previous.content.push(...others);
      // thinking 块必须先于 text/tool_use；合并后统一归位（保持相对次序）
      const thinkingBlocks = previous.content.filter((b) => b.type === "thinking");
      if (thinkingBlocks.length > 0) {
        const rest = previous.content.filter((b) => b.type !== "thinking");
        previous.content = [...thinkingBlocks, ...rest];
      }
      return;
    }
    result.push({ role, content: blocks });
  };

  for (const message of messages) {
    if (message.role === "system") continue;

    if (message.role === "tool") {
      const block: AnthropicToolResultBlock = {
        type: "tool_result",
        tool_use_id: typeof message.tool_call_id === "string" ? message.tool_call_id : "",
        content: contentToText(message.content)
      };
      pushBlocks("user", [block]);
      continue;
    }

    if (message.role === "assistant") {
      const blocks: AnthropicContentBlock[] = [...extractSignedThinkingBlocks(message)];
      const text = contentToText(message.content);
      if (text) blocks.push({ type: "text", text });
      for (const call of extractToolCalls(message)) {
        blocks.push({
          type: "tool_use",
          id: call.id || `toolu_${Math.random().toString(36).slice(2, 10)}`,
          name: call.function!.name!,
          input: parseToolArguments(call.function!.arguments)
        });
      }
      pushBlocks("assistant", blocks);
      continue;
    }

    const text = contentToText(message.content);
    if (!text) continue;
    pushBlocks("user", [{ type: "text", text }]);
  }

  if (result.length === 0) {
    result.push({ role: "user", content: [{ type: "text", text: "" }] });
  }

  return result;
}

/**
 * thinking 降级守卫：extended thinking 开启时，若末位 assistant 消息含 tool_use
 * 却没有可回放的带签名 thinking 块，上游会以 400（signature 缺失）拒绝。
 * 客户端未能回传 thinking_blocks 时，对本轮请求关闭 thinking 以保通路。
 */
function shouldDisableThinkingForToolHistory(messages: AnthropicMessage[]): boolean {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (message.role !== "assistant") continue;
    const hasToolUse = message.content.some((block) => block.type === "tool_use");
    if (!hasToolUse) return false;
    const hasSignedThinking = message.content.some((block) => block.type === "thinking");
    return !hasSignedThinking;
  }
  return false;
}

export function getAnthropicNativeUrl(env: Env): string {
  return `${normalizeAiGatewayBaseUrl(env)}/anthropic/v1/messages`;
}

export function getAnthropicUrlForModel(env: Env, targetModel: string): string {
  const customProvider = parseCustomProviderModel(targetModel);
  if (!customProvider) return getAnthropicNativeUrl(env);
  return `${normalizeAiGatewayBaseUrl(env)}/custom-${customProvider.slug}/${getCustomAnthropicMessagesPath(env)}`;
}

export function buildAnthropicHeaders(env: Env): Headers {
  const headers = new Headers({
    "content-type": "application/json",
    "anthropic-version": "2023-06-01",
    "anthropic-beta": "prompt-caching-2024-07-31",
    "cf-aig-skip-cache": "true",
  });

  if (env.CF_AIG_TOKEN) {
    headers.set("cf-aig-authorization", `Bearer ${env.CF_AIG_TOKEN}`);
  }

  return headers;
}

export async function buildAnthropicNativeRequest(
  req: OpenAIChatRequest,
  input: { env: Env; targetModel: string; namespace: string; memories: MemoryApiRecord[] }
): Promise<AnthropicRequest> {
  const stableMemoryPack = await buildStableMemoryPack(input.env, input.namespace);
  const stableBlock: AnthropicTextBlock = {
    type: "text",
    text: stableMemoryPack
  };

  if (input.env.ANTHROPIC_CACHE_STABLE_SYSTEM !== "false") {
    stableBlock.cache_control = buildCacheControl(input.env);
  }

  const dynamicMemoryPatch = formatMemoryPatch(input.memories);
  const system: AnthropicTextBlock[] = [
    ...extractSystemBlocks(req.messages),
    {
      type: "text",
      text: [
        "以下长期记忆来自代理层。",
        "你可以自然使用它们，但不要提到记忆系统、数据库、RAG、代理层。",
        "如果记忆与当前用户消息无关，不要强行提起。"
      ].join("\n")
    },
    stableBlock
  ];

  const messages = convertMessages(req.messages);
  const thinking = shouldDisableThinkingForToolHistory(messages)
    ? undefined
    : buildThinkingConfig(input.env, req);
  applyRollingMessageCache(messages, input.env);
  appendUncachedUserContext(messages, dynamicMemoryPatch);

  return {
    model: stripAnthropicModelPrefix(input.targetModel),
    max_tokens: getAnthropicMaxTokens(req, input.env, thinking),
    cache_control: buildAutomaticCacheControl(input.env),
    temperature: thinking ? undefined : typeof req.temperature === "number" ? req.temperature : undefined,
    stream: Boolean(req.stream),
    thinking,
    tools: convertOpenAITools(req),
    tool_choice: convertOpenAIToolChoice(req, Boolean(thinking)),
    system,
    messages
  };
}

/**
 * Build an Anthropic native request from an AssembledPrompt.
 *
 * - System blocks are converted via assembledToAnthropicSystem
 * - Messages via assembledToAnthropicMessages
 *   (structured content like image_url is JSON.stringify'd — temporary fallback)
 * - dynamic_memory_patch is moved out of system and appended after the
 *   rolling cache point, so changing RAG hits do not poison cached prefixes
 * - cache_control is applied to the client_system anchor block and the
 *   rolling user/window block, respecting ANTHROPIC_CACHE_ENABLED and
 *   ANTHROPIC_CACHE_TTL
 */
export function buildAnthropicRequestFromAssembled(
  req: OpenAIChatRequest,
  targetModel: string,
  assembled: AssembledPrompt,
  env: Env
): AnthropicRequest {
  const thinking = buildThinkingConfig(env, req);
  const { systemBlocks, dynamicMemoryPatch } = splitDynamicMemorySystemBlock(assembled);
  const system = assembledToAnthropicSystem(systemBlocks);
  const messages = assembledToAnthropicMessages(assembled.messages);
  applyCacheOverrides(system, env);
  applyRollingMessageCache(messages, env);
  appendUncachedUserContext(messages, dynamicMemoryPatch);

  return {
    model: stripAnthropicModelPrefix(targetModel),
    max_tokens: getAnthropicMaxTokens(req, env, thinking),
    cache_control: buildAutomaticCacheControl(env),
    temperature: thinking ? undefined : typeof req.temperature === "number" ? req.temperature : undefined,
    stream: Boolean(req.stream),
    thinking,
    tools: convertOpenAITools(req),
    tool_choice: convertOpenAIToolChoice(req, Boolean(thinking)),
    system,
    messages,
  };
}

function applyCacheOverrides(systemBlocks: AnthropicTextBlock[], env: Env): void {
  const anchor = systemBlocks.find((b) => b.cache_control);
  if (!anchor) return;

  if (env.ANTHROPIC_CACHE_ENABLED === "false") {
    delete anchor.cache_control;
    return;
  }

  const ttl = env.ANTHROPIC_CACHE_TTL === "1h" ? "1h" : "5m";
  anchor.cache_control = { type: "ephemeral", ttl };
}

export async function callAnthropicNative(env: Env, body: AnthropicRequest, targetModel?: string): Promise<Response> {
  return fetch(getAnthropicUrlForModel(env, targetModel || body.model), {
    method: "POST",
    headers: buildAnthropicHeaders(env),
    body: JSON.stringify(body)
  });
}

export function parseAnthropicNonStream(response: AnthropicResponse): {
  openai: OpenAIChatResponse;
  content: string;
  finishReason: string | null;
  usage?: TokenUsage;
} {
  const blocks = response.content ?? [];
  const content = blocks
    .filter((block) => block.type === "text" && typeof block.text === "string")
    .map((block) => block.text)
    .join("");
  const reasoningContent = blocks
    .filter((block) => block.type === "thinking" && typeof block.thinking === "string")
    .map((block) => block.thinking)
    .join("");

  // 带签名的 thinking 块原样返还（LiteLLM 风格），供客户端在工具回环中回传
  const thinkingBlocks = blocks
    .filter((block) => block.type === "thinking" && typeof block.thinking === "string")
    .map((block) => ({
      type: "thinking" as const,
      thinking: block.thinking ?? "",
      ...(block.signature ? { signature: block.signature } : {})
    }));

  // tool_use → OpenAI tool_calls
  const toolCalls = blocks
    .filter((block) => block.type === "tool_use" && typeof block.name === "string")
    .map((block, index) => ({
      index,
      id: block.id || `toolu_${index}`,
      type: "function" as const,
      function: {
        name: block.name as string,
        arguments: JSON.stringify(block.input ?? {})
      }
    }));

  const usage = normalizeAnthropicUsage(response.usage);

  return {
    content,
    finishReason: response.stop_reason ?? null,
    usage,
    openai: {
      id: response.id,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: response.model,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content,
            ...(reasoningContent ? { reasoning_content: reasoningContent } : {}),
            ...(thinkingBlocks.length > 0 ? { thinking_blocks: thinkingBlocks } : {}),
            ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {})
          },
          finish_reason: mapAnthropicStopReason(response.stop_reason)
        }
      ],
      usage
    }
  };
}

export function normalizeAnthropicUsage(usage: TokenUsage | undefined): TokenUsage | undefined {
  if (!usage) return undefined;

  const input = usage.input_tokens ?? usage.prompt_tokens;
  const output = usage.output_tokens ?? usage.completion_tokens;

  return {
    ...usage,
    prompt_tokens: input,
    completion_tokens: output,
    total_tokens: typeof input === "number" && typeof output === "number" ? input + output : usage.total_tokens
  };
}
