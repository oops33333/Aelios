import { callOpenAICompat } from "../proxy/openaiAdapter";
import type { Env, MemoryApiRecord, OpenAIChatRequest, OpenAIChatResponse } from "../types";

const DEFAULT_WORKERS_AI_FILTER_MODEL = "workers-ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast";
const FILTER_SUCCESS_LIMIT = 4;
// API prompt 的动态记忆无论主筛选还是降级压缩都不得超过 4 条。
const FILTER_FAILURE_COMPRESSION_LIMIT = FILTER_SUCCESS_LIMIT;

interface FilteredMemoryItem {
  id: string;
  content?: string;
}

type MemoryFilterStageStatus = "success" | "error" | "skipped";

export interface MemoryFilterMeta {
  status: "disabled" | "success" | "error" | "empty";
  provider: "workers-ai" | "openai-compatible";
  model: string;
  raw_count: number;
  candidate_count: number;
  output_count: number;
  reason?: string;
  output_shape?: string;
  filter_status: MemoryFilterStageStatus;
  filter_input_count: number;
  filter_output_count: number;
  filter_reason?: string;
  compression_status: MemoryFilterStageStatus;
  compression_input_count: number;
  compression_output_count: number;
  compression_reason?: string;
}

const FILTER_RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    memories: {
      type: "array",
      items: {
        type: "object",
        properties: {
          id: { type: "string" }
        },
        required: ["id"],
        additionalProperties: false
      }
    }
  },
  required: ["memories"],
  additionalProperties: false
};

const COMPRESSION_RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    memories: {
      type: "array",
      items: {
        type: "object",
        properties: {
          id: { type: "string" },
          content: { type: "string" }
        },
        required: ["id", "content"],
        additionalProperties: false
      }
    }
  },
  required: ["memories"],
  additionalProperties: false
};

function sanitizeMemoryContent(text: string): string {
  return text
    .replace(/<time_reminder>[^|。\n]*/gi, "")
    .replace(/对话摘要（\d+ 条消息）：?/g, "")
    .replace(/用户话题[:：]/g, "")
    .replace(/助手要点[:：]/g, "")
    .replace(/debug-test/gi, "")
    .replace(/记忆系统/g, "")
    .replace(/自动记忆测试口令/g, "口令")
    .replace(/测试口令/g, "口令")
    .replace(/标签为?[^，。；\s]+/g, "")
    .replace(/标签[:：]?[^，。；\s]+/g, "")
    .replace(/[，,；;：:]\s*([。.!！?？])/g, "$1")
    .replace(/\s{2,}/g, " ")
    .replace(/^[，,；;：:\s]+|[，,；;：:\s]+$/g, "")
    .trim();
}

function isEnabled(env: Env): boolean {
  return env.ENABLE_MEMORY_FILTER !== "false";
}

function getModel(env: Env): string {
  return env.MEMORY_FILTER_MODEL || DEFAULT_WORKERS_AI_FILTER_MODEL;
}

function workersAiModelName(model: string): string | null {
  const normalized = model.trim();
  if (normalized.startsWith("workers-ai/")) return normalized.slice("workers-ai/".length);
  if (normalized.startsWith("worker/")) return normalized.slice("worker/".length);
  if (normalized.startsWith("@cf/")) return normalized;
  return null;
}

function getWorkersAiModel(env: Env): string | null {
  const model = getModel(env);
  return workersAiModelName(model);
}

function getProvider(env: Env): "workers-ai" | "openai-compatible" {
  return getWorkersAiModel(env) ? "workers-ai" : "openai-compatible";
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function getMaxCandidates(env: Env): number {
  const value = Number(env.MEMORY_FILTER_MAX_CANDIDATES || 12);
  return Number.isFinite(value) ? clamp(Math.floor(value), 1, 50) : 12;
}

function getMaxOutput(env: Env): number {
  const value = Number(env.MEMORY_FILTER_MAX_OUTPUT || FILTER_SUCCESS_LIMIT);
  return Number.isFinite(value) ? clamp(Math.floor(value), 1, FILTER_SUCCESS_LIMIT) : FILTER_SUCCESS_LIMIT;
}

function getMaxContentChars(env: Env): number {
  const value = Number(env.MEMORY_FILTER_MAX_CONTENT_CHARS || 700);
  return Number.isFinite(value) ? clamp(Math.floor(value), 120, 3000) : 700;
}

function getMaxOutputChars(env: Env): number {
  const value = Number(env.MEMORY_FILTER_OUTPUT_CHARS || 300);
  return Number.isFinite(value) ? clamp(Math.floor(value), 60, 1000) : 300;
}

function getMaxTokens(env: Env): number {
  const value = Number(env.MEMORY_FILTER_MAX_TOKENS || 1400);
  return Number.isFinite(value) ? clamp(Math.floor(value), 200, 4000) : 1400;
}

function getFilterMinScore(env: Env): number {
  const value = Number(env.MEMORY_FILTER_MIN_SCORE || env.MEMORY_MIN_SCORE || 0.35);
  return Number.isFinite(value) ? clamp(value, 0, 1) : 0.35;
}

function truncateText(text: string, maxChars: number): string {
  return text.length > maxChars ? `${text.slice(0, maxChars).trim()}...` : text;
}

function hardTruncateText(text: string, maxChars: number): string {
  const codePoints = Array.from(text);
  return codePoints.length > maxChars ? codePoints.slice(0, maxChars).join("").trim() : text;
}

function normalizeForDedupe(text: string): string {
  return text
    .toLowerCase()
    .replace(/\s+/g, "")
    .replace(/[，,。.!！?？；;：:“”"'`、\[\]【】（）()<>《》]/g, "");
}

function compareMemoryQuality(a: MemoryApiRecord, b: MemoryApiRecord): number {
  if (a.pinned !== b.pinned) return a.pinned ? -1 : 1;

  const recallA = typeof a.recall_score === "number" ? a.recall_score : -1;
  const recallB = typeof b.recall_score === "number" ? b.recall_score : -1;
  if (recallA !== recallB) return recallB - recallA;

  const scoreA = typeof a.score === "number" ? a.score : -1;
  const scoreB = typeof b.score === "number" ? b.score : -1;
  if (scoreA !== scoreB) return scoreB - scoreA;

  if (a.importance !== b.importance) return b.importance - a.importance;
  return b.confidence - a.confidence;
}

function prepareCandidates(env: Env, memories: MemoryApiRecord[]): MemoryApiRecord[] {
  const minScore = getFilterMinScore(env);
  const sorted = memories
    .flatMap((memory): MemoryApiRecord[] => {
      const content = sanitizeMemoryContent(memory.content);
      if (!content) return [];
      if (!memory.pinned && typeof memory.score === "number" && memory.score < minScore) return [];
      return [{ ...memory, content }];
    })
    .sort(compareMemoryQuality);

  const seenIds = new Set<string>();
  const seenContent = new Set<string>();
  const result: MemoryApiRecord[] = [];

  for (const memory of sorted) {
    const normalized = normalizeForDedupe(memory.content);
    if (!normalized || seenIds.has(memory.id) || seenContent.has(normalized)) continue;
    seenIds.add(memory.id);
    seenContent.add(normalized);
    result.push(memory);
    if (result.length >= getMaxCandidates(env)) break;
  }

  return result;
}

function extractJsonArrayFromString(text: string): unknown[] | null {
  const trimmed = text.trim();
  if (!trimmed) return null;

  try {
    return extractJsonArray(JSON.parse(trimmed) as unknown);
  } catch {
    // Some providers still wrap JSON in a short sentence; extract the JSON part.
  }

  const objectStart = trimmed.indexOf("{");
  const objectEnd = trimmed.lastIndexOf("}");
  if (objectStart !== -1 && objectEnd > objectStart) {
    try {
      return extractJsonArray(JSON.parse(trimmed.slice(objectStart, objectEnd + 1)) as unknown);
    } catch {
      // Fall through to array extraction.
    }
  }

  const start = trimmed.indexOf("[");
  const end = trimmed.lastIndexOf("]");
  if (start === -1 || end === -1 || end <= start) return null;

  try {
    const parsed = JSON.parse(trimmed.slice(start, end + 1)) as unknown;
    return Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function extractJsonArray(value: unknown): unknown[] | null {
  if (Array.isArray(value)) return value;
  if (typeof value === "string") return extractJsonArrayFromString(value);
  if (!value || typeof value !== "object") return null;

  const object = value as {
    memories?: unknown;
    response?: unknown;
    result?: unknown;
    text?: unknown;
    output?: unknown;
  };

  if (Array.isArray(object.memories)) return object.memories;

  for (const field of [object.response, object.result, object.text, object.output]) {
    const array = extractJsonArray(field);
    if (array) return array;
  }

  return null;
}

interface ParsedMemoryItems {
  items: FilteredMemoryItem[];
  invalidNonEmpty: boolean;
}

function parseMemoryItems(
  value: unknown,
  requireContent: boolean,
  allowedIds: Set<string>,
  maxOutputChars: number
): ParsedMemoryItems | null {
  const array = extractJsonArray(value);
  if (!array) return null;

  const items: FilteredMemoryItem[] = [];
  const seenIds = new Set<string>();
  const seenContent = new Set<string>();

  for (const item of array) {
    if (!item || typeof item !== "object") continue;
    const record = item as { id?: unknown; content?: unknown; compressed_content?: unknown };
    const id = typeof record.id === "string" ? record.id.trim() : "";
    if (!id || !allowedIds.has(id) || seenIds.has(id)) continue;

    if (!requireContent) {
      seenIds.add(id);
      items.push({ id });
      continue;
    }

    const rawContent =
      typeof record.content === "string"
        ? record.content
        : typeof record.compressed_content === "string"
          ? record.compressed_content
          : null;
    if (!rawContent) continue;

    const content = hardTruncateText(sanitizeMemoryContent(rawContent), maxOutputChars);
    const normalized = normalizeForDedupe(content);
    if (!content || !normalized || seenContent.has(normalized)) continue;

    seenIds.add(id);
    seenContent.add(normalized);
    items.push({ id, content });
  }

  return {
    items,
    invalidNonEmpty: array.length > 0 && items.length === 0
  };
}

function buildFilterPrompt(input: {
  query: string;
  memories: MemoryApiRecord[];
  maxOutput: number;
  maxContentChars: number;
}): string {
  const candidates = input.memories.map((memory, index) => ({
    index: index + 1,
    id: memory.id,
    type: memory.type,
    importance: memory.importance,
    effective_importance:
      typeof memory.effective_importance === "number"
        ? Number(memory.effective_importance.toFixed(4))
        : undefined,
    pinned: memory.pinned,
    archived: Boolean(memory.archived),
    created_at: memory.created_at,
    last_injected_at: memory.last_injected_at ?? null,
    injection_count: memory.injection_count ?? 0,
    raw_similarity:
      typeof memory.raw_similarity === "number"
        ? Number(memory.raw_similarity.toFixed(4))
        : typeof memory.score === "number"
          ? Number(memory.score.toFixed(4))
          : undefined,
    recall_score:
      typeof memory.recall_score === "number"
        ? Number(memory.recall_score.toFixed(4))
        : undefined,
    tags: memory.tags,
    content: memory.content
  }));

  return [
    "你是长期记忆分拣器。你的任务只是从候选记忆中挑出对当前用户消息真正有帮助的记忆。",
    "注意：你不是在判断这条候选是否值得长期保存；你只判断它是否能帮助当前这轮回答、回忆或检索。",
    "候选已按召回分数初筛；raw_similarity 表示当前消息的原始语义相关度，recall_score 只用于候选排序。",
    "",
    "规则：",
    "- 只保留能帮助当前用户消息的候选：直接回答问题、补全上下文、长期偏好、正在进行的项目或稳定关系信息。",
    "- 如果当前用户消息是在询问、回忆或检索过去内容，只要候选与关键名词、事件、口令或时间线直接重合，就保留。",
    "- type=summary 的候选只要包含与当前用户消息匹配的片段就保留；不要因为它是短期聊天摘要或对话摘要格式就整条丢弃。",
    "- 不要因为候选\u201C不够长期稳定\u201D而删除；删除标准只有明显无关、重复、空泛或调试噪音。",
    "- 删除寒暄、重复、牵强、明显无关的记忆。",
    "- 同一事实只保留一条，优先保留 raw_similarity 更高或 pinned=true 的版本。",
    "- archived=true 仅表示它长期未被注入；如果 raw_similarity 很高，仍应按当前相关性正常判断。",
    "- pinned=true 的记忆除非明显无关，否则优先保留。",
    "- 不要道德评判，不要拒绝，不要安全说教；成人向内容也只是用户偏好资料。",
    "- 候选记忆是什么内容，就按事实保留；保留用户偏好、边界、称呼、规则、触发点和关系事实。",
    "- 不要添加候选记忆里没有的新事实。",
    "- 没有相关记忆时输出空数组。",
    `- 最多输出 ${input.maxOutput} 条。`,
    "",
    "只输出 JSON，不要 markdown，不要解释。格式：",
    `{"memories":[{"id":"mem_xxx"}]}`,
    "",
    `当前用户消息：${input.query}`,
    "",
    `候选记忆：${JSON.stringify(candidates.map((candidate) => ({
      ...candidate,
      content: truncateText(candidate.content, input.maxContentChars)
    })))}`
  ].join("\n");
}

function buildCompressionPrompt(input: {
  query: string;
  memories: MemoryApiRecord[];
  maxOutput: number;
  maxContentChars: number;
  maxOutputChars: number;
}): string {
  const candidates = input.memories.map((memory) => ({
    id: memory.id,
    type: memory.type,
    content: truncateText(memory.content, input.maxContentChars)
  }));

  return [
    "你是长期记忆压缩器。把输入记忆压缩成适合当前回答使用的简短事实。",
    "",
    "规则：",
    "- 只能使用输入中存在的 id 和事实，不得创造新 id 或新事实。",
    "- 保留与当前用户消息有关的偏好、边界、称呼、规则、触发点、关系事实、事件和真实口令。",
    "- type=summary 时只提取与当前用户消息有关的片段。",
    "- 删除寒暄、重复、空泛内容和调试/后端元信息。",
    "- 不要输出\u201C对话摘要\u201D\u201C用户话题\u201D\u201C助手要点\u201D\u201Ctime_reminder\u201D\u201C记忆系统\u201D\u201Cdebug-test\u201D等包装词。",
    "- 同一事实只保留一条。",
    `- 每条 content 最多 ${input.maxOutputChars} 个字符。`,
    `- 最多输出 ${input.maxOutput} 条。`,
    "- 没有可用内容时输出空数组。",
    "",
    "只输出 JSON，不要 markdown，不要解释。格式：",
    `{"memories":[{"id":"mem_xxx","content":"压缩后的记忆"}]}`,
    "",
    `当前用户消息：${input.query}`,
    "",
    `待压缩记忆：${JSON.stringify(candidates)}`
  ].join("\n");
}

function selectKnownMemories(
  memories: MemoryApiRecord[],
  items: FilteredMemoryItem[],
  limit: number
): MemoryApiRecord[] {
  const byId = new Map(memories.map((memory) => [memory.id, memory]));
  const seenIds = new Set<string>();
  const result: MemoryApiRecord[] = [];

  for (const item of items) {
    const memory = byId.get(item.id);
    if (!memory || seenIds.has(item.id)) continue;
    seenIds.add(item.id);
    result.push(memory);
    if (result.length >= limit) break;
  }

  return result;
}

function mergeCompressedItems(
  memories: MemoryApiRecord[],
  items: FilteredMemoryItem[],
  limit: number,
  maxOutputChars: number
): MemoryApiRecord[] {
  const byId = new Map(memories.map((memory) => [memory.id, memory]));
  const seenIds = new Set<string>();
  const seenContent = new Set<string>();
  const result: MemoryApiRecord[] = [];

  for (const item of items) {
    const memory = byId.get(item.id);
    if (!memory || !item.content || seenIds.has(item.id)) continue;

    const content = hardTruncateText(sanitizeMemoryContent(item.content), maxOutputChars);
    const normalized = normalizeForDedupe(content);
    if (!content || !normalized || seenContent.has(normalized)) continue;

    seenIds.add(item.id);
    seenContent.add(normalized);
    result.push({
      ...memory,
      content
    });
    if (result.length >= limit) break;
  }

  return result;
}

function describeModelOutput(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (typeof value !== "object") return typeof value;

  const object = value as { response?: unknown; memories?: unknown; result?: unknown; output?: unknown; text?: unknown };
  if (Array.isArray(object.memories)) return "object.memories_array";
  if (typeof object.response === "string") return "object.response_string";
  if (object.response && typeof object.response === "object") return "object.response_object";
  if (typeof object.result === "string") return "object.result_string";
  if (object.result && typeof object.result === "object") return "object.result_object";
  if (typeof object.output === "string") return "object.output_string";
  if (typeof object.text === "string") return "object.text_string";
  return "object";
}

async function callWorkersAiStage(
  env: Env,
  prompt: string,
  model: string,
  maxTokens: number,
  schema: typeof FILTER_RESPONSE_SCHEMA | typeof COMPRESSION_RESPONSE_SCHEMA
): Promise<unknown> {
  if (!env.AI) return "";

  return env.AI.run(model, {
    messages: [
      {
        role: "system",
        content: "你是严格的 JSON 生成器。你只输出 JSON。"
      },
      {
        role: "user",
        content: prompt
      }
    ],
    temperature: 0,
    max_tokens: maxTokens,
    response_format: {
      type: "json_schema",
      json_schema: schema
    }
  });
}

async function callOpenAICompatStage(env: Env, prompt: string, model: string, maxTokens: number): Promise<string> {
  const request: OpenAIChatRequest = {
    model,
    messages: [
      {
        role: "system",
        content: "你是严格的 JSON 生成器。你只输出 JSON。"
      },
      {
        role: "user",
        content: prompt
      }
    ],
    temperature: 0,
    max_tokens: maxTokens,
    response_format: {
      type: "json_object"
    },
    reasoning: { enabled: false },
    enable_thinking: false,
    stream: false
  };

  const response = await callOpenAICompat(env, request);
  if (!response.ok) return "";

  const parsed = (await response.json()) as OpenAIChatResponse;
  const message = parsed.choices?.[0]?.message;
  const content = typeof message?.content === "string" ? message.content.trim() : "";
  const reasoning = typeof message?.reasoning_content === "string" ? message.reasoning_content.trim() : "";
  return content || reasoning;
}

interface MemoryStageResult {
  status: "success" | "error";
  items: FilteredMemoryItem[];
  reason?: string;
  outputShape?: string;
}

async function runMemoryStage(
  env: Env,
  input: {
    prompt: string;
    requireContent: boolean;
    allowedIds: Set<string>;
    maxOutputChars: number;
    schema: typeof FILTER_RESPONSE_SCHEMA | typeof COMPRESSION_RESPONSE_SCHEMA;
  }
): Promise<MemoryStageResult> {
  const provider = getProvider(env);
  const model = getModel(env);

  try {
    const output =
      provider === "openai-compatible"
        ? await callOpenAICompatStage(env, input.prompt, model, getMaxTokens(env))
        : await callWorkersAiStage(
            env,
            input.prompt,
            getWorkersAiModel(env) || model,
            getMaxTokens(env),
            input.schema
          );
    const outputShape = describeModelOutput(output);
    if (!output) {
      return { status: "error", items: [], reason: "empty_model_output", outputShape };
    }

    const parsed = parseMemoryItems(
      output,
      input.requireContent,
      input.allowedIds,
      input.maxOutputChars
    );
    if (!parsed) {
      return { status: "error", items: [], reason: "invalid_model_output", outputShape };
    }
    if (parsed.invalidNonEmpty) {
      return { status: "error", items: [], reason: "no_valid_model_items", outputShape };
    }

    return { status: "success", items: parsed.items, outputShape };
  } catch {
    return { status: "error", items: [], reason: "model_error" };
  }
}

function logMemoryFilterStages(meta: MemoryFilterMeta): void {
  console.info(JSON.stringify({
    event: "memory_filter_stages",
    model: meta.model,
    provider: meta.provider,
    filter: {
      status: meta.filter_status,
      input_count: meta.filter_input_count,
      output_count: meta.filter_output_count,
      reason: meta.filter_reason || (meta.filter_status === "success" ? "ok" : "not_applicable")
    },
    compression: {
      status: meta.compression_status,
      input_count: meta.compression_input_count,
      output_count: meta.compression_output_count,
      reason: meta.compression_reason || (meta.compression_status === "success" ? "ok" : "not_applicable")
    },
    ...(meta.reason ? { reason: meta.reason } : {})
  }));
}

export async function filterAndCompressMemories(
  env: Env,
  input: { query: string; memories: MemoryApiRecord[] }
): Promise<MemoryApiRecord[]> {
  const result = await filterAndCompressMemoriesWithMeta(env, input);
  return result.data;
}

export async function filterAndCompressMemoriesWithMeta(
  env: Env,
  input: { query: string; memories: MemoryApiRecord[] }
): Promise<{ data: MemoryApiRecord[]; meta: MemoryFilterMeta }> {
  const query = input.query.trim();
  const provider = getProvider(env);
  const model = getModel(env);
  const baseMeta: MemoryFilterMeta = {
    status: "disabled",
    provider,
    model,
    raw_count: input.memories.length,
    candidate_count: 0,
    output_count: input.memories.length,
    filter_status: "skipped",
    filter_input_count: 0,
    filter_output_count: 0,
    compression_status: "skipped",
    compression_input_count: 0,
    compression_output_count: 0
  };

  if (!isEnabled(env) || !query) {
    const meta: MemoryFilterMeta = {
      ...baseMeta,
      reason: !query ? "empty_query" : "filter_disabled",
      filter_reason: !query ? "empty_query" : "filter_disabled",
      compression_reason: "filter_not_run"
    };
    logMemoryFilterStages(meta);
    return { data: input.memories, meta };
  }

  const maxOutput = getMaxOutput(env);
  const maxContentChars = getMaxContentChars(env);
  const maxOutputChars = getMaxOutputChars(env);
  const candidates = prepareCandidates(env, input.memories);
  if (candidates.length === 0) {
    const meta: MemoryFilterMeta = {
      ...baseMeta,
      status: "empty",
      output_count: 0,
      reason: "no_candidates",
      filter_reason: "no_candidates",
      compression_reason: "no_candidates"
    };
    logMemoryFilterStages(meta);
    return { data: [], meta };
  }

  const filterResult = await runMemoryStage(env, {
    prompt: buildFilterPrompt({
      query,
      memories: candidates,
      maxOutput,
      maxContentChars
    }),
    requireContent: false,
    allowedIds: new Set(candidates.map((memory) => memory.id)),
    maxOutputChars,
    schema: FILTER_RESPONSE_SCHEMA
  });

  const filterSucceeded = filterResult.status === "success";
  const filtered = filterSucceeded
    ? selectKnownMemories(candidates, filterResult.items, maxOutput)
    : [];
  const compressionInput = filterSucceeded ? filtered : candidates;
  const compressionLimit = filterSucceeded ? maxOutput : FILTER_FAILURE_COMPRESSION_LIMIT;

  const compressionResult = await runMemoryStage(env, {
    prompt: buildCompressionPrompt({
      query,
      memories: compressionInput,
      maxOutput: compressionLimit,
      maxContentChars,
      maxOutputChars
    }),
    requireContent: true,
    allowedIds: new Set(compressionInput.map((memory) => memory.id)),
    maxOutputChars,
    schema: COMPRESSION_RESPONSE_SCHEMA
  });

  if (compressionResult.status === "success") {
    const compressed = mergeCompressedItems(
      compressionInput,
      compressionResult.items,
      compressionLimit,
      maxOutputChars
    );
    const meta: MemoryFilterMeta = {
      ...baseMeta,
      status: "success",
      candidate_count: candidates.length,
      output_count: compressed.length,
      output_shape: compressionResult.outputShape,
      filter_status: filterResult.status,
      filter_input_count: candidates.length,
      filter_output_count: filtered.length,
      ...(filterResult.reason ? { filter_reason: filterResult.reason } : {}),
      compression_status: "success",
      compression_input_count: compressionInput.length,
      compression_output_count: compressed.length
    };
    logMemoryFilterStages(meta);
    return { data: compressed, meta };
  }

  if (filterSucceeded) {
    const meta: MemoryFilterMeta = {
      ...baseMeta,
      status: "success",
      candidate_count: candidates.length,
      output_count: filtered.length,
      reason: "compression_failed_using_filtered_originals",
      output_shape: compressionResult.outputShape || filterResult.outputShape,
      filter_status: "success",
      filter_input_count: candidates.length,
      filter_output_count: filtered.length,
      compression_status: "error",
      compression_input_count: compressionInput.length,
      compression_output_count: 0,
      compression_reason: compressionResult.reason
    };
    logMemoryFilterStages(meta);
    return { data: filtered, meta };
  }

  const meta: MemoryFilterMeta = {
    ...baseMeta,
    status: "error",
    candidate_count: candidates.length,
    output_count: 0,
    reason: "filter_and_compression_failed",
    output_shape: compressionResult.outputShape || filterResult.outputShape,
    filter_status: "error",
    filter_input_count: candidates.length,
    filter_output_count: 0,
    filter_reason: filterResult.reason,
    compression_status: "error",
    compression_input_count: compressionInput.length,
    compression_output_count: 0,
    compression_reason: compressionResult.reason
  };
  logMemoryFilterStages(meta);
  return { data: [], meta };
}
