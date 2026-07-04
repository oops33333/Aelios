/**
 * compress.ts — 对话历史自动压缩（级联版）
 *
 * 当 RikkaHub 发来的完整历史超过阈值（默认 30 轮）时，
 * 按 windowSize（默认 25）条一段做级联压缩：
 *
 *   seg 1: messages[0..24]           → summary_1
 *   seg 2: summary_1 + messages[25..49] → summary_2
 *   seg 3: summary_2 + messages[50..74] → summary_3
 *   ...
 *
 * 每段输入量固定（~上轮摘要 800 字 + 25 条消息），
 * 不随对话长度线性增长，可以支撑任意长对话。
 *
 * 缓存策略：
 *   - 每段独立缓存在 D1 cache_entries
 *   - key 含段号 + 首条消息指纹，防跨对话缓存碰撞
 *   - 快路径：先查最终段缓存，命中则直接返回
 *   - 慢路径：从最后一个命中段开始级联，逐段压缩+缓存
 */

import { getCacheEntry, putCacheEntry, parseCacheEntryValue } from "../db/cacheEntries";
import { callOpenAICompat } from "../proxy/openaiAdapter";
import type { Env, OpenAIChatMessage } from "../types";

// ---------------------------------------------------------------------------
// 配置
// ---------------------------------------------------------------------------

const DEFAULT_THRESHOLD = 30;
const DEFAULT_KEEP_RECENT = 5;
const DEFAULT_COMPRESS_MAX_CHARS = 2000;
const DEFAULT_CACHE_TTL = 86400;
const DEFAULT_COMPRESS_MODEL = "workers-ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast";

function getThreshold(env: Env): number {
  const v = Number(env.HISTORY_COMPRESS_THRESHOLD || DEFAULT_THRESHOLD);
  return Number.isFinite(v) ? Math.max(v, 10) : DEFAULT_THRESHOLD;
}

function getKeepRecent(env: Env): number {
  const v = Number(env.HISTORY_KEEP_RECENT || DEFAULT_KEEP_RECENT);
  return Number.isFinite(v) ? Math.max(v, 2) : DEFAULT_KEEP_RECENT;
}

function getCompressModel(env: Env): string {
  return env.HISTORY_COMPRESS_MODEL || env.MEMORY_FILTER_MODEL || DEFAULT_COMPRESS_MODEL;
}

function getMaxChars(env: Env): number {
  const v = Number(env.HISTORY_COMPRESS_MAX_CHARS || DEFAULT_COMPRESS_MAX_CHARS);
  return Number.isFinite(v) ? v : DEFAULT_COMPRESS_MAX_CHARS;
}

function getCacheTtl(env: Env): number {
  const v = Number(env.HISTORY_COMPRESS_CACHE_TTL || DEFAULT_CACHE_TTL);
  return Number.isFinite(v) ? v : DEFAULT_CACHE_TTL;
}

function isEnabled(env: Env): boolean {
  return env.ENABLE_HISTORY_COMPRESSION === "true";
}

// ---------------------------------------------------------------------------
// 公开接口
// ---------------------------------------------------------------------------

export interface CompressResult {
  summary: string | null;
  messages: OpenAIChatMessage[];
  meta: {
    original_count: number;
    compressed_count: number;
    kept_count: number;
    cache_hit: boolean;
    compress_boundary: number;
    total_segments: number;
    segments_computed: number;
  };
}

/**
 * 主入口：检查阈值 → 级联压缩 → 返回裁剪后的消息。
 */
export async function compressHistoryIfNeeded(
  env: Env,
  messages: OpenAIChatMessage[],
  namespace: string
): Promise<CompressResult> {
  const noopResult: CompressResult = {
    summary: null,
    messages,
    meta: {
      original_count: 0, compressed_count: 0, kept_count: 0,
      cache_hit: false, compress_boundary: 0,
      total_segments: 0, segments_computed: 0,
    },
  };

  if (!isEnabled(env)) {
    console.log("[compress] DISABLED - ENABLE_HISTORY_COMPRESSION is not 'true'");
    return noopResult;
  }
  console.log("[compress] ENABLED, checking messages...");

  const systemMessages = messages.filter(m => m.role === "system");
  const chatMessages = messages.filter(m => m.role === "user" || m.role === "assistant");

  noopResult.meta.original_count = chatMessages.length;
  noopResult.meta.kept_count = chatMessages.length;

  const threshold = getThreshold(env);
  const keepRecent = getKeepRecent(env);

  console.log(`[compress] chatMessages=${chatMessages.length}, threshold=${threshold}, keepRecent=${keepRecent}`);
  if (chatMessages.length <= threshold) {
    console.log("[compress] below threshold, skipping");
    return noopResult;
  }

  const windowSize = threshold - keepRecent;
  const completeWindows = Math.floor((chatMessages.length - keepRecent) / windowSize);
  const compressEnd = completeWindows * windowSize;

  if (compressEnd < windowSize) return noopResult;

  const recent = chatMessages.slice(compressEnd);

  // -----------------------------------------------------------------------
  // 级联压缩
  // -----------------------------------------------------------------------

  let summary: string | null = null;
  let segmentsComputed = 0;
  let cacheHit = true;

  // 快路径：最终段缓存命中 → 直接返回
  const finalKey = buildSegmentCacheKey(namespace, completeWindows, chatMessages, windowSize);
  const finalCached = await tryGetCache(env.DB, namespace, finalKey);
  if (finalCached) {
    console.log("[compress] CACHE HIT on final segment");
    return {
      summary: finalCached,
      messages: [...systemMessages, ...recent],
      meta: {
        original_count: chatMessages.length,
        compressed_count: compressEnd,
        kept_count: recent.length,
        cache_hit: true,
        compress_boundary: compressEnd,
        total_segments: completeWindows,
        segments_computed: 0,
      },
    };
  }

  // 慢路径：从后往前找最后一个缓存命中的段
  let lastCachedSeg = 0;
  let lastCachedSummary: string | null = null;

  for (let seg = completeWindows - 1; seg >= 1; seg--) {
    const key = buildSegmentCacheKey(namespace, seg, chatMessages, windowSize);
    const cached = await tryGetCache(env.DB, namespace, key);
    if (cached) {
      lastCachedSeg = seg;
      lastCachedSummary = cached;
      break;
    }
  }

  // 从 lastCachedSeg+1 开始级联压缩到 completeWindows
  summary = lastCachedSummary;
  cacheHit = false;

  for (let seg = lastCachedSeg + 1; seg <= completeWindows; seg++) {
    const segStart = (seg - 1) * windowSize;
    const segEnd = seg * windowSize;
    const segMessages = chatMessages.slice(segStart, segEnd);

    try {
      console.log(`[compress] calling model for segment ${seg}, model=${getCompressModel(env)}`);
      summary = await callCompressModel(env, segMessages, summary);
      console.log(`[compress] segment ${seg} compressed OK, summary length=${summary.length}`);
      segmentsComputed++;
    } catch (error) {
      console.error(`[compress] segment ${seg} failed:`, error);
      // 压缩失败 → 降级为不压缩
      return { ...noopResult, meta: { ...noopResult.meta, original_count: chatMessages.length, kept_count: chatMessages.length } };
    }

    // 写缓存（非阻塞）
    const key = buildSegmentCacheKey(namespace, seg, chatMessages, windowSize);
    try {
      await putCacheEntry(env.DB, {
        namespace,
        key,
        value: summary,
        tags: ["history_compression"],
        ttlSeconds: getCacheTtl(env),
      });
    } catch {
      // 写缓存失败不影响返回
    }
  }

  return {
    summary,
    messages: [...systemMessages, ...recent],
    meta: {
      original_count: chatMessages.length,
      compressed_count: compressEnd,
      kept_count: recent.length,
      cache_hit: cacheHit,
      compress_boundary: compressEnd,
      total_segments: completeWindows,
      segments_computed: segmentsComputed,
    },
  };
}

// ---------------------------------------------------------------------------
// 缓存辅助
// ---------------------------------------------------------------------------

function simpleHash(text: string): string {
  let hash = 5381;
  for (let i = 0; i < text.length; i++) {
    hash = ((hash << 5) + hash + text.charCodeAt(i)) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

/**
 * 段缓存 key：含段号 + 首条消息指纹。
 * 不同对话的同一段号内容不同，指纹不同，不会碰撞。
 * 同一对话的同一段，消息不变，指纹不变，缓存命中。
 */
function buildSegmentCacheKey(
  namespace: string,
  segNum: number,
  chatMessages: OpenAIChatMessage[],
  windowSize: number
): string {
  const segStart = (segNum - 1) * windowSize;
  const firstContent = contentToText(chatMessages[segStart]?.content).slice(0, 200);
  const fp = simpleHash(firstContent);
  return `compress:${namespace}:seg${segNum}_${fp}`;
}

async function tryGetCache(
  db: D1Database,
  namespace: string,
  key: string
): Promise<string | null> {
  try {
    const record = await getCacheEntry(db, { namespace, key });
    if (!record) return null;
    const value = parseCacheEntryValue(record);
    return typeof value === "string" && value.trim() ? value : null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// 压缩模型调用
// ---------------------------------------------------------------------------

function contentToText(content: OpenAIChatMessage["content"]): string {
  if (typeof content === "string") return content;
  if (content == null) return "";
  return (content as unknown[])
    .flatMap((part: unknown) => {
      if (!part || typeof part !== "object" || Array.isArray(part)) return [];
      const value = part as { type?: unknown; text?: unknown };
      return value.type === "text" && typeof value.text === "string" ? [value.text] : [];
    })
    .join("\n");
}

function formatMessagesForCompression(messages: OpenAIChatMessage[]): string {
  return messages
    .map(m => `${m.role === "user" ? "用户" : "助手"}：${contentToText(m.content)}`)
    .join("\n\n");
}

function extractWorkersAiModel(model: string): string | null {
  const normalized = model.trim();
  if (normalized.startsWith("workers-ai/")) return normalized.slice("workers-ai/".length);
  if (normalized.startsWith("worker/")) return normalized.slice("worker/".length);
  if (normalized.startsWith("@cf/")) return normalized;
  return null;
}

/**
 * 调小模型做压缩。
 *
 * 级联模式：
 *   - previousSummary = null → 首段，直接压缩消息
 *   - previousSummary 存在  → 后续段，合并旧摘要 + 新消息
 */
async function callCompressModel(
  env: Env,
  messages: OpenAIChatMessage[],
  previousSummary: string | null = null
): Promise<string> {
  const model = getCompressModel(env);
  const maxChars = getMaxChars(env);
  const formatted = formatMessagesForCompression(messages);

  const requirements = [
    `1. 用中文输出，目标 ${maxChars} 字以内`,
    "2. 保留：关键事实、做出的决定、未完成的话题、情绪状态变化、亲密互动的情感脉络",
    "3. 保留彼此使用的称呼和语气特征",
    "4. 只写摘要本身，不加任何解释、分析或元评论",
    "5. 写成可以直接续接对话的上下文，不是旁观者的总结报告",
    "6. 不要用学术化口吻解读对话，记录发生了什么、感受是什么",
    "7. 丢弃心跳探测消息（仅含 'ping' 或类似无实质内容的对话轮次），不将其纳入摘要",
  ].join("\n");

  let compressPrompt: string;

  if (previousSummary) {
    compressPrompt = [
      "你是对话压缩器。把旧摘要和新对话段合并成一份更新的摘要，让读到这份摘要的人能无缝接续对话。",
      "",
      "要求：",
      requirements,
      "",
      "旧摘要：",
      previousSummary,
      "",
      "新对话段：",
      formatted,
    ].join("\n");
  } else {
    compressPrompt = [
      "你是对话压缩器。把以下对话压缩成一份简洁摘要，让读到这份摘要的人能无缝接续对话。",
      "",
      "要求：",
      requirements,
      "",
      "对话：",
      formatted,
    ].join("\n");
  }

  const workersAiName = extractWorkersAiModel(model);

  if (workersAiName && env.AI) {
    try {
      const result = await (env.AI as any).run(workersAiName, {
        messages: [{ role: "user", content: compressPrompt }],
        max_tokens: 2000,
      });
      const text = (result as any)?.response ?? "";
      if (typeof text === "string" && text.trim()) return text.trim();
    } catch {
      // Workers AI 失败则 fallback 到 OpenAI compat
    }
  }

  const request = {
    model,
    messages: [{ role: "user", content: compressPrompt }],
    max_tokens: 2000,
    temperature: 0.3,
    stream: false,
  };

  const response = await callOpenAICompat(env, request as any);
  if (!response.ok) {
    throw new Error(`Compress model returned ${response.status}`);
  }

  const data = await response.json() as any;
  const text = data?.choices?.[0]?.message?.content;
  if (typeof text !== "string" || !text.trim()) {
    throw new Error("Compress model returned empty content");
  }

  return text.trim();
}
