/**
 * compress.ts — 对话历史自动压缩
 *
 * 当 RikkaHub 发来的完整历史超过阈值（默认 30 轮）时，
 * 把旧消息压缩成摘要，只保留最近 N 轮原文。
 *
 * 分段压缩策略：按 windowSize（= threshold - keepRecent）条消息为一段，
 * 只在完成一个完整段时触发压缩。段内多次请求复用 D1 缓存。
 *
 * 与记忆搜索并行执行（Promise.all），不增加总延迟。
 */

import { getCacheEntry, putCacheEntry, parseCacheEntryValue } from "../db/cacheEntries";
import { callOpenAICompat } from "../proxy/openaiAdapter";
import type { Env, OpenAIChatMessage } from "../types";

// ---------------------------------------------------------------------------
// 配置读取
// ---------------------------------------------------------------------------

const DEFAULT_THRESHOLD = 30;
const DEFAULT_KEEP_RECENT = 5;
const DEFAULT_COMPRESS_MAX_CHARS = 800;
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
  /** 压缩摘要文本，null 表示无需压缩 */
  summary: string | null;
  /** 传给 assembler 的 messages（保留 system + 最近 user/assistant） */
  messages: OpenAIChatMessage[];
  /** 压缩统计元数据 */
  meta: {
    original_count: number;
    compressed_count: number;
    kept_count: number;
    cache_hit: boolean;
    compress_boundary: number;
  };
}

/**
 * 主入口：检查是否需要压缩，如需要则执行压缩并返回裁剪后的消息。
 *
 * 分段策略：
 *   windowSize = threshold - keepRecent (默认 25)
 *   compressEnd = floor((chatCount - keepRecent) / windowSize) * windowSize
 *   只在 compressEnd >= windowSize 时触发。
 *
 * 这样在同一个段内（约 25 轮请求），缓存持续命中。
 */
export async function compressHistoryIfNeeded(
  env: Env,
  messages: OpenAIChatMessage[],
  namespace: string
): Promise<CompressResult> {
  const noopResult: CompressResult = {
    summary: null,
    messages,
    meta: { original_count: 0, compressed_count: 0, kept_count: 0, cache_hit: false, compress_boundary: 0 },
  };

  if (!isEnabled(env)) return noopResult;

  const systemMessages = messages.filter(m => m.role === "system");
  const chatMessages = messages.filter(m => m.role === "user" || m.role === "assistant");

  noopResult.meta.original_count = chatMessages.length;
  noopResult.meta.kept_count = chatMessages.length;

  const threshold = getThreshold(env);
  const keepRecent = getKeepRecent(env);

  if (chatMessages.length <= threshold) return noopResult;

  const windowSize = threshold - keepRecent;
  const completeWindows = Math.floor((chatMessages.length - keepRecent) / windowSize);
  const compressEnd = completeWindows * windowSize;

  if (compressEnd < windowSize) return noopResult;

  const toCompress = chatMessages.slice(0, compressEnd);
  const recent = chatMessages.slice(compressEnd);

  // D1 缓存查找
  const cacheKey = `compress:${namespace}:boundary_${compressEnd}`;
  let summary: string | null = null;
  let cacheHit = false;

  try {
    const cached = await getCacheEntry(env.DB, { namespace, key: cacheKey });
    if (cached) {
      const value = parseCacheEntryValue(cached);
      if (typeof value === "string" && value.trim()) {
        summary = value;
        cacheHit = true;
      }
    }
  } catch {
    // 缓存读取失败不阻断主流程
  }

  if (!summary) {
    try {
      summary = await callCompressModel(env, toCompress);
    } catch (error) {
      // 压缩失败 → 降级为不压缩，原样通过
      console.error("[compress] model call failed:", error);
      return noopResult;
    }

    // 异步写缓存
    try {
      await putCacheEntry(env.DB, {
        namespace,
        key: cacheKey,
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
      compressed_count: toCompress.length,
      kept_count: recent.length,
      cache_hit: cacheHit,
      compress_boundary: compressEnd,
    },
  };
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

async function callCompressModel(env: Env, messages: OpenAIChatMessage[]): Promise<string> {
  const model = getCompressModel(env);
  const maxChars = getMaxChars(env);
  const formatted = formatMessagesForCompression(messages);

  const compressPrompt = [
    "你是对话压缩助手。将以下对话历史压缩为简明摘要。",
    "",
    "要求：",
    "1. 保留关键事实、决策、重要上下文",
    "2. 保留情感状态变化和关系动态",
    "3. 保留未完成的话题线索",
    "4. 丢弃重复内容、寒暄、已解决的问题",
    `5. 输出控制在${maxChars}字以内`,
    "6. 直接输出摘要文本，不要加任何前缀或标记",
    "",
    "对话历史：",
    formatted,
  ].join("\n");

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
