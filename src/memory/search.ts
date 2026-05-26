/**
 * search.ts — 记忆搜索层，改为直接使用 sweepy 向量搜索
 *
 * 原版先走 Vectorize 再回退 D1 文本搜索。
 * 现在直接调 sweepy 的 /api/memories/search，
 * 由 sweepy 的 qwen3-embedding 提供语义匹配。
 */

import { fetchMemoriesByIds, markMemoriesRecalled, searchMemoriesByText } from "../db/memories";
import type { Env, MemoryApiRecord, MemoryRecord } from "../types";

function parseJsonArray(value: string | null): string[] {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value) as unknown;
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string") : [];
  } catch {
    return [];
  }
}

export function toMemoryApiRecord(record: MemoryRecord, score?: number): MemoryApiRecord {
  return {
    id: record.id,
    namespace: record.namespace,
    type: record.type,
    content: record.content,
    summary: record.summary,
    importance: record.importance,
    confidence: record.confidence,
    status: record.status,
    pinned: Boolean(record.pinned),
    tags: parseJsonArray(record.tags),
    source: record.source,
    source_message_ids: parseJsonArray(record.source_message_ids),
    vector_id: record.vector_id,
    last_recalled_at: record.last_recalled_at,
    recall_count: record.recall_count,
    created_at: record.created_at,
    updated_at: record.updated_at,
    expires_at: record.expires_at,
    ...(score === undefined ? {} : { score }),
  };
}

function getTopK(env: Env, requested?: number): number {
  const fallback = Number(env.MEMORY_TOP_K || 12);
  const value = requested || fallback;
  return Math.min(Math.max(value, 1), 200);
}

/**
 * 搜索记忆 — 直接走 sweepy 语义搜索
 *
 * sweepy 的 /api/memories/search 底层是 qwen3-embedding-8b 向量搜索，
 * 余弦相似度阈值 0.4，比 Aelios 原版的 Cloudflare Vectorize 更适合中文语境。
 */
export async function searchMemories(
  env: Env,
  input: { namespace: string; query: string; types?: string[]; topK?: number }
): Promise<MemoryApiRecord[]> {
  const topK = getTopK(env, input.topK);

  // 直接调 sweepy 搜索（已在 db/memories.ts 里封装）
  const records = await searchMemoriesByText(env, {
    namespace: input.namespace,
    query: input.query,
    types: input.types,
    limit: topK,
  });

  // 标记被召回（sweepy 内部处理）
  await markMemoriesRecalled(env, {
    namespace: input.namespace,
    ids: records.map((record) => record.id),
  });

  return records.map((record) => toMemoryApiRecord(record, record.score));
}
