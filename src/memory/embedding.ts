/**
 * embedding.ts — sweepy 适配版
 *
 * sweepy 在写入记忆时自动生成 embedding（qwen3-embedding-8b），
 * 不需要 Aelios 侧额外调用 Vectorize 或 Workers AI。
 * 保留函数签名以兼容上游调用，但实现为空操作。
 */

import type { Env, MemoryRecord } from "../types";

/**
 * createEmbedding — 不再需要，sweepy 自动处理
 * 保留供 inject.ts 的 searchVectorMemories 使用（如果还有残留调用）
 */
export async function createEmbedding(_env: Env, _text: string): Promise<number[] | null> {
  return null;
}

/**
 * upsertMemoryEmbedding — 空操作
 * sweepy 在 POST /api/memories 时自动生成 embedding
 */
export async function upsertMemoryEmbedding(_env: Env, _memory: MemoryRecord): Promise<boolean> {
  return true;
}

/**
 * deleteMemoryEmbedding — 空操作
 * sweepy 在删除记忆时自动清理 embedding
 */
export async function deleteMemoryEmbedding(_env: Env, _memory: MemoryRecord): Promise<boolean> {
  return true;
}
