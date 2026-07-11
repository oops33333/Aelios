/**
 * memories.ts — 对接 aeliosmemory（sweepy.cloud/aeliosmemory）的记忆适配层
 *
 * 2026-07-11 记忆分家：存储后端从 sweepy 主库切换到 aelios 专属库。
 * 服务端即原生 schema（importance 0-1 实数、tags JSON 数组、真实 status/pinned 列、
 * namespace 区分 sweepy-mirror 镜像与 default 自产），字段直通，
 * 旧版 SweepyRecord 映射补丁（tags 模拟 status、content>700 过滤、importance 量纲混用）全部移除。
 * 镜像行由服务端同步脚本独占管辖：对镜像行的更新会在下一轮全量同步被主库冲回。
 */

import type { Env, MemoryRecord } from "../types";
import { nowIso } from "../utils/time";

// ─── HTTP helpers ───

function memoryApiBase(env: Env): string {
  return (env.SWEEPY_URL || "https://sweepy.cloud/aeliosmemory").replace(/\/+$/, "");
}

function memoryApiHeaders(env: Env): HeadersInit {
  const auth = env.SWEEPY_AUTH || "";
  return {
    "content-type": "application/json",
    authorization: "Basic " + btoa(auth),
  };
}

async function memoryApiFetch(env: Env, path: string, init?: RequestInit): Promise<Response> {
  const base = memoryApiBase(env);
  return fetch(`${base}${path}`, {
    ...init,
    headers: { ...memoryApiHeaders(env), ...(init?.headers || {}) },
  });
}

// ─── 服务端记录（原生 API 形状）↔ MemoryRecord 转换 ───

interface ApiMemory {
  id: string;
  namespace: string;
  type: string;
  content: string;
  summary: string | null;
  importance: number;
  confidence: number;
  status: string;
  pinned: number;
  tags: string[];
  source: string | null;
  source_message_ids: string[];
  vector_id: string | null;
  last_recalled_at: string | null;
  recall_count: number;
  created_at: string;
  updated_at: string;
  expires_at: string | null;
  score?: number;
  similarity?: number;
}

function apiToRecord(m: ApiMemory): MemoryRecord {
  return {
    id: String(m.id),
    namespace: m.namespace || "default",
    type: m.type || "note",
    content: m.content,
    summary: m.summary ?? null,
    importance: typeof m.importance === "number" ? m.importance : 0.5,
    confidence: typeof m.confidence === "number" ? m.confidence : 0.8,
    status: m.status || "active",
    pinned: m.pinned ? 1 : 0,
    tags: JSON.stringify(Array.isArray(m.tags) ? m.tags : []),
    source: m.source ?? null,
    source_message_ids: JSON.stringify(Array.isArray(m.source_message_ids) ? m.source_message_ids : []),
    vector_id: m.vector_id ?? null,
    last_recalled_at: m.last_recalled_at || null,
    recall_count: m.recall_count ?? 0,
    created_at: m.created_at,
    updated_at: m.updated_at,
    expires_at: m.expires_at ?? null,
  };
}

// ─── 公开接口（签名与旧版一致，调用方零改动）───

export interface CreateMemoryInput {
  namespace: string;
  type: string;
  content: string;
  summary?: string | null;
  importance?: number;
  confidence?: number;
  status?: string;
  pinned?: boolean;
  tags?: string[];
  source?: string | null;
  sourceMessageIds?: string[];
  expiresAt?: string | null;
}

export interface ListMemoryFilters {
  namespace: string;
  type?: string;
  status?: string;
  limit: number;
  offset?: number;
}

export interface ListMemoryPage {
  records: MemoryRecord[];
  hasMore: boolean;
  nextOffset: number | null;
}

export interface UpdateMemoryInput {
  type?: string;
  content?: string;
  summary?: string | null;
  importance?: number;
  confidence?: number;
  status?: string;
  pinned?: boolean;
  tags?: string[];
  sourceMessageIds?: string[];
  expiresAt?: string | null;
}

/**
 * 写入一条记忆（namespace='default' 自产区；embedding 由服务端异步生成）
 */
export async function createMemory(env: Env, input: CreateMemoryInput): Promise<MemoryRecord> {
  const body = {
    namespace: input.namespace || "default",
    type: input.type || "note",
    content: input.content,
    summary: input.summary ?? null,
    importance: input.importance ?? 0.5,
    confidence: input.confidence ?? 0.8,
    status: input.status || "active",
    pinned: input.pinned === true,
    tags: input.tags ?? [],
    source: input.source || "aelios",
    source_message_ids: input.sourceMessageIds ?? [],
    expires_at: input.expiresAt ?? null,
  };

  const res = await memoryApiFetch(env, "/api/memories", {
    method: "POST",
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    console.error("aeliosmemory createMemory failed", res.status, await res.text());
    // 返回一个占位记录，不阻断上游流程
    const now = nowIso();
    return {
      id: `err_${Date.now()}`,
      namespace: input.namespace,
      type: input.type,
      content: input.content,
      summary: input.summary ?? null,
      importance: input.importance ?? 0.5,
      confidence: input.confidence ?? 0.8,
      status: "active",
      pinned: input.pinned ? 1 : 0,
      tags: JSON.stringify(input.tags ?? []),
      source: input.source ?? null,
      source_message_ids: "[]",
      vector_id: null,
      last_recalled_at: null,
      recall_count: 0,
      created_at: now,
      updated_at: now,
      expires_at: null,
    };
  }

  const data = (await res.json()) as { ok: boolean; memory: ApiMemory };
  return apiToRecord(data.memory);
}

/**
 * 列出记忆（服务端排序：pinned DESC, importance DESC, created_at DESC；默认仅 active）
 * 注意不传 namespace 过滤——aelios 需要同时看见镜像区与自产区。
 */
export async function listMemoriesPage(env: Env, filters: ListMemoryFilters): Promise<ListMemoryPage> {
  const offset = Math.max(filters.offset ?? 0, 0);
  const limit = Math.max(filters.limit, 1);

  const params: string[] = [`limit=${offset + limit + 1}`];
  if (filters.type) params.push(`type=${encodeURIComponent(filters.type)}`);
  if (filters.status) params.push(`status=${encodeURIComponent(filters.status)}`);

  const res = await memoryApiFetch(env, `/api/memories?${params.join("&")}`);

  if (!res.ok) {
    console.error("aeliosmemory listMemories failed", res.status);
    return { records: [], hasMore: false, nextOffset: null };
  }

  const all = (await res.json()) as ApiMemory[];
  const sliced = all.slice(offset, offset + limit + 1);
  const records = sliced.slice(0, limit).map(apiToRecord);

  return {
    records,
    hasMore: sliced.length > limit,
    nextOffset: sliced.length > limit ? offset + limit : null,
  };
}

export async function listMemories(env: Env, filters: ListMemoryFilters): Promise<MemoryRecord[]> {
  const page = await listMemoriesPage(env, filters);
  return page.records;
}

/**
 * 按 ID 读取单条记忆（id 为 TEXT：am- / sweepy- / legacy- 前缀）
 */
export async function getMemoryById(
  env: Env,
  input: { namespace: string; id: string }
): Promise<MemoryRecord | null> {
  const res = await memoryApiFetch(env, `/api/memories/${encodeURIComponent(input.id)}`);
  if (!res.ok) return null;

  const record = (await res.json()) as ApiMemory;
  return apiToRecord(record);
}

/**
 * 批量按 ID 读取（服务端无批量接口，逐条读取）
 */
export async function fetchMemoriesByIds(
  env: Env,
  input: { namespace: string; ids: string[] }
): Promise<MemoryRecord[]> {
  if (input.ids.length === 0) return [];

  const results = await Promise.allSettled(
    input.ids.map((id) => getMemoryById(env, { namespace: input.namespace, id }))
  );

  return results
    .filter((r): r is PromiseFulfilledResult<MemoryRecord | null> => r.status === "fulfilled")
    .map((r) => r.value)
    .filter((r): r is MemoryRecord => r !== null);
}

/**
 * 更新记忆（原生字段直通；status 为真实列。
 * 若目标是 sweepy-mirror 镜像行，更新会生效但将在下一轮同步被主库冲回。）
 */
export async function updateMemory(
  env: Env,
  input: { namespace: string; id: string; patch: UpdateMemoryInput }
): Promise<MemoryRecord | null> {
  const body: Record<string, unknown> = {};

  if (input.patch.content !== undefined) body.content = input.patch.content;
  if (input.patch.type !== undefined) body.type = input.patch.type;
  if (input.patch.summary !== undefined) body.summary = input.patch.summary;
  if (input.patch.importance !== undefined) body.importance = input.patch.importance;
  if (input.patch.confidence !== undefined) body.confidence = input.patch.confidence;
  if (input.patch.status !== undefined) body.status = input.patch.status;
  if (input.patch.pinned !== undefined) body.pinned = input.patch.pinned;
  if (input.patch.tags !== undefined) body.tags = input.patch.tags;
  if (input.patch.expiresAt !== undefined) body.expires_at = input.patch.expiresAt;

  if (Object.keys(body).length === 0) {
    return getMemoryById(env, input);
  }

  const res = await memoryApiFetch(env, `/api/memories/${encodeURIComponent(input.id)}`, {
    method: "PUT",
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    console.error("aeliosmemory updateMemory failed", res.status, await res.text());
    return getMemoryById(env, input);
  }

  const updated = (await res.json()) as ApiMemory;
  return apiToRecord(updated);
}

/**
 * 软删除（status='deleted'，服务端真实列）
 */
export async function softDeleteMemory(
  env: Env,
  input: { namespace: string; id: string }
): Promise<MemoryRecord | null> {
  return updateMemory(env, {
    namespace: input.namespace,
    id: input.id,
    patch: { status: "deleted" },
  });
}

/**
 * 语义检索 — 服务端 qwen3-embedding-8b 余弦（阈值 0.4），命中自动累加 recall_count
 */
export async function searchMemoriesByText(
  env: Env,
  input: { namespace: string; query: string; types?: string[]; limit: number }
): Promise<Array<MemoryRecord & { score: number }>> {
  const query = input.query.trim().slice(0, 500);
  if (!query) return [];

  const res = await memoryApiFetch(
    env,
    `/api/memories/search?q=${encodeURIComponent(query)}&limit=${Math.max(input.limit, 1)}`
  );

  if (!res.ok) {
    console.error("aeliosmemory search failed", res.status);
    return [];
  }

  const results = (await res.json()) as ApiMemory[];

  return results.slice(0, input.limit).map((r) => ({
    ...apiToRecord(r),
    score: r.score ?? r.similarity ?? 0.7,
  }));
}

/**
 * 标记记忆被召回（服务端 search/单读已自增 recall_count，此处无事可做）
 */
export async function markMemoriesRecalled(
  _env: Env,
  _input: { namespace: string; ids: string[] }
): Promise<void> {
  // aeliosmemory 在 search 与单读时自增 recall_count，无需额外调用
}
