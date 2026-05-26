/**
 * memories.ts — 对接 sweepy.cloud 的记忆适配层
 *
 * 原版通过 D1 SQL 读写，现改为 HTTP 调用 sweepy API。
 * 函数签名从 (db: D1Database, ...) 改为 (env: Env, ...)，
 * 调用方需将 env.DB 改为 env。
 */

import type { Env, MemoryRecord } from "../types";
import { nowIso } from "../utils/time";

// ─── sweepy HTTP helpers ───

function sweepyUrl(env: Env): string {
  return (env.SWEEPY_URL || "https://sweepy.cloud").replace(/\/+$/, "");
}

function sweepyHeaders(env: Env): HeadersInit {
  const auth = env.SWEEPY_AUTH || "";
  return {
    "content-type": "application/json",
    authorization: "Basic " + btoa(auth),
  };
}

async function sweepyFetch(env: Env, path: string, init?: RequestInit): Promise<Response> {
  const base = sweepyUrl(env);
  return fetch(`${base}${path}`, {
    ...init,
    headers: { ...sweepyHeaders(env), ...(init?.headers || {}) },
  });
}

// ─── sweepy record ↔ MemoryRecord 转换 ───

interface SweepyRecord {
  id: number;
  content: string;
  category: string;
  tags: string;
  source: string | null;
  created_at: string;
  updated_at: string;
  image_path: string | null;
  valence: number;
  arousal: number;
  importance: number;
  activation_count: number;
  last_active: string | null;
  resolved: number;
}

function sweepyToMemoryRecord(s: SweepyRecord, namespace: string): MemoryRecord {
  const tags = s.tags
    ? JSON.stringify(s.tags.split(",").map((t: string) => t.trim()).filter(Boolean))
    : "[]";

  const isPinned = s.tags?.includes("pinned") || s.tags?.includes("必读") ? 1 : 0;

  return {
    id: String(s.id),
    namespace,
    type: s.category || "note",
    content: s.content,
    summary: null,
    importance: s.importance ?? 0.5,
    confidence: 0.8,
    status: "active",
    pinned: isPinned,
    tags,
    source: s.source,
    source_message_ids: "[]",
    vector_id: `sweepy_${s.id}`,
    last_recalled_at: s.last_active,
    recall_count: s.activation_count || 0,
    created_at: s.created_at,
    updated_at: s.updated_at,
    expires_at: null,
  };
}

// ─── 公开接口 ───

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
 * 写入一条记忆到 sweepy
 */
export async function createMemory(env: Env, input: CreateMemoryInput): Promise<MemoryRecord> {
  const tagsArray = input.tags ?? [];
  if (input.pinned && !tagsArray.includes("pinned")) tagsArray.push("pinned");

  const body = {
    content: input.content,
    category: input.type || "note",
    tags: ["aelios", ...tagsArray].join(","),
    source: input.source || "aelios",
    importance: input.importance ?? 0.5,
    valence: 0,
    arousal: 0.3,
  };

  const res = await sweepyFetch(env, "/api/memories", {
    method: "POST",
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    console.error("sweepy createMemory failed", res.status, await res.text());
    // 返回一个占位记录，不阻断上游流程
    const now = nowIso();
    return {
      id: `err_${Date.now()}`,
      namespace: input.namespace,
      type: input.type,
      content: input.content,
      summary: null,
      importance: input.importance ?? 0.5,
      confidence: 0.8,
      status: "active",
      pinned: input.pinned ? 1 : 0,
      tags: JSON.stringify(tagsArray),
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

  const created = (await res.json()) as SweepyRecord;
  return sweepyToMemoryRecord(created, input.namespace);
}

/**
 * 列出记忆（按 importance 降序）
 */
export async function listMemoriesPage(env: Env, filters: ListMemoryFilters): Promise<ListMemoryPage> {
  let path = "/api/memories?";
  const params: string[] = [];

  if (filters.type) params.push(`keyword=${encodeURIComponent(filters.type)}`);

  const res = await sweepyFetch(env, `/api/memories${params.length ? "?" + params.join("&") : ""}`);

  if (!res.ok) {
    console.error("sweepy listMemories failed", res.status);
    return { records: [], hasMore: false, nextOffset: null };
  }

  const all = (await res.json()) as SweepyRecord[];
  const offset = Math.max(filters.offset ?? 0, 0);
  const limit = Math.max(filters.limit, 1);

  // 过滤状态（sweepy 没有 status 字段，用 tags 模拟）
  let filtered = all;
  if (filters.status === "active") {
    filtered = all.filter((r) => !r.tags?.includes("deleted") && !r.tags?.includes("superseded"));
  }

  const sliced = filtered.slice(offset, offset + limit + 1);
  const records = sliced.slice(0, limit).map((r) => sweepyToMemoryRecord(r, filters.namespace));

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
 * 按 ID 读取单条记忆
 */
export async function getMemoryById(
  env: Env,
  input: { namespace: string; id: string }
): Promise<MemoryRecord | null> {
  const res = await sweepyFetch(env, `/api/memories/${input.id}`);
  if (!res.ok) return null;

  const record = (await res.json()) as SweepyRecord;
  return sweepyToMemoryRecord(record, input.namespace);
}

/**
 * 批量按 ID 读取
 */
export async function fetchMemoriesByIds(
  env: Env,
  input: { namespace: string; ids: string[] }
): Promise<MemoryRecord[]> {
  if (input.ids.length === 0) return [];

  // sweepy 没有批量读取接口，逐条读取
  const results = await Promise.allSettled(
    input.ids.map((id) => getMemoryById(env, { namespace: input.namespace, id }))
  );

  return results
    .filter((r): r is PromiseFulfilledResult<MemoryRecord | null> => r.status === "fulfilled")
    .map((r) => r.value)
    .filter((r): r is MemoryRecord => r !== null);
}

/**
 * 更新记忆
 */
export async function updateMemory(
  env: Env,
  input: { namespace: string; id: string; patch: UpdateMemoryInput }
): Promise<MemoryRecord | null> {
  const body: Record<string, unknown> = {};

  if (input.patch.content !== undefined) body.content = input.patch.content;
  if (input.patch.type !== undefined) body.category = input.patch.type;
  if (input.patch.importance !== undefined) body.importance = input.patch.importance;
  if (input.patch.tags !== undefined) body.tags = input.patch.tags.join(",");
  if (input.patch.status !== undefined) {
    // sweepy 没有 status 字段，用 tags 标记
    const currentTags = typeof body.tags === "string" ? body.tags : "";
    if (input.patch.status === "deleted" || input.patch.status === "superseded") {
      body.tags = currentTags ? `${currentTags},${input.patch.status}` : input.patch.status;
    }
  }

  if (Object.keys(body).length === 0) {
    return getMemoryById(env, input);
  }

  const res = await sweepyFetch(env, `/api/memories/${input.id}`, {
    method: "PUT",
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    console.error("sweepy updateMemory failed", res.status, await res.text());
    return getMemoryById(env, input);
  }

  const updated = (await res.json()) as SweepyRecord;
  return sweepyToMemoryRecord(updated, input.namespace);
}

/**
 * 软删除
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
 * 文本搜索记忆 — 调用 sweepy 的语义向量搜索
 */
export async function searchMemoriesByText(
  env: Env,
  input: { namespace: string; query: string; types?: string[]; limit: number }
): Promise<Array<MemoryRecord & { score: number }>> {
  const query = input.query.trim().slice(0, 500);
  if (!query) return [];

  const encodedQuery = encodeURIComponent(query);
  const res = await sweepyFetch(env, `/api/memories/search?q=${encodedQuery}`);

  if (!res.ok) {
    console.error("sweepy search failed", res.status);
    return [];
  }

  const results = (await res.json()) as Array<SweepyRecord & { similarity?: number }>;

  return results
    .filter((r) => !r.tags?.includes("deleted") && !r.tags?.includes("superseded"))
    .slice(0, input.limit)
    .map((r) => ({
      ...sweepyToMemoryRecord(r, input.namespace),
      score: r.similarity ?? 0.7,
    }));
}

/**
 * 标记记忆被召回（sweepy 没有专用接口，跳过）
 */
export async function markMemoriesRecalled(
  _env: Env,
  _input: { namespace: string; ids: string[] }
): Promise<void> {
  // sweepy 有自己的 activation_count 机制，这里不额外处理
}
