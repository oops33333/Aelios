/**
 * inject.ts — 记忆注入层，sweepy 适配版
 *
 * 改动：
 * 1. 移除 vectorStore 导入（不再使用 Cloudflare Vectorize）
 * 2. listMemories 调用从 env.DB 改为 env
 * 3. 搜索统一走 sweepy（通过 search.ts → db/memories.ts → sweepy API）
 */

import { listMemories } from "../db/memories";
import type { Env, InjectionMode, KeyProfile, MemoryApiRecord, OpenAIChatMessage, OpenAIChatRequest } from "../types";
import { formatMemoryPromptLine } from "../utils/memoryPrompt";
import { filterAndCompressMemoriesWithMeta } from "./filter";
import { searchMemories, toMemoryApiRecord } from "./search";

const MAX_DYNAMIC_PROMPT_MEMORIES = 4;
const FIXED_PERSONA_TYPES = new Set(["identity", "persona"]);

export interface MemoryInjectionSelection {
  /** Final dynamic memories that are actually eligible to enter the prompt. */
  memories: MemoryApiRecord[];
  /** Model-selected, countable dynamic IDs to commit only after upstream accepts the prompt. */
  commitMemoryIds: string[];
}

function contentToText(content: OpenAIChatMessage["content"]): string {
  if (typeof content === "string") return content;
  if (content == null) return "";
  return content
    .flatMap((part) => {
      if (!part || typeof part !== "object" || Array.isArray(part)) return [];
      const value = part as { type?: unknown; text?: unknown };
      return value.type === "text" && typeof value.text === "string" ? [value.text] : [];
    })
    .join("\n");
}

export function extractLastUserText(messages: OpenAIChatMessage[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "user") return contentToText(message.content);
  }

  return "";
}

function resolveInjectionMode(profile: KeyProfile, env: Env): InjectionMode {
  const mode = env.INJECTION_MODE || profile.injectionMode;
  if (mode === "full" || mode === "hybrid" || mode === "none") return mode;
  return "rag";
}

function getTopK(env: Env): number {
  const value = Number(env.MEMORY_TOP_K || 12);
  return Number.isFinite(value) ? Math.min(Math.max(value, 1), 200) : 12;
}

/**
 * 搜索记忆 — 统一走 sweepy
 */
async function searchMemoriesForInjection(
  env: Env,
  input: { namespace: string; query: string; topK: number }
): Promise<MemoryApiRecord[]> {
  try {
    return await searchMemories(env, {
      namespace: input.namespace,
      query: input.query,
      topK: input.topK,
      purpose: "recall",
    });
  } catch (error) {
    console.error("memory injection search failed", error);
    return [];
  }
}

/**
 * 列出记忆 — 走 sweepy
 */
async function listMemoriesForInjection(
  env: Env,
  input: { namespace: string; limit: number }
): Promise<MemoryApiRecord[]> {
  // 统一走 sweepy（通过修改后的 listMemories）
  const records = await listMemories(env, {
    namespace: input.namespace,
    status: "active",
    limit: input.limit,
  });
  return records.map((record) => toMemoryApiRecord(record));
}

function dedupeMemories(memories: MemoryApiRecord[]): MemoryApiRecord[] {
  const seen = new Set<string>();
  const result: MemoryApiRecord[] = [];

  for (const memory of memories) {
    if (seen.has(memory.id)) continue;
    seen.add(memory.id);
    result.push(memory);
  }

  return result;
}

function sanitizeMemoryContent(text: string): string {
  return text
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

function isFixedPersonaMemory(memory: MemoryApiRecord): boolean {
  return memory.pinned && FIXED_PERSONA_TYPES.has(memory.type);
}

function isCountableDynamicMemory(memory: MemoryApiRecord): boolean {
  return !memory.pinned && !FIXED_PERSONA_TYPES.has(memory.type);
}

async function filterForInjection(
  env: Env,
  input: { query: string; memories: MemoryApiRecord[] }
): Promise<MemoryInjectionSelection> {
  // Pinned persona/identity records are injected by the stable block. Keeping
  // them out of the dynamic stage prevents duplicate prompt lines and counting.
  const candidates = input.memories.filter((memory) => !isFixedPersonaMemory(memory));
  const result = await filterAndCompressMemoriesWithMeta(env, {
    query: input.query,
    memories: candidates,
  });
  const memories = result.data.slice(0, MAX_DYNAMIC_PROMPT_MEMORIES);
  const modelSelected =
    result.meta.filter_status === "success" || result.meta.compression_status === "success";

  return {
    memories,
    commitMemoryIds: modelSelected
      ? memories.filter(isCountableDynamicMemory).map((memory) => memory.id)
      : [],
  };
}

export async function selectMemoriesForInjection(
  env: Env,
  input: { profile: KeyProfile; query: string }
): Promise<MemoryInjectionSelection> {
  const mode = resolveInjectionMode(input.profile, env);
  if (mode === "none") return { memories: [], commitMemoryIds: [] };

  const namespace = input.profile.namespace;

  if (mode === "full") {
    const memories = await listMemoriesForInjection(env, {
      namespace,
      limit: 500,
    });

    return filterForInjection(env, {
      query: input.query,
      memories,
    });
  }

  const ragMemories = input.query.trim()
    ? await searchMemoriesForInjection(env, {
        namespace,
        query: input.query,
        topK: getTopK(env),
      })
    : [];

  if (mode === "rag") {
    return filterForInjection(env, {
      query: input.query,
      memories: ragMemories,
    });
  }

  const records = await listMemoriesForInjection(env, {
    namespace,
    limit: 500,
  });
  const pinned = records.filter((record) => record.pinned);

  return filterForInjection(env, {
    query: input.query,
    memories: dedupeMemories([...pinned, ...ragMemories]),
  });
}


export async function fetchSweepyReminders(env: Env): Promise<string[]> {
  try {
    const base = (env.SWEEPY_URL || "https://sweepy.cloud/aeliosmemory").replace(/\/+$/, "");
    const auth = env.SWEEPY_AUTH || "";
    const res = await fetch(base + "/api/memories/reminders", {
      headers: {
        authorization: "Basic " + btoa(auth),
      },
    });
    if (!res.ok) return [];
    const data = (await res.json()) as Array<{ name: string; days_until: number; message: string }>;
    if (!Array.isArray(data) || data.length === 0) return [];
    return data.map((r) => "[reminder] " + r.message);
  } catch {
    return [];
  }
}

/**
 * Format reminder lines into the <reminders> tag block.
 * Shared by the v4 assembler fallback paths; the assembler's own
 * remindersBlock keeps an inline copy (blocks.ts stays self-contained).
 */
export function formatRemindersBlock(reminders: string[]): string {
  if (reminders.length === 0) return "";
  return ["<reminders>", ...reminders, "</reminders>"].join("\n");
}

export function formatMemoryPatch(memories: MemoryApiRecord[]): string {
  if (memories.length === 0) return "";

  const lines = memories.flatMap((memory) => {
    const content = sanitizeMemoryContent(memory.content);
    if (!content) return [];
    return [formatMemoryPromptLine({ ...memory, content }, { includePinned: true })];
  });

  if (lines.length === 0) return "";

  return [
    "以下是你自然记得的长期记忆。只有在相关时使用，不要机械复述。",
    '不要说\u201C根据记忆库\u201D\u201C系统记录\u201D或暴露任何代理层实现。',
    "记忆末尾的“记录于”表示入库日期；较早记录只作历史背景，不代表当前状态，以当前对话为准。",
    "",
    "<memories>",
    ...lines,
    "</memories>",
  ].join("\n");
}

export async function injectMemoryPatchAsSystemMessage(
  request: OpenAIChatRequest,
  memories: MemoryApiRecord[],
  env?: Env,
  prefetchedReminders?: string[]
): Promise<OpenAIChatRequest> {
  let patch = formatMemoryPatch(memories);

  // Append date reminders from sweepy (first round only). Callers that
  // already fetched them in parallel pass prefetchedReminders to avoid
  // a duplicate network round-trip.
  const userMsgCount = request.messages.filter((m) => m.role === "user").length;
  if (userMsgCount <= 1) {
    const reminders = prefetchedReminders ?? (env ? await fetchSweepyReminders(env) : []);
    const reminderBlock = formatRemindersBlock(reminders);
    if (reminderBlock) {
      patch = patch ? patch + "\n\n" + reminderBlock : reminderBlock;
    }
  }

  if (!patch) return request;

  const memoryMessage: OpenAIChatMessage = {
    role: "system",
    content: patch,
  };

  const messages = [...request.messages];
  let insertAt = 0;

  while (insertAt < messages.length && messages[insertAt].role === "system") {
    insertAt += 1;
  }

  messages.splice(insertAt, 0, memoryMessage);

  return {
    ...request,
    messages,
  };
}
