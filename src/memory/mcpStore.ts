import type { Env } from "../types";

export interface McpMemoryInput {
  content: string;
  type?: string;
  summary?: string | null;
  importance?: number;
  confidence?: number;
  pinned?: boolean;
  tags?: string[];
  expires_at?: string | null;
}

export type McpMemoryPatch = Partial<McpMemoryInput>;

export interface McpMemoryView extends McpMemoryInput {
  id: string;
  namespace: string;
  status: string;
  read_only: boolean;
  score?: number;
  raw_similarity?: number;
}

export type McpStoreErrorCode =
  | "INVALID_ARGUMENT"
  | "NOT_CONFIGURED"
  | "NOT_FOUND"
  | "READ_ONLY"
  | "BACKEND_HTTP"
  | "BACKEND_UNAVAILABLE"
  | "BACKEND_TIMEOUT"
  | "INVALID_RESPONSE";

export class McpStoreError extends Error {
  constructor(
    public readonly code: McpStoreErrorCode,
    message: string,
    public readonly outcomeUnknown = false,
    public readonly status?: number
  ) {
    super(message);
    this.name = "McpStoreError";
  }
}

const TIMEOUT_MS = 30_000;
const EDITABLE_FIELDS = new Set([
  "content", "type", "summary", "importance", "confidence",
  "pinned", "tags", "expires_at"
]);

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function invalidArgument(): never {
  throw new McpStoreError("INVALID_ARGUMENT", "参数缺失、类型错误或包含不支持的字段。");
}

function invalidResponse(write = false): never {
  throw new McpStoreError(
    "INVALID_RESPONSE",
    write ? "记忆服务响应异常，写入结果未确认；请先搜索核对，不要直接重复创建。" : "记忆服务响应格式异常。",
    write
  );
}

function nonempty(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function checkObject(value: unknown, allowed: ReadonlySet<string>): Record<string, unknown> {
  if (!isObject(value) || Object.keys(value).some((key) => !allowed.has(key))) {
    invalidArgument();
  }
  return value;
}

function validatePatch(value: unknown, creating: boolean): McpMemoryPatch {
  const input = checkObject(value, EDITABLE_FIELDS);
  const output: McpMemoryPatch = {};
  for (const [key, item] of Object.entries(input)) {
    if (key === "content" || key === "type") {
      if (!nonempty(item)) invalidArgument();
      output[key] = item;
    } else if (key === "summary") {
      if (item !== null && typeof item !== "string") invalidArgument();
      output.summary = item;
    } else if (key === "importance" || key === "confidence") {
      if (typeof item !== "number" || !Number.isFinite(item) || item < 0 || item > 1) {
        invalidArgument();
      }
      output[key] = item;
    } else if (key === "pinned") {
      if (typeof item !== "boolean") invalidArgument();
      output.pinned = item;
    } else if (key === "tags") {
      if (!Array.isArray(item) || !item.every((tag) => typeof tag === "string")) {
        invalidArgument();
      }
      output.tags = item.map((tag: string) => tag.trim()).filter(Boolean);
    } else if (key === "expires_at") {
      if (item !== null && (!nonempty(item) || !Number.isFinite(Date.parse(item)))) {
        invalidArgument();
      }
      output.expires_at = item;
    }
  }
  if (Object.keys(output).length === 0 || (creating && !nonempty(output.content))) {
    invalidArgument();
  }
  return output;
}

function isWritable(record: { namespace: string; id: string }): boolean {
  return record.namespace === "default" && !record.id.startsWith("sweepy-");
}

function toView(value: unknown, write = false): McpMemoryView {
  if (!isObject(value) || !nonempty(value.id) || !nonempty(value.namespace) ||
      !nonempty(value.status) || !nonempty(value.type) || !nonempty(value.content)) {
    invalidResponse(write);
  }
  if (value.pinned !== true && value.pinned !== false && value.pinned !== 0 && value.pinned !== 1) {
    invalidResponse(write);
  }
  const selected: Record<string, unknown> = {
    content: value.content,
    type: value.type,
    pinned: value.pinned === true || value.pinned === 1
  };
  for (const key of ["summary", "importance", "confidence", "tags", "expires_at"]) {
    if (value[key] !== undefined) selected[key] = value[key];
  }
  let fields: McpMemoryPatch;
  try {
    fields = validatePatch(selected, true);
  } catch {
    invalidResponse(write);
  }
  const result: McpMemoryView = {
    ...fields,
    content: value.content,
    id: value.id,
    namespace: value.namespace,
    status: value.status,
    read_only: !isWritable({ namespace: value.namespace, id: value.id })
  };
  for (const key of ["score", "raw_similarity"] as const) {
    if (value[key] !== undefined) {
      if (typeof value[key] !== "number" || !Number.isFinite(value[key])) invalidResponse(write);
      result[key] = value[key];
    }
  }
  return result;
}

async function requestMemory(
  env: Env,
  path: string,
  method: "GET" | "POST" | "PUT" = "GET",
  body?: Record<string, unknown>
): Promise<unknown> {
  let base: string;
  let authorization: string;
  try {
    if (!env.SWEEPY_AUTH?.trim()) throw new Error();
    const url = new URL(env.SWEEPY_URL?.trim() || "https://sweepy.cloud/aeliosmemory");
    if (url.protocol !== "https:" || url.username || url.password || url.search || url.hash) {
      throw new Error();
    }
    base = url.toString().replace(/\/+$/, "");
    authorization = "Basic " + btoa(env.SWEEPY_AUTH);
  } catch {
    throw new McpStoreError("NOT_CONFIGURED", "记忆服务地址或鉴权配置不可用。");
  }
  const write = method !== "GET";
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const response = await fetch(base + path, {
      method,
      headers: { "content-type": "application/json", authorization },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      redirect: "error",
      signal: controller.signal
    });
    if (!response.ok) {
      if (response.status === 404) {
        throw new McpStoreError("NOT_FOUND", "记忆不存在或记忆服务路径不可用。", false, 404);
      }
      const uncertain = write && response.status >= 500;
      throw new McpStoreError(
        "BACKEND_HTTP",
        uncertain ? "记忆服务错误，写入结果未确认；请先核对。" : "记忆服务拒绝了请求。",
        uncertain,
        response.status
      );
    }
    try {
      return await response.json();
    } catch (error) {
      if (controller.signal.aborted) throw error;
      invalidResponse(write);
    }
  } catch (error) {
    if (error instanceof McpStoreError) throw error;
    if (controller.signal.aborted) {
      throw new McpStoreError(
        "BACKEND_TIMEOUT",
        write ? "记忆服务请求超时，写入结果未确认；请先核对，不要直接重复创建。" : "记忆搜索或读取超时。",
        write
      );
    }
    throw new McpStoreError(
      "BACKEND_UNAVAILABLE",
      write ? "记忆服务连接异常，写入结果未确认；请先核对。" : "暂时无法连接记忆服务。",
      write
    );
  } finally {
    clearTimeout(timer);
  }
}

export async function searchMcpMemories(
  env: Env,
  input: { query: string; top_k?: number }
): Promise<McpMemoryView[]> {
  const args = checkObject(input, new Set(["query", "top_k"]));
  if (!nonempty(args.query) || Array.from(args.query).length > 500) invalidArgument();
  const limit = args.top_k === undefined ? 12 : args.top_k;
  if (typeof limit !== "number" || !Number.isInteger(limit) || limit < 1 || limit > 50) {
    invalidArgument();
  }
  const data = await requestMemory(
    env,
    `/api/memories/search?q=${encodeURIComponent(args.query.trim())}&limit=${limit}&purpose=semantic`
  );
  if (!Array.isArray(data)) invalidResponse();
  const records: McpMemoryView[] = [];
  for (const item of data) {
    if (!isObject(item) || !nonempty(item.namespace)) invalidResponse();
    if (item.namespace !== "default" && item.namespace !== "sweepy-mirror") continue;
    records.push(toView(item));
  }
  return records.slice(0, limit);
}

export async function createMcpMemory(
  env: Env,
  input: McpMemoryInput
): Promise<McpMemoryView> {
  const fields = validatePatch(input, true);
  const data = await requestMemory(env, "/api/memories", "POST", {
    ...fields,
    namespace: "default",
    source: "mcp",
    status: "active"
  });
  if (!isObject(data) || data.ok !== true) invalidResponse(true);
  const memory = toView(data.memory, true);
  if (!memory.id.startsWith("am-") || memory.read_only || memory.status !== "active") {
    invalidResponse(true);
  }
  return memory;
}

export async function updateMcpMemory(
  env: Env,
  input: { id: string; patch: McpMemoryPatch }
): Promise<McpMemoryView> {
  const args = checkObject(input, new Set(["id", "patch"]));
  if (!nonempty(args.id)) invalidArgument();
  const patch = validatePatch(args.patch, false);
  const path = `/api/memories/${encodeURIComponent(args.id)}`;
  const existing = toView(await requestMemory(env, path));
  if (existing.id !== args.id) invalidResponse();
  if (existing.read_only) {
    throw new McpStoreError("READ_ONLY", "从主库同步的记忆及非自产命名空间只读，不能修改任何字段，包括置顶。");
  }
  // Existing HTTP updates cannot change id/namespace; synchronization only writes sweepy-* IDs.
  const updated = toView(await requestMemory(env, path, "PUT", { ...patch }), true);
  if (updated.id !== existing.id || updated.namespace !== "default" || updated.read_only) {
    invalidResponse(true);
  }
  return updated;
}
