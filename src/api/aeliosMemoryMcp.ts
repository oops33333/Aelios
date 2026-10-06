import { authenticate } from "../auth/apiKey";
import {
  createMcpMemory,
  McpStoreError,
  searchMcpMemories,
  updateMcpMemory,
  type McpMemoryInput,
  type McpMemoryPatch
} from "../memory/mcpStore";
import type { Env } from "../types";

type RpcId = string | number;
const PROTOCOL_VERSION = "2025-06-18";

const EDITABLE_PROPERTIES = {
  content: { type: "string", minLength: 1, description: "记忆正文，不能只有空白。" },
  type: { type: "string", minLength: 1 },
  summary: { type: ["string", "null"] },
  importance: { type: "number", minimum: 0, maximum: 1 },
  confidence: { type: "number", minimum: 0, maximum: 1 },
  pinned: { type: "boolean" },
  tags: { type: "array", items: { type: "string" } },
  expires_at: {
    type: ["string", "null"],
    description: "有效日期字符串；null 表示无到期时间。"
  }
};

const TOOLS = [
  {
    name: "memory_search",
    description: "搜索现有记忆，返回真实 ID 和 read_only 标记。主库同步记忆只读，不能修改，包括置顶。",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", minLength: 1, maxLength: 500 },
        top_k: { type: "integer", minimum: 1, maximum: 50, default: 12 }
      },
      required: ["query"],
      additionalProperties: false
    }
  },
  {
    name: "memory_create",
    description: "在 aeliosmemory 自产区新增记忆，不写入主库或同步区。正文保存后向量异步生成；写入结果未确认时先搜索核对，不要直接重复创建。",
    inputSchema: {
      type: "object",
      properties: EDITABLE_PROPERTIES,
      required: ["content"],
      additionalProperties: false
    }
  },
  {
    name: "memory_update",
    description: "按真实 ID 修改已有自产记忆，仅提交需要修改的字段。主库同步记忆的全部字段只读，包括置顶。不能修改来源、命名空间或删除记忆。",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", minLength: 1 },
        patch: {
          type: "object",
          properties: EDITABLE_PROPERTIES,
          minProperties: 1,
          additionalProperties: false
        }
      },
      required: ["id", "patch"],
      additionalProperties: false
    }
  }
];

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function reply(body: unknown, status = 200, extra: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      ...extra
    }
  });
}

function rpcError(id: RpcId | null, code: number, message: string, status = 200): Response {
  return reply({ jsonrpc: "2.0", id, error: { code, message } }, status);
}

function rpcResult(id: RpcId, result: unknown): Response {
  return reply({ jsonrpc: "2.0", id, result });
}

function toolResult(data: unknown): Record<string, unknown> {
  return {
    content: [{ type: "text", text: JSON.stringify({ data }) }]
  };
}

function toolFailure(error: unknown): Record<string, unknown> {
  const known = error instanceof McpStoreError;
  return {
    isError: true,
    content: [{
      type: "text",
      text: JSON.stringify({
        error: known ? error.code : "INTERNAL_ERROR",
        message: known ? error.message : "工具执行异常，结果未确认；请先核对。",
        outcome_unknown: known ? error.outcomeUnknown : true
      })
    }]
  };
}

async function callTool(env: Env, params: Record<string, unknown>): Promise<Record<string, unknown>> {
  if (!isObject(params.arguments)) {
    throw new McpStoreError("INVALID_ARGUMENT", "工具 arguments 必须是对象。");
  }
  // Runtime validation lives in the strict adapter, including unknown-field rejection.
  if (params.name === "memory_search") {
    return toolResult(await searchMcpMemories(
      env, params.arguments as unknown as { query: string; top_k?: number }
    ));
  }
  if (params.name === "memory_create") {
    return toolResult(await createMcpMemory(env, params.arguments as unknown as McpMemoryInput));
  }
  if (params.name === "memory_update") {
    return toolResult(await updateMcpMemory(
      env, params.arguments as unknown as { id: string; patch: McpMemoryPatch }
    ));
  }
  throw new McpStoreError("INVALID_ARGUMENT", "未知工具。");
}

export async function handleAeliosMemoryMcp(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  const origin = request.headers.get("origin");
  // Native clients normally omit Origin. Browser requests must be same-origin.
  if (origin !== null && origin !== url.origin) {
    return rpcError(null, -32000, "Origin not allowed", 403);
  }
  if (request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: { Allow: "POST, OPTIONS", "cache-control": "no-store" }
    });
  }
  const authorization = request.headers.get("authorization");
  if (url.searchParams.has("token") || !authorization?.toLowerCase().startsWith("bearer ")) {
    return reply(
      { error: "Unauthorized" },
      401,
      { "WWW-Authenticate": 'Bearer realm="aelios-memory-mcp"' }
    );
  }
  const auth = await authenticate(request, env);
  if (!auth.ok || auth.keyName !== "MEMORY_MCP_API_KEY") {
    return reply(
      { error: "Unauthorized" },
      401,
      { "WWW-Authenticate": 'Bearer realm="aelios-memory-mcp"' }
    );
  }
  if (request.method !== "POST") {
    return reply({ error: "Method not allowed" }, 405, { Allow: "POST, OPTIONS" });
  }
  const contentType = request.headers.get("content-type")?.split(";")[0].trim().toLowerCase();
  if (contentType !== "application/json") {
    return rpcError(null, -32600, "Content-Type must be application/json", 415);
  }
  const accepted = (request.headers.get("accept") || "")
    .split(",").map((part) => part.split(";")[0].trim().toLowerCase());
  if (!accepted.includes("application/json") || !accepted.includes("text/event-stream")) {
    return rpcError(null, -32600, "Accept must include application/json and text/event-stream", 406);
  }
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return rpcError(null, -32700, "Parse error", 400);
  }
  if (!isObject(body) || body.jsonrpc !== "2.0" || typeof body.method !== "string" ||
      (body.params !== undefined && !isObject(body.params))) {
    return rpcError(null, -32600, "Invalid Request", 400);
  }
  const version = request.headers.get("MCP-Protocol-Version");
  // initialize negotiates a supported version even when the client proposes a newer one.
  if (body.method !== "initialize" && version !== null && version !== PROTOCOL_VERSION) {
    return rpcError(null, -32600, "Unsupported MCP-Protocol-Version", 400);
  }
  if (!Object.prototype.hasOwnProperty.call(body, "id")) {
    if (!body.method.startsWith("notifications/")) {
      return rpcError(null, -32600, "Request id required", 400);
    }
    return new Response(null, { status: 202, headers: { "cache-control": "no-store" } });
  }
  if (typeof body.id !== "string" &&
      !(typeof body.id === "number" && Number.isFinite(body.id))) {
    return rpcError(null, -32600, "Invalid request id", 400);
  }
  const id = body.id;
  const params = isObject(body.params) ? body.params : {};
  if (body.method === "initialize") {
    if (typeof params.protocolVersion !== "string" || !params.protocolVersion.trim()) {
      return rpcError(id, -32602, "protocolVersion is required");
    }
    return rpcResult(id, {
      protocolVersion: PROTOCOL_VERSION,
      capabilities: { tools: { listChanged: false } },
      serverInfo: { name: "aelios-memory-mcp", version: "0.1.0" }
    });
  }
  if (body.method === "ping") return rpcResult(id, {});
  if (body.method === "tools/list") return rpcResult(id, { tools: TOOLS });
  if (body.method === "tools/call") {
    if (typeof params.name !== "string" || !TOOLS.some((tool) => tool.name === params.name)) {
      return rpcError(id, -32602, "Unknown tool");
    }
    const scope = params.name === "memory_search" ? "memory:read" : "memory:write";
    if (!auth.profile.scopes.includes(scope)) {
      return rpcResult(id, {
        isError: true,
        content: [{ type: "text", text: "没有执行此记忆工具的权限。" }]
      });
    }
    try {
      return rpcResult(id, await callTool(env, params));
    } catch (error) {
      return rpcResult(id, toolFailure(error));
    }
  }
  return rpcError(id, -32601, "Method not found");
}
