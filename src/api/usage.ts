/**
 * GET /v1/usage — 供 aelios.html 监控面板读取最近真实请求的 token 用量
 * 与缓存命中情况（usage_logs 表，心跳请求不落此表）。
 *
 * 认证两种方式：
 * - Authorization: Bearer <key>（标准路径）
 * - ?key=<CHATBOX_API_KEY>（浏览器简单 GET，免 CORS 预检）
 * 响应带 Access-Control-Allow-Origin: *，页面在 sweepy.cloud 域下直接 fetch。
 */

import { authenticate } from "../auth/apiKey";
import { json } from "../utils/json";
import type { Env } from "../types";

interface UsageRow {
  created_at: string;
  provider: string | null;
  model: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
  cache_read_tokens: number | null;
  cache_creation_tokens: number | null;
  cache_mode: string | null;
  cache_ttl: string | null;
}

export async function handleUsage(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);

  let namespace = "default";
  const queryKey = url.searchParams.get("key");
  if (queryKey && env.CHATBOX_API_KEY && queryKey === env.CHATBOX_API_KEY) {
    // 面板路径：query key 等同 chatbox 档位（namespace: default）
  } else {
    const auth = await authenticate(request, env);
    if (!auth.ok) return json({ error: "Unauthorized" }, { status: 401 });
    namespace = auth.profile.namespace;
  }

  const parsedLimit = parseInt(url.searchParams.get("limit") || "50", 10);
  const limit = Math.min(Math.max(Number.isFinite(parsedLimit) ? parsedLimit : 50, 1), 200);

  const result = await env.DB.prepare(
    `SELECT created_at, provider, model, input_tokens, output_tokens,
            cache_read_tokens, cache_creation_tokens, cache_mode, cache_ttl
     FROM usage_logs
     WHERE namespace = ?
     ORDER BY created_at DESC
     LIMIT ?`
  )
    .bind(namespace, limit)
    .all<UsageRow>();

  const entries = (result.results || []).map((row) => {
    const input = row.input_tokens || 0;
    const read = row.cache_read_tokens || 0;
    const creation = row.cache_creation_tokens || 0;
    const promptTotal = input + read + creation;
    return {
      ...row,
      prompt_total: promptTotal,
      // 命中率 = 缓存读 / (未缓存输入 + 缓存读 + 缓存写)
      cache_hit_rate: promptTotal > 0 ? Math.round((read / promptTotal) * 1000) / 10 : null
    };
  });

  return new Response(JSON.stringify({ entries }), {
    headers: {
      "content-type": "application/json; charset=utf-8",
      "access-control-allow-origin": "*"
    }
  });
}
