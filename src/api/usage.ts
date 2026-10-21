/**
 * GET /v1/usage — 供 aelios.html 监控面板读取最近真实请求的 token 用量
 * 与缓存命中情况；kind=heartbeat 单独查询心跳用量和费用。
 *
 * 认证两种方式：
 * - Authorization: Bearer <key>（标准路径）
 * - ?key=<CHATBOX_API_KEY>（浏览器简单 GET，免 CORS 预检）
 * 响应带 Access-Control-Allow-Origin: *，页面在 sweepy.cloud 域下直接 fetch。
 */

import { authenticate } from "../auth/apiKey";
import { json } from "../utils/json";
import type { Env } from "../types";
import { fetchGatewayCost, type GatewayCost } from "../proxy/gatewayCost";

interface UsageRow {
  id: string;
  raw_usage_json: string | null;
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

function parseUsageMetadata(raw: string | null): Record<string, unknown> {
  try {
    const value: unknown = JSON.parse(raw || "{}");
    return value && typeof value === "object" && !Array.isArray(value)
      ? value as Record<string, unknown>
      : {};
  } catch {
    return {};
  }
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
  const kind = url.searchParams.get("kind") === "heartbeat" ? "heartbeat" : "chat";

  const result = await env.DB.prepare(
    `SELECT id, raw_usage_json, created_at, provider, model, input_tokens, output_tokens,
            cache_read_tokens, cache_creation_tokens, cache_mode, cache_ttl
     FROM usage_logs
     WHERE namespace = ?
       AND COALESCE(
         CASE WHEN json_valid(raw_usage_json)
           THEN json_extract(raw_usage_json, '$.cf_aig_kind') END,
         'chat'
       ) = ?
     ORDER BY created_at DESC
     LIMIT ?`
  )
    .bind(namespace, kind, limit)
    .all<UsageRow>();

  let lookupBudget = 12;
  const entries = await Promise.all((result.results || []).map(async (row) => {
    const { id, raw_usage_json, ...visibleRow } = row;
    const metadata = parseUsageMetadata(raw_usage_json);
    const logId = typeof metadata.cf_aig_log_id === "string"
      ? metadata.cf_aig_log_id
      : null;
    const storedCost = metadata.cf_aig_cost;
    let pricing: GatewayCost =
      typeof storedCost === "number" && Number.isFinite(storedCost) && storedCost >= 0
        ? { cost: storedCost, status: "ok" }
        : { cost: null, status: logId ? "pending" : "unavailable" };

    if (pricing.cost === null && logId && lookupBudget > 0) {
      lookupBudget--;
      pricing = await fetchGatewayCost(env, logId);
      if (pricing.cost !== null) {
        try {
          await env.DB.prepare(
            `UPDATE usage_logs
             SET raw_usage_json = json_set(raw_usage_json, '$.cf_aig_cost', ?)
             WHERE id = ? AND namespace = ?`
          ).bind(pricing.cost, id, namespace).run();
        } catch {
          // A persistence failure must not hide the fetched cost or token usage.
        }
      }
    }

    const input = row.input_tokens || 0;
    const read = row.cache_read_tokens || 0;
    const creation = row.cache_creation_tokens || 0;
    const promptTotal = input + read + creation;
    return {
      ...visibleRow,
      gateway_log_id: logId,
      cost: pricing.cost,
      cost_status: pricing.status,
      cost_reason: pricing.reason ?? null,
      cost_http_status: pricing.http_status ?? null,
      cost_error_codes: pricing.error_codes ?? [],
      prompt_total: promptTotal,
      // 命中率 = 缓存读 / (未缓存输入 + 缓存读 + 缓存写)
      cache_hit_rate: promptTotal > 0 ? Math.round((read / promptTotal) * 1000) / 10 : null
    };
  }));

  return new Response(JSON.stringify({ entries }), {
    headers: {
      "content-type": "application/json; charset=utf-8",
      "access-control-allow-origin": "*"
    }
  });
}
