import type { Env } from "../types";

export interface GatewayCost {
  cost: number | null;
  status: "ok" | "unavailable" | "not_configured" | "forbidden" | "pending";
  reason?: "http_error" | "api_error" | "log_id_mismatch" | "timeout" | "request_error";
  http_status?: number;
  error_codes?: number[];
}

export async function fetchGatewayCost(
  env: Env,
  logId: string | null | undefined
): Promise<GatewayCost> {
  if (!logId) return { cost: null, status: "unavailable" };

  let account = env.CLOUDFLARE_ACCOUNT_ID?.trim();
  let gateway = env.AI_GATEWAY_ID?.trim();
  try {
    const parts = new URL(env.AI_GATEWAY_BASE_URL || "")
      .pathname.split("/").filter(Boolean);
    const index = parts.indexOf("v1");
    if (index >= 0) {
      account ||= parts[index + 1];
      gateway ||= parts[index + 2]
        ? decodeURIComponent(parts[index + 2])
        : undefined;
    }
  } catch {
    // Explicit account and gateway settings can still be used.
  }
  gateway ||= "default";
  const token = env.CLOUDFLARE_API_TOKEN;
  if (!account || !token) {
    return { cost: null, status: "not_configured" };
  }

  const path = [account, gateway, logId].map(encodeURIComponent);
  const url = `https://api.cloudflare.com/client/v4/accounts/${path[0]}`
    + `/ai-gateway/gateways/${path[1]}/logs/${path[2]}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 2000);
  try {
    const response = await fetch(url, {
      headers: { authorization: `Bearer ${token}` },
      signal: controller.signal,
      redirect: "error"
    });
    if (response.status === 401 || response.status === 403) {
      return { cost: null, status: "forbidden", http_status: response.status };
    }
    if (response.status === 404) {
      return { cost: null, status: "pending", http_status: 404 };
    }
    if (!response.ok) {
      return {
        cost: null,
        status: "unavailable",
        reason: "http_error",
        http_status: response.status
      };
    }
    const data = await response.json() as {
      success?: boolean;
      result?: { id?: string; cost?: number };
      errors?: Array<{ code?: unknown }>;
    };
    if (!data.success) {
      return {
        cost: null,
        status: "unavailable",
        reason: "api_error",
        http_status: response.status,
        error_codes: Array.isArray(data.errors)
          ? data.errors.map(error => error?.code)
              .filter((code): code is number => typeof code === "number" && Number.isFinite(code))
              .slice(0, 5)
          : []
      };
    }
    if (data.result?.id !== logId) {
      return {
        cost: null,
        status: "unavailable",
        reason: "log_id_mismatch",
        http_status: response.status
      };
    }
    const cost = data.result.cost;
    return typeof cost === "number" && Number.isFinite(cost) && cost >= 0
      ? { cost, status: "ok" }
      : { cost: null, status: "pending" };
  } catch {
    return {
      cost: null,
      status: "unavailable",
      reason: controller.signal.aborted ? "timeout" : "request_error"
    };
  } finally {
    clearTimeout(timer);
  }
}
