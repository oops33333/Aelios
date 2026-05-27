/**
 * openaiAdapter.ts — 增加 OpenRouter 支持
 *
 * 原版所有模型调用走 Cloudflare AI Gateway。
 * 改为：优先走 OpenRouter（如果配了 OPENROUTER_API_KEY），
 * 回退到 AI Gateway。
 */

import type { AssembledPrompt } from "../assembler/types";
import { assembledToOpenAIChatMessages } from "../assembler/toOpenAI";
import type { Env, OpenAIChatRequest } from "../types";

function stripClaudeNativeThinkingFields(req: OpenAIChatRequest): OpenAIChatRequest {
  const cleaned: OpenAIChatRequest = { ...req };
  delete cleaned.thinking;
  return cleaned;
}

export function buildOpenAICompatRequest(req: OpenAIChatRequest, targetModel: string): OpenAIChatRequest {
  const cleaned = stripClaudeNativeThinkingFields(req);
  return {
    ...cleaned,
    model: targetModel,
    stream: Boolean(cleaned.stream),
  };
}

export function buildOpenAIRequestFromAssembled(
  req: OpenAIChatRequest,
  targetModel: string,
  assembled: AssembledPrompt
): OpenAIChatRequest {
  const messages = assembledToOpenAIChatMessages(assembled);
  return buildOpenAICompatRequest({ ...req, messages }, targetModel);
}

// ─── URL 和 Headers 路由 ───

function useOpenRouter(env: Env): boolean {
  return Boolean(env.OPENROUTER_API_KEY);
}

function getOpenRouterBaseUrl(env: Env): string {
  return (env.OPENROUTER_BASE_URL || "https://openrouter.ai/api").replace(/\/+$/, "");
}

export function normalizeAiGatewayBaseUrl(env: Env): string {
  const base = env.AI_GATEWAY_BASE_URL;
  if (!base) {
    return "";
  }

  return base
    .replace(/\/+$/, "")
    .replace(/\/compat$/i, "")
    .replace(/\/compat\/chat\/completions$/i, "")
    .replace(/\/compat\/embeddings$/i, "")
    .replace(/\/anthropic\/v1\/messages$/i, "");
}

export function getOpenAICompatUrl(env: Env): string {
  if (useOpenRouter(env)) {
    return `${getOpenRouterBaseUrl(env)}/v1/chat/completions`;
  }
  return `${normalizeAiGatewayBaseUrl(env)}/compat/chat/completions`;
}

export function buildOpenAICompatHeaders(env: Env): Headers {
  const headers = new Headers({
    "content-type": "application/json",
  });

  if (useOpenRouter(env)) {
    headers.set("authorization", `Bearer ${env.OPENROUTER_API_KEY}`);
    // OpenRouter 推荐设置的 header
    headers.set("http-referer", "https://sweepy.cloud");
    headers.set("x-title", "sweepy-companion");
  } else {
    if (env.CF_AIG_TOKEN) {
      headers.set("cf-aig-authorization", `Bearer ${env.CF_AIG_TOKEN}`);
    }
  }

  return headers;
}


// ─── Chat 专用：走 AI Gateway，不走 OpenRouter ───

export async function callChatViaGateway(env: Env, body: OpenAIChatRequest): Promise<Response> {
  const baseUrl = normalizeAiGatewayBaseUrl(env);
  if (!baseUrl) throw new Error("AI_GATEWAY_BASE_URL is not configured");

  const url = `${baseUrl}/compat/chat/completions`;
  const headers = new Headers({ "content-type": "application/json" });
  if (env.CF_GATEWAY_CHAT_TOKEN) {
    headers.set("cf-aig-authorization", `Bearer ${env.CF_GATEWAY_CHAT_TOKEN}`);
  }

  return fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
}

export async function callOpenAICompat(env: Env, body: OpenAIChatRequest): Promise<Response> {
  return fetch(getOpenAICompatUrl(env), {
    method: "POST",
    headers: buildOpenAICompatHeaders(env),
    body: JSON.stringify(body),
  });
}

export async function callOpenAICompatEmbeddings(
  env: Env,
  body: { model: string; input: string | string[]; dimensions?: number }
): Promise<Response> {
  // Embedding 仍走 AI Gateway / Workers AI（不走 OpenRouter）
  const headers = buildOpenAICompatHeaders(env);
  if (body.model.startsWith("workers-ai/") && env.CLOUDFLARE_API_TOKEN) {
    headers.set("authorization", `Bearer ${env.CLOUDFLARE_API_TOKEN}`);
  }

  // 如果是 OpenRouter 模式但需要 embedding，回退到 AI Gateway
  if (useOpenRouter(env) && env.AI_GATEWAY_BASE_URL) {
    const gatewayHeaders = new Headers({ "content-type": "application/json" });
    if (env.CF_AIG_TOKEN) {
      gatewayHeaders.set("cf-aig-authorization", `Bearer ${env.CF_AIG_TOKEN}`);
    }
    return fetch(`${normalizeAiGatewayBaseUrl(env)}/compat/embeddings`, {
      method: "POST",
      headers: gatewayHeaders,
      body: JSON.stringify(body),
    });
  }

  return fetch(`${normalizeAiGatewayBaseUrl(env)}/compat/embeddings`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
}
