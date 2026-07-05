/**
 * openaiAdapter.ts — 增加 OpenRouter 支持
 *
 * 原版所有模型调用走 Cloudflare AI Gateway。
 * 改为：优先走 OpenRouter（如果配了 OPENROUTER_API_KEY），
 * 回退到 AI Gateway。
 */

import type { AssembledPrompt } from "../assembler/types";
import { assembledToOpenAISystem, assembledToOpenAIMessages } from "../assembler/toOpenAI";
import type { Env, OpenAIChatMessage, OpenAIChatRequest } from "../types";

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
  const stableBlocks: typeof assembled.system_blocks = [];
  const dynamicTexts: string[] = [];

  for (let i = 0; i < assembled.system_blocks.length; i++) {
    const blockId = assembled.meta.block_ids[i];
    if (blockId === "client_volatile_context" || blockId === "dynamic_memory_patch") {
      dynamicTexts.push(assembled.system_blocks[i].text);
    } else {
      stableBlocks.push(assembled.system_blocks[i]);
    }
  }

  const systemMsg = assembledToOpenAISystem(stableBlocks);
  const messages: OpenAIChatMessage[] = [];
  if (systemMsg) messages.push(systemMsg);
  messages.push(...assembledToOpenAIMessages(assembled.messages));

  if (dynamicTexts.length > 0) {
    const extra = dynamicTexts.join("\n\n");
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        const current = messages[i].content;
        if (Array.isArray(current)) {
          (current as unknown[]).push({ type: "text", text: extra });
        } else {
          messages[i].content = (typeof current === "string" ? current : "") + "\n\n" + extra;
        }
        break;
      }
    }
  }

  return buildOpenAICompatRequest({ ...req, messages }, targetModel);
}

// ─── URL 和 Headers 路由 ───

function useOpenRouter(env: Env, model?: string): boolean {
  if (!env.OPENROUTER_API_KEY) return false;
  if (!model) return true;
  const lower = model.toLowerCase();
  if (lower.startsWith("workers-ai/") || lower.startsWith("@cf/")) return false;
  return true;
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

export function getOpenAICompatUrl(env: Env, model?: string): string {
  if (useOpenRouter(env, model)) {
    return `${getOpenRouterBaseUrl(env)}/v1/chat/completions`;
  }
  return `${normalizeAiGatewayBaseUrl(env)}/compat/chat/completions`;
}

export function buildOpenAICompatHeaders(env: Env, model?: string): Headers {
  const headers = new Headers({
    "content-type": "application/json",
  });

  if (useOpenRouter(env, model)) {
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

export async function callOpenAICompat(env: Env, body: OpenAIChatRequest): Promise<Response> {
  const model = body.model;
  return fetch(getOpenAICompatUrl(env, model), {
    method: "POST",
    headers: buildOpenAICompatHeaders(env, model),
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
