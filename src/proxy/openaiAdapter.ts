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
  if (!model) return false;
  const lower = model.toLowerCase();
  if (lower.startsWith("@cf/") || lower.startsWith("workers-ai/") || lower.startsWith("google/")) return false;
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

function getWorkersAiModel(model: string): string | null {
  if (model.startsWith("@cf/")) return model;
  if (model.startsWith("workers-ai/")) return model.slice("workers-ai/".length);
  return null;
}

function isGeminiModel(model: string): boolean {
  const lower = model.toLowerCase();
  return lower.startsWith("google/gemini") || lower.startsWith("gemini");
}

interface GeminiPart {
  text?: string;
  inline_data?: { mime_type: string; data: string };
}

function openAiMsgToGeminiContents(messages: Array<{ role: string; content: unknown }>): Array<{ role: string; parts: GeminiPart[] }> {
  const contents: Array<{ role: string; parts: GeminiPart[] }> = [];
  for (const m of messages) {
    if (m.role !== "user" && m.role !== "assistant") continue;
    const role = m.role === "assistant" ? "model" : "user";
    const parts: GeminiPart[] = [];
    if (typeof m.content === "string") {
      parts.push({ text: m.content });
    } else if (Array.isArray(m.content)) {
      for (const p of m.content as Array<Record<string, unknown>>) {
        if (p.type === "text" && typeof p.text === "string") {
          parts.push({ text: p.text });
        } else if (p.type === "image_url") {
          const url = (p.image_url as Record<string, unknown>)?.url;
          if (typeof url === "string" && url.startsWith("data:")) {
            const sep = url.indexOf(";base64,");
            if (sep > 5) {
              parts.push({ inline_data: { mime_type: url.slice(5, sep), data: url.slice(sep + 8) } });
            }
          }
        }
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  return contents;
}

function extractGeminiText(result: unknown): string {
  if (!result || typeof result !== "object") return "";
  const r = result as Record<string, unknown>;
  if (typeof r.response === "string") return r.response;
  const candidates = r.candidates;
  if (Array.isArray(candidates) && candidates.length > 0) {
    const c = candidates[0] as Record<string, unknown>;
    const content = c?.content as Record<string, unknown> | undefined;
    const parts = content?.parts;
    if (Array.isArray(parts)) {
      return parts.map((p: Record<string, unknown>) => typeof p.text === "string" ? p.text : "").join("");
    }
  }
  return "";
}

export async function callOpenAICompat(env: Env, body: OpenAIChatRequest): Promise<Response> {
  const model = body.model;

  if (model && env.AI) {
    const waiModel = getWorkersAiModel(model);
    if (waiModel) {
      try {
        const result = await (env.AI as any).run(waiModel, {
          messages: body.messages,
          max_tokens: typeof body.max_tokens === "number" ? body.max_tokens : 1024,
          temperature: typeof body.temperature === "number" ? body.temperature : undefined,
        });
        const text = typeof result?.response === "string" ? result.response : "";
        return new Response(JSON.stringify({
          id: "wai-" + Date.now(), object: "chat.completion",
          created: Math.floor(Date.now() / 1000), model: waiModel,
          choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
        }), { status: 200, headers: { "content-type": "application/json" } });
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Workers AI error";
        return new Response(JSON.stringify({ error: { message: msg, type: "workers_ai_error" } }), {
          status: 502, headers: { "content-type": "application/json" },
        });
      }
    }

    if (isGeminiModel(model)) {
      try {
        const contents = openAiMsgToGeminiContents(body.messages as Array<{ role: string; content: unknown }>);
        const result = await (env.AI as any).run(model, { contents });
        const text = extractGeminiText(result);
        return new Response(JSON.stringify({
          id: "wai-" + Date.now(), object: "chat.completion",
          created: Math.floor(Date.now() / 1000), model,
          choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
        }), { status: 200, headers: { "content-type": "application/json" } });
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Workers AI error";
        return new Response(JSON.stringify({ error: { message: msg, type: "workers_ai_error" } }), {
          status: 502, headers: { "content-type": "application/json" },
        });
      }
    }
  }

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
