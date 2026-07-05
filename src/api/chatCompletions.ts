import { authenticate } from "../auth/apiKey";
import { requireScope } from "../auth/scopes";
import { getOrCreateConversation } from "../db/conversations";
import { listMemories } from "../db/memories";
import { saveAssistantMessage, saveUserMessages } from "../db/messages";
import { getLatestSummary } from "../db/summaries";
import { saveUsageLog } from "../db/usageLogs";
import { extractLastUserText, injectMemoryPatchAsSystemMessage, selectMemoriesForInjection } from "../memory/inject";
import { compressHistoryIfNeeded } from "../memory/compress";
import { toMemoryApiRecord } from "../memory/search";
import { assemble } from "../assembler/assemble";
import { PERSONA_MEMORY_TYPES } from "../assembler/types";
import { enqueueMemoryMaintenanceIfNeeded, enqueueRetentionIfNeeded } from "../queue/producer";
import {
  buildAnthropicNativeRequest,
  buildAnthropicRequestFromAssembled,
  callAnthropicNative,
  getAnthropicCacheMode,
  parseAnthropicNonStream
} from "../proxy/anthropicAdapter";
import { buildOpenAICompatRequest, buildOpenAIRequestFromAssembled, callOpenAICompat } from "../proxy/openaiAdapter";
import { classifyProvider, resolveTargetModel } from "../proxy/resolveModel";
import { streamAnthropicToOpenAI } from "../proxy/streamAnthropic";
import { streamOpenAIWithTee } from "../proxy/streamOpenAI";
import { CONTENT_RULES } from "../preset/regexRules";
import { applyRegexRules } from "../preset/regexPipeline";
import type { Env, MemoryApiRecord, OpenAIChatRequest, OpenAIChatResponse } from "../types";
import { openAiError } from "../utils/json";
import { hasImageContent } from "../utils/messages";

function extractAssistantText(response: OpenAIChatResponse): string {
  const message = response.choices?.[0]?.message;
  if (!message) return "";

  if (typeof message.content === "string") return message.content;
  if (message.content == null) return "";
  return JSON.stringify(message.content);
}

export function hasToolContent(body: OpenAIChatRequest): boolean {
  return body.messages.some(
    (m) => m.role === "tool" || (m.role === "assistant" && m.tool_calls != null)
  );
}

/**
 * Fetch pinned memories whose type is "persona" or "identity" from D1.
 * Returns MemoryApiRecord[] for the assembler's persona_pinned block.
 * Deterministic sort is applied later by the assembler itself.
 */
async function fetchPinnedPersonaMemories(
  env: Env,
  namespace: string
): Promise<MemoryApiRecord[]> {
  const records = await listMemories(env, {
    namespace,
    status: "active",
    limit: 100,
  });

  return records
    .filter((r) => r.pinned && PERSONA_MEMORY_TYPES.includes(r.type))
    .map((r) => toMemoryApiRecord(r));
}

export async function handleChatCompletions(
  request: Request,
  env: Env,
  ctx: ExecutionContext
): Promise<Response> {
  const auth = await authenticate(request, env);
  if (!auth.ok) return openAiError("Unauthorized", 401, "authentication_error");

  const scopeError = requireScope(auth.profile, "chat:proxy");
  if (scopeError) return scopeError;

  let body: OpenAIChatRequest;
  try {
    body = (await request.json()) as OpenAIChatRequest;
  } catch {
    return openAiError("Request body must be valid JSON", 400);
  }

  if (!Array.isArray(body.messages)) {
    return openAiError("messages must be an array", 400);
  }

  const isHeartbeat = request.headers.get("x-heartbeat") === "true";

  let targetModel: string;
  try {
    targetModel = resolveTargetModel(body.model, auth.profile, env);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to resolve target model";
    return openAiError(message, 500);
  }

  const provider = classifyProvider(targetModel);

  let visionOutput: string | null = null;
  if (hasImageContent(body) && env.VISION_MODEL) {
    const lastUserMsg = [...body.messages].reverse().find(m => m.role === "user");
    if (lastUserMsg && Array.isArray(lastUserMsg.content)) {
      const imageParts = (lastUserMsg.content as Array<Record<string, unknown>>).filter(
        p => p.type === "image_url" || p.type === "input_image"
      );
      if (imageParts.length > 0) {
        try {
          const visionRes = await callOpenAICompat(env, {
            model: env.VISION_MODEL,
            messages: [
              { role: "system", content: "你是图片描述工具。如实、详细地描述图片中看到的所有内容：物体、人物、场景、文字、颜色、布局。只输出描述本身，不要加任何分析、解读、评论或开场白。" },
              { role: "user", content: [{ type: "text", text: "请描述这张图片。" }, ...imageParts] },
            ],
            max_tokens: 500,
            stream: false,
          } as OpenAIChatRequest);
          if (visionRes.ok) {
            const vd = await visionRes.json() as { choices?: Array<{ message?: { content?: string } }> };
            visionOutput = vd?.choices?.[0]?.message?.content || null;
            console.log("[vision] output:", visionOutput ? visionOutput.slice(0, 80) + "..." : "empty");
          } else {
            console.log("[vision] error status:", visionRes.status, await visionRes.text().catch(() => ""));
          }
        } catch (e) {
          console.log("[vision] exception:", e instanceof Error ? e.message : e);
        }
      }
    }
    body = {
      ...body,
      messages: body.messages.map(m => {
        if (!Array.isArray(m.content)) return m;
        const texts = (m.content as Array<Record<string, unknown>>).filter(p => p.type !== "image_url" && p.type !== "input_image");
        if (texts.length === 1 && texts[0].type === "text") return { ...m, content: texts[0].text as string };
        return { ...m, content: texts.length > 0 ? texts : "" };
      }),
    };
    console.log("[vision] visionOutput injecting:", visionOutput ? "yes" : "no");
  }

  const conversation = isHeartbeat ? null : await getOrCreateConversation(env.DB, {
    namespace: auth.profile.namespace
  });

  let latestUserMessageId: string | undefined;
  if (!isHeartbeat) {
    const savedUserMessageIds = await saveUserMessages(env.DB, {
      conversationId: conversation!.id,
      namespace: auth.profile.namespace,
      source: auth.profile.source,
      messages: body.messages,
      requestModel: body.model,
      upstreamModel: targetModel,
      upstreamProvider: provider,
      stream: Boolean(body.stream)
    });
    latestUserMessageId = savedUserMessageIds[savedUserMessageIds.length - 1];
  }

  // History compression + memory search + persona + summary in parallel
  const userQuery = extractLastUserText(body.messages);
  const memoryQuery = visionOutput ? `${userQuery}\n${visionOutput}`.slice(0, 500) : userQuery;
  const [compressResult, memories, pinnedPersonaMemories, latestSummary] = await Promise.all([
    compressHistoryIfNeeded(env, body.messages, auth.profile.namespace),
    selectMemoriesForInjection(env, { profile: auth.profile, query: memoryQuery }),
    fetchPinnedPersonaMemories(env, auth.profile.namespace),
    getLatestSummary(env.DB, auth.profile.namespace),
  ]);

  // If compression happened, use trimmed messages for the assembler
  const assemblerBody = compressResult.summary
    ? { ...body, messages: compressResult.messages }
    : body;
  const compressedSummary = compressResult.summary;
  const summaryEntry = latestSummary ? { content: latestSummary.content } : null;

  let upstream: Response;
  let clientSystemHash: string | null = null;
  let cacheAnchorBlock: string | null = null;
  try {
    if (provider === "anthropic") {
      if (hasToolContent(body)) {
        // Tool messages / tool_calls not yet supported by assembler — fall back
        const anthropicRequest = await buildAnthropicNativeRequest(body, {
          env,
          targetModel,
          namespace: auth.profile.namespace,
          memories
        });
        upstream = await callAnthropicNative(env, anthropicRequest, targetModel);
      } else {
        const assembled = assemble({
          request: assemblerBody,
          pinnedPersonaMemories,
          summaryEntry,
          ragMemories: memories,
          visionOutput,
          compressedSummary,
        });
        clientSystemHash = assembled.meta.client_system_hash;
        cacheAnchorBlock = assembled.meta.anchor_index >= 0 ? "client_system" : null;
        // NOTE: Anthropic adapter stringifies structured content (image_url etc.)
        // as a temporary fallback; native Anthropic image support will be added
        // when the vision pipeline is wired in.
        upstream = await callAnthropicNative(env, buildAnthropicRequestFromAssembled(body, targetModel, assembled, env), targetModel);
      }
    } else {
      if (hasToolContent(body)) {
        // Tool messages / tool_calls not yet supported by assembler — fall back
        const patchedBody = await injectMemoryPatchAsSystemMessage(body, memories, env);
        const upstreamRequest = buildOpenAICompatRequest(patchedBody, targetModel);
        upstream = await callOpenAICompat(env, upstreamRequest);
      } else {
        const assembled = assemble({
          request: assemblerBody,
          pinnedPersonaMemories,
          summaryEntry,
          ragMemories: memories,
          visionOutput,
          compressedSummary,
        });
        clientSystemHash = assembled.meta.client_system_hash;
        upstream = await callOpenAICompat(env, buildOpenAIRequestFromAssembled(body, targetModel, assembled));
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to call upstream";
    return openAiError(message, 502);
  }

  if (!upstream.ok) {
    const errorText = await upstream.text();
    return new Response(errorText, {
      status: upstream.status,
      headers: {
        "content-type": upstream.headers.get("content-type") || "application/json; charset=utf-8"
      }
    });
  }

  if (body.stream) {
    if (provider === "anthropic") {
      return streamAnthropicToOpenAI(upstream, {
        env,
        ctx,
        profile: auth.profile,
        conversationId: conversation!.id,
        fromMessageId: latestUserMessageId,
        requestModel: body.model,
        upstreamModel: targetModel,
        provider,
        clientSystemHash,
        cacheAnchorBlock
      });
    }

    return streamOpenAIWithTee(upstream, {
      env,
      ctx,
      profile: auth.profile,
      conversationId: conversation!.id,
      fromMessageId: latestUserMessageId,
      requestModel: body.model,
      upstreamModel: targetModel,
      provider,
      clientSystemHash,
      cacheAnchorBlock
    });
  }

  const responseText = await upstream.text();

  if (provider === "anthropic") {
    let anthropicParsed: unknown;
    try {
      anthropicParsed = JSON.parse(responseText) as unknown;
    } catch {
      return openAiError("Upstream returned invalid JSON", 502);
    }

    const parsed = parseAnthropicNonStream(anthropicParsed as never);
    const anthropicCacheMode = getAnthropicCacheMode(env);
    // Filter visible content only — reasoning_content is preserved upstream.
    const filteredContent = applyRegexRules(parsed.content, CONTENT_RULES);
    if (parsed.openai.choices?.[0]?.message) {
      parsed.openai.choices[0].message.content = filteredContent;
    }
    if (!isHeartbeat) {
      const assistantMessageId = await saveAssistantMessage(env.DB, {
        conversationId: conversation!.id,
        namespace: auth.profile.namespace,
        source: auth.profile.source,
        content: filteredContent,
        requestModel: body.model,
        upstreamModel: targetModel,
        provider,
        stream: false,
        finishReason: parsed.finishReason,
        usage: parsed.usage,
        cacheMode: anthropicCacheMode,
        cacheTtl: env.ANTHROPIC_CACHE_TTL || "5m"
      });

      ctx.waitUntil(
        Promise.all([
          saveUsageLog(env.DB, {
            messageId: assistantMessageId,
            namespace: auth.profile.namespace,
            provider,
            model: targetModel,
            usage: parsed.usage,
            cacheMode: anthropicCacheMode,
            cacheTtl: env.ANTHROPIC_CACHE_TTL || "5m",
            clientSystemHash,
            cacheAnchorBlock
          }),
          enqueueMemoryMaintenanceIfNeeded(env, {
            namespace: auth.profile.namespace,
            conversationId: conversation!.id,
            fromMessageId: latestUserMessageId!,
            toMessageId: assistantMessageId,
            source: auth.profile.source
          }),
          enqueueRetentionIfNeeded(env, auth.profile.namespace)
        ])
      );
    }

    return new Response(JSON.stringify(parsed.openai), {
      status: 200,
      headers: {
        "content-type": "application/json; charset=utf-8"
      }
    });
  }

  let parsed: OpenAIChatResponse;
  try {
    parsed = JSON.parse(responseText) as OpenAIChatResponse;
  } catch {
    return openAiError("Upstream returned invalid JSON", 502);
  }

  const assistantContent = extractAssistantText(parsed);
  const filteredContent = applyRegexRules(assistantContent, CONTENT_RULES);
  // Patch the response that goes back to the client.
  if (parsed.choices?.[0]?.message) {
    parsed.choices[0].message.content = filteredContent;
  }
  if (!isHeartbeat) {
    const assistantMessageId = await saveAssistantMessage(env.DB, {
      conversationId: conversation!.id,
      namespace: auth.profile.namespace,
      source: auth.profile.source,
      content: filteredContent,
      requestModel: body.model,
      upstreamModel: targetModel,
      provider,
      stream: false,
      finishReason: parsed.choices?.[0]?.finish_reason,
      usage: parsed.usage
    });

    ctx.waitUntil(
      Promise.all([
        saveUsageLog(env.DB, {
          messageId: assistantMessageId,
          namespace: auth.profile.namespace,
          provider,
          model: targetModel,
          usage: parsed.usage,
          clientSystemHash,
          cacheAnchorBlock
        }),
        enqueueMemoryMaintenanceIfNeeded(env, {
          namespace: auth.profile.namespace,
          conversationId: conversation!.id,
          fromMessageId: latestUserMessageId!,
          toMessageId: assistantMessageId,
          source: auth.profile.source
        }),
        enqueueRetentionIfNeeded(env, auth.profile.namespace)
      ])
    );
  }

  return new Response(JSON.stringify(parsed), {
    status: 200,
    headers: {
      "content-type": "application/json; charset=utf-8"
    }
  });
}
