import { authenticate } from "../auth/apiKey";
import { requireScope } from "../auth/scopes";
import { getOrCreateConversation } from "../db/conversations";
import { finishIdempotentTask, tryStartIdempotentTask } from "../db/idempotency";
import { commitMemoryInjection, listMemories } from "../db/memories";
import { saveAssistantMessage, saveUserMessages } from "../db/messages";
import { getLatestSummary } from "../db/summaries";
import { saveUsageLog } from "../db/usageLogs";
import { extractLastUserText, fetchSweepyReminders, injectMemoryPatchAsSystemMessage, selectMemoriesForInjection } from "../memory/inject";
import { compressHistoryIfNeeded, type CompressResult } from "../memory/compress";
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
import { loadHeartbeatBody, makeReplayable, storeHeartbeatBody } from "../proxy/heartbeatPrefix";
import { buildOpenAICompatRequest, buildOpenAIRequestFromAssembled, callOpenAICompat } from "../proxy/openaiAdapter";
import { classifyProvider, isFable5Model, resolveTargetModel } from "../proxy/resolveModel";
import { streamAnthropicToOpenAI } from "../proxy/streamAnthropic";
import { stashThinkingSignature } from "../proxy/thinkingStash";
import { streamOpenAIWithTee } from "../proxy/streamOpenAI";
import { CONTENT_RULES } from "../preset/regexRules";
import { applyRegexRules } from "../preset/regexPipeline";
import type { Env, MemoryApiRecord, OpenAIChatMessage, OpenAIChatRequest, OpenAIChatResponse } from "../types";
import { openAiError } from "../utils/json";
import {
  getLastUserVisionImageParts,
  hasNonToolVisionImageContent,
  stripNonToolVisionImages
} from "../utils/messages";

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

export function prepareHistoryForTargetModel(
  env: Env,
  messages: OpenAIChatMessage[],
  namespace: string,
  targetModel: string
): Promise<CompressResult> {
  if (!isFable5Model(targetModel)) {
    return compressHistoryIfNeeded(env, messages, namespace);
  }

  const chatMessageCount = messages.filter(
    (message) => message.role === "user" || message.role === "assistant"
  ).length;
  return Promise.resolve({
    summary: null,
    messages,
    meta: {
      original_count: chatMessageCount,
      compressed_count: 0,
      kept_count: chatMessageCount,
      cache_hit: false,
      compress_boundary: 0,
      total_segments: 0,
      segments_computed: 0,
    },
  });
}

/**
 * 将压缩摘要注入消息区头部（回退路径用）。以 user 消息承载而非 system 块，
 * 使其落在滚动缓存前缀内（摘要只在压缩边界跳变时才变化，缓存读按一折计费）；
 * 与首条 user 消息合并以保持角色交替合法。
 */
function injectCompressedSummary(
  messages: OpenAIChatMessage[],
  summary: string
): OpenAIChatMessage[] {
  const text = `<conversation_summary>\n以下是本次对话较早部分的第三人称档案摘要，更早的原文已省略。摘要仅供回忆事实与脉络；对话的语气、称呼与情感状态以其后的近期消息原文为准，不要模仿摘要的记录口吻：\n${summary}\n</conversation_summary>`;
  const result = [...messages];
  const first = result.findIndex((m) => m.role !== "system");
  if (first === -1) {
    result.push({ role: "user", content: text });
    return result;
  }
  const msg = result[first];
  if (msg.role !== "user") {
    result.splice(first, 0, { role: "user", content: text });
  } else if (typeof msg.content === "string") {
    result[first] = { ...msg, content: `${text}\n\n${msg.content}` };
  } else if (Array.isArray(msg.content)) {
    result[first] = { ...msg, content: [{ type: "text", text }, ...msg.content] } as OpenAIChatMessage;
  } else {
    result[first] = { ...msg, content: text };
  }
  return result;
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

function shanghaiDateKey(date = new Date()): string {
  const parts = new Intl.DateTimeFormat("en", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  }).formatToParts(date);
  const values = Object.fromEntries(
    parts
      .filter((part) => part.type !== "literal")
      .map((part) => [part.type, part.value])
  );
  return `${values.year}-${values.month}-${values.day}`;
}

async function claimDailyReminders(
  env: Env,
  namespace: string,
  reminders: string[]
): Promise<string[]> {
  if (reminders.length === 0) return [];

  const key = `date_reminders:${namespace}:${shanghaiDateKey()}`;
  let claimed: boolean;
  try {
    claimed = await tryStartIdempotentTask(env.DB, {
      key,
      taskType: "date_reminders"
    });
  } catch (error) {
    console.error(
      "daily reminder claim failed",
      error instanceof Error ? error.message : error
    );
    return [];
  }

  if (!claimed) return [];

  try {
    await finishIdempotentTask(env.DB, { key, status: "done" });
  } catch (error) {
    // The unique INSERT already consumed today's claim. Keep injecting this
    // request so a status-only update failure cannot silently lose reminders.
    console.error(
      "daily reminder claim status update failed",
      error instanceof Error ? error.message : error
    );
  }

  return reminders;
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

  if (isHeartbeat) {
    // 心跳不落库也不该走流式；防止下游 conversation!.id 空指针
    body = { ...body, stream: false };

    // 用最近一次真实请求的客户端 body 顶替裸 ping，下方走与真实请求完全
    // 相同的拼装管线（共用消息模板）：拼装结构随代码演进，心跳前缀永远
    // 与下一轮真实请求同构。发送前经 makeReplayable 截断到最后锚点并压
    // max_tokens。无存储时保持裸 ping 旧路径。
    const storedBody = await loadHeartbeatBody(env, auth.profile.namespace);
    if (storedBody) {
      body = { ...storedBody, stream: false };
      console.log("[heartbeat] rebuilding from stored client body,", body.messages.length, "messages");
    }
  }

  let targetModel: string;
  try {
    targetModel = resolveTargetModel(body.model, auth.profile, env);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to resolve target model";
    return openAiError(message, 500);
  }

  const provider = classifyProvider(targetModel);

  if (isHeartbeat && provider !== "anthropic") {
    // 缓存保活只对 Anthropic 有意义；万一模型映射改到别家，
    // 存储的完整历史不该在这里跑出一次全价生成
    body = { ...body, max_tokens: 8 };
  }

  let visionOutput: string | null = null;
  const userImageParts = getLastUserVisionImageParts(body);
  if (hasNonToolVisionImageContent(body) && env.VISION_MODEL) {
    if (userImageParts.length > 0) {
      try {
        const visionRes = await callOpenAICompat(env, {
          model: env.VISION_MODEL,
          messages: [
            { role: "system", content: "你是图片描述工具。如实、详细地描述图片中看到的所有内容：物体、人物、场景、文字、颜色、布局。只输出描述本身，不要加任何分析、解读、评论或开场白。" },
            { role: "user", content: [{ type: "text", text: "请描述这张图片。" }, ...userImageParts] },
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
    body = {
      ...body,
      messages: stripNonToolVisionImages(body.messages),
    };
    if (visionOutput) {
      const visionTag = `\n\n<vision_context>\n用户发送了一张图片。以下是图片的描述：\n${visionOutput}\n</vision_context>`;
      const msgs = [...body.messages];
      for (let i = msgs.length - 1; i >= 0; i--) {
        if (msgs[i].role === "user") {
          const cur = msgs[i].content;
          if (Array.isArray(cur)) {
            msgs[i] = { ...msgs[i], content: [...cur, { type: "text", text: visionTag }] };
          } else {
            msgs[i] = { ...msgs[i], content: (typeof cur === "string" ? cur : "") + visionTag };
          }
          break;
        }
      }
      body = { ...body, messages: msgs };
    }
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
  const [compressResult, memorySelection, pinnedPersonaMemories, latestSummary, fetchedReminders] = await Promise.all([
    prepareHistoryForTargetModel(env, body.messages, auth.profile.namespace, targetModel),
    selectMemoriesForInjection(env, { profile: auth.profile, query: memoryQuery }),
    fetchPinnedPersonaMemories(env, auth.profile.namespace),
    getLatestSummary(env.DB, auth.profile.namespace),
    isHeartbeat ? Promise.resolve([]) : fetchSweepyReminders(env),
  ]);
  const memories = memorySelection.memories;
  const reminders = await claimDailyReminders(env, auth.profile.namespace, fetchedReminders);

  // If compression happened, use trimmed messages for the assembler
  const assemblerBody = compressResult.summary
    ? { ...body, messages: compressResult.messages }
    : body;
  const compressedSummary = compressResult.summary;
  // 回退路径（历史含 tool 消息时）同样必须吃到压缩结果，否则完整历史原样上行；
  // 摘要注入消息区头部而非 system，见 injectCompressedSummary。
  const fallbackBody = compressedSummary
    ? { ...assemblerBody, messages: injectCompressedSummary(assemblerBody.messages, compressedSummary) }
    : assemblerBody;
  const summaryEntry = latestSummary ? { content: latestSummary.content } : null;

  let upstream: Response;
  let clientSystemHash: string | null = null;
  let cacheAnchorBlock: string | null = null;
  try {
    if (provider === "anthropic") {
      if (!isHeartbeat) {
        // 存客户端原始 body（vision 剥离后、压缩前），不存组装成品：
        // 心跳重建走同一条管线，结构变更不再搁浅，见 heartbeatPrefix.ts
        ctx.waitUntil(storeHeartbeatBody(env, auth.profile.namespace, body));
      }
      if (hasToolContent(assemblerBody)) {
        // Tool messages / tool_calls not yet supported by assembler — fall back
        const anthropicRequest = await buildAnthropicNativeRequest(fallbackBody, {
          env,
          targetModel,
          namespace: auth.profile.namespace,
          memories,
          reminders
        });
        upstream = await callAnthropicNative(
          env,
          isHeartbeat ? makeReplayable(anthropicRequest) : anthropicRequest,
          targetModel
        );
      } else {
        const assembled = assemble({
          request: assemblerBody,
          pinnedPersonaMemories,
          summaryEntry,
          ragMemories: memories,
          reminders,
          visionOutput: null,
          compressedSummary,
        });
        clientSystemHash = assembled.meta.client_system_hash;
        cacheAnchorBlock = assembled.meta.anchor_index >= 0 ? "client_system" : null;
        // NOTE: Anthropic adapter stringifies structured content (image_url etc.)
        // as a temporary fallback; native Anthropic image support will be added
        // when the vision pipeline is wired in.
        const anthropicRequest = buildAnthropicRequestFromAssembled(body, targetModel, assembled, env);
        upstream = await callAnthropicNative(
          env,
          isHeartbeat ? makeReplayable(anthropicRequest) : anthropicRequest,
          targetModel
        );
      }
    } else {
      if (hasToolContent(assemblerBody)) {
        // Tool messages / tool_calls not yet supported by assembler — fall back
        // The dynamic selector intentionally excludes fixed pinned persona /
        // identity records. The assembler normally injects those separately;
        // this fallback must add them explicitly without adding them to the
        // dynamic commitMemoryIds set.
        const patchedBody = await injectMemoryPatchAsSystemMessage(
          fallbackBody,
          [...pinnedPersonaMemories, ...memories],
          env,
          reminders
        );
        const upstreamRequest = buildOpenAICompatRequest(patchedBody, targetModel);
        upstream = await callOpenAICompat(env, upstreamRequest);
      } else {
        const assembled = assemble({
          request: assemblerBody,
          pinnedPersonaMemories,
          summaryEntry,
          ragMemories: memories,
          reminders,
          visionOutput: null,
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

  // 只有成功送达上游 prompt 的最终动态记忆才计活跃。候选搜索、静态
  // pinned/persona、filter disabled 和 heartbeat 都不会进入 commitMemoryIds。
  if (!isHeartbeat && latestUserMessageId && memorySelection.commitMemoryIds.length > 0) {
    ctx.waitUntil(
      commitMemoryInjection(env, {
        commitId: latestUserMessageId,
        memoryIds: memorySelection.commitMemoryIds,
      }).catch((error) => {
        console.error("memory injection commit failed", error);
        return null;
      })
    );
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

    if (isHeartbeat) {
      console.log(
        "[heartbeat] replay ok, cache_read =",
        parsed.usage?.cache_read_input_tokens ?? 0,
        "cache_creation =",
        parsed.usage?.cache_creation_input_tokens ?? 0
      );
    }

    // 服务端兜底：非流式同样按 tool_use id 暂存签名 thinking 块（与流式路径对齐）。
    // await：写入必须先于客户端携带 tool_result 的下一次请求。
    {
      const rawBlocks =
        (anthropicParsed as { content?: Array<{ type?: string; id?: string; thinking?: string; signature?: string }> })
          .content ?? [];
      const signedThinking = rawBlocks.find(
        (b) => b.type === "thinking" && typeof b.signature === "string" && b.signature
      );
      const toolUseIds = rawBlocks
        .filter((b) => b.type === "tool_use" && typeof b.id === "string" && b.id)
        .map((b) => b.id as string);
      if (signedThinking && toolUseIds.length > 0) {
        try {
          await stashThinkingSignature(env.DB, {
            namespace: auth.profile.namespace,
            toolUseIds,
            thinking: signedThinking.thinking ?? "",
            signature: signedThinking.signature as string
          });
        } catch (error) {
          console.error("failed to stash thinking signature", error);
        }
      }
    }
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
