import { saveAssistantMessage } from "../db/messages";
import { saveUsageLog } from "../db/usageLogs";
import { enqueueMemoryMaintenanceIfNeeded, enqueueRetentionIfNeeded } from "../queue/producer";
import { getAnthropicCacheMode, mapAnthropicStopReason, normalizeAnthropicUsage } from "./anthropicAdapter";
import {
  createThinkingFilterState,
  flushStreamFilter,
  processStreamChunk,
  type ThinkingFilterState,
} from "../preset/streamFilters";
import type { Env, KeyProfile, TokenUsage } from "../types";
import { getSseData, splitSseEvents } from "../utils/sseParser";

interface StreamAnthropicOptions {
  env: Env;
  ctx: ExecutionContext;
  profile: KeyProfile;
  conversationId: string;
  fromMessageId?: string;
  requestModel: string;
  upstreamModel: string;
  provider: string;
  clientSystemHash?: string | null;
  cacheAnchorBlock?: string | null;
}

interface StreamToolCallDelta {
  index: number;
  id?: string;
  type?: "function";
  function?: { name?: string; arguments?: string };
}

interface StreamDelta {
  content?: string;
  reasoning_content?: string;
  tool_calls?: StreamToolCallDelta[];
  thinking_blocks?: Array<{ type: "thinking"; thinking: string; signature?: string }>;
}

interface StreamState {
  assistantText: string;
  reasoningText: string;
  thinkingSignature: string;
  finishReason: string | null;
  usage?: TokenUsage;
  thinkingFilter: ThinkingFilterState;
  /** Anthropic content block index → OpenAI tool_calls index */
  toolCallIndexByBlock: Map<number, number>;
  toolCallCount: number;
}

function openAIChunk(delta: StreamDelta, finishReason: string | null = null): Uint8Array {
  return new TextEncoder().encode(
    `data: ${JSON.stringify({
      choices: [
        {
          index: 0,
          delta,
          finish_reason: finishReason
        }
      ]
    })}\n\n`
  );
}

function doneChunk(): Uint8Array {
  return new TextEncoder().encode("data: [DONE]\n\n");
}

function consumeAnthropicData(data: string, state: StreamState): StreamDelta | null {
  try {
    const parsed = JSON.parse(data) as {
      type?: string;
      index?: number;
      content_block?: {
        type?: string;
        id?: string;
        name?: string;
      };
      delta?: {
        type?: string;
        text?: string;
        thinking?: string;
        signature?: string;
        partial_json?: string;
        stop_reason?: string | null;
      };
      usage?: TokenUsage;
      message?: {
        usage?: TokenUsage;
      };
    };

    if (parsed.type === "message_start" && parsed.message?.usage) {
      state.usage = normalizeAnthropicUsage(parsed.message.usage);
    }

    // tool_use 块开场：登记块号→工具序号映射，向客户端发出 id/name 帧
    if (
      parsed.type === "content_block_start" &&
      parsed.content_block?.type === "tool_use" &&
      typeof parsed.index === "number"
    ) {
      const toolIndex = state.toolCallCount;
      state.toolCallCount += 1;
      state.toolCallIndexByBlock.set(parsed.index, toolIndex);
      return {
        tool_calls: [
          {
            index: toolIndex,
            id: parsed.content_block.id || `toolu_${toolIndex}`,
            type: "function",
            function: { name: parsed.content_block.name || "", arguments: "" }
          }
        ]
      };
    }

    // tool_use 入参增量：partial_json → arguments 增量帧
    if (
      parsed.type === "content_block_delta" &&
      parsed.delta?.type === "input_json_delta" &&
      typeof parsed.index === "number" &&
      state.toolCallIndexByBlock.has(parsed.index)
    ) {
      const partial = parsed.delta.partial_json ?? "";
      if (!partial) return null;
      return {
        tool_calls: [
          {
            index: state.toolCallIndexByBlock.get(parsed.index)!,
            function: { arguments: partial }
          }
        ]
      };
    }

    if (parsed.type === "content_block_delta" && parsed.delta?.type === "text_delta" && parsed.delta.text) {
      // Filter visible content: strip leaked <thinking>, replace dashes, remove ■.
      // reasoning_content (thinking_delta) is handled separately and never filtered.
      const filtered = processStreamChunk(parsed.delta.text, state.thinkingFilter);
      if (!filtered) return null;
      state.assistantText += filtered;
      return { content: filtered };
    }

    if (parsed.type === "content_block_delta" && parsed.delta?.type === "thinking_delta" && parsed.delta.thinking) {
      // reasoning_content is NEVER filtered — pass through as-is.
      state.reasoningText += parsed.delta.thinking;
      return { reasoning_content: parsed.delta.thinking };
    }

    // thinking 签名：暂存，流末以 thinking_blocks 帧整体返还，供工具回环回传
    if (parsed.type === "content_block_delta" && parsed.delta?.type === "signature_delta" && parsed.delta.signature) {
      state.thinkingSignature += parsed.delta.signature;
      return null;
    }

    if (parsed.type === "message_delta") {
      if (parsed.delta?.stop_reason) state.finishReason = parsed.delta.stop_reason;
      if (parsed.usage) {
        state.usage = normalizeAnthropicUsage({
          ...(state.usage ?? {}),
          ...parsed.usage
        });
      }
    }
  } catch {
    // Ignore malformed provider events while keeping the client stream alive.
  }

  return null;
}

async function persistStreamResult(options: StreamAnthropicOptions, state: StreamState): Promise<void> {
  const messageId = await saveAssistantMessage(options.env.DB, {
    conversationId: options.conversationId,
    namespace: options.profile.namespace,
    source: options.profile.source,
    content: state.assistantText,
    requestModel: options.requestModel,
    upstreamModel: options.upstreamModel,
    provider: options.provider,
    stream: true,
    finishReason: state.finishReason,
    usage: state.usage,
    cacheMode: getAnthropicCacheMode(options.env),
    cacheTtl: options.env.ANTHROPIC_CACHE_TTL || "5m"
  });

  await saveUsageLog(options.env.DB, {
    messageId,
    namespace: options.profile.namespace,
    provider: options.provider,
    model: options.upstreamModel,
    usage: state.usage,
    cacheMode: getAnthropicCacheMode(options.env),
    cacheTtl: options.env.ANTHROPIC_CACHE_TTL || "5m",
    clientSystemHash: options.clientSystemHash ?? null,
    cacheAnchorBlock: options.cacheAnchorBlock ?? null
  });

  await enqueueMemoryMaintenanceIfNeeded(options.env, {
    namespace: options.profile.namespace,
    conversationId: options.conversationId,
    fromMessageId: options.fromMessageId,
    toMessageId: messageId,
    source: options.profile.source
  });

  await enqueueRetentionIfNeeded(options.env, options.profile.namespace);
}

export function streamAnthropicToOpenAI(upstream: Response, options: StreamAnthropicOptions): Response {
  if (!upstream.body) {
    return new Response(upstream.body, {
      status: upstream.status,
      headers: upstream.headers
    });
  }

  const { readable, writable } = new TransformStream<Uint8Array, Uint8Array>();
  const reader = upstream.body.getReader();
  const writer = writable.getWriter();
  const decoder = new TextDecoder();
  const state: StreamState = {
    assistantText: "",
    reasoningText: "",
    thinkingSignature: "",
    finishReason: null,
    thinkingFilter: createThinkingFilterState(),
    toolCallIndexByBlock: new Map(),
    toolCallCount: 0
  };

  void (async () => {
    let buffered = "";

    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!value) continue;

        buffered += decoder.decode(value, { stream: true });
        const parsed = splitSseEvents(buffered);
        buffered = parsed.rest;

        for (const event of parsed.events) {
          const data = getSseData(event);
          if (!data) continue;
          const delta = consumeAnthropicData(data, state);
          if (delta) await writer.write(openAIChunk(delta));
        }
      }

      buffered += decoder.decode();
      const parsed = splitSseEvents(buffered);
      for (const event of parsed.events) {
        const data = getSseData(event);
        if (!data) continue;
        const delta = consumeAnthropicData(data, state);
        if (delta) await writer.write(openAIChunk(delta));
      }

      // Flush held trailing dash or unclosed <think> text at stream end.
      const trailing = flushStreamFilter(state.thinkingFilter);
      if (trailing) {
        state.assistantText += trailing;
        await writer.write(openAIChunk({ content: trailing }));
      }

      // 带签名的 thinking 整块返还（LiteLLM 风格），供客户端工具回环时回传
      if (state.thinkingSignature && state.reasoningText) {
        await writer.write(
          openAIChunk({
            thinking_blocks: [
              {
                type: "thinking",
                thinking: state.reasoningText,
                signature: state.thinkingSignature
              }
            ]
          })
        );
      }

      // 收束帧：tool_use → tool_calls，end_turn → stop
      await writer.write(
        openAIChunk(
          {},
          mapAnthropicStopReason(state.finishReason) ?? (state.toolCallCount > 0 ? "tool_calls" : "stop")
        )
      );

      await writer.write(doneChunk());
      await writer.close();
      options.ctx.waitUntil(
        persistStreamResult(options, state).catch((error) => {
          console.error("failed to persist anthropic stream result", error);
        })
      );
    } catch (error) {
      console.error("anthropic stream proxy error", error);
      await writer.abort(error);
    } finally {
      reader.releaseLock();
      writer.releaseLock();
    }
  })();

  return new Response(readable, {
    status: upstream.status,
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache"
    }
  });
}
