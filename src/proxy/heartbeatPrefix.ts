/**
 * heartbeatPrefix — 存储并重放最近一次真实 Anthropic 请求的缓存前缀。
 *
 * 背景：Anthropic prompt cache 按前缀哈希匹配。心跳若只发一条裸 "ping"
 * （无 client system、无历史），组装出的前缀与真实对话在第一个缓存断点
 * 之前就分叉，永远刷不到对话的缓存条目——心跳等于空转。
 *
 * 方案：每次真实请求发往 Anthropic 前，把组装好的请求体截断到最后一个
 * cache_control 锚点后存入 D1 cache_entries；心跳到来时原样重放
 * （max_tokens=1，关 thinking），前缀逐字节一致 → 命中所有断点并刷新
 * 其 TTL，成本仅为 0.1x 的缓存读。
 */

import { getCacheEntry, parseCacheEntryValue, putCacheEntry } from "../db/cacheEntries";
import { callAnthropicNative, type AnthropicRequest } from "./anthropicAdapter";
import type { Env } from "../types";

const PREFIX_KEY = "heartbeat:anthropic_prefix";
// D1 单值上限 2MB，留出余量
const MAX_STORED_BYTES = 1_500_000;
const STORE_TTL_SECONDS = 7 * 24 * 3600;

interface StoredPrefix {
  model: string;
  request: AnthropicRequest;
}

function hasCacheControl(request: AnthropicRequest): boolean {
  if (request.system?.some((block) => block.cache_control)) return true;
  return request.messages.some((message) =>
    message.content.some((block) => "cache_control" in block && (block as { cache_control?: unknown }).cache_control)
  );
}

/**
 * 截断到消息里最后一个 cache_control 锚点：锚点之后的内容（动态记忆补丁、
 * volatile 上下文）不在缓存前缀内，重放时带上只会按全价重算，去掉即可。
 */
function truncateToLastAnchor(request: AnthropicRequest): AnthropicRequest {
  const messages = request.messages;
  let msgIdx = -1;
  let blockIdx = -1;

  for (let i = messages.length - 1; i >= 0 && msgIdx < 0; i -= 1) {
    const content = messages[i].content;
    for (let j = content.length - 1; j >= 0; j -= 1) {
      if ((content[j] as { cache_control?: unknown }).cache_control) {
        msgIdx = i;
        blockIdx = j;
        break;
      }
    }
  }

  if (msgIdx < 0) return request;

  const kept = messages.slice(0, msgIdx + 1);
  kept[msgIdx] = { ...kept[msgIdx], content: kept[msgIdx].content.slice(0, blockIdx + 1) };
  return { ...request, messages: kept };
}

/**
 * 真实请求发出前调用（ctx.waitUntil，非阻塞）。
 * 无任何 cache_control 锚点（缓存关闭）时不存。
 */
export async function storeHeartbeatPrefix(
  env: Env,
  namespace: string,
  model: string,
  request: AnthropicRequest
): Promise<void> {
  try {
    if (!hasCacheControl(request)) return;

    const truncated = truncateToLastAnchor(request);
    const replayable: AnthropicRequest = {
      ...truncated,
      stream: false,
      max_tokens: 1,
      thinking: undefined,
      tool_choice: undefined,
      temperature: undefined,
      cache_control: undefined
    };

    const value: StoredPrefix = { model, request: replayable };
    const serialized = JSON.stringify(value);
    if (serialized.length > MAX_STORED_BYTES) {
      console.log("[heartbeat] prefix too large to store:", serialized.length);
      return;
    }

    await putCacheEntry(env.DB, {
      namespace,
      key: PREFIX_KEY,
      value,
      contentType: "application/json",
      tags: ["heartbeat"],
      ttlSeconds: STORE_TTL_SECONDS
    });
  } catch (error) {
    console.error("[heartbeat] store prefix failed", error);
  }
}

/**
 * 心跳到来时重放存储的前缀。没有存储或调用失败返回 null，
 * 调用方回退到旧的 ping 组装路径。
 */
export async function replayHeartbeatPrefix(env: Env, namespace: string): Promise<Response | null> {
  try {
    const record = await getCacheEntry(env.DB, { namespace, key: PREFIX_KEY });
    if (!record) return null;

    const value = parseCacheEntryValue(record) as StoredPrefix | null;
    if (!value || typeof value !== "object" || !value.model || !value.request) return null;

    return await callAnthropicNative(env, value.request, value.model);
  } catch (error) {
    console.error("[heartbeat] prefix replay failed", error);
    return null;
  }
}
