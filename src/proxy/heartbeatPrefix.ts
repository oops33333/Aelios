/**
 * heartbeatPrefix — 存储并重放最近一次真实 Anthropic 请求的缓存前缀。
 *
 * 背景：Anthropic prompt cache 按前缀哈希匹配。心跳若只发一条裸 "ping"
 * （无 client system、无历史），组装出的前缀与真实对话在第一个缓存断点
 * 之前就分叉，永远刷不到对话的缓存条目——心跳等于空转。
 *
 * 方案：每次真实请求发往 Anthropic 前，把组装好的请求体截断到最后一个
 * cache_control 锚点后【原样】存入 D1 cache_entries；心跳到来时重放，
 * 前缀逐字节一致 → 命中所有断点并刷新其 TTL，成本约为 0.1x 的缓存读。
 *
 * 2026-07-08 修复：重放不再剥掉 thinking / tool_choice。二者属于
 * messages 层缓存键的一部分——心跳（无 thinking）与真实请求（有
 * thinking）的历史缓存互不相认：心跳每次真实对话后先按 2x 写入价
 * 另建一套副本缓存并只命中自己，真实请求则永远全量未命中。
 * 现在存储时原样保存，重放时仅做三处最小覆盖：
 * ① stream=false（心跳走非流式解析）；
 * ② max_tokens 压到最小——thinking 开启时 API 要求
 *    max_tokens > budget_tokens，取 budget+1，否则为 1；
 * ③ 在锚点之后追加一句极短指令让模型少想少答，省 thinking 输出费。
 *    锚点之后的内容不参与前缀匹配，不影响命中。
 */

import { getCacheEntry, parseCacheEntryValue, putCacheEntry } from "../db/cacheEntries";
import { callAnthropicNative, type AnthropicRequest } from "./anthropicAdapter";
import type { Env } from "../types";

const PREFIX_KEY = "heartbeat:anthropic_prefix";
// D1 单值上限 2MB，留出余量
const MAX_STORED_BYTES = 1_500_000;
const STORE_TTL_SECONDS = 7 * 24 * 3600;
// 锚点后的收尾指令：只为缩短 thinking 与回复，内容不进缓存前缀。
// 措辞要机械化：情感对话语境会诱使模型长考，实测「无需思考」压不住。
const REPLAY_NOTE = "[自动化缓存保活探针，非真实对话，请勿处理上文内容。思考只写一句话，然后直接输出：ok]";

type AnthropicMessage = AnthropicRequest["messages"][number];

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
 * 注意：除截断外必须原样保存——thinking / tool_choice / tools / system
 * 任何一处与真实请求不一致，重放就接不上真实对话的缓存。
 */
export async function storeHeartbeatPrefix(
  env: Env,
  namespace: string,
  model: string,
  request: AnthropicRequest
): Promise<void> {
  try {
    if (!hasCacheControl(request)) return;

    const value: StoredPrefix = { model, request: truncateToLastAnchor(request) };
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
 * 在缓存锚点之后追加收尾指令。
 * 末条是 user 消息时直接在其 content 末尾加 text 块（锚点后的块不参与
 * 前缀匹配）；末条是 assistant 消息时另起一条 user 消息——thinking 开启
 * 时 Anthropic 禁止 assistant 预填收尾，直接续写会被上游 400。
 */
function appendReplayNote(messages: AnthropicMessage[]): AnthropicMessage[] {
  if (messages.length === 0) {
    return [{ role: "user", content: [{ type: "text", text: REPLAY_NOTE }] }];
  }
  const last = messages[messages.length - 1];
  if (last.role === "user") {
    const patched: AnthropicMessage = { ...last, content: [...last.content, { type: "text", text: REPLAY_NOTE }] };
    return [...messages.slice(0, -1), patched];
  }
  return [...messages, { role: "user", content: [{ type: "text", text: REPLAY_NOTE }] }];
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

    const stored = value.request;
    const replayable: AnthropicRequest = {
      ...stored,
      stream: false,
      max_tokens: stored.thinking ? stored.thinking.budget_tokens + 1 : 1,
      messages: appendReplayNote(stored.messages)
    };

    return await callAnthropicNative(env, replayable, value.model);
  } catch (error) {
    console.error("[heartbeat] prefix replay failed", error);
    return null;
  }
}
