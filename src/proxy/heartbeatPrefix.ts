/**
 * heartbeatPrefix — 存储最近一次真实请求的客户端原始 body，心跳时经由与
 * 真实请求完全相同的拼装管线重建请求（共用同一个消息模板）。
 *
 * 旧方案是把组装好的 Anthropic 请求体逐字节存储、心跳时原样重放。字节重放
 * 在稳态下最优（精确刷新上一轮写下的缓存条目），但拼装结构一变（部署新块序、
 * 改 prompt 模板），存储的旧字节就成了孤儿：心跳整夜按废前缀空转，直到下一次
 * 真实对话覆盖存储才自愈。2026-07-11 实测：8ce0d87 重排块序部署后，
 * 19:11–03:28 UTC 七跳心跳刷的全是旧结构前缀，真实请求一次也用不上。
 *
 * 现方案存「原料」不存「成品」：真实请求进 anthropic 分支时把客户端 body
 * （vision 剥离后、压缩前）存入 D1；心跳到来时取出顶替裸 ping，从管线头部
 * 灌入——压缩查 D1 缓存、persona/summary 现取、同一套 BLOCK_ORDER 组装。
 * 代码怎么演进，心跳前缀跟着怎么变，永不掉队；部署后第一跳还能抢在真实
 * 对话之前把新结构的前缀写热。
 *
 * 稳态字节一致性：压缩按 (消息, COMPRESS_PROMPT_VERSION) 命中 D1 缓存，
 * persona/summary 只在写库时变化，RAG 查询词取自同一条末条用户消息——
 * 两次真实对话之间无写入则重建结果逐字节等于上一轮，成本仍是 0.1x 缓存读。
 *
 * 2026-07-08 的教训（thinking / tool_choice 属于缓存键，重放不得剥除或
 * 另配一套）在此方案下自动满足：参数派生走的就是真实管线那一份代码。
 * 发送前仅做三处最小覆盖，见 makeReplayable。
 */

import { getCacheEntry, parseCacheEntryValue, putCacheEntry } from "../db/cacheEntries";
import type { AnthropicRequest } from "./anthropicAdapter";
import type { Env, OpenAIChatRequest } from "../types";

const BODY_KEY = "heartbeat:client_body";
// D1 单值上限 2MB，留出余量
const MAX_STORED_BYTES = 1_500_000;
const STORE_TTL_SECONDS = 7 * 24 * 3600;
// 锚点后的收尾指令：只为缩短 thinking 与回复，内容不进缓存前缀。
// 措辞要机械化：情感对话语境会诱使模型长考，实测「无需思考」压不住。
const REPLAY_NOTE = "[自动化缓存保活探针，非真实对话，请勿处理上文内容。思考只写一句话，然后直接输出：ok]";

type AnthropicMessage = AnthropicRequest["messages"][number];

interface StoredHeartbeatBody {
  body: OpenAIChatRequest;
}

/**
 * 真实请求发出前调用（ctx.waitUntil，非阻塞）。只在 anthropic 分支挂载——
 * 缓存保活只对 Anthropic 有意义，也保证心跳重建时模型解析回同一提供方。
 */
export async function storeHeartbeatBody(
  env: Env,
  namespace: string,
  body: OpenAIChatRequest
): Promise<void> {
  try {
    const value: StoredHeartbeatBody = { body: { ...body, stream: false } };
    const serialized = JSON.stringify(value);
    if (serialized.length > MAX_STORED_BYTES) {
      console.log("[heartbeat] client body too large to store:", serialized.length);
      return;
    }

    await putCacheEntry(env.DB, {
      namespace,
      key: BODY_KEY,
      value,
      contentType: "application/json",
      tags: ["heartbeat"],
      ttlSeconds: STORE_TTL_SECONDS
    });
  } catch (error) {
    console.error("[heartbeat] store body failed", error);
  }
}

/**
 * 心跳到来时取出存储的客户端 body。没有存储或解析失败返回 null，
 * 调用方保持裸 ping 旧路径。
 */
export async function loadHeartbeatBody(env: Env, namespace: string): Promise<OpenAIChatRequest | null> {
  try {
    const record = await getCacheEntry(env.DB, { namespace, key: BODY_KEY });
    if (!record) return null;

    const value = parseCacheEntryValue(record) as StoredHeartbeatBody | null;
    if (!value || typeof value !== "object" || !value.body || !Array.isArray(value.body.messages)) {
      return null;
    }
    return value.body;
  } catch (error) {
    console.error("[heartbeat] load body failed", error);
    return null;
  }
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
 * 管线组装完毕、发往上游前对心跳请求做三处最小覆盖：
 * ① 截断到最后一个 cache_control 锚点（锚点后按全价重算，去掉）；
 * ② stream=false（心跳走非流式解析）；
 * ③ max_tokens 压到最小——thinking 开启时 API 要求
 *    max_tokens > budget_tokens，取 budget+1，否则为 1；
 *    并在锚点后追加收尾指令让模型少想少答，省 thinking 输出费。
 */
export function makeReplayable(request: AnthropicRequest): AnthropicRequest {
  const truncated = truncateToLastAnchor(request);
  return {
    ...truncated,
    stream: false,
    max_tokens: truncated.thinking ? truncated.thinking.budget_tokens + 1 : 1,
    messages: appendReplayNote(truncated.messages)
  };
}
