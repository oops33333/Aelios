import { getCacheEntry, parseCacheEntryValue, putCacheEntry } from "../db/cacheEntries";

/**
 * 签名 thinking 块的服务端暂存（D1 cache_entries 表）。
 *
 * 背景：extended thinking 开启时，工具回环的下一轮请求要求末位 assistant
 * tool_use 消息携带原样的签名 thinking 块，否则上游 400。aelios 在流末已
 * 以 LiteLLM 风格 thinking_blocks 帧返还，但多数 OpenAI 兼容客户端
 * （如 RikkaHub）会丢弃该非标准字段，导致 thinking 被降级守卫关闭——
 * thinking 参数逐轮开关横跳会打掉 messages 级缓存断点，缓存全 miss。
 *
 * 方案：响应侧按 tool_use id 暂存 {thinking, signature}，请求侧在守卫
 * 触发前先按 id 回填。内容与上游返回逐字节一致，回填是确定性的，
 * 不影响滚动缓存前缀的稳定性。
 */

const KEY_PREFIX = "thinking-sig:";
/** 工具回环通常在数秒内完成；给足余量以覆盖客户端重试与长工具执行。 */
const TTL_SECONDS = 6 * 60 * 60;

export interface StashedThinking {
  thinking: string;
  signature: string;
}

/** 将本轮的签名 thinking 块按每个 tool_use id 各存一份。 */
export async function stashThinkingSignature(
  db: D1Database,
  input: { namespace: string; toolUseIds: string[]; thinking: string; signature: string }
): Promise<void> {
  if (!input.signature || input.toolUseIds.length === 0) return;
  await Promise.all(
    input.toolUseIds.map((id) =>
      putCacheEntry(db, {
        namespace: input.namespace,
        key: `${KEY_PREFIX}${id}`,
        value: { thinking: input.thinking, signature: input.signature },
        contentType: "application/json",
        ttlSeconds: TTL_SECONDS
      })
    )
  );
}

/** 按 tool_use id 取回暂存的签名 thinking 块；未命中或已过期返回 null。 */
export async function getStashedThinking(
  db: D1Database,
  namespace: string,
  toolUseId: string
): Promise<StashedThinking | null> {
  const record = await getCacheEntry(db, { namespace, key: `${KEY_PREFIX}${toolUseId}` });
  if (!record) return null;
  const value = parseCacheEntryValue(record);
  if (!value || typeof value !== "object") return null;
  const { thinking, signature } = value as { thinking?: unknown; signature?: unknown };
  if (typeof signature !== "string" || !signature) return null;
  return { thinking: typeof thinking === "string" ? thinking : "", signature };
}
