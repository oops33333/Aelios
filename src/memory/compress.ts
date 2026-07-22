/**
 * compress.ts — 对话历史自动压缩（级联版）
 *
 * 当 RikkaHub 发来的完整历史超过阈值（默认 30 轮）时，
 * 按 windowSize（默认 25）条一段做级联压缩：
 *
 *   seg 1: messages[0..24]           → summary_1
 *   seg 2: summary_1 + messages[25..49] → summary_2
 *   seg 3: summary_2 + messages[50..74] → summary_3
 *   ...
 *
 * 每段输入量固定（~上轮摘要 800 字 + 25 条消息），
 * 不随对话长度线性增长，可以支撑任意长对话。
 *
 * 缓存策略：
 *   - 每段独立缓存在 D1 cache_entries
 *   - key 含段号 + 首条消息指纹，防跨对话缓存碰撞
 *   - 快路径：先查最终段缓存，命中则直接返回
 *   - 慢路径：从最后一个命中段开始级联，逐段压缩+缓存
 */

import { getCacheEntry, putCacheEntry, parseCacheEntryValue } from "../db/cacheEntries";
import { callOpenAICompat } from "../proxy/openaiAdapter";
import type { Env, OpenAIChatMessage } from "../types";

// ---------------------------------------------------------------------------
// 配置
// ---------------------------------------------------------------------------

const DEFAULT_THRESHOLD = 30;
const DEFAULT_KEEP_RECENT = 5;
const DEFAULT_COMPRESS_MAX_CHARS = 2000;
const DEFAULT_CACHE_TTL = 86400;
const DEFAULT_COMPRESS_MODEL = "workers-ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast";

function getThreshold(env: Env): number {
  const v = Number(env.HISTORY_COMPRESS_THRESHOLD || DEFAULT_THRESHOLD);
  return Number.isFinite(v) ? Math.max(v, 10) : DEFAULT_THRESHOLD;
}

function getKeepRecent(env: Env): number {
  const v = Number(env.HISTORY_KEEP_RECENT || DEFAULT_KEEP_RECENT);
  return Number.isFinite(v) ? Math.max(v, 2) : DEFAULT_KEEP_RECENT;
}

function getCompressModel(env: Env): string {
  return env.HISTORY_COMPRESS_MODEL || env.MEMORY_FILTER_MODEL || DEFAULT_COMPRESS_MODEL;
}

function getMaxChars(env: Env): number {
  const v = Number(env.HISTORY_COMPRESS_MAX_CHARS || DEFAULT_COMPRESS_MAX_CHARS);
  return Number.isFinite(v) ? v : DEFAULT_COMPRESS_MAX_CHARS;
}

function getCacheTtl(env: Env): number {
  const v = Number(env.HISTORY_COMPRESS_CACHE_TTL || DEFAULT_CACHE_TTL);
  return Number.isFinite(v) ? v : DEFAULT_CACHE_TTL;
}

function isEnabled(env: Env): boolean {
  return env.ENABLE_HISTORY_COMPRESSION === "true";
}

// ---------------------------------------------------------------------------
// 公开接口
// ---------------------------------------------------------------------------

export interface CompressResult {
  summary: string | null;
  messages: OpenAIChatMessage[];
  meta: {
    original_count: number;
    compressed_count: number;
    kept_count: number;
    cache_hit: boolean;
    compress_boundary: number;
    total_segments: number;
    segments_computed: number;
  };
}

/**
 * 从 chatMessages 的压缩边界映射回原始消息数组。
 * 必须从原始数组切片，才能保留边界后的 tool 消息及其配对关系。
 */
function sliceRecentFromBoundary(
  messages: OpenAIChatMessage[],
  chatMessages: OpenAIChatMessage[],
  compressEnd: number
): OpenAIChatMessage[] {
  const boundaryMessage = chatMessages[compressEnd];
  if (!boundaryMessage) return [];

  const origStart = messages.indexOf(boundaryMessage);
  if (origStart < 0) return [];

  return messages.slice(origStart).filter((m) => m.role !== "system");
}

/**
 * 主入口：检查阈值 → 级联压缩 → 返回裁剪后的消息。
 */
export async function compressHistoryIfNeeded(
  env: Env,
  messages: OpenAIChatMessage[],
  namespace: string
): Promise<CompressResult> {
  const noopResult: CompressResult = {
    summary: null,
    messages,
    meta: {
      original_count: 0, compressed_count: 0, kept_count: 0,
      cache_hit: false, compress_boundary: 0,
      total_segments: 0, segments_computed: 0,
    },
  };

  if (!isEnabled(env)) {
    console.log("[compress] DISABLED - ENABLE_HISTORY_COMPRESSION is not 'true'");
    return noopResult;
  }
  console.log("[compress] ENABLED, checking messages...");

  const systemMessages = messages.filter(m => m.role === "system");
  const chatMessages = messages.filter(m => m.role === "user" || m.role === "assistant");

  noopResult.meta.original_count = chatMessages.length;
  noopResult.meta.kept_count = chatMessages.length;

  const threshold = getThreshold(env);
  const keepRecent = getKeepRecent(env);

  console.log(`[compress] chatMessages=${chatMessages.length}, threshold=${threshold}, keepRecent=${keepRecent}`);
  if (chatMessages.length <= threshold) {
    console.log("[compress] below threshold, skipping");
    return noopResult;
  }

  const windowSize = threshold - keepRecent;
  const completeWindows = Math.floor((chatMessages.length - keepRecent) / windowSize);
  const compressEnd = completeWindows * windowSize;

  if (compressEnd < windowSize) return noopResult;

  // recent 从原始数组切片：保留窗口内 tool 消息及其与 assistant tool_calls 的配对。
  // chatMessages[compressEnd] 必为 user/assistant；其之前的 tool 结果属于更早的
  // assistant（随摘要一并丢弃），因此切片内不会出现孤儿 tool_result。
  const recent = sliceRecentFromBoundary(messages, chatMessages, compressEnd);

  // -----------------------------------------------------------------------
  // 级联压缩
  // -----------------------------------------------------------------------

  let summary: string | null = null;
  let segmentsComputed = 0;
  let cacheHit = true;

  // 快路径：最终段缓存命中 → 直接返回
  const finalKey = buildSegmentCacheKey(namespace, completeWindows, chatMessages, windowSize);
  const finalCached = await tryGetCache(env.DB, namespace, finalKey);
  if (finalCached) {
    console.log("[compress] CACHE HIT on final segment");
    return {
      summary: finalCached,
      messages: [...systemMessages, ...recent],
      meta: {
        original_count: chatMessages.length,
        compressed_count: compressEnd,
        kept_count: recent.length,
        cache_hit: true,
        compress_boundary: compressEnd,
        total_segments: completeWindows,
        segments_computed: 0,
      },
    };
  }

  // 慢路径：从后往前找最后一个缓存命中的段
  let lastCachedSeg = 0;
  let lastCachedSummary: string | null = null;

  for (let seg = completeWindows - 1; seg >= 1; seg--) {
    const key = buildSegmentCacheKey(namespace, seg, chatMessages, windowSize);
    const cached = await tryGetCache(env.DB, namespace, key);
    if (cached) {
      lastCachedSeg = seg;
      lastCachedSummary = cached;
      break;
    }
  }

  // 从 lastCachedSeg+1 开始级联压缩到 completeWindows
  summary = lastCachedSummary;
  cacheHit = false;

  for (let seg = lastCachedSeg + 1; seg <= completeWindows; seg++) {
    const segStart = (seg - 1) * windowSize;
    const segEnd = seg * windowSize;
    const segMessages = chatMessages.slice(segStart, segEnd);

    try {
      console.log(`[compress] calling model for segment ${seg}, model=${getCompressModel(env)}`);
      summary = await callCompressModel(env, segMessages, summary);
      console.log(`[compress] segment ${seg} compressed OK, summary length=${summary.length}`);
      segmentsComputed++;
    } catch (error) {
      console.error(`[compress] segment ${seg} failed:`, error);

      // 已有前序段摘要时，沿用最后成功结果，并把裁剪边界退回该段末尾。
      // 失败段及之后的消息保留原文，避免丢消息，也避免整段历史重新上行。
      const completedSegments = seg - 1;
      if (summary && completedSegments > 0) {
        const partialCompressEnd = completedSegments * windowSize;
        const partialRecent = sliceRecentFromBoundary(
          messages,
          chatMessages,
          partialCompressEnd
        );

        console.warn(
          `[compress] using last successful segment ${completedSegments}; ` +
          `keeping ${partialRecent.length} recent messages until segment ${seg} succeeds`
        );

        return {
          summary,
          messages: [...systemMessages, ...partialRecent],
          meta: {
            original_count: chatMessages.length,
            compressed_count: partialCompressEnd,
            kept_count: partialRecent.length,
            cache_hit: false,
            compress_boundary: partialCompressEnd,
            total_segments: completeWindows,
            segments_computed: segmentsComputed,
          },
        };
      }

      // 第一段就失败、没有任何可沿用摘要时，只能保留完整历史。
      return { ...noopResult, meta: { ...noopResult.meta, original_count: chatMessages.length, kept_count: chatMessages.length } };
    }

    // 写缓存（非阻塞）
    const key = buildSegmentCacheKey(namespace, seg, chatMessages, windowSize);
    try {
      await putCacheEntry(env.DB, {
        namespace,
        key,
        value: summary,
        tags: ["history_compression"],
        ttlSeconds: getCacheTtl(env),
      });
    } catch {
      // 写缓存失败不影响返回
    }
  }

  return {
    summary,
    messages: [...systemMessages, ...recent],
    meta: {
      original_count: chatMessages.length,
      compressed_count: compressEnd,
      kept_count: recent.length,
      cache_hit: cacheHit,
      compress_boundary: compressEnd,
      total_segments: completeWindows,
      segments_computed: segmentsComputed,
    },
  };
}

// ---------------------------------------------------------------------------
// 缓存辅助
// ---------------------------------------------------------------------------

function simpleHash(text: string): string {
  let hash = 5381;
  for (let i = 0; i < text.length; i++) {
    hash = ((hash << 5) + hash + text.charCodeAt(i)) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

/**
 * 段缓存 key：含 prompt 版本 + 段号 + 首条消息指纹。
 * 不同对话的同一段号内容不同，指纹不同，不会碰撞。
 * 同一对话的同一段，消息不变，指纹不变，缓存命中。
 * 压缩 prompt 改动时递增版本号，强制全量重压缩，避免旧风格摘要经缓存沿用。
 */
const COMPRESS_PROMPT_VERSION = 5;

/**
 * 剥离内联思考链。qwen 等模型可能把思考以 <think>…</think> 内联在 content 里，
 * 不剥会随摘要入 D1 缓存并每轮注入上游。reasoning 参数已关，此处是兜底。
 */
function stripThinkingBlocks(text: string): string {
  let out = text.replace(/<think(?:ing)?>[\s\S]*?<\/think(?:ing)?>/gi, "");
  // 开标签缺失、只剩前导闭合标签的残链（部分模型省略开标签）
  if (/<\/think(?:ing)?>/i.test(out)) {
    out = out.replace(/^[\s\S]*?<\/think(?:ing)?>\s*/i, "");
  }
  return out.trim();
}

function buildSegmentCacheKey(
  namespace: string,
  segNum: number,
  chatMessages: OpenAIChatMessage[],
  windowSize: number
): string {
  const segStart = (segNum - 1) * windowSize;
  const firstContent = contentToText(chatMessages[segStart]?.content).slice(0, 200);
  const fp = simpleHash(firstContent);
  return `compress:${namespace}:v${COMPRESS_PROMPT_VERSION}:seg${segNum}_${fp}`;
}

async function tryGetCache(
  db: D1Database,
  namespace: string,
  key: string
): Promise<string | null> {
  try {
    const record = await getCacheEntry(db, { namespace, key });
    if (!record) return null;
    const value = parseCacheEntryValue(record);
    return typeof value === "string" && value.trim() ? value : null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// 压缩模型调用
// ---------------------------------------------------------------------------

function contentToText(content: OpenAIChatMessage["content"]): string {
  if (typeof content === "string") return content;
  if (content == null) return "";
  return (content as unknown[])
    .flatMap((part: unknown) => {
      if (!part || typeof part !== "object" || Array.isArray(part)) return [];
      const value = part as { type?: unknown; text?: unknown };
      return value.type === "text" && typeof value.text === "string" ? [value.text] : [];
    })
    .join("\n");
}

function formatMessagesForCompression(messages: OpenAIChatMessage[]): string {
  return messages
    .map(m => `${m.role === "user" ? "用户" : "助手"}：${contentToText(m.content)}`)
    .join("\n\n");
}

function extractWorkersAiModel(model: string): string | null {
  const normalized = model.trim();
  if (normalized.startsWith("workers-ai/")) return normalized.slice("workers-ai/".length);
  if (normalized.startsWith("worker/")) return normalized.slice("worker/".length);
  if (normalized.startsWith("@cf/")) return normalized;
  return null;
}

/**
 * 调小模型做压缩。
 *
 * 级联模式：
 *   - previousSummary = null → 首段，直接压缩消息
 *   - previousSummary 存在  → 后续段，合并旧摘要 + 新消息
 */
/**
 * 长度保险丝：prompt 里的字数目标是软约束，模型可能越界。
 * 字符数不依赖任何 tokenizer，这里才是权威上限（max_tokens 只是成本保险丝，
 * 且其计数单位是压缩模型自己的 tokenizer，与注入后主模型的计数不一致）。
 * 超过上限 20% 即在句子边界截断——截尾部（最新段）损失最小，因为紧邻的
 * 近期消息以原文形式在上下文里，与摘要尾部覆盖的时间段相接。
 */
function enforceMaxChars(text: string, maxChars: number): string {
  const hardCap = Math.floor(maxChars * 1.2);
  if (text.length <= hardCap) return text;
  const head = text.slice(0, hardCap);
  const lastEnd = Math.max(
    head.lastIndexOf("。"), head.lastIndexOf("！"), head.lastIndexOf("？"),
    head.lastIndexOf("；"), head.lastIndexOf("\n")
  );
  const cut = lastEnd > hardCap * 0.6 ? head.slice(0, lastEnd + 1) : head;
  console.log(`[compress] summary over cap: ${text.length} > ${hardCap} chars, truncated at sentence boundary (${cut.length})`);
  return `${cut}\n（摘要超长，已截断）`;
}

async function callCompressModel(
  env: Env,
  messages: OpenAIChatMessage[],
  previousSummary: string | null = null
): Promise<string> {
  const model = getCompressModel(env);
  const maxChars = getMaxChars(env);
  // 成本保险丝：中文在主流 tokenizer 下 ≲1 token/字，maxChars×1.2 足以写满目标区间；
  // 权威长度上限由 enforceMaxChars 的字符截断保证，与 tokenizer 无关。
  const maxTokens = Math.ceil(maxChars * 1.2);
  const formatted = formatMessagesForCompression(messages);

  const stance = [
    "你是对话档案员，不是对话的参与者。你的任务是把对话记录压缩成第三人称的档案摘要，供后续对话作为背景参考。",
    "铁律：全程以第三人称旁述（\"用户……\"\"助手……\"），禁止模仿、续写或代入任何一方的口吻，禁止使用第一人称。",
  ].join("\n");

  const targetFloor = Math.round((maxChars * 2) / 3 / 100) * 100;
  const requirements = [
    `1. 用中文输出，目标 ${targetFloor}–${maxChars} 字`,
    "2. 事实优先：保留关键事实、数据、做出的决定、达成的约定、未完成的话题与待办事项",
    "3. 情绪与关系动态同样是事实，用陈述句记录（如\"用户对某事感到不安，助手予以安抚\"），记录其存在与走向即可，不要渲染、不要重演",
    "4. 对话中提及的用户过去经历、往事、记忆内容、引用的外部材料：不逐字复述，一两句概括其内容与它在对话中的作用",
    "5. 双方使用的称呼、约定的说法作为事实记录（如\"助手称用户为某某\"），你自己不要使用这些称呼",
    "6. 只写摘要本身，不加任何解释、评价或元说明",
    "7. 丢弃心跳探测消息（仅含 'ping' 或类似无实质内容的对话轮次），不将其纳入摘要",
  ].join("\n");

  let compressPrompt: string;

  if (previousSummary) {
    compressPrompt = [
      stance,
      "把旧摘要与新对话段合并为一份更新的档案摘要，总长仍受同一目标区间约束。合并时按时间衰减取舍：越早的内容写得越简，为新内容腾出篇幅，但每段早期内容至少保留其结论与情感基调各一句。若旧摘要中存在第一人称或角色口吻，一并改写为第三人称。",
      "",
      "要求：",
      requirements,
      "",
      "旧摘要：",
      previousSummary,
      "",
      "新对话段：",
      formatted,
    ].join("\n");
  } else {
    compressPrompt = [
      stance,
      "",
      "要求：",
      requirements,
      "",
      "对话：",
      formatted,
    ].join("\n");
  }

  const workersAiName = extractWorkersAiModel(model);

  if (workersAiName && env.AI) {
    try {
      const result = await (env.AI as any).run(workersAiName, {
        messages: [{ role: "user", content: compressPrompt }],
        max_tokens: maxTokens,
      });
      const text = (result as any)?.response ?? "";
      if (typeof text === "string" && text.trim()) return enforceMaxChars(stripThinkingBlocks(text), maxChars);
    } catch {
      // Workers AI 失败则 fallback 到 OpenAI compat
    }
  }

  const request = {
    model,
    messages: [{ role: "user", content: compressPrompt }],
    max_tokens: maxTokens,
    temperature: 0.3,
    stream: false,
    // qwen3.7-plus 默认开思考链，reasoning tokens 挤占 max_tokens 预算；
    // OpenRouter 统一 reasoning 参数关闭，enable_thinking 兜底直连 dashscope 风格上游
    reasoning: { enabled: false },
    enable_thinking: false,
  };

  const response = await callOpenAICompat(env, request as any);
  if (!response.ok) {
    throw new Error(`Compress model returned ${response.status}`);
  }

  const data = await response.json() as any;
  const text = data?.choices?.[0]?.message?.content;
  if (typeof text !== "string" || !text.trim()) {
    throw new Error("Compress model returned empty content");
  }

  const stripped = stripThinkingBlocks(text);
  if (!stripped) {
    throw new Error("Compress model returned only thinking content");
  }

  return enforceMaxChars(stripped, maxChars);
}
