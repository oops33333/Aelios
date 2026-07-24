# aelios 工程档案（项目宪法）

> 本文件是 aelios 的不可变约束清单。每一条铁律都是真实事故换来的，
> 改动涉及缓存、拼装、压缩、心跳四个部件中任何一个之前，先通读第三节。
> 修订本文件中的「铁律」需要糖糖点头；「事实」段落随代码演进如实更新。

## 一、目的

aelios 是跑在 Cloudflare Workers 上的私人聊天代理：对客户端（RikkaHub 等）
暴露 OpenAI 兼容 API，对上游走 Anthropic 原生协议。它在转发之余做四件事：

1. **记忆注入**：从 sweepy / Vectorize 取长期记忆、persona、摘要、日期提醒，拼进 prompt；
2. **历史压缩**：超长历史级联压缩成摘要，只保留近期窗口原文；
3. **prompt 缓存管理**：精心排布 cache_control 断点，让每一轮请求至少命中首断点；
4. **缓存保活**：无对话时由心跳定期重放前缀，防止 TTL 过期。

一切设计的最终度量是**账单**：cache_read 高、cache_creation 低、input 全价 tokens 少。

## 二、请求管线（数据流事实）

```
客户端请求 (OpenAI 格式)
  → vision 剥离（最后 user 图片由小模型转述；所有 non-tool 图片照旧移除，
    tool 图片原位保留）
  → [此处存心跳原料 body → D1]（仅 anthropic 分支、非心跳轮）
  → 并行取数：压缩(compressHistoryIfNeeded) / RAG记忆 / persona / summary / reminders
  → 路径判定 hasToolContent(裁剪后消息)：
      含 tool 消息 → 回退路径 buildAnthropicNativeRequest（摘要注入消息区头部）
      不含       → v4 assembler（BLOCK_ORDER 拼装）
  → callAnthropicNative → 响应解析 / thinking 签名暂存 / 落库
```

关键事实：**路径判定发生在压缩裁剪之后**——裁掉旧 tool 消息可能使请求在
两条路径间轮换（横跳），这是设计内行为，靠「双路径共享首断点字节」兜底。

核心文件：

| 文件 | 职责 |
|---|---|
| `src/api/chatCompletions.ts` | 主入口、路径分发、心跳分支 |
| `src/assembler/types.ts` | BLOCK_ORDER、锚点常量（**改前必读第三节**） |
| `src/assembler/blocks.ts` | 各块 content_fn、splitClientSystem、assemble |
| `src/proxy/anthropicAdapter.ts` | 回退路径、协议转换、滚动消息缓存 |
| `src/memory/compress.ts` | 级联压缩、COMPRESS_PROMPT_VERSION、D1 段缓存 |
| `src/proxy/heartbeatPrefix.ts` | 心跳存原料 / 管线重建 / makeReplayable |
| `scripts/verify-assembler.mjs` | assembler 的手写镜像 + 209 条测试 |

## 三、缓存前缀契约（铁律）

Anthropic prompt cache 按前缀逐字节哈希匹配，断点（cache_control）每请求最多 4 个。
当前断点预算（≤4）：

- **BP1 = client_system**（客户端 prompt 的 stable 段，`\n\n` 拼接为单块）
- **BP2 = compressed_summary**（assembler 路径）/ 稳定记忆包（回退路径）
- **BP3 = 滚动末条 user 消息**（applyRollingMessageCache）

### 铁律（违反任何一条 = 前缀零命中级事故）

1. **锚点之前只许放逐字节不变的内容。** 每轮现取的东西（persona 的
   importance、summary 更新、时间戳）出现在任何断点之前，该断点及此前
   全部投资即报废。persona/summary 漂移只允许伤及 BP2 及之后。
   （事故：8ce0d87 前 BLOCK_ORDER 把四个现取块压在 client_system 前，
   潜伏两个半月后压缩生效当天引爆。）
2. **client_system 必须位于首位且独占 BP1**，两条拼装路径（assembler /
   回退）必须经 `splitClientSystem` 产出**逐字节同构**的首块。路径横跳
   是常态，首断点是唯一的公共保底。
3. **客户端 system 里的易变时间/日期行必须剥出**（splitClientSystemTexts /
   isVolatileTimeLine），沉到消息尾部非缓存区，绝不许坐在断点之前。
4. **断点总数不得超过 4。** 新增缓存块前先数一遍现有断点。
5. **content_fn 必须确定性**：同一 ctx 必须产出同一字符串。禁止时间戳、
   请求 id、Map 迭代序。
6. **改 `src/assembler/` 必须同步修改 `scripts/verify-assembler.mjs` 镜像**
   并跑 `node scripts/verify-assembler.mjs`（209/209 才算过）。镜像曾漂移
   （漏 compressed_summary），漂移的镜像等于没有测试。
7. **thinking / tool_choice / tools / system 属于消息级缓存键**：任何重放、
   探针、旁路请求与真实请求在这些字段上不一致，缓存互不相认（各自翻倍
   写库、永不命中对方）。max_tokens / temperature 不进缓存键。

### 当前 BLOCK_ORDER（事实，改动须过第 1、2、6 条）

```
client_system → persona_pinned → long_term_summary → proxy_static_rules
→ preset_lite → compressed_summary → client_volatile_context
→ dynamic_memory_patch → reminders → vision_context → recent_history → current_user
```

## 四、历史压缩契约（铁律）

1. **改压缩 prompt 必须同步递增 `COMPRESS_PROMPT_VERSION`**（compress.ts，
   现为 v5）。段缓存键含版本号，不递增则级联永远命中旧缓存、新 prompt 形同虚设。
2. **给默认带思考链的模型（qwen / deepseek-r1 类）做内部调用，一律显式关
   reasoning**（`reasoning:{enabled:false}` + `enable_thinking:false`），并保留
   stripThinkingBlocks 兜底（剥空则抛错不入缓存）。事故：`<think>` 随摘要腌入
   D1、每轮全价注入上游。
3. **跨模型长度约束用字符数（enforceMaxChars），不用 max_tokens**——
   max_tokens 数的是压缩模型自己的 tokenizer，跨模型换算必然失准；
   max_tokens 只作成本保险丝（maxChars×1.2）。
4. **长度指令忌恐吓措辞**（「宁短勿超」类）：小模型长度服从是双峰的，
   恐吓即塌零。用带地板的目标区间（现为 1000–1500 字）。
5. recent 窗口从**原始数组**切片，保证 tool_calls 与 tool_result 配对不出孤儿；
   压缩结果必须同时喂给两条拼装路径（回退路径经 injectCompressedSummary）。

## 五、心跳保活契约（铁律）

1. **探针必须与真实请求共用生成代码：存原料（客户端原始 body），不存
   成品（组装好的请求体）。** 成品字节在拼装结构变更后即成孤儿——
   事故：8ce0d87 部署后心跳整夜七跳全刷废前缀。心跳重建走真实管线，
   发送前仅 makeReplayable 三处最小覆盖（截断到末锚点 / stream=false /
   max_tokens 压最小）。
2. `x-heartbeat: true` 是**重放已存前缀**，不是免副作用直通——不能当
   干净测试通道用（所有 key 共用 namespace "default"，没有干净通道）。
3. 心跳解析到非 anthropic 提供方时压 max_tokens（保活只对 Anthropic 有意义）。
4. 心跳发送端在 `/root/memory-server/heartbeat.js`（55 分钟间隔，
   失败 5 分钟重试，配合 1h TTL）。

## 六、thinking 契约（事实 + 铁律）

- RikkaHub 不回传 thinking_blocks；服务端按 tool_use id 在 D1 暂存带签名
  thinking 块，工具回环轮回填，保 thinking 常开（开关横跳会打掉消息级缓存）。
- 流式与非流式两条返回路径的暂存逻辑必须保持对齐。
- `role:"tool"` 的 base64 图片只允许留在对应 `tool_result.content` 内；不得
  新建消息、移动 `tool_result`、改变 `tool_use_id` 配对或缓存锚点。用户主动
  发送的图片仍走既有小视觉模型描述链路。

## 七、开发与部署规范

- **部署 = `git push origin main`** → Cloudflare Workers Builds 自动部署。
  本机无 wrangler、无 CLOUDFLARE_API_TOKEN；secrets 只在 Dashboard 改。
- 改动后必跑：`npx tsc --noEmit` + `node scripts/verify-assembler.mjs`。
- 诊断入口：`GET /v1/usage?key=<CHATBOX_API_KEY>`（key 在
  /root/memory-server/public/aelios.html）看 cache_read / cache_creation 走势；
  心跳日志 `https://sweepy.cloud/api/heartbeat/log`。
- 验收标准：任何一轮（含压缩边界轮、路径横跳轮）的 cache_read 至少覆盖
  客户端 prompt 长度；部署后首轮全量写缓存属正常冷启动。
- 每次实质改动写 sweepy 维护日志（tag `维护日志,aelios`，tags 传逗号字符串）。
- 事故史与归因细节见 sweepy 维护日志 #600+ 及本机 memory
  `aelios-compression-never-enabled.md`。
