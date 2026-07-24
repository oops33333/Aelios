import type { OpenAIChatMessage, OpenAIChatRequest } from "../types";

export type SupportedImageMediaType = "image/jpeg" | "image/png" | "image/gif" | "image/webp";

export interface AnthropicBase64ImageBlock {
  type: "image";
  source: {
    type: "base64";
    media_type: SupportedImageMediaType;
    data: string;
  };
}

export interface AnthropicTextContentBlock {
  type: "text";
  text: string;
}

export type AnthropicToolResultContentBlock = AnthropicTextContentBlock | AnthropicBase64ImageBlock;

export interface AnthropicToolResultValue {
  type: "tool_result";
  tool_use_id: string;
  content: string | AnthropicToolResultContentBlock[];
}

export const TOOL_IMAGE_OMITTED = "[unsupported tool image omitted]";

const SUPPORTED_IMAGE_MEDIA_TYPES = new Set<SupportedImageMediaType>([
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/webp",
]);

function isObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function isOpenAIVisionImagePart(part: unknown): part is Record<string, unknown> {
  return isObject(part) && (part.type === "image_url" || part.type === "input_image");
}

export function hasImageContent(body: OpenAIChatRequest): boolean {
  return body.messages.some((message) => {
    if (!Array.isArray(message.content)) return false;
    return message.content.some(isOpenAIVisionImagePart);
  });
}

export function hasNonToolVisionImageContent(body: OpenAIChatRequest): boolean {
  return body.messages.some((message) => {
    if (message.role === "tool" || !Array.isArray(message.content)) return false;
    return message.content.some(isOpenAIVisionImagePart);
  });
}

/**
 * The existing vision-description pipeline only describes images attached to
 * the latest user message. Tool-result images are intentionally excluded: they
 * are sent to the main Anthropic model inside their original tool_result.
 */
export function getLastUserVisionImageParts(body: OpenAIChatRequest): Record<string, unknown>[] {
  const lastUser = [...body.messages].reverse().find((message) => message.role === "user");
  if (!lastUser || !Array.isArray(lastUser.content)) return [];
  return lastUser.content.filter(isOpenAIVisionImagePart);
}

/**
 * Remove OpenAI vision parts from non-tool messages after the small vision
 * model has described them. Tool content is returned byte-for-byte unchanged.
 */
export function stripNonToolVisionImages(messages: OpenAIChatMessage[]): OpenAIChatMessage[] {
  return messages.map((message) => {
    if (message.role === "tool" || !Array.isArray(message.content)) return message;
    const kept = message.content.filter((part) => !isOpenAIVisionImagePart(part));
    if (kept.length === 1 && isObject(kept[0]) && kept[0].type === "text" && typeof kept[0].text === "string") {
      return { ...message, content: kept[0].text };
    }
    return { ...message, content: kept.length > 0 ? kept : "" };
  });
}

/**
 * Strict RFC 4648 base64 shape:
 * - complete four-character quanta only;
 * - no whitespace;
 * - padding only in the final quantum as "==" after two data characters or
 *   "=" after three data characters;
 * - empty data is rejected.
 */
export function isStrictBase64(data: string): boolean {
  return data.length > 0 &&
    data.length % 4 === 0 &&
    /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(data);
}

function toSupportedMediaType(value: unknown): SupportedImageMediaType | null {
  if (typeof value !== "string") return null;
  const normalized = value.toLowerCase() as SupportedImageMediaType;
  return SUPPORTED_IMAGE_MEDIA_TYPES.has(normalized) ? normalized : null;
}

function parseImageDataUrl(value: unknown): AnthropicBase64ImageBlock | null {
  if (typeof value !== "string") return null;
  const match = /^data:(image\/(?:jpeg|png|gif|webp));base64,(.*)$/i.exec(value);
  if (!match || !isStrictBase64(match[2])) return null;
  const mediaType = toSupportedMediaType(match[1]);
  if (!mediaType) return null;
  return {
    type: "image",
    source: { type: "base64", media_type: mediaType, data: match[2] }
  };
}

function imageUrlFromPart(part: Record<string, unknown>): unknown {
  const imageUrl = part.image_url;
  if (typeof imageUrl === "string") return imageUrl;
  if (isObject(imageUrl)) return imageUrl.url;
  return undefined;
}

export function parseToolImagePart(part: Record<string, unknown>): AnthropicBase64ImageBlock | null {
  if (part.type === "image_url" || part.type === "input_image") {
    return parseImageDataUrl(imageUrlFromPart(part));
  }
  if (part.type !== "image" || !isObject(part.source) || part.source.type !== "base64") {
    return null;
  }
  const mediaType = toSupportedMediaType(part.source.media_type);
  const data = part.source.data;
  if (!mediaType || typeof data !== "string" || !isStrictBase64(data)) return null;
  return {
    type: "image",
    source: { type: "base64", media_type: mediaType, data }
  };
}

function isToolImagePart(part: Record<string, unknown>): boolean {
  return part.type === "image_url" || part.type === "input_image" || part.type === "image";
}

export function contentToAnthropicToolResult(
  content: OpenAIChatMessage["content"]
): string | AnthropicToolResultContentBlock[] {
  if (typeof content === "string") return content;
  if (content == null) return "";

  const blocks: AnthropicToolResultContentBlock[] = content.map((part) => {
    if (typeof part === "string") return { type: "text", text: part };
    if (isObject(part)) {
      if (part.type === "text" && typeof part.text === "string") {
        return { type: "text", text: part.text };
      }
      if (isToolImagePart(part)) {
        return parseToolImagePart(part) || { type: "text", text: TOOL_IMAGE_OMITTED };
      }
    }
    return { type: "text", text: JSON.stringify(part) ?? String(part) };
  });
  return blocks.length > 0 ? blocks : "";
}

export function convertToolMessageToAnthropicToolResult(
  message: OpenAIChatMessage
): AnthropicToolResultValue {
  return {
    type: "tool_result",
    tool_use_id: typeof message.tool_call_id === "string" ? message.tool_call_id : "",
    content: contentToAnthropicToolResult(message.content),
  };
}
