import type { MemoryApiRecord } from "../types";

const SHANGHAI_TIME_ZONE = "Asia/Shanghai";
const MEMORY_TIMESTAMP_RE =
  /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?(?:(Z)|([+-])(\d{2}):?(\d{2}))?)?$/;

const SHANGHAI_DATE_FORMATTER = new Intl.DateTimeFormat("en-US", {
  timeZone: SHANGHAI_TIME_ZONE,
  year: "numeric",
  month: "numeric",
  day: "numeric",
});

type MemoryPromptRecord = Pick<
  MemoryApiRecord,
  "type" | "importance" | "content" | "created_at"
> &
  Partial<Pick<MemoryApiRecord, "pinned">>;

export interface FormatMemoryPromptLineOptions {
  includePinned?: boolean;
}

function parseMemoryTimestamp(value: unknown): Date | null {
  if (typeof value !== "string") return null;
  const match = MEMORY_TIMESTAMP_RE.exec(value.trim());
  if (!match) return null;

  const [, yearText, monthText, dayText, hourText, minuteText, secondText, fraction = "", zulu, offsetSign, offsetHourText, offsetMinuteText] = match;
  const year = Number(yearText);
  const month = Number(monthText);
  const day = Number(dayText);
  const hour = Number(hourText || 0);
  const minute = Number(minuteText || 0);
  const second = Number(secondText || 0);
  const millisecond = Number((fraction + "000").slice(0, 3));

  const calendarDate = new Date(Date.UTC(year, month - 1, day));
  const validCalendarDate =
    calendarDate.getUTCFullYear() === year &&
    calendarDate.getUTCMonth() === month - 1 &&
    calendarDate.getUTCDate() === day;
  if (!validCalendarDate || hour > 23 || minute > 59 || second > 59) return null;

  const offsetHour = Number(offsetHourText || 0);
  const offsetMinute = Number(offsetMinuteText || 0);
  if (offsetHour > 23 || offsetMinute > 59) return null;

  // Database timestamps without an explicit zone (including SQLite
  // CURRENT_TIMESTAMP) are UTC. Date-only values are treated as UTC midnight.
  let timestamp = Date.UTC(year, month - 1, day, hour, minute, second, millisecond);
  if (!zulu && offsetSign) {
    const offsetMs = (offsetHour * 60 + offsetMinute) * 60_000;
    timestamp += offsetSign === "+" ? -offsetMs : offsetMs;
  }

  const date = new Date(timestamp);
  return Number.isFinite(date.getTime()) ? date : null;
}

/** Format a persisted memory timestamp as a stable Shanghai calendar date. */
export function formatMemoryRecordedDate(value: unknown): string | null {
  const date = parseMemoryTimestamp(value);
  if (!date) return null;

  const parts = SHANGHAI_DATE_FORMATTER.formatToParts(date);
  const values = Object.fromEntries(
    parts
      .filter((part) => part.type !== "literal")
      .map((part) => [part.type, part.value])
  );
  if (!values.year || !values.month || !values.day) return null;
  return `${values.year}年${values.month}月${values.day}日`;
}

/**
 * Render one prompt-facing memory line. The date is the record creation date,
 * not a claim that the remembered event happened on that day.
 */
export function formatMemoryPromptLine(
  memory: MemoryPromptRecord,
  options: FormatMemoryPromptLineOptions = {}
): string {
  const pinned = options.includePinned && memory.pinned ? "[pinned]" : "";
  const recordedDate = formatMemoryRecordedDate(memory.created_at);
  const dateSuffix = recordedDate ? `（记录于${recordedDate}）` : "";
  return `- [${memory.type}][importance=${memory.importance.toFixed(2)}]${pinned} ${memory.content}${dateSuffix}`;
}
