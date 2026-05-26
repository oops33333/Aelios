#!/usr/bin/env python3
"""
Aelios sweepy 适配 - 自动修补脚本
处理所有 env.DB → env 的机械替换和 types.ts 更新
"""
import re, os

BASE = "/root/aelios/src"

def patch(filepath, replacements):
    path = os.path.join(BASE, filepath)
    with open(path, "r") as f:
        content = f.read()
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"  ✓ {filepath}: '{old[:50]}...' → patched")
        else:
            print(f"  - {filepath}: '{old[:50]}...' not found, skipping")
    with open(path, "w") as f:
        f.write(content)

print("=== 1. Patching merge.ts ===")
patch("memory/merge.ts", [
    ("await createMemory(env.DB, {", "await createMemory(env, {"),
    ("await getMemoryById(env.DB, { namespace:", "await getMemoryById(env, { namespace:"),
    ("const merged = await updateMemory(env.DB, {", "const merged = await updateMemory(env, {"),
    ("const superseded = await updateMemory(env.DB, {", "const superseded = await updateMemory(env, {"),
])

print("=== 2. Patching maintenance.ts ===")
patch("memory/maintenance.ts", [
    ("await searchMemoriesByText(env.DB, {", "await searchMemoriesByText(env, {"),
])

print("=== 3. Patching stablePack.ts ===")
patch("memory/stablePack.ts", [
    ("await listMemories(env.DB, {", "await listMemories(env, {"),
])

print("=== 4. Patching chatCompletions.ts ===")
patch("api/chatCompletions.ts", [
    ("await listMemories(env.DB, {", "await listMemories(env, {"),
])

print("=== 5. Patching api/memories.ts ===")
# This file has many env.DB references for memory CRUD
path = os.path.join(BASE, "api/memories.ts")
with open(path, "r") as f:
    content = f.read()
# Replace all memory function calls that pass env.DB
for func in ["createMemory", "listMemories", "listMemoriesPage", "getMemoryById",
             "fetchMemoriesByIds", "updateMemory", "softDeleteMemory",
             "searchMemoriesByText", "markMemoriesRecalled"]:
    old = f"{func}(env.DB,"
    new = f"{func}(env,"
    if old in content:
        content = content.replace(old, new)
        print(f"  ✓ api/memories.ts: {func}(env.DB,) → patched")
count = content.count("env.DB")
if count > 0:
    print(f"  ⚠ api/memories.ts: {count} remaining env.DB references (may be non-memory)")
with open(path, "w") as f:
    f.write(content)

print("=== 6. Updating types.ts ===")
path = os.path.join(BASE, "types.ts")
with open(path, "r") as f:
    content = f.read()
# Add new env vars before the closing brace of Env interface
new_vars = """  // sweepy
  SWEEPY_URL?: string;
  SWEEPY_AUTH?: string;
  // OpenRouter
  OPENROUTER_API_KEY?: string;
  OPENROUTER_BASE_URL?: string;
"""
# Find SUMMARY_MODEL line and insert after it
if "SWEEPY_URL" not in content:
    content = content.replace("  SUMMARY_MODEL?: string;\n}", f"  SUMMARY_MODEL?: string;\n{new_vars}}}")
    print("  ✓ types.ts: added SWEEPY_URL, SWEEPY_AUTH, OPENROUTER_API_KEY, OPENROUTER_BASE_URL")
else:
    print("  - types.ts: already has SWEEPY vars")
with open(path, "w") as f:
    f.write(content)

print("=== 7. Removing vectorStore import from filter.ts if needed ===")
# Check if filter.ts imports from vectorStore
filter_path = os.path.join(BASE, "memory/filter.ts")
if os.path.exists(filter_path):
    print("  - filter.ts exists, no changes needed (inject.ts already handles routing)")

print("\n=== All patches applied! ===")
