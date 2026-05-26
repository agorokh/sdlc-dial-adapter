## Summary

One canonical `/post-merge` behaviour across Claude Code, Cursor, Codex, and Antigravity. Follow-up to [agorokh/agent-factory#246](https://github.com/agorokh/agent-factory/issues/246).

All four surfaces now delegate to the same source of truth: `.claude/agents/post-merge-steward.md` (149 lines, identical to agent-factory's canonical).

| Surface | File |
|---------|------|
| canonical source | `.claude/agents/post-merge-steward.md` |
| Claude Code | `.claude/skills/post-merge/SKILL.md` |
| Cursor | `.cursor/skills/post-merge/SKILL.md` |
| Codex | `.codex/skills/post-merge/SKILL.md` |
| Antigravity | `.agents/skills/post-merge/SKILL.md` (only if `.agents/` already present) |

The old Codex format `.codex/agents/post-merge-steward.toml` is replaced by `.codex/skills/post-merge/SKILL.md` matching the convention established by `.codex/skills/account-intake-overview/` in agent-factory.

## Test plan

- [ ] CI green
- [x] `/post-merge` invocation flows through identical procedure on every surface (verified by content equality)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
