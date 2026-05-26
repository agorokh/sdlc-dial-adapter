## Summary

Propagates the workflow-skill collapse from [agorokh/agent-factory#256](https://github.com/agorokh/agent-factory/pull/256). Same hard-break design: the 5 workflow agents (post-merge-steward, issue-driven-coding-orchestrator, pr-resolution-follow-up, dependency-review, learner) become canonical skills under `.claude/skills/<slash>/SKILL.md` with mirrors in `.cursor/skills/`, `.codex/skills/`, `.agents/skills/`.

After this lands, `/post-merge`, `/orchestrate`, `/resolve-pr`, `/dependency-review`, `/learner` do the identical thing in every agent surface in this repo. No more agent vs skill divergence.

## Test plan

- [ ] CI green
