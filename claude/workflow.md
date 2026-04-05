# Workflow

## Plan First
- Enter plan mode for any non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan — do not keep pushing a broken approach
- Write the plan to `tasks/todo.md` with checkable items before starting
- Check in with the user before implementing if the plan has trade-offs

## Build Clean
- Simplicity first — make every change as simple as possible, touch minimal code
- Find root causes, not temporary fixes. Senior developer standards.
- Changes should only touch what is necessary
- Do not refactor unrelated areas, do not sneak in "helpful" extras
- Skip over-engineering — three similar lines beat a premature abstraction

## Verify Before Done
- Never mark a task complete without proving it works
- Run `pytest`, check that the MCP server starts, verify hooks fire
- State: what changed, what was verified, what was NOT verified, remaining risks
- Ask: "Would a staff engineer approve this?"

## Demand Elegance (When It Matters)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky, step back and implement the clean solution
- Skip this for simple, obvious fixes — do not over-engineer

## Autonomous Bug Fixing
- When given a bug report: just fix it. Do not ask for hand-holding.
- Point at logs, errors, failing tests — then resolve them
- Go fix failing CI/tests without being told how

## Learn From Mistakes
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules that prevent the same mistake recurring
- Review `tasks/lessons.md` at session start for relevant context

## Git Rules
- Commit locally after verified work. **Do not push unless explicitly asked.**
- Descriptive commit messages — what and why, not just what
- Do not amend published commits
- This is a community-facing repo — commits and PRs will be read by strangers. Write them accordingly.
