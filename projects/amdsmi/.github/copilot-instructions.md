# Copilot Instructions — AMD SMI

See [CLAUDE.md](../CLAUDE.md) for project overview, critical rules, and behavioral guidelines.
See [.github/CONTRIBUTING.md](CONTRIBUTING.md) for PR workflow and coding standards.

## Quick Reference

- **Never edit `py-interface/amdsmi_wrapper.py` manually** — regenerate with `tools/update_wrapper.sh`
- **PRs target `develop`** branch
- **Pre-commit must pass**: `pre-commit run --all-files`
- **Version** defined in `include/amd_smi/amdsmi.h`
- **Excluded from linting**: `docs/`, `build/`, `esmi_ib_library/`, `third_party/`

## Agent Settings

- **`agent-max-nesting-depth: 3`** — the deepest a subagent chain may nest.
  Subagents may dispatch their own subagents, but only while within this budget.
  Depth is counted from the first orchestrator (Planning / Development / Review),
  which runs at depth 1. An agent running
  at the max depth is a **leaf** — it must do the work itself and must not
  dispatch further. The mechanism lives in the `agent-handoff` skill ("Nesting
  Budget"). To change the cap, edit the number here — it is the single source of
  truth. (VS Code does not enforce this natively; the agents honor it by
  convention.)
