# Repo Hygiene Review (Opus 4.7)

**Date:** 2026-04-27
**Reviewer:** Opus 4.7 (read-only audit)
**Scope:** `/home/dhawal/turboQuantDC/` — repo hygiene for vLLM integration + open-source readiness
**Posture:** Archival > deletion. Zero files removed; recommendations only.

---

## Summary at a glance

- **Tracked files:** 306 (git ls-files)
- **Disk size of `.mypy_cache`:** 97 MB (correctly gitignored, but on disk)
- **Disk size of `.venv-vllm`:** 68 MB (NOT gitignored — at risk of being committed)
- **Root-level `.py` scripts:** 16 (≈7,653 LOC of one-shot demos / autoresearch / benchmarks)
- **Root-level `.md` files:** 8 (several stale/marketing-only)
- **Tracked `autoresearch_results_*.jsonl` files in repo root:** 5 (~5.4 MB raw, including ones explicitly named `_buggy`, `_broken_scoring`, `_pre_fixes`, `_pre_gradient`, `_pre_hybrid`)
- **CLAUDE.md is severely stale:** says "All source files in `turboquantdc/` and `tests/` are empty stubs" — repo actually has 67 modules + 43 test files, v0.3.0 shipped, 1,818+ tests.
- **Reference is a git submodule** (160000 mode pointer), already isolated — fine to keep.
- **Backup file `.backup`** — single instance, NOT tracked (`*.backup` rule fires), can simply be deleted from disk.

---

## Top-Level File Inventory

### Root scripts (`*.py`)

| File / Dir | LOC | Category | Recommendation |
|---|---:|---|---|
| `setup.py` | ~50 | PRODUCTION | KEEP at root |
| `autoresearch.py` | 991 | RESEARCH-ARTIFACT | Move to `archive/autoresearch/` |
| `autoresearch_real.py` | 650 | RESEARCH-ARTIFACT | Move to `archive/autoresearch/` |
| `autoresearch_dashboard.py` | 295 | RESEARCH-ARTIFACT | Move to `archive/autoresearch/` |
| `benchmark.py` | 944 | DEV-TIME | Move to `benchmarks/legacy/` |
| `benchmark_entropy.py` | 331 | DEV-TIME | Move to `benchmarks/legacy/` |
| `benchmark_speed.py` | 127 | DEV-TIME | Move to `benchmarks/` (it's small/relevant) |
| `demo.py` | 542 | DEV-TIME | Move to `examples/legacy/` |
| `demo_70b.py` | 251 | DEV-TIME | Move to `examples/legacy/` |
| `demo_app.py` | 667 | DEV-TIME (Gradio app) | Move to `examples/` |
| `demo_e8.py` | 147 | DEV-TIME (Apr 15) | Move to `examples/e8/` (recent + cited from README) |
| `demo_final.py` | 151 | STALE | Move to `archive/demos/` |
| `demo_gemma4.py` | 332 | DEV-TIME | Move to `examples/gemma4/` |
| `run_70b.py` | 655 | DEV-TIME | Move to `examples/legacy/` |
| `run_infinite_context.py` | 305 | DEV-TIME | Move to `examples/` |
| `showcase.py` | 813 | DEV-TIME | Move to `examples/` (cited as Phase 4 deliverable) |
| `test_retrieval_scale.py` | 452 | STALE — looks like a test file living in root, not in `tests/` | Move into `tests/` OR `benchmarks/` |

**Insight:** 13 of 16 root scripts are one-shot dev artifacts. A clean repo root should expose `setup.py` only (plus configuration files). New visitors currently see 16 entry points and can't tell which is "real."

### Root data / log files

| File | Size | Tracked? | Category | Recommendation |
|---|---|---|---|---|
| `autoresearch.log` | 9.3k | NO (gitignored) | GITIGNORE-LEAK on disk only | Delete from disk |
| `autoresearch_real.log` | 33k | NO (gitignored) | GITIGNORE-LEAK on disk only | Delete from disk |
| `autoresearch_real_results_buggy.jsonl` | 8.3k | NO (gitignored) | RESEARCH-ARTIFACT | Delete from disk |
| `autoresearch_results.jsonl` | 580k | NO (gitignored) | RESEARCH-ARTIFACT | Delete from disk |
| `autoresearch_results_buggy.jsonl` | 121k | NO (gitignored) | RESEARCH-ARTIFACT | Delete from disk |
| `autoresearch_results_old_modules.jsonl` | 699k | **YES** | RESEARCH-ARTIFACT (named "_old_modules") | Untrack + move to `archive/autoresearch/results/` or delete |
| `autoresearch_results_pre_fixes.jsonl` | 694k | **YES** | RESEARCH-ARTIFACT (named "_pre_fixes") | Untrack + archive |
| `autoresearch_results_v1.jsonl` | 47k | **YES** | RESEARCH-ARTIFACT | Untrack + archive |
| `autoresearch_results_v2_broken_scoring.jsonl` | 1.7M | **YES** | RESEARCH-ARTIFACT (named "_broken_scoring") | Untrack + archive |
| `autoresearch_results_v3_pre_gradient.jsonl` | 2.2M | **YES** | RESEARCH-ARTIFACT (named "_pre_gradient") | Untrack + archive |
| `autoresearch_results_v4_pre_hybrid.jsonl` | 1.8M | **YES** | RESEARCH-ARTIFACT (named "_pre_hybrid") | Untrack + archive |
| `index.html` | 586 B | YES | PRODUCTION (redirect to visualization/) | Keep — trivial redirect |

### Root markdown files

| File | Tracked? | Status | Recommendation |
|---|---|---|---|
| `README.md` | YES | LIVE (Apr 15, 9.8k) | KEEP at root |
| `CLAUDE.md` | YES | **STALE** (Apr 15, 7.9k, contradicts reality) | UPDATE in place — see staleness section |
| `LICENSE` | YES | LIVE | KEEP |
| `HANDOFF.md` | YES | LIVE (Apr 15) | KEEP at root or move to `docs/HANDOFF.md` |
| `PLAN.md` | YES | STALE (Mar 28; Phase 5 was claimed "complete" but real state is v0.3.0 + many post-Phase-5 modules) | Move to `docs/archive/PLAN_phase1-5.md` |
| `MEMORY.md` | YES | STUB (only 690 bytes, mostly placeholders) | Delete or move to `docs/archive/` |
| `PUBLISH.md` | YES | TINY (323 bytes) | Inline into `README.md` "Publishing" section, then delete |
| `OVERNIGHT_PLAN.md` | YES | STALE (Apr 2 plan, work completed) | Move to `docs/archive/` |
| `GROWTH_PLAYBOOK.md` | YES | MARKETING DOC (30k, codename "MACHIAVELLI", Apr-era hype playbook) | Move to `docs/archive/` — outsider-facing repo shouldn't have this at root |

### Root config files (KEEP all)

| File | Recommendation |
|---|---|
| `.gitignore` | KEEP, but extend (see audit below) |
| `pyproject.toml`, `setup.py` | KEEP |
| `requirements.txt`, `requirements-dev.txt` | KEEP |
| `requirements_demo.txt` | Consolidate into `requirements-dev.txt` then delete |
| `MANIFEST.in` | KEEP |

### Subdirectories

| Dir | Category | Recommendation |
|---|---|---|
| `turboquantdc/` (67 modules, 4.4 MB) | PRODUCTION (with research deadweight inside — see other reviews for module pruning) | KEEP, prune internally per separate review |
| `tests/` (43 files, 6.6 MB) | PRODUCTION | KEEP |
| `benchmarks/` (50+ scripts, 6.9 MB) | MIXED | KEEP, but split into `current/` vs `legacy/` |
| `benchmarks/results/` | RESEARCH-ARTIFACT (mixed: some cited from README, some orphaned) | KEEP cited, archive rest |
| `docs/` (24 tracked files + paper PDF + HTML) | MIXED | See "Proposed Archival Plan" |
| `reference/tonbistudio-ref/` | EXTERNAL (git submodule, mode 160000) | KEEP — already isolated; CLAUDE.md correctly points to it |
| `examples/` (1 file: `hf_turboquant_example.py`) | DEV-TIME | KEEP, expand by absorbing root demos |
| `notebooks/` (1 file: `TurboQuantDC_Demo.ipynb`) | DEV-TIME | KEEP |
| `assets/` (1 file: `twitter_card.html`) | MARKETING | Move to `docs/assets/` (where `docs/assets/` already exists, currently empty) |
| `tools/` (3 files: gguf export, C ref, validator) | PRODUCTION (llama.cpp interop) | KEEP — it's legitimately tool code |
| `visualization/` (50k `index.html`) | PRODUCTION (GitHub Pages source per `.github/workflows/pages.yml`) | KEEP — it's the deployed site |
| `warroom/` (jsonl + 4 GPU result MDs + html + server) | RESEARCH-ARTIFACT (multi-agent coordination from Mar 25-29) | Move entire dir to `archive/warroom/` |
| `overnight_results/` (16 files, 172 KB; phase1-5 reports + scripts) | RESEARCH-ARTIFACT (April 2 sprint) | Move to `archive/overnight_apr2/` |
| `logs/2026-04-27/` | TRANSIENT (current session) | KEEP, but `logs/` should be gitignored entirely |
| `dist/` (4 wheel/sdist files, 924 KB) | BUILD-OUTPUT (gitignored, on disk only) | Delete from disk; rebuild for next release |
| `turboquantdc.egg-info/` | BUILD-OUTPUT (gitignored, on disk only) | Delete from disk |
| `__pycache__/` (root-level), `turboquantdc/__pycache__/`, `cuda/__pycache__/` | GITIGNORE-LEAK on disk | Delete from disk (`find . -name __pycache__ -exec rm -rf {} +`) |
| `.mypy_cache/` (97 MB), `.ruff_cache/`, `.pytest_cache/` | GITIGNORE-LEAK on disk | Delete from disk; they regenerate |
| `.venv-vllm/` (68 MB) | VIRTUALENV — **NOT in .gitignore** | Add to .gitignore IMMEDIATELY (see CRITICAL) |

---

## .gitignore Audit

### Positives

`.gitignore` covers the essentials: `__pycache__/`, `*.egg-info/`, `dist/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`, `*.log`, `*.backup`, plus explicit named entries for several `autoresearch_results_*.jsonl` files and `nohup.out`.

### What's tracked that shouldn't be

**Six intermediate research result files are committed despite being objectively "buggy / pre-fix":**

```
autoresearch_results_old_modules.jsonl       (699 KB)
autoresearch_results_pre_fixes.jsonl         (694 KB)
autoresearch_results_v1.jsonl                ( 47 KB)
autoresearch_results_v2_broken_scoring.jsonl ( 1.7 MB)
autoresearch_results_v3_pre_gradient.jsonl   ( 2.2 MB)
autoresearch_results_v4_pre_hybrid.jsonl     ( 1.8 MB)
```

The current `.gitignore` lists `autoresearch_results_buggy.jsonl`, `autoresearch_results_v1.jsonl`, etc., but `autoresearch_results_v1.jsonl` is **already committed**. Adding to `.gitignore` after a file is tracked does nothing; it has to be `git rm --cached`'d.

Combined: ~7 MB of tracked junk explicitly tagged as broken / superseded.

### What's missing from .gitignore

Add to `.gitignore`:

```gitignore
# Virtualenvs (additional patterns)
.venv-*/
.venv-vllm/

# Build outputs that occasionally appear
dist/
build/
*.whl
*.tar.gz

# Logs directory (currently tracked piecemeal)
logs/

# Editor/OS files
.DS_Store
Thumbs.db
```

`.claude/` exists at repo root — verify if it's user-local or repo-shared; if user-local, gitignore it.

### What's tracked but should likely be `git rm --cached`

| File | Reason |
|---|---|
| `autoresearch_results_old_modules.jsonl` | Tagged "old_modules" |
| `autoresearch_results_pre_fixes.jsonl` | Tagged "pre_fixes" |
| `autoresearch_results_v1.jsonl` | Superseded; v3/v4 also exist |
| `autoresearch_results_v2_broken_scoring.jsonl` | **Self-labeled "broken"** |
| `autoresearch_results_v3_pre_gradient.jsonl` | Tagged "pre_gradient" |
| `autoresearch_results_v4_pre_hybrid.jsonl` | Tagged "pre_hybrid" |
| `autoresearch.py` / `_real.py` / `_dashboard.py` | If autoresearch is dead, they belong in `archive/` |

---

## CLAUDE.md Staleness

`CLAUDE.md` was last touched **April 15, 2026** but its content reflects the **March 25 pre-Phase-1 state**. Specific lies:

| Line | Current text | Reality |
|---:|---|---|
| 11 | "All source files in `turboquantdc/` and `tests/` are empty stubs." | 67 source modules (4.4 MB), 43 test files (6.6 MB), 1,818+ tests passing. |
| 11 | "`setup.py` is empty." | `setup.py` is 1.8 KB and has built v0.2.0 + v0.3.0 wheels (in `dist/`). |
| 11 | "Phase 1 (Core Algorithm) has not started." | Phases 1-5 complete (per PLAN.md), v0.3.0 shipped, GitHub Pages deployed. |
| 17 | "Dependencies (not yet in a requirements.txt — create one when starting)" | `requirements.txt`, `requirements-dev.txt`, `requirements_demo.txt`, `pyproject.toml` all exist. |
| 64-101 | "Module Pipeline: codebook.py → rotation.py → polarquant.py → qjl.py → estimator.py → kv_cache.py" | Accurate at the foundational layer, but ignores 60+ other modules: e8_lattice, generation_core, expected_attention, cayley_quant, residual_quant, learned_quant, block_rotation, cross_layer_kv, vllm_integration, ... |
| 130 | "Don't try to integrate with vLLM/SGLang until the core algorithm is validated." | `turboquantdc/vllm_integration.py` is 37 KB and the explicit current goal is vLLM integration. |

### Proposed CLAUDE.md update

Replace the "Current Status" section (line 10-13) with:

```markdown
## Current Status (as of 2026-04-27)

v0.3.0 shipped. 67 source modules in `turboquantdc/`, 43 test files in `tests/`,
1,818+ tests passing. PyPI/GitHub Pages live.

Active focus: vLLM integration (`turboquantdc/vllm_integration.py`), E8 lattice
VQ paper draft, llama.cpp K-cache flash-attention bug investigation.

See `HANDOFF.md` for the latest engineering state and `docs/RESEARCH_FINDINGS_APR15.md`
for the most recent results. `PLAN.md` is archived (Phase 1-5 historical record).
```

Replace the "Don't" item on line 130:
- Remove "Don't try to integrate with vLLM/SGLang until the core algorithm is validated."
- Replace with: "Don't add new compression modules without head-to-head benchmarks against existing ones (`benchmarks/rotorquant_comprehensive.py`)."

Add a new "Module Map" pointing to the high-value production modules (per HANDOFF.md):
- `generation_core.py` — production cache (STABLE)
- `e8_lattice.py` + `e8p_codec.py` — current best (E8 VQ)
- `expected_attention.py` — EA pruning
- `cache_distillation.py` — KVSculpt
- ... vs deprecated experimental ones a code reviewer should NOT touch.

---

## Proposed Archival Plan

### New top-level layout

```
turboQuantDC/
├── README.md                # KEEP (live, Apr 15)
├── CLAUDE.md                # KEEP, updated
├── HANDOFF.md               # KEEP (or move to docs/)
├── LICENSE                  # KEEP
├── setup.py, pyproject.toml, MANIFEST.in
├── requirements*.txt        # KEEP, consolidate _demo into -dev
├── .gitignore               # KEEP, extend
├── index.html               # KEEP (trivial redirect to visualization/)
│
├── turboquantdc/            # PACKAGE (separate review for internal pruning)
├── tests/                   # TESTS
├── benchmarks/
│   ├── current/             # Active benchmarks (rotorquant_comprehensive, ppl_for_tom, niah_for_tom, ...)
│   ├── legacy/              # Pre-Apr 15 superseded benchmarks
│   └── results/
│       ├── current/         # Cited from README/HANDOFF
│       └── archive/         # Old runs
├── examples/
│   ├── hf_turboquant_example.py     # existing
│   ├── e8_demo.py                   # from root demo_e8.py
│   ├── gemma4_demo.py               # from root demo_gemma4.py
│   ├── infinite_context.py          # from root run_infinite_context.py
│   ├── showcase.py                  # from root showcase.py
│   └── legacy/
│       ├── demo.py                  # original demo (Mar 26)
│       ├── demo_70b.py
│       ├── demo_app.py              # gradio app
│       ├── demo_final.py
│       └── run_70b.py
├── notebooks/               # KEEP
├── tools/                   # KEEP (llama.cpp interop)
├── visualization/           # KEEP (GitHub Pages source)
├── reference/               # KEEP (git submodule)
│
├── docs/
│   ├── HANDOFF.md           # if moved here
│   ├── README.md            # link index for docs/
│   ├── algorithm/
│   │   ├── MATH_SPEC.md
│   │   ├── REFERENCE_ANALYSIS.md
│   │   ├── IMPLEMENTATION_PLAN.md
│   │   └── turboquant_paper.pdf
│   ├── research/
│   │   ├── ARXIV_OUTLINE.md
│   │   ├── ASYMPTOTIC_LAW_REPORT.md
│   │   ├── ATTENTION_FIX_PLAN.md
│   │   ├── KV_COMPRESSION_SURVEY_2026.md
│   │   ├── RESEARCH_FINDINGS_APR15.md
│   │   ├── RESEARCH_LANDSCAPE.md
│   │   ├── RESIDUALQUANT_LLAMA_CPP_SPEC.md
│   │   ├── COMPARISON_TABLE.md
│   │   └── IMPOSSIBLE_INFERENCE.md
│   ├── outreach/
│   │   ├── HF_PR_DESCRIPTION.md
│   │   ├── LLAMA_CPP_COMMENT.md
│   │   ├── VLLM_COMMENT.md
│   │   ├── REDDIT_POST.md
│   │   ├── REDDIT_POST_V2.md
│   │   ├── TWEETS.md
│   │   └── WARROOM_TRANSCRIPT.md
│   ├── archive/
│   │   ├── PLAN_phase1-5.md           # from root PLAN.md
│   │   ├── MEMORY.md                  # from root MEMORY.md (stub)
│   │   ├── OVERNIGHT_PLAN.md          # from root
│   │   ├── GROWTH_PLAYBOOK.md         # from root (marketing-only)
│   │   └── PUBLISH.md                 # if not inlined into README
│   ├── assets/              # existing (currently empty) — receive twitter_card.html
│   ├── code_review/         # existing
│   ├── superpowers/         # existing
│   ├── google_blog.html     # paper-related blog snapshot
│   └── index.html           # docs landing
│
└── archive/                 # NEW top-level dir for one-shot research artifacts
    ├── autoresearch/
    │   ├── autoresearch.py
    │   ├── autoresearch_real.py
    │   ├── autoresearch_dashboard.py
    │   ├── README.md (explain what autoresearch was)
    │   └── results/
    │       ├── autoresearch_results_v1.jsonl
    │       ├── autoresearch_results_v2_broken_scoring.jsonl
    │       ├── autoresearch_results_v3_pre_gradient.jsonl
    │       ├── autoresearch_results_v4_pre_hybrid.jsonl
    │       ├── autoresearch_results_old_modules.jsonl
    │       └── autoresearch_results_pre_fixes.jsonl
    ├── overnight_apr2/      # from root overnight_results/
    │   └── (existing 16 files)
    └── warroom/             # from root warroom/
        └── (existing 9 files)
```

### Estimated LOC moved out of repo root

| Source | Lines | Files |
|---|---:|---:|
| Root `*.py` scripts → `examples/`, `examples/legacy/`, `archive/autoresearch/` | 7,653 | 16 |
| Root marketing/stale `*.md` → `docs/archive/` | ~3,000 | 6 |
| Tracked `autoresearch_results_*.jsonl` → archive (also untrack) | n/a (binary-ish) | 6 |
| `warroom/` → `archive/warroom/` | ~600 | 9 |
| `overnight_results/` → `archive/overnight_apr2/` | ~1,500 | 16 |

Net effect on repo root: **drops from 17 `.py` + 8 `.md` files to 1 `.py` (`setup.py`) + 3-4 `.md` files** (`README`, `CLAUDE`, `HANDOFF`, `LICENSE`).

LOC moved out of `turboquantdc/` itself: **0** — this hygiene review only touches the repo root and ancillary directories. Module pruning inside `turboquantdc/` is left to a separate review (the package directory has 67 modules totalling 4.4 MB — needs a dedicated audit).

### Disk-only cleanup (not in git, but should be removed)

```bash
rm -rf __pycache__/
find . -type d -name __pycache__ -exec rm -rf {} +
rm -rf .mypy_cache/ .ruff_cache/ .pytest_cache/
rm -rf dist/ turboquantdc.egg-info/
rm -f autoresearch.log autoresearch_real.log
rm -f autoresearch_real_results_buggy.jsonl autoresearch_results.jsonl autoresearch_results_buggy.jsonl
rm -f turboquantdc/evolving_compressor.py.backup
# .venv-vllm/ — keep if developer is mid-session, but add to .gitignore first
```

This frees ~166 MB of disk (.mypy_cache 97M + .venv-vllm 68M, plus smaller ones).

---

## CRITICAL findings (anything that would embarrass us if a stranger looked)

1. **CLAUDE.md is a brazen lie to a fresh agent / external reader.** First thing a curious open-source contributor reads is "All source files in `turboquantdc/` and `tests/` are empty stubs. setup.py is empty. Phase 1 has not started." Then they see `dist/turboquantdc-0.3.0.tar.gz`, 67 modules, and a deployed GitHub Pages site. Either we're lying or we don't know our own state. Both look bad.

2. **Six tracked `.jsonl` files self-labeled "buggy / broken_scoring / pre_fixes / pre_gradient / pre_hybrid / old_modules" totaling ~7 MB.** Anyone browsing the repo sees `autoresearch_results_v2_broken_scoring.jsonl` literally in the root listing. This signals "we ship broken artifacts and don't clean up."

3. **Repo root has 17 Python entry-point scripts.** A senior staff engineer skimming the project cannot tell what the package's intended entry point is. `demo.py`, `demo_70b.py`, `demo_app.py`, `demo_e8.py`, `demo_final.py`, `demo_gemma4.py`, `run_70b.py`, `run_infinite_context.py`, `showcase.py`, `benchmark.py`, `benchmark_entropy.py`, `benchmark_speed.py`, `autoresearch.py`, `autoresearch_real.py`, `autoresearch_dashboard.py`, `test_retrieval_scale.py`, `setup.py` — fifteen of these belong in `examples/` or `archive/`.

4. **`GROWTH_PLAYBOOK.md` ("Codename: MACHIAVELLI") is at repo root, tracked, 30 KB.** External readers see explicit "killer hook" / "killer demo" / "what NOT to do" growth-hacking marketing copy front and center. This is fine for an internal doc but actively repels open-source contributors who expect a serious technical project.

5. **`docs/PATENT_DRAFTS.md` and `docs/PATH_TO_100M.md` are in `.gitignore` but the directory still exists and could leak.** Sensitive / strategic docs adjacent to public docs. If they're sensitive, they should be outside the repo entirely (e.g., `~/turboquantdc-private/`). If they're not, the .gitignore entry is misleading.

6. **`test_retrieval_scale.py` (452 lines) lives in repo root, not `tests/`.** It looks like a real test, not a benchmark. Either it's broken/excluded for a reason (then archive it) or it should be in the test suite (then move it). Currently it's in limbo.

---

## HIGH

1. **`.venv-vllm/` (68 MB) is on disk and not gitignored.** Single `git add .` from a developer would commit a virtualenv. Add `.venv-*/` to `.gitignore` before any other action.

2. **PLAN.md says "Phase 5 ✅" but lists "[ ]" unchecked items inside Phase 3.** The plan's truth-state is internally inconsistent. Move PLAN.md to `docs/archive/` and let `HANDOFF.md` be the single source of project state.

3. **`MEMORY.md` (root, 690 bytes) is a stub** — only contains placeholder text "_(accumulated during implementation)_" and "_(analysis of tonbistudio/turboquant-pytorch goes here)_". Delete or move to `docs/archive/`. The real memory is in `~/.claude/projects/.../memory/MEMORY.md` per the system reminder.

4. **`PUBLISH.md` (root, 323 bytes)** is just two `twine upload` commands. Inline into README or `CONTRIBUTING.md` and delete.

5. **`requirements_demo.txt` is a third (vestigial) requirements file** — fold into `requirements-dev.txt`.

6. **`logs/2026-04-27/` is tracked under `logs/`** but `.gitignore` has `*.log` — date-stamped log dirs aren't covered. Add `logs/` to `.gitignore`.

7. **`turboquantdc/evolving_compressor.py.backup` exists on disk.** Pattern `*.backup` is gitignored, so it's not in git, but it's still on disk. Delete it. (`evolving_compressor.py` itself is tracked.)

8. **`docs/code_review/` is currently untracked** (`?? docs/code_review/` in `git status`). The audit reports being written today need to be intentionally added (or not) — flag for the reviewer's user to decide whether code-review reports are part of the repo or transient working artifacts.

---

## MEDIUM

1. **`reference/tonbistudio-ref/` is a git submodule pointer with mode 160000** — fine, but no `.gitmodules` file at root. A fresh `git clone` won't pull the submodule because there's no `.gitmodules` registration. Either add `.gitmodules` or convert to a vendored read-only copy + a `reference/SOURCE.md` saying "snapshot of github.com/tonbistudio/turboquant-pytorch at commit 07bd848". Current setup will silently fail for new contributors.

2. **`overnight_results/` at repo root is misleading.** It contains run scripts (`run_phase1.py` through `run_200b.py`) interleaved with markdown reports. The naming suggests "outputs" but it actually contains code. Move under `archive/overnight_apr2/` and split scripts vs reports.

3. **`benchmarks/` has 50+ scripts with 2026 dates spanning Mar 26 → Apr 15** with no internal organization. A reader can't tell which benchmark is current vs deprecated. At minimum split into `benchmarks/current/` (rotorquant_comprehensive, ppl_for_tom, niah_for_tom, adversarial_validation, e8 stuff) vs `benchmarks/legacy/` (everything pre-Apr 4).

4. **`benchmarks/results/v2_long_ctx_pca.pt` and `v2_pca_rotations.pt` (2.4 MB each)** — large `.pt` tensor files in the repo. `.gitignore` has `*.pt` listed, so these are likely tracked from before the rule was added. Verify with `git ls-files | grep .pt` and `git rm --cached` if so.

5. **`docs/google_blog.html` (119 KB)** — saved external blog page. Either explain its provenance in a `docs/external/README.md` or delete; raw HTML scrape of someone else's blog at root of `docs/` looks weird.

6. **`assets/twitter_card.html` (single file)** — move into `docs/assets/` (which exists and is empty).

7. **`__pycache__/` at the repo root** contains 7 `.pyc` files for `autoresearch.py`, `benchmark.py`, `demo_app.py`, `run_70b.py`, `showcase.py`. Confirms those scripts have been run from repo root recently, reinforcing they're acting as entry points. Argues even more strongly for organizing them.

8. **`docs/superpowers/` contains `plans/` and `specs/` directories that are empty in the snapshot.** Clean up or delete.

---

## LOW

1. `notebooks/` has only one `.ipynb`. Fine, but make sure `*.ipynb_checkpoints/` is gitignored (it is).

2. `index.html` at repo root is a 5-line redirect to `visualization/index.html`. Could be removed if GitHub Pages redirect is configured server-side, but it's harmless.

3. `MANIFEST.in` is 217 bytes — verify it matches the new directory layout after archival; otherwise sdist will pick up archived files.

4. `warroom/.gitignore` ignores `messages.jsonl` only — fine, but `warroom/messages.jsonl` is also explicitly listed in root `.gitignore`, redundant.

5. `requirements.txt` uses 70 bytes — likely just a few package names. Make sure it's pinned or at least version-bounded for reproducibility before any open-source push.

6. The repo's `.github/workflows/` has only `pages.yml`. No CI for tests / linting on PR. Adding a basic `pytest` workflow before opening up to external contributions would be a high-leverage low-cost addition.

---

## Verification trail

- `git ls-files | wc -l` → 306 tracked files
- `git ls-files | grep -E "\.(jsonl|log|backup)$"` → 5 jsonl files tracked at root, no `.log` or `.backup` (correctly gitignored)
- `git check-ignore -v` confirms `__pycache__/`, `*.egg-info/`, `dist/`, `.mypy_cache/`, `*.backup`, `autoresearch.log`, `autoresearch_results.jsonl` are all properly ignored
- `git ls-tree HEAD reference/` → `160000 commit ... reference/tonbistudio-ref` (submodule pointer, no `.gitmodules` at root)
- `git status --porcelain` → only `.venv-vllm/` and `docs/code_review/` untracked
- `du -sh` on cache dirs → 97 MB `.mypy_cache`, 68 MB `.venv-vllm`, 2.9 MB `turboquantdc/__pycache__/`

## Assumptions

- The user's goal is open-source readiness + vLLM integration polish, per the prompt.
- "Archival > deletion" — recommendations preserve all artifacts under `archive/` rather than removing them.
- `HANDOFF.md` is the current source of truth for project state (Apr 15, 2026), per its dating and detail level.
- The five tracked `_v*` jsonl files are not load-bearing for any current code path. If `autoresearch.py` reads them at startup, that would change the recommendation — quick grep before untracking.
- `docs/PATENT_DRAFTS.md` / `PATH_TO_100M.md` are listed in .gitignore but their on-disk presence wasn't directly verified in this audit.

## Unverified / risky

- Did not run `pytest` or `ruff` to verify what's actually green in `tests/`. The "1,818+ tests" figure comes from HANDOFF.md, not measured here.
- Did not check git history depth — repo has 163 commits per `git log --oneline | wc -l`, but did not check whether large `.jsonl` blobs are also bloating the pack files (would need `git count-objects -vH`).
- Did not enumerate `turboquantdc/` for deprecated modules (e.g., `ultra_compress.py`, `v2_cache.py`, `xquant_cache.py`, `ultimate_cache.py` — 4 separate "ultimate cache" attempts visible in the listing). That's a separate audit.
- The reference submodule (`reference/tonbistudio-ref`) has a tracked `.git` directory inside. If `.gitmodules` is missing, this may already be in a broken state for fresh clones.
