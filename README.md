# poker-llm-irt

Item-response-theory (IRT) and Q-matrix multidimensional factor analysis of a
panel of large language models evaluated on
[PokerBench](https://huggingface.co/datasets/RZ412/PokerBench) (Zhuang et al.,
2025; arXiv:2501.08328). The codebase fits a 2PL IRT baseline, a free
K-factor model, and a series of Q-matrix-constrained variants (action class,
game phase, heuristic skill, LLM-labeled skill, joint Action x Skill) to test
whether PokerBench accuracy is well described as a unidimensional skill or
conflates separable sub-skills. Stanford CS321M final project.

## Setup

A conda environment file (`environment.yml`) and a pip fallback
(`requirements.txt`) are provided. Python 3.11 is required.

```bash
conda env create -f environment.yml
conda activate poker-irt
# or, with pip only:
# python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

API keys for the model panel and the labeling helper are read from a `.env`
file. Copy `.env.example` and fill in the credentials you intend to use. These
keys are only required to re-run the LLM evaluation step; reproducing the
paper tables and figures from the cached response shards does **not** need any
API access.

```bash
cp .env.example .env
```

### Datasets

The PokerBench test sets (1k preflop + 10k postflop) are pulled from
[Hugging Face](https://huggingface.co/datasets/RZ412/PokerBench):

```bash
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('RZ412/PokerBench', repo_type='dataset', \
local_dir='data/raw/pokerbench')"
```

Per-model response shards used to build the response matrices live under
`data/responses/main_11k_seed42_7models/` and are already in the repo.
Regenerating them from scratch (running the 7-model panel over the full 11k
test set) requires the API keys above and the
`scripts/paper/materialize_gemini_flash_lite.py` materialization step for
Gemini's preflop shard.

### Hardware and runtime

All paper experiments run on a single CPU. No GPU is required. Peak memory
is roughly 4 GB. Approximate wall times on a modern laptop (Apple M-series):

| Step | Time |
|---|---|
| Heuristic skill rebuild (`build_heuristic_qmatrix.py`) | < 2 s |
| Single Q-matrix fit (`qmatrix_comparison.py --dimension action`) | ~30 s |
| 5-fold CV LL sweep (`qmatrix_cv_ll.py`) | ~5 min |
| Joint Action × Skill CV (`qmatrix_within_action.py`) | ~5 min |
| Bootstrap CIs (`qmatrix_bootstrap.py`, default `--n-boot=200`) | ~30 min |
| Permutation tests | ~10 min each |
| Full re-eval of the 7-model panel on 11k items | hours, API-bound |

## Repository layout

```
poker-llm-irt/
├── src/poker_irt/              # library
│   ├── data.py                 # PokerBench loader and stratified sampler
│   ├── prompting.py            # zero-shot prompt construction
│   ├── parsing.py              # regex parsing for GTO labels and LLM outputs
│   ├── scoring.py              # action accuracy / exact match
│   ├── matrix.py               # parquet shard -> long + wide AA matrix
│   ├── features.py             # Phase 1 instruction-text features
│   ├── features_phase2.py      # Phase 2 hand/board interaction features
│   ├── features_phase3.py      # Phase 3 equity features via treys
│   ├── irt.py                  # 1PL / 2PL IRT via py-irt
│   ├── factor.py               # free K-factor multidim IRT (Pyro SVI)
│   ├── qmatrix.py              # Q-matrix constrained multidim IRT
│   ├── runner.py               # eval orchestrator
│   ├── logging_setup.py        # per-run logging helpers
│   └── providers/              # OpenAI / Anthropic / Gemini / Together / DeepSeek
├── scripts/
│   └── paper/                  # scripts that produce paper artifacts (see below)
├── data/
│   ├── raw/                    # PokerBench JSON (gitignored)
│   └── responses/              # per-(model, run) parquet response shards
├── outputs/
│   └── main_11k_seed42_7models/# 7-model run artifacts (paper)
└── environment.yml
```

## Quickstart

The following two commands take under five minutes total on a laptop and
regenerate the heuristic skill labels plus Figure 1 from artifacts already
checked in to `outputs/main_11k_seed42_7models/`:

```bash
conda activate poker-irt

# Rebuild the rule-based 8-skill Q-matrix labels (< 2 s).
python scripts/paper/build_heuristic_qmatrix.py

# Regenerate Figure 1 (ΔCV LL per cell bar chart) from the cached CV LL CSV.
python scripts/paper/fig1_qmatrix_cvll_with_joint.py
# -> outputs/main_11k_seed42_7models/figs/paper_fig1_qmatrix_cvll_with_joint.png
```

If the first command prints the same 8-row skill distribution shown in the
paper (folding_discipline 2500, pot_control_check 2096, bluff_catching 1977,
value_betting 1815, preflop_open_3bet_4bet 1000, bluff_betting 685,
pot_odds_drawing 523, position_aware_continuation 404) and the second writes
a PNG, the environment is set up correctly.

## Reproducing the paper results

The paper artifacts live under `outputs/main_11k_seed42_7models/`. The fit
pickles and response matrices required for the headline numbers in Table 2
and Figures 1-3 are already on disk; the commands below regenerate the
derived tables and figures from those artifacts.

Script → paper artifact mapping:

| Paper artifact | Script |
|---|---|
| Table 2, ΔBIC rows (Action / Phase / Heuristic skill / Gemini skill) | `scripts/paper/qmatrix_comparison.py` |
| Table 2, held-out CV LL per cell | `scripts/paper/qmatrix_cv_ll.py` |
| Table 2, Joint Action × Skill row | `scripts/paper/qmatrix_within_action.py` |
| Figure 1 (ΔCV LL per cell, all Q-matrices vs K=1) | `scripts/paper/fig1_qmatrix_cvll_with_joint.py` |
| Figure 2 (per-(model, action-class) κ heatmap) | `scripts/paper/paper_figures_v2_7models.py` (the `fig4b_kappa` function; output `paper_fig4b_kappa.png`) |
| Figure 3 (ability profile heatmaps with bootstrap CIs) | `scripts/paper/qmatrix_bootstrap.py` |
| Appendix permutation null | `scripts/paper/paper_figures_v2_7models.py` + `permutation_test.py` |

```bash
RUN=main_11k_seed42_7models

# Table 2 (Q-matrix ΔBIC rows): action, phase, heuristic skill, Gemini skill
python scripts/paper/qmatrix_comparison.py --run $RUN --dimension action
python scripts/paper/qmatrix_comparison.py --run $RUN --dimension phase
python scripts/paper/qmatrix_comparison.py --run $RUN --dimension skill \
    --skill-file data/responses/main_11k_seed42_7models/item_skills_HEURISTIC.parquet
python scripts/paper/qmatrix_comparison.py --run $RUN --dimension skill \
    --skill-file data/responses/main_11k_seed42_7models/item_skills_at_11k.parquet

# Table 2 (held-out CV LL per cell)
python scripts/paper/qmatrix_cv_ll.py --run $RUN

# Table 2 (joint Action x Skill row)
python scripts/paper/qmatrix_within_action.py --run $RUN

# Permutation tests
python scripts/paper/permutation_test.py --run $RUN \
    --skill-file data/responses/main_11k_seed42_7models/item_skills_at_11k.parquet
python scripts/paper/permutation_test_restricted.py --run $RUN

# Figure 1 (Q-matrix held-out ΔCV LL per cell bar chart including joint variant).
# Reads the CV LL CSV produced by the two commands above, so run them first.
python scripts/paper/fig1_qmatrix_cvll_with_joint.py

# Figures 2-3 (per-action AA heatmap, ability profile heatmap, permutation null)
python scripts/paper/paper_figures_v2_7models.py

# Bootstrap CIs for the ability profile figures
python scripts/paper/qmatrix_bootstrap.py --run $RUN

# Appendix: reliability and K-factor sweep
python scripts/paper/reliability_7models.py
python scripts/paper/factor_sweep_7model.py
```

The heuristic Q-matrix labels can be rebuilt from item features with:

```bash
python scripts/paper/build_heuristic_qmatrix.py
```

## Citation

```bibtex
@misc{poker-llm-irt,
  title  = {Item Response Theory and Q-matrix Factor Analysis of LLMs on PokerBench},
  author = {Sattiraju, Abhinav},
  year   = {2026},
  note   = {Stanford CS321M final project}
}
```
