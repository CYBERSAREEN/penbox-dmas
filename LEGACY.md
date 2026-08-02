# Superseded v1 package

The first version of this repository (commit `d2609b7`, May 2026) shipped a
`penbox_dmas/` package, `scripts/`, `tests/`, `data/` and `outputs/` that
accompanied an earlier draft of the paper. Those files were removed from `HEAD`
in the revision that introduced the current backend. They remain in git history
and can be recovered at any time:

```sh
git show d2609b7 --stat
git checkout d2609b7 -- penbox_dmas/          # restore into the working tree
```

## Why they were removed rather than kept alongside

They contradict the current paper, and a reader who ran them would get results
that do not match it.

1. **Different mathematical framework.** The v1 package implements **17**
   equations. The current paper has **13**, several of which are new and several
   of the old ones no longer exist. `penbox_dmas/equations.py` is not a subset
   of the current framework; it is a different one.

2. **Hardware assumptions throughout.** The v1 simulation models a tiered
   x86-plus-ARM appliance. The current paper is software-only and moves the
   hardware realisation to future work, so the v1 latency and fault-tolerance
   results describe a system the paper no longer claims.

3. **`data/references.csv` carried citations that do not exist.** Four entries
   could not be verified against the live record and have been removed from the
   paper entirely:

   | v1 ref | Claimed venue | Status |
   |---|---|---|
   | Wang *et al.*, "EdgeAI: Hardware-Software Co-Design…" | ACM CCS 2023 | not found |
   | Chen *et al.*, "Autonomous Attack Graph Generation Using Deep RL" | IEEE TDSC 2024 | not found |
   | Li *et al.*, "Hardware-Based Security for IoT: Survey" | IEEE IoT-J 2023 | DOI does not resolve to this title |
   | Liu *et al.*, "Token-Aware Routing for Hybrid Cloud-Edge LLM Inference" | ACM SEC 2025 | not found |

   A fifth, "Simonetto *et al.*, Automated CVE-to-ATT&CK Mapping"
   (*Information* 15(4):214), is a real paper but by Brănescu, Grigorescu and
   Dascălu. It is now cited correctly.

   Leaving that CSV published under an author's name was the single strongest
   reason not to keep the v1 tree at `HEAD`.

4. **Results tables no longer hold.** `data/table*.csv` and
   `outputs/csv/*.csv` encode the v1 numbers. The current results come from
   `dmas_model.py` and are regenerated on every build; `results.json` is the
   only results file now.

## What replaced it

| v1 | now |
|---|---|
| `penbox_dmas/equations.py`, `simulation.py` | `dmas_model.py` — one file, all 13 equations evaluated |
| `scripts/run_simulation.py`, `image_generator.py` | `dmas_model.py` (figures included) |
| `tests/test_equations.py` | `verify.py` — 26 checks, incl. paper↔model agreement and citation order |
| `data/*.csv`, `outputs/` | `results.json`, `numbers.tex`, `table_groups.tex`, `figs/*.pdf`, all generated |

The substantive improvement is that the paper can no longer disagree with the
code: `PAPER.tex` reads its numbers from `numbers.tex`, which only
`dmas_model.py` writes.
