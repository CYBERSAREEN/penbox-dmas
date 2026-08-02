# PenBox-DMAS

Reproducibility repository for the paper *PenBox-DMAS: A Distributed Multi-Agent
Security Appliance for Autonomous Vulnerability Assessment and Remediation at
the Edge*.

**Vedant Sareen**, Department of Computer Science and Engineering
(Cybersecurity), Chitkara University, Punjab.

---

## What this repository is for

Papers usually ask you to trust their numbers. This one does not.

Every figure, table and quoted value in `PAPER.pdf` is produced by
`dmas_model.py`. The LaTeX source contains no hand-typed results: the model
writes each value as a macro into `numbers.tex`, and the paper cites the macro.
Change the model and the abstract changes on the next build. There is no path
by which a stale number survives.

`verify.py` tests that claim rather than asserting it, and CI runs it on a
clean machine on every push.

The distinction CI enforces is deliberate. `numbers.tex` — the file the paper
actually reads — must regenerate **byte-identically on a different machine**.
Figure PDFs and full-precision `results.json` are held to a weaker standard,
because matplotlib writes version-specific bytes and 4000 episodes of
floating-point accumulation can differ in the last bits across NumPy and BLAS
builds. Claiming byte equality there would be claiming something untrue of any
Python numerical stack. The published values are unaffected.

```
$ python3 verify.py
...
28 passed, 0 failed
```

## What it checks

| Group | Checks |
|---|---|
| **Reproducibility** | The paper's numbers regenerate byte-identically from the committed code on any machine (`numbers.tex`, `table_groups.tex`); full-precision results agree to 1e-6; a second run in the same environment is byte-identical across all 12 generated files, figures included. |
| **Paper ↔ model** | Every generated macro the paper uses is defined; 16 headline values in the text are compared against `results.json` directly. |
| **Internal consistency** | Confusion matrices close and their derived metrics recompute; recall is monotonic in worker count; speed-up never exceeds `N`; every reported probability lies in `[0,1]`; Eq. (1) satisfies `P = E·O/T`; Eq. (13) satisfies `V* = v_new/µ`; gated routing costs less than ungated. |
| **Figures** | All included figures exist, are vector (not embedded bitmaps), carry no orphans, and every one is referenced from the text. |
| **Citations** | Every `\cite` resolves, every reference is cited, and references are numbered in order of first citation as IEEE requires. |
| **Compilation** | The paper builds in a clean temporary directory with zero errors, zero overfull boxes and zero undefined references. |

One check exists specifically to keep the paper honest about its own weakest
result: the random control in the chain-prediction experiment reaches **91.7 %
recall**, higher than the proposed method. That happens because calibrating a
meaningless score for maximum F1 drives it to label nearly everything positive.
`verify.py` asserts the control is beaten on **F1** while confirming its recall
is high, so the reason the paper reports F1 rather than recall is enforced by a
test instead of buried in prose.

## Build

```sh
pip install -r requirements.txt
make            # runs the model, builds the paper, runs every check
```

or by hand:

```sh
python3 dmas_model.py                 # regenerates numbers, tables, figures
pdflatex PAPER.tex && pdflatex PAPER.tex
python3 verify.py
```

Needs TeX Live with `texlive-publishers` (IEEEtran) and `texlive-science`
(algorithm/algpseudocode).

## Layout

| Path | Role |
|---|---|
| `PAPER.tex` | The paper. Reads `numbers.tex` and `table_groups.tex`. |
| `PAPER.pdf` | Built artefact, 12 pages, IEEE two-column. |
| `dmas_model.py` | The evaluation backend. Produces everything below. |
| `verify.py` | Authenticity checks. |
| `numbers.tex` | **Generated** — 130 `\newcommand` macros. Do not edit. |
| `table_groups.tex` | **Generated** — the per-group detection table. Do not edit. |
| `results.json` | **Generated** — the full result set, for archival. |
| `figs/*.pdf` | **Generated** — nine vector figures, greyscale-safe. |

## What the model actually does

Not a Monte-Carlo wrapper around chosen answers. It is an executable model of
the assessment pipeline:

- a synthetic inventory of **325 findings** across 16 hosts and 227 services,
  with CVSS v3.1 scores drawn per severity band and attack complexity
  negatively correlated with severity;
- a **discrete-event scheduler** implementing the assignment rule of Eq. (11),
  measured over 40 independent task streams for 1 to 8 workers;
- a **chain-prediction agent** trained by semi-gradient Q-learning over 4000
  episodes, with 25 % of escalation edges withheld so the 1805 evaluation
  chains are genuinely unseen;
- a **token and cost model** priced at published list rates for
  `claude-opus-5` and `claude-fable-5`.

All thirteen equations in the paper are evaluated numerically. None is stated
without a value.

## The limitation you should read first

**This is a software-in-the-loop evaluation. No hardware was built and no
production network was touched.** The results are the behaviour of a model
against a synthetic inventory. They are reproducible and their assumptions are
stated, but a synthetic inventory cannot reproduce real services under real
load.

Section VI of the paper should be read as *the architecture is internally
consistent and its trade-offs behave as the equations predict* — not as a
measurement of a deployed system. Field validation against instrumented live
networks with analyst-adjudicated ground truth is stated as required future
work, and the paper says so in both the threat model and the limitations.

Nine of the parameters in Table VIII are **set**, not measured — the
exploitation skill factor, the blast-radius weights, the priority mix,
remediation efficacy and the arrival rate among them. The table marks each one.
Results that depend on them inherit their uncertainty, and a sensitivity
analysis is outstanding.

## Citation hygiene

Every reference was verified against the live record before inclusion. Four
entries in an earlier draft could not be found and were **removed rather than
replaced with lookalikes**:

- "EdgeAI: A Hardware-Software Co-Design for On-Device AI Security
  Applications", CCS '23
- "Autonomous Attack Graph Generation and Analysis Using Deep Reinforcement
  Learning", IEEE TDSC 21(2)
- "A Survey of Hardware-Based Security for IoT Devices", IEEE IoT-J
- "Token-Aware Routing for Hybrid Cloud-Edge LLM Inference in Security
  Appliances", ACM SEC 2025

Three more were real but carried wrong metadata, now corrected: *Information*
15(4):214 is by Brănescu, Grigorescu and Dascălu; the ESWA reinforcement-learning
survey is vol. 300, art. 130219; CHECKMATE (arXiv:2512.11143) is December 2025.

Where a removed citation had been supporting a claim, a verified paper was
substituted — Yousefi *et al.* (TrustCom 2018) for attack-graph reinforcement
learning, Hao *et al.* (EdgeFM '24) for edge-cloud inference splitting.

## Scope and intended use

Research artefact for authorised security assessment. The framework requires an
operator-supplied scope before it will start, models persistence rather than
establishing it, and makes no claim of autonomous security-operations
capability. See Section II of the paper.

## Licence

MIT — see `LICENSE`.
