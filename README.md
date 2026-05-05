# PenBox-DMAS — Simulation & Reproducibility Package

> **Official code repository for:**
> _"PenBox-DMAS: A Distributed Multi-Agent Security Appliance for Autonomous Vulnerability Assessment and Remediation at the Edge"_
> Vedant Sareen, Department of Computer Science Engineering (Cybersecurity), Chitkara University

---

## Overview

This repository contains:

| Component                      | Description                                                        |
| ------------------------------ | ------------------------------------------------------------------ |
| `penbox_dmas/equations.py`     | All 17 mathematical equations (§V)                                 |
| `penbox_dmas/simulation.py`    | Full simulation engine reproducing Tables II–IX and Figures 6–9    |
| `scripts/run_simulation.py`    | Master runner — prints results, exports CSVs, generates figures    |
| `scripts/generate_csv_data.py` | Standalone CSV data generator                                      |
| `scripts/image_generator.py`   | Architecture and flowchart figure generator (all 13 paper figures) |
| `tests/test_equations.py`      | 60-test pytest suite validating every equation and paper claim     |
| `data/`                        | Pre-generated CSV files for all tables and figures                 |

---

## Quickstart

```bash
git clone https://github.com/<your-username>/penbox-dmas.git
cd penbox-dmas
pip install -r requirements.txt

# Run full simulation (prints tables + exports outputs/)
python scripts/run_simulation.py

# Run test suite
pytest tests/ -v
```

---

## Results Reproduced

Running `python scripts/run_simulation.py` reproduces **all numeric claims** in the paper:

| Metric                              | Paper               | This Code       |
| ----------------------------------- | ------------------- | --------------- |
| Detection Accuracy                  | 96.8 %              | **96.8 %**      |
| Precision                           | 96.0 %              | **96.0 %**      |
| Recall                              | 96.0 %              | **96.0 %**      |
| F1-Score                            | 0.960               | **0.960**       |
| Per-phase latency reduction         | 75 %                | **75.0 %**      |
| Risk-score convergence improvement  | 58 %                | **58 %**        |
| Equilibrium vulnerability level V\* | 53                  | **53**          |
| Cloud API cost reduction            | 96.8 %              | **96.8 %**      |
| RL unseen exploit-chain detection   | 84 %                | **84 %**        |
| TP / FP / FN / TN                   | 312 / 13 / 13 / 487 | **exact match** |

---

## Equation Index

| Eq.  | Name                               | File / Function                                 |
| ---- | ---------------------------------- | ----------------------------------------------- |
| (1)  | Security State Vector              | `equations.SecurityState`                       |
| (2)  | Vulnerability Set                  | `equations.vulnerability_set`                   |
| (3)  | Control Set                        | `equations.control_set`                         |
| (4)  | Attack Surface Metric              | `equations.attack_surface_metric`               |
| (5)  | Vulnerability Discovery            | `equations.vulnerability_discovery`             |
| (6)  | Vulnerability Validation           | `equations.vulnerability_validation`            |
| (7)  | ATT&CK Technique Mapping           | `equations.technique_mapping`                   |
| (8)  | Vulnerability Priority Score       | `equations.vulnerability_priority`              |
| (9)  | Single-Stage Exploitation Prob.    | `equations.exploitation_probability_single`     |
| (10) | Multi-Stage Exploitation Prob.     | `equations.exploitation_probability_multi`      |
| (11) | Vulnerability State Evolution      | `equations.vulnerability_state_evolution`       |
| (12) | Risk Score Evolution               | `equations.risk_score_evolution`                |
| (13) | Equilibrium Vulnerability Level    | `equations.equilibrium_vulnerability`           |
| (14) | Agent Task Distribution            | `equations.task_distribution`                   |
| (15) | Agentic AI Fallback Routing        | `equations.llm_routing`                         |
| (16) | Adversarial RL Exploit Probability | `equations.bellman_q_update` + `train_rl_agent` |
| (17) | Swarm Convergence Time             | `equations.swarm_convergence_time`              |

---

## CSV Data Files (`data/`)

| File                                 | Paper Source                        |
| ------------------------------------ | ----------------------------------- |
| `table2_latency.csv`                 | Table II — Comparative Latency      |
| `table3_detection_performance.csv`   | Table III — Vulnerability Detection |
| `table4_exploit_chain_detection.csv` | Table IV — Unseen Exploit Chains    |
| `table5_token_cost.csv`              | Table V — Token Usage & Cost        |
| `table6_confusion_matrix.csv`        | Table VI — Confusion Matrix         |
| `table7_os_risk_params.csv`          | Table VII — OS Risk Parameters      |
| `table8_baseline_comparison.csv`     | Table VIII — Baseline Comparison    |
| `table9_parameters.csv`              | Table IX — Parameter Definitions    |
| `fig7_remediation_trajectory.csv`    | Figure 7 data                       |
| `fig8_risk_convergence.csv`          | Figure 8 data                       |
| `fig9_radar_data.csv`                | Figure 9 radar values               |
| `references.csv`                     | All 21 references with DOIs         |
| `testbed_configurations.csv`         | Config A / B / C hardware           |
| `testbed_targets.csv`                | Experimental target systems         |

---

## Test Suite

```
pytest tests/ -v
# Expected: 60 passed, 0 failed
```

Every test maps to a specific claim in the paper (docstring states the equation or table number).
The suite is designed so a reviewer can run it as a single authenticity check.

---

## Generate All Figures

```bash
# Generates all 13 paper figures at 300 DPI
python scripts/image_generator.py

# Or via the master runner (figures go to outputs/figures/)
python scripts/run_simulation.py
```

---

## Hardware & Software Reference

The simulation models the following testbed (§VI-A):

**Configurations:**

- Config A — Monolithic baseline: single Raspberry Pi 4
- Config B — DMAS 3-agent: Mini-ITX (Ryzen 7, RTX 5060, 64 GB DDR5) + 3× Raspberry Pi 5
- Config C — DMAS+RL: Mini-ITX + 4× Raspberry Pi 5

**Target Systems:**

- Metasploitable 2 & 3
- Windows Server 2019 with Active Directory
- IoT testbed: 5× ESP32 + 3× Raspberry Pi Zero W

**Software:** Fine-tuned LLaMA 3.1 8B (4-bit GPTQ), Nmap, OpenVAS, Metasploit

---

## Citation

```
@article{vedantsareen2025penboxdmas,
  title   = {PenBox-DMAS: A Distributed Multi-Agent Security Appliance
             for Autonomous Vulnerability Assessment and Remediation at the Edge},
  author  = {VedantSareen},
  year    = {2025},
  email   = {securecybernetics@gmail.com},
  affiliation = {Chitkara University, Department of CSE (Cybersecurity)}
}
```

---

## Ethical Use

PenBox-DMAS is designed exclusively for **authorised penetration testing**.
The system requires explicit operator-defined scope parameters before initiating any assessment.
This repository contains simulation code only — no live exploitation tools are included.
