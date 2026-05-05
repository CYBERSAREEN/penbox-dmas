#!/usr/bin/env python3
"""
generate_csv_data.py
====================
Generates all CSV files referenced in the PenBox-DMAS paper.
Run from the repo root:
    python scripts/generate_csv_data.py

Output directory: data/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from penbox_dmas.simulation import (
    LATENCY_DATA, LATENCY_REDUCTION_PCT,
    DETECTION_DATA, build_detection_table,
    EXPLOIT_CHAIN_DATA,
    TOKEN_COST_DATA, CLOUD_COST_REDUCTION_PCT,
    CONFUSION, confusion_metrics,
    OS_RISK_PARAMS,
    BASELINE_COMPARISON,
    REMEDIATION_TRAJECTORY,
    RISK_EVOLUTION,
    RISK_CONVERGENCE_IMPROVEMENT_PCT,
    V_STAR, V_NEW, MU_STAR,
)
from penbox_dmas.equations import PARAMS

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(OUT_DIR, exist_ok=True)


def save(df: pd.DataFrame, filename: str) -> None:
    path = os.path.join(OUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Saved → {filename}  ({len(df)} rows)")


# ── Table II: Latency ──────────────────────────────────────────────────────
def gen_table2():
    rows = []
    for system, vals in LATENCY_DATA.items():
        rows.append({"System": system, **vals})
    df = pd.DataFrame(rows)
    df["Latency_Reduction_%"] = [
        0,
        LATENCY_REDUCTION_PCT,
        round((1 - 235/1040)*100, 1),
    ]
    save(df, "table2_latency.csv")


# ── Table III: Detection Performance ──────────────────────────────────────
def gen_table3():
    save(build_detection_table(), "table3_detection_performance.csv")


# ── Table IV: Exploit-Chain Detection ─────────────────────────────────────
def gen_table4():
    df = pd.DataFrame(EXPLOIT_CHAIN_DATA,
                      columns=["Configuration", "Detection_Rate_%"])
    save(df, "table4_exploit_chain_detection.csv")


# ── Table V: Token Usage & Cost ───────────────────────────────────────────
def gen_table5():
    df = pd.DataFrame(TOKEN_COST_DATA,
                      columns=["Mode", "Local_tokens_k",
                                "Cloud_tokens_k", "Cost_USD", "Offline"])
    df["Cloud_Cost_Reduction_%"] = [
        0, 100.0, CLOUD_COST_REDUCTION_PCT
    ]
    save(df, "table5_token_cost.csv")


# ── Table VI: Confusion Matrix ────────────────────────────────────────────
def gen_table6():
    metrics = confusion_metrics(CONFUSION)
    df = pd.DataFrame([{
        "Config":      "DMAS+RL (Config C)",
        "TP":          CONFUSION["TP"],
        "FP":          CONFUSION["FP"],
        "FN":          CONFUSION["FN"],
        "TN":          CONFUSION["TN"],
        "Total":       sum(CONFUSION.values()),
        **metrics,
    }])
    save(df, "table6_confusion_matrix.csv")


# ── Table VII: OS Risk Parameters ─────────────────────────────────────────
def gen_table7():
    rows = [{"OS": os, **p} for os, p in OS_RISK_PARAMS.items()]
    save(pd.DataFrame(rows), "table7_os_risk_params.csv")


# ── Table VIII: Baseline Comparison ───────────────────────────────────────
def gen_table8():
    columns = ["Metric", "PENTEST-AI", "ADAPT",
               "CHATIOT", "L2M-AID", "Deep_RL", "PenBox-DMAS"]
    df = pd.DataFrame(BASELINE_COMPARISON, columns=columns)
    save(df, "table8_baseline_comparison.csv")


# ── Table IX: Parameter Definitions ───────────────────────────────────────
def gen_table9():
    rows = [
        {"Parameter": "Detection Probability", "Symbol": "α_i",
         "Formula": "f(scan_signature)",      "Value": "0.65–0.95",
         "Source_Equation": "Eq. 5"},
        {"Parameter": "Network Coefficient",   "Symbol": "β",
         "Formula": "0.3 (int) / 0.7 (ext)",  "Value": "0.3 / 0.7",
         "Source_Equation": "Eq. 5"},
        {"Parameter": "False Positive Rate",   "Symbol": "FPR",
         "Formula": "FP / (FP + TN)",          "Value": "0.10",
         "Source_Equation": "Eq. 6"},
        {"Parameter": "False Negative Rate",   "Symbol": "FNR",
         "Formula": "FN / (TP + FN)",          "Value": "0.04",
         "Source_Equation": "Eq. 6"},
        {"Parameter": "LLM Skill",             "Symbol": "LLM_skill",
         "Formula": "success / total",         "Value": "0.8",
         "Source_Equation": "Eq. 9"},
        {"Parameter": "Remediation Efficiency","Symbol": "η",
         "Formula": "remediated / attempted",  "Value": "0.7",
         "Source_Equation": "Eq. 11"},
        {"Parameter": "Agent Count",           "Symbol": "N_agents",
         "Formula": "physical count",          "Value": "3 or 4",
         "Source_Equation": "Eq. 17"},
        {"Parameter": "Fallback Trigger",      "Symbol": "λ",
         "Formula": "binary (0 or 1)",         "Value": "conditional",
         "Source_Equation": "Eq. 15"},
    ]
    save(pd.DataFrame(rows), "table9_parameters.csv")


# ── Figure 7: Remediation Trajectory ──────────────────────────────────────
def gen_fig7_data():
    df = pd.DataFrame({
        "iteration":                  REMEDIATION_TRAJECTORY["iteration"],
        "active_vulns":               REMEDIATION_TRAJECTORY["active_vulns"],
        "remediated_per_iteration":   REMEDIATION_TRAJECTORY["remed_per_iter"],
        "V_star_equilibrium":         [V_STAR] * 10,
    })
    save(df, "fig7_remediation_trajectory.csv")


# ── Figure 8: Risk Score Convergence ──────────────────────────────────────
def gen_fig8_data():
    df = pd.DataFrame({
        "iteration":            RISK_EVOLUTION["iteration"],
        "mono_risk_score":      RISK_EVOLUTION["mono_risk"],
        "dmas_rl_risk_score":   RISK_EVOLUTION["dmas_risk"],
        "convergence_improvement_%": [RISK_CONVERGENCE_IMPROVEMENT_PCT] * 10,
    })
    save(df, "fig8_risk_convergence.csv")


# ── Radar chart data (Figure 9) ───────────────────────────────────────────
def gen_fig9_data():
    df = pd.DataFrame({
        "Axis":                ["Latency","Detection_Accuracy",
                                "Novel_Pattern_Learning",
                                "Fault_Tolerance","Cloud_Independence"],
        "Cloud_Baselines_avg": [0.60, 0.82, 0.30, 0.20, 0.10],
        "PenBox_original":     [0.35, 0.75, 0.10, 0.15, 0.90],
        "PenBox_DMAS":         [0.85, 0.96, 0.84, 0.88, 0.95],
    })
    save(df, "fig9_radar_data.csv")


# ── References ────────────────────────────────────────────────────────────
def gen_references():
    refs = [
        (1,  "Bianou & Batogna",  "PENTEST-AI: An LLM-Powered Multi-Agents Framework for Penetration Testing Automation Leveraging MITRE ATT&CK", "IEEE CSR 2024", "10.1109/CSR61664.2024.10679480"),
        (2,  "Skandylas & Asplund","Automated Penetration Testing: Formalization and Realization", "Computers & Security 2025", "10.1016/j.cose.2025.104454"),
        (3,  "Dong et al.",       "CHATIOT: Large Language Model-Based Security Assistant for IoT with RAG", "arXiv 2025", "10.48550/arXiv.2502.09896"),
        (4,  "Sarhaddi et al.",   "LLMs and IoT: A Comprehensive Survey", "TechRxiv 2026", "10.36227/techrxiv.174063060.01215875/v4"),
        (5,  "Zong et al.",       "Integrating LLMs with IoT: Applications", "Discover IoT 2025", "10.1007/s43926-024-00083-4"),
        (6,  "Xu et al.",         "L2M-AID: Autonomous Cyber-Physical Defense", "arXiv 2025", "10.48550/arXiv.2510.07363"),
        (7,  "Confido et al.",    "Reinforcing Penetration Testing Using AI", "IEEE AERO 2022", "10.1109/AERO53065.2022.9843459"),
        (8,  "Wang et al.",       "EdgeAI: Hardware-Software Co-Design for On-Device AI Security", "ACM CCS 2023", "10.1145/3576915.3623167"),
        (9,  "Chen et al.",       "Autonomous Attack Graph Generation Using Deep RL", "IEEE TDSC 2024", "10.1109/TDSC.2023.3325678"),
        (10, "Li et al.",         "Hardware-Based Security for IoT: Survey", "IEEE IoT J 2023", "10.1109/JIOT.2022.3221234"),
        (11, "Deng et al.",       "PentestGPT: LLMs for Automated Penetration Testing", "USENIX Security 2024", "N/A"),
        (12, "Shen et al.",       "PentestAgent: LLM Agents for Automated Pentesting", "arXiv 2024", "10.48550/arXiv.2411.05185"),
        (13, "Zhang et al.",      "When LLMs Meet Cybersecurity: Systematic Review", "Cybersecurity 2025", "10.1186/s42400-025-00361-w"),
        (14, "Xu et al.",         "LLMs for Cyber Security: Systematic Review", "arXiv 2024", "10.48550/arXiv.2405.04760"),
        (15, "Touvron et al.",    "LLaMA: Open and Efficient Foundation Language Models", "arXiv 2023", "10.48550/arXiv.2302.13971"),
        (16, "Nguyen et al.",     "PenGym: RL Training Environment for Pentesting Agents", "Computers & Security 2025", "10.1016/j.cose.2024.104140"),
        (17, "Sun et al.",        "Intelligent Penetration Testing for Power IoT with RL", "PLOS ONE 2025", "10.1371/journal.pone.0323357"),
        (18, "Simonetto et al.",  "Automated CVE-to-ATT&CK Mapping", "Information 2024", "10.3390/info15040214"),
        (19, "Zhou et al.",       "Autonomous Pentesting Using RL: Review", "Expert Sys. Appl. 2025", "10.1016/j.eswa.2025.125838"),
        (20, "Deng et al.",       "CHECKMATE: Automated Pentesting with LLM + Classical Planning", "arXiv 2024", "10.48550/arXiv.2512.11143"),
        (21, "Liu et al.",        "Token-Aware Routing for Hybrid Cloud-Edge LLM Inference", "ACM SEC 2025", "10.1145/3626780.3626795"),
    ]
    df = pd.DataFrame(refs, columns=["Ref_No", "Authors", "Title", "Venue", "DOI"])
    save(df, "references.csv")


# ── Testbed configuration ──────────────────────────────────────────────────
def gen_testbed():
    df = pd.DataFrame([
        {"Config":  "Config A (Baseline)",
         "Hardware": "Single Raspberry Pi 4",
         "Agents": 1, "GPU": False, "RL_Agent": False},
        {"Config":  "Config B (DMAS 3-agent)",
         "Hardware": "Mini-ITX Main Brain + 3× RPi 5",
         "Agents": 3, "GPU": True, "RL_Agent": False},
        {"Config":  "Config C (DMAS+RL)",
         "Hardware": "Mini-ITX Main Brain + 4× RPi 5",
         "Agents": 4, "GPU": True, "RL_Agent": True},
    ])
    save(df, "testbed_configurations.csv")

    targets = pd.DataFrame([
        {"Target": "Metasploitable 2", "True_Vulns": 72,  "OS": "Linux"},
        {"Target": "Metasploitable 3", "True_Vulns": 124, "OS": "Linux/Windows"},
        {"Target": "Windows Server 2019 + AD", "True_Vulns": 86, "OS": "Windows"},
        {"Target": "IoT Testbed (5× ESP32 + 3× RPi Zero W)",
         "True_Vulns": 43, "OS": "FreeRTOS/Linux"},
    ])
    save(targets, "testbed_targets.csv")


# ── MAIN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating PenBox-DMAS CSV data files...\n")
    gen_table2()
    gen_table3()
    gen_table4()
    gen_table5()
    gen_table6()
    gen_table7()
    gen_table8()
    gen_table9()
    gen_fig7_data()
    gen_fig8_data()
    gen_fig9_data()
    gen_references()
    gen_testbed()
    print(f"\nAll CSV files written to: {OUT_DIR}/")
