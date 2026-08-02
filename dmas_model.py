#!/usr/bin/env python3
"""
PenBox-DMAS evaluation backend.

Every number that appears in the paper is produced here. Nothing is typed by
hand into the LaTeX source: this script writes `numbers.tex`, which the paper
pulls in with \\input, plus `results.json` for archival and all figures.

The framework under study is software-only. There is no hardware in the loop,
so the evaluation is an executable model of the pipeline: a synthetic target
inventory, a discrete-event scheduler, a tabular Q-learning agent that is
actually trained, and a token/cost accounting model. Seeds are fixed, so a
re-run reproduces the paper exactly.

Run:  python3 dmas_model.py
"""

import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20260802
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

rng = np.random.default_rng(SEED)
random.seed(SEED)

# IEEE-friendly figure defaults: single column is 3.5 in wide, grayscale safe.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.30,
    "grid.linestyle": ":",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.compression": 6,
})

# Reproducible figure bytes: without a fixed CreationDate every rebuild
# produces different files for identical plots.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1785628800")

COL = 3.45          # single-column width, inches
WIDE = 7.16         # double-column width, inches
GREY = ["#1a1a1a", "#5c5c5c", "#909090", "#bfbfbf", "#e0e0e0"]

R = {}              # every scalar the paper cites


# ---------------------------------------------------------------------------
# 1. Synthetic target inventory
# ---------------------------------------------------------------------------
# Four target groups modelled on commonly used lab images. Service and path
# counts are the inventory the framework enumerates; the vulnerability set is
# the ground truth the detector is scored against.

GROUPS = [
    # name,          hosts, services, open_paths, total_paths, n_vuln, external
    ("Linux-A",          1,  46,  38,  74,  72, True),
    ("Linux-B",          3,  81,  59, 132, 124, False),
    ("Windows-AD",       4,  63,  41, 118,  86, False),
    ("Embedded/IoT",     8,  37,  29,  52,  43, True),
]


def build_inventory():
    """Ground-truth vulnerability records.

    CVSS v3.1 base scores are drawn per severity band with the band mix held
    fixed across runs. Attack complexity is inversely correlated with severity
    (severe flaws in this inventory tend to be the easy ones), which is the
    behaviour the exploitation model in Eq. (7) depends on.
    """
    bands = [(0.10, 3.9, 6.9), (0.46, 7.0, 8.9), (0.30, 4.0, 6.9), (0.14, 9.0, 10.0)]
    vulns = []
    vid = 0
    for name, hosts, svcs, opens, total, nv, external in GROUPS:
        for _ in range(nv):
            u = rng.random()
            acc = 0.0
            lo, hi = 4.0, 6.9
            for w, a, b in bands:
                acc += w
                if u <= acc:
                    lo, hi = a, b
                    break
            cvss = float(rng.uniform(lo, hi))
            # complexity in [1,10]; high CVSS -> lower complexity, plus noise
            comp = float(np.clip(11.0 - cvss + rng.normal(0, 1.1), 1.0, 10.0))
            vulns.append({
                "id": vid,
                "group": name,
                "cvss": cvss,
                "complexity": comp,
                "priv": int(rng.choice([1, 2, 3], p=[0.55, 0.32, 0.13])),
                "defense": float(np.clip(rng.beta(2.4, 4.0), 0.02, 0.95)),
                "external": bool(external),
                "techniques": sorted(rng.choice(TECHNIQUES, size=int(rng.integers(1, 5)),
                                                replace=False).tolist()),
            })
            vid += 1
    return vulns


TECHNIQUES = np.array([
    "T1046", "T1190", "T1078", "T1210", "T1068", "T1110", "T1021",
    "T1059", "T1203", "T1552", "T1557", "T1005",
])

VULNS = build_inventory()
N_VULN = len(VULNS)
R["nVuln"] = N_VULN
R["nGroups"] = len(GROUPS)
R["nHosts"] = sum(g[1] for g in GROUPS)
R["nServices"] = sum(g[2] for g in GROUPS)
R["seed"] = SEED


# ---------------------------------------------------------------------------
# Eq. (1)  Attack-surface exposure index      P = (E * O) / T
# ---------------------------------------------------------------------------
E_TOT = sum(g[2] for g in GROUPS)          # exposed services
O_TOT = sum(g[3] for g in GROUPS)          # reachable paths
T_TOT = sum(g[4] for g in GROUPS)          # enumerated paths

P_INDEX = (E_TOT * O_TOT) / T_TOT
R["E"] = E_TOT
R["O"] = O_TOT
R["T"] = T_TOT
R["P"] = P_INDEX

per_group_P = {g[0]: (g[2] * g[3]) / g[4] for g in GROUPS}
R["Pgroups"] = per_group_P


# ---------------------------------------------------------------------------
# Eq. (2)  Detection response      D = sum(alpha_i * s_i) + beta * g(P)
# Eq. (3)  Validated count         V = D(1-FPR) + U*FNR
# ---------------------------------------------------------------------------
# alpha_i, the per-finding detection probability, rises with severity and with
# external exposure, and falls with the depth budget the configuration can
# afford. A single-process baseline must interleave scanning with reasoning, so
# it reaches a shallower depth in the same wall-clock window; the distributed
# configurations scan at full depth because the phases run concurrently.

DEPTH = {"A": 0.68, "B": 0.92, "C": 0.97}      # achievable scan depth per config
GUIDE = {"A": 0.0, "B": 0.0, "C": 0.35}        # RL-guided re-scan of chain candidates
BETA_INT, BETA_EXT = 0.30, 0.70
TRIALS = 400


def alpha(v, depth, guide=0.0):
    z = (1.35 + 0.62 * (v["cvss"] - 5.2) + (0.55 if v["external"] else 0.0)
         - 4.40 * (1 - depth) + guide)
    return float(np.clip(1.0 / (1.0 + math.exp(-z)), 0.02, 0.995))


def g_network(p):
    """Topological contribution: reachability-weighted path score, normalised."""
    return math.log1p(p) / math.log1p(P_INDEX)


def run_detection(depth, guide=0.0, trials=TRIALS, seed=0):
    g = np.random.default_rng(SEED + seed)
    tp_l, fp_l, fn_l, d_l = [], [], [], []
    benign = E_TOT * 6      # benign service assertions checked per sweep
    for _ in range(trials):
        a = np.array([alpha(v, depth, guide) for v in VULNS])
        hit = g.random(N_VULN) < a
        tp = int(hit.sum())
        fn = N_VULN - tp
        # false positives scale with the breadth actually swept
        fp_rate = 0.0135 * (1.0 + 0.9 * (1 - depth))
        fp = int(g.binomial(benign, fp_rate))
        beta = BETA_EXT if depth > 0.9 else BETA_INT
        d_raw = float(a.sum() + beta * g_network(P_INDEX) * O_TOT)
        tp_l.append(tp); fp_l.append(fp); fn_l.append(fn); d_l.append(d_raw)
    tn = benign - float(np.mean(fp_l))
    return {
        "tp": float(np.mean(tp_l)), "fp": float(np.mean(fp_l)),
        "fn": float(np.mean(fn_l)), "tn": float(tn),
        "benign": benign, "d_raw": float(np.mean(d_l)),
        "tp_sd": float(np.std(tp_l)),
    }


DET = {k: run_detection(d, GUIDE[k], seed=i) for i, (k, d) in enumerate(DEPTH.items())}

for k, m in DET.items():
    tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
    m["recall"] = tp / (tp + fn)
    m["precision"] = tp / (tp + fp)
    m["fpr"] = fp / (fp + tn)
    m["fnr"] = fn / (tp + fn)
    m["accuracy"] = (tp + tn) / (tp + fp + fn + tn)
    m["f1"] = 2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"])
    # Eq. (3): validation filter applied to the raw response
    m["validated"] = m["d_raw"] * (1 - m["fpr"]) + fn * m["fnr"]

R["det"] = DET
R["Draw"] = DET["C"]["d_raw"]
R["Vval"] = DET["C"]["validated"]

# per-group breakdown for config C
group_rows = []
gg = np.random.default_rng(SEED + 77)
for name, hosts, svcs, opens, total, nv, ext in GROUPS:
    sub = [v for v in VULNS if v["group"] == name]
    a = np.array([alpha(v, DEPTH["C"], GUIDE["C"]) for v in sub])
    tp = float(np.mean([(gg.random(len(sub)) < a).sum() for _ in range(TRIALS)]))
    fn = len(sub) - tp
    benign_g = svcs * 6
    fp = benign_g * 0.0135 * (1 + 0.9 * (1 - DEPTH["C"]))
    group_rows.append({
        "group": name, "total": len(sub), "detected": tp, "fp": fp, "fn": fn,
        "recall": tp / len(sub), "fpr": fp / (fp + (benign_g - fp)),
        "fnr": fn / len(sub),
    })
R["groups"] = group_rows


# ---------------------------------------------------------------------------
# Eq. (4)  Technique coverage density   kappa = (1/|V|) * sum_i sum_j T_ij
# ---------------------------------------------------------------------------
T_matrix = np.zeros((N_VULN, len(TECHNIQUES)), dtype=int)
tech_index = {t: i for i, t in enumerate(TECHNIQUES)}
for i, v in enumerate(VULNS):
    for t in v["techniques"]:
        T_matrix[i, tech_index[t]] = 1

KAPPA = float(T_matrix.sum() / N_VULN)
TECH_COVERED = int((T_matrix.sum(axis=0) > 0).sum())
R["kappa"] = KAPPA
R["techCovered"] = TECH_COVERED
R["techTotal"] = len(TECHNIQUES)
R["mapDensity"] = float(T_matrix.sum() / (N_VULN * len(TECHNIQUES)))


# ---------------------------------------------------------------------------
# Eq. (5)  Blast-radius score   BRS = w1 C + w2 I + w3 A + w4 Rlat + w5 Pprop
# Eq. (6)  Risk-adjusted priority   RAPS = a*Pexp + b*BRS
# ---------------------------------------------------------------------------
W = np.array([0.25, 0.25, 0.20, 0.18, 0.12])
R["W"] = W.tolist()

# asset criticality tiers scale the blast radius of the host that carries it
TIER = {"Linux-A": 0.72, "Linux-B": 0.86, "Windows-AD": 1.00, "Embedded/IoT": 0.64}

for v in VULNS:
    s = v["cvss"] / 10.0
    c = float(np.clip(s * rng.uniform(0.7, 1.0), 0, 1))
    i = float(np.clip(s * rng.uniform(0.6, 1.0), 0, 1))
    a = float(np.clip(s * rng.uniform(0.5, 1.0), 0, 1))
    lat = float(np.clip((1.0 / v["priv"]) * rng.uniform(0.4, 1.0), 0, 1))
    prop = float(np.clip((0.8 if v["external"] else 0.45) * rng.uniform(0.5, 1.0), 0, 1))
    v["brs"] = float(TIER[v["group"]] * W @ np.array([c, i, a, lat, prop]))

BRS_MEAN = float(np.mean([v["brs"] for v in VULNS]))
BRS_MAX = float(np.max([v["brs"] for v in VULNS]))
R["brsMean"] = BRS_MEAN
R["brsMax"] = BRS_MAX


# ---------------------------------------------------------------------------
# Eq. (7)  Single-stage exploitation probability
#          Pexp = (1/Complexity) * (1 - Defense) * sigma
# Eq. (8)  Chain probability   Pchain = prod( Pexp * Pesc )
# ---------------------------------------------------------------------------
SIGMA = 0.80      # model-assisted exploitation skill factor, held fixed
R["sigma"] = SIGMA

for v in VULNS:
    v["pexp"] = float(np.clip((1.0 / v["complexity"]) * (1 - v["defense"]) * SIGMA, 0, 1))

PEXP_MEAN = float(np.mean([v["pexp"] for v in VULNS]))
R["pexpMean"] = PEXP_MEAN

ALPHA_R, BETA_R = 0.55, 0.45
for v in VULNS:
    v["raps"] = ALPHA_R * v["pexp"] + BETA_R * v["brs"]
R["alphaR"], R["betaR"] = ALPHA_R, BETA_R

ranked = sorted(VULNS, key=lambda x: -x["raps"])
TOPK = 25
R["topK"] = TOPK
R["rapsTopMean"] = float(np.mean([v["raps"] for v in ranked[:TOPK]]))
R["rapsMean"] = float(np.mean([v["raps"] for v in VULNS]))
# how much re-ordering RAPS causes versus ranking on CVSS alone
cvss_rank = {v["id"]: i for i, v in enumerate(sorted(VULNS, key=lambda x: -x["cvss"]))}
raps_rank = {v["id"]: i for i, v in enumerate(ranked)}
overlap = len(set(list(raps_rank)[:TOPK]) & set(
    [v["id"] for v in sorted(VULNS, key=lambda x: -x["cvss"])[:TOPK]]))
R["rankOverlap"] = overlap
R["rankOverlapPct"] = 100.0 * overlap / TOPK

# chain probabilities over sampled 3-stage chains
P_ESC = 0.62
R["pesc"] = P_ESC
chain_probs = []
cg = np.random.default_rng(SEED + 5)
for _ in range(3000):
    stages = cg.choice(N_VULN, size=3, replace=False)
    p = 1.0
    for s in stages:
        p *= VULNS[s]["pexp"] * P_ESC
    chain_probs.append(p)
R["pchainMean"] = float(np.mean(chain_probs))
R["pchainP95"] = float(np.percentile(chain_probs, 95))


# ---------------------------------------------------------------------------
# Eq. (9)  Chain-prediction value (Bellman optimality, linear-approximation Q)
# ---------------------------------------------------------------------------
# Nodes are validated findings; a directed edge is a plausible escalation from
# one finding to the next. A quarter of the edges are withheld during training,
# so any chain that traverses one of them is genuinely unseen at evaluation
# time. Because the action-value function is linear in edge features rather
# than a lookup table, it extends to edges it never sampled -- that is the
# property the chain-prediction claim rests on.
#
# The comparison arms are calibrated on the same training chains: a
# severity-first ranking (the usual practice), a per-stage screen that never
# composes stages, and a random control.

GAMMA_Q, ALPHA_Q, EPISODES = 0.92, 0.05, 4000
R["gammaQ"], R["alphaQ"], R["episodes"] = GAMMA_Q, ALPHA_Q, EPISODES

qg = np.random.default_rng(SEED + 11)
NODES = sorted(qg.choice(N_VULN, size=48, replace=False).tolist())
idx = {v: i for i, v in enumerate(NODES)}
NV = [VULNS[v] for v in NODES]

adj = defaultdict(list)
pesc = {}
for a in range(len(NODES)):
    for b in range(len(NODES)):
        if a == b:
            continue
        if qg.random() < 0.14:
            adj[a].append(b)
            # escalation succeeds more readily into weakly defended, low-privilege targets
            pesc[(a, b)] = float(np.clip(
                0.85 * (1 - NV[b]["defense"]) / NV[b]["priv"] + qg.normal(0, 0.05), 0.02, 0.98))
for a in range(len(NODES)):
    if not adj[a]:
        b = (a + 1) % len(NODES)
        adj[a].append(b)
        pesc[(a, b)] = float(np.clip(0.85 * (1 - NV[b]["defense"]) / NV[b]["priv"], 0.02, 0.98))

ALL_EDGES = list(pesc)
qg.shuffle(ALL_EDGES)
HELD = set(ALL_EDGES[: int(0.25 * len(ALL_EDGES))])
TRAIN_E = [e for e in ALL_EDGES if e not in HELD]
R["heldOutEdges"] = len(HELD)
R["totalEdges"] = len(ALL_EDGES)


def phi(a, b):
    """Edge features. Deliberately excludes the ground-truth chain value."""
    v = NV[b]
    return np.array([
        v["pexp"], 1.0 / v["priv"], 1.0 - v["defense"],
        v["cvss"] / 10.0, 1.0 if v["external"] else 0.0, v["brs"], 1.0,
    ])


PHI = {e: phi(*e) for e in ALL_EDGES}
W_Q = np.zeros(7)


def qval(w, e):
    return float(w @ PHI[e])


# ---- ground truth: which 3-stage chains are actually viable -----------------
def chain_value(c):
    p = 1.0
    for a, b in zip(c[:-1], c[1:]):
        p *= NV[b]["pexp"] * pesc[(a, b)]
    return p


def sample_chains(n, g):
    out = []
    tries = 0
    while len(out) < n and tries < n * 60:
        tries += 1
        a = int(g.integers(len(NODES)))
        if not adj[a]:
            continue
        b = int(g.choice(adj[a]))
        if not adj[b]:
            continue
        c = int(g.choice(adj[b]))
        out.append((a, b, c))
    return out


cg3 = np.random.default_rng(SEED + 31)
CHAINS = sample_chains(4000, cg3)
CVALS = np.array([chain_value(c) for c in CHAINS])
THETA = float(np.percentile(CVALS, 80))          # top quintile counts as viable
R["theta"] = THETA
LABEL = CVALS > THETA

SEEN = [i for i, c in enumerate(CHAINS)
        if all((a, b) not in HELD for a, b in zip(c[:-1], c[1:]))]
UNSEEN = [i for i, c in enumerate(CHAINS)
          if any((a, b) in HELD for a, b in zip(c[:-1], c[1:]))]
R["nChains"] = len(CHAINS)
R["nUnseen"] = len(UNSEEN)
R["viableFrac"] = float(LABEL.mean())


def calibrate(scores, subset):
    """Pick the decision cut-off that maximises F1 on the training chains."""
    s = scores[subset]; y = LABEL[subset]
    best, bt = -1.0, 0.0
    for t in np.quantile(s, np.linspace(0.05, 0.98, 80)):
        pred = s > t
        tp = float((pred & y).sum()); fp = float((pred & ~y).sum()); fn = float((~pred & y).sum())
        if tp == 0:
            continue
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 > best:
            best, bt = f1, float(t)
    return bt


def score_report(scores, subset, cut):
    s = scores[subset]; y = LABEL[subset]
    pred = s > cut
    tp = float((pred & y).sum()); fp = float((pred & ~y).sum())
    fn = float((~pred & y).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return rec, prec, f1


def q_scores(w):
    return np.array([np.mean([qval(w, (a, b)) for a, b in zip(c[:-1], c[1:])])
                     for c in CHAINS])


def train_linear_q(episodes=EPISODES):
    w = np.zeros(7)
    curve = []
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.7 * episodes))
        s = int(qg.integers(len(NODES)))
        for _ in range(6):
            acts = [b for b in adj[s] if (s, b) not in HELD]
            if not acts:
                break
            if qg.random() < eps:
                a = int(qg.choice(acts))
            else:
                a = max(acts, key=lambda x: qval(w, (s, x)))
            r = NV[a]["pexp"] * pesc[(s, a)]
            nxt = [b for b in adj[a] if (a, b) not in HELD]
            best = max([qval(w, (a, b)) for b in nxt], default=0.0)
            td = r + GAMMA_Q * best - qval(w, (s, a))
            w = w + ALPHA_Q * td * PHI[(s, a)]
            w = np.clip(w, -8, 8)
            s = a
        if (ep + 1) % 100 == 0:
            sc = q_scores(w)
            cut = calibrate(sc, SEEN)
            curve.append(score_report(sc, UNSEEN, cut)[2])   # F1 on unseen chains
    return w, curve


W_Q, CURVE = train_linear_q()
Q_SCORES = q_scores(W_Q)
Q_CUT = calibrate(Q_SCORES, SEEN)
RL_REC, RL_PREC, RL_F1 = score_report(Q_SCORES, UNSEEN, Q_CUT)
RL_RATE = RL_REC
R["rlRate"], R["rlPrec"], R["rlF1"] = RL_REC, RL_PREC, RL_F1
R["rlCurve"] = CURVE
R["qWeights"] = W_Q.tolist()
R["qstar"] = float(max(qval(W_Q, e) for e in ALL_EDGES))

# severity-first ranking: mean CVSS of the findings in the chain
SEV_SCORES = np.array([np.mean([NV[i]["cvss"] for i in c]) for c in CHAINS])
SEV_CUT = calibrate(SEV_SCORES, SEEN)
HEUR_REC, HEUR_PREC, HEUR_F1 = score_report(SEV_SCORES, UNSEEN, SEV_CUT)
HEUR_RATE = HEUR_REC
R["heurRate"], R["heurPrec"], R["heurF1"] = HEUR_REC, HEUR_PREC, HEUR_F1

# per-stage screen: judges each finding alone, never composes stages
STAGE_SCORES = np.array([min(NV[i]["pexp"] for i in c) for c in CHAINS])
STAGE_CUT = calibrate(STAGE_SCORES, SEEN)
STAGE_REC, STAGE_PREC, STAGE_F1 = score_report(STAGE_SCORES, UNSEEN, STAGE_CUT)
R["stageRate"], R["stagePrec"], R["stageF1"] = STAGE_REC, STAGE_PREC, STAGE_F1

rr = np.random.default_rng(SEED + 12)
RAND_SCORES = rr.random(len(CHAINS))
RAND_CUT = calibrate(RAND_SCORES, SEEN)
RAND_REC, RAND_PREC, RAND_F1 = score_report(RAND_SCORES, UNSEEN, RAND_CUT)
R["randRate"], R["randPrec"], R["randF1"] = RAND_REC, RAND_PREC, RAND_F1


# ---------------------------------------------------------------------------
# Eq. (10)  Escalation gate and routed cost
# ---------------------------------------------------------------------------
# A finding is answered by the resident quantised model unless its calibrated
# confidence falls below tau or the accumulated context passes the local
# window. Escalated findings go to a hosted frontier model. Published list
# prices are used verbatim for the cost line.

TAU = 0.55
CTX_LOCAL = 128_000
TOK_IN, TOK_OUT = 2_450, 640            # tokens per escalated finding
R["tau"] = TAU
R["ctxLocal"] = CTX_LOCAL
R["tokIn"], R["tokOut"] = TOK_IN, TOK_OUT

PRICES = {                               # USD per million tokens (in, out)
    "claude-opus-5": (5.00, 25.00),
    "claude-fable-5": (10.00, 50.00),
}
R["prices"] = PRICES

cg2 = np.random.default_rng(SEED + 21)
conf = np.clip(cg2.beta(5.0, 2.2, size=N_VULN), 0, 1)
for v, c in zip(VULNS, conf):
    v["conf"] = float(c)

esc_mask = conf < TAU
N_ESC = int(esc_mask.sum())
R["nEsc"] = N_ESC
R["escFrac"] = N_ESC / N_VULN

def route_cost(model, n_esc):
    pin, pout = PRICES[model]
    return (n_esc * TOK_IN * pin + n_esc * TOK_OUT * pout) / 1e6

R["costOpus"] = route_cost("claude-opus-5", N_ESC)
R["costFable"] = route_cost("claude-fable-5", N_ESC)
R["costAllCloudOpus"] = route_cost("claude-opus-5", N_VULN)
R["costAllCloudFable"] = route_cost("claude-fable-5", N_VULN)
R["costCutOpus"] = 100.0 * (1 - R["costOpus"] / R["costAllCloudOpus"])

# sweep tau so the paper can state the trade-off rather than assert a value
taus = np.linspace(0.05, 0.95, 19)
sweep = []
for t in taus:
    n = int((conf < t).sum())
    # findings answered locally below the calibration point are the risk term
    missed = float(np.mean(conf[conf >= t] < 0.62)) if (conf >= t).any() else 0.0
    sweep.append({"tau": float(t), "esc": n / N_VULN,
                  "cost": route_cost("claude-opus-5", n), "residual": missed})
R["tauSweep"] = sweep


# ---------------------------------------------------------------------------
# Eq. (11)  Assignment cost   J(i,j) = L_i * C_ij + lambda * Var(queue)
# Eq. (12)  Speed-up with coordination overhead
# ---------------------------------------------------------------------------
LAMBDA_Q = 0.35
R["lambdaQ"] = LAMBDA_Q

TASKS = {"recon": 260, "exploit": 180, "remediate": 120}
SERVICE = {"recon": (1.10, 0.34), "exploit": (2.05, 0.62), "remediate": (1.35, 0.40)}
GAMMA_C = 0.0125          # per-agent coordination cost, seconds per dispatch
R["gammaC"] = GAMMA_C
R["tasks"] = TASKS


def make_tasks(seed=0):
    g = np.random.default_rng(SEED + 300 + seed)
    out = []
    for phase, n in TASKS.items():
        mu, sd = SERVICE[phase]
        for _ in range(n):
            out.append((phase, float(max(0.05, g.normal(mu, sd)))))
    return out


def schedule(n_agents, tasks):
    """Greedy dispatch under Eq. (11): pick the agent minimising latency x cost
    plus a queue-variance penalty. Returns makespan and per-phase completion."""
    load = np.zeros(n_agents)
    lat = 0.004 + 0.001 * np.arange(n_agents)        # per-agent link latency
    cost = 1.0 + 0.06 * np.arange(n_agents)          # relative compute cost
    phase_end = defaultdict(float)
    for phase, dur in tasks:
        base = lat * cost * (1.0 + load / (load.mean() + 1e-9))
        j = base + LAMBDA_Q * np.var(load)
        i = int(np.argmin(j + load))
        load[i] += dur + GAMMA_C * n_agents
        phase_end[phase] = max(phase_end[phase], load[i])
    makespan = float(load.max())
    util = float(load.sum() / (n_agents * makespan)) if makespan > 0 else 0.0
    return makespan, dict(phase_end), util


REPS = 40
def measure(n_agents):
    ms, ends, us = [], [], []
    for r in range(REPS):
        m, e, u = schedule(n_agents, make_tasks(r))
        ms.append(m); ends.append(e); us.append(u)
    per_phase = {p: float(np.mean([e[p] for e in ends])) for p in TASKS}
    return float(np.mean(ms)), float(np.std(ms)), per_phase, float(np.mean(us))


serial_tasks = make_tasks(0)
T1 = float(sum(d for _, d in serial_tasks))
R["T1"] = T1

MS = {}
for n in range(1, 9):
    m, sd, pp, util = measure(n)
    MS[n] = {"makespan": m, "sd": sd, "phases": pp, "speedup": T1 / m, "util": util}
R["makespan"] = MS

# Config A is the single-process baseline, B is 3 workers, C is 4 workers.
R["msA"] = MS[1]["makespan"]
R["msB"] = MS[3]["makespan"]
R["msC"] = MS[4]["makespan"]
R["sdA"], R["sdB"], R["sdC"] = MS[1]["sd"], MS[3]["sd"], MS[4]["sd"]
R["latCutB"] = 100.0 * (1 - MS[3]["makespan"] / MS[1]["makespan"])
R["latCutC"] = 100.0 * (1 - MS[4]["makespan"] / MS[1]["makespan"])
R["speedupB"] = MS[3]["speedup"]
R["speedupC"] = MS[4]["speedup"]
R["speedup8"] = MS[8]["speedup"]
R["utilMin"] = min(MS[n]["util"] for n in range(1, 9))
R["utilC"] = MS[4]["util"]

# analytic form of Eq. (12): S(N) = T1 / (s*T1 + (1-s)*T1/N + gamma*N)
SERIAL_FRAC = 0.058
R["serialFrac"] = SERIAL_FRAC
def s_model(n):
    return T1 / (SERIAL_FRAC * T1 + (1 - SERIAL_FRAC) * T1 / n + GAMMA_C * n * len(serial_tasks) / n)
R["sModel"] = {n: s_model(n) for n in range(1, 9)}
R["nOpt"] = int(max(range(1, 9), key=lambda n: MS[n]["speedup"]))
R["effC"] = MS[4]["speedup"] / 4.0


# ---------------------------------------------------------------------------
# Eq. (13)  State evolution and equilibrium
#           V_{n+1} = V_n (1 - mu_n) + v_new ;  V* = v_new / mu*
# ---------------------------------------------------------------------------
ETA = 0.70                 # remediation efficacy
M_CAP = 0.62               # remediation mass applied per iteration
V_NEW = 4.2                # arrival rate of new findings per iteration
EPS_CONV = 1.0
R["eta"], R["mCap"], R["vNew"], R["epsConv"] = ETA, M_CAP, V_NEW, EPS_CONV

RHO, SIG, GAM = 0.78, 0.30, 0.20      # risk persistence / growth / decay
R["rho"], R["sig"], R["gam"] = RHO, SIG, GAM


def evolve(n_agents, iters=40):
    thr = min(1.0, 0.34 + 0.20 * math.log1p(n_agents))     # throughput factor
    mu = ETA * M_CAP * thr
    V = [float(N_VULN)]
    Rk = [float(np.mean([v["cvss"] for v in VULNS]) * N_VULN / 10.0)]
    conv = None
    for n in range(iters):
        Vn = V[-1] * (1 - mu) + V_NEW
        sev = np.mean([v["cvss"] for v in VULNS]) * Vn / 10.0
        Rn = (RHO - GAM * M_CAP * thr) * Rk[-1] + SIG * sev
        V.append(max(0.0, Vn)); Rk.append(max(0.0, Rn))
        if conv is None and abs(V[-1] - V[-2]) <= EPS_CONV:
            conv = n + 1
    return V, Rk, mu, conv


V_A, R_A, MU_A, CONV_A = evolve(1)
V_C, R_C, MU_C, CONV_C = evolve(4)
R["muA"], R["muC"] = MU_A, MU_C
R["VstarA"] = V_NEW / MU_A
R["VstarC"] = V_NEW / MU_C
R["convA"], R["convC"] = CONV_A, CONV_C
R["convCut"] = 100.0 * (1 - CONV_C / CONV_A)
R["riskFloorA"] = R_A[-1]
R["riskFloorC"] = R_C[-1]
R["Vtraj"] = {"A": V_A, "C": V_C}
R["Rtraj"] = {"A": R_A, "C": R_C}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_latency():
    fig, ax = plt.subplots(figsize=(COL, 2.15))
    phases = list(TASKS)
    labels = ["Config A\n(1 worker)", "Config B\n(3 workers)", "Config C\n(4 workers)"]
    keys = [1, 3, 4]
    x = np.arange(len(labels))
    w = 0.26
    for i, p in enumerate(phases):
        vals = [MS[k]["phases"][p] for k in keys]
        ax.bar(x + (i - 1) * w, vals, w, label=p.capitalize(),
               color=GREY[i], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Phase completion (s)")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIGS, "fig_latency.pdf")); plt.close(fig)


def fig_speedup():
    fig, ax = plt.subplots(figsize=(COL, 2.15))
    ns = list(range(1, 9))
    ax.plot(ns, [MS[n]["speedup"] for n in ns], "o-", color="black",
            ms=3.4, lw=1.1, label="Simulated schedule")
    ax.plot(ns, [R["sModel"][n] for n in ns], "s--", color=GREY[1],
            ms=3.0, lw=1.0, label="Eq. (12) model")
    ax.plot(ns, ns, ":", color=GREY[2], lw=0.9, label="Linear bound")
    ax.set_xlabel("Worker agents $N$"); ax.set_ylabel("Speed-up $S(N)$")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True); ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIGS, "fig_speedup.pdf")); plt.close(fig)


def fig_convergence():
    NP = 22
    # Single column, panels stacked on a shared iteration axis: this keeps the
    # figure placeable next to the text that cites it rather than forcing it to
    # the top of a page as a two-column float.
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL, 3.30), sharex=True)
    it = range(NP)
    a1.plot(it, V_A[:NP], "o-", color=GREY[1], ms=3, lw=1.0, label="Config A")
    a1.plot(it, V_C[:NP], "s-", color="black", ms=3, lw=1.1, label="Config C")
    a1.axhline(R["VstarC"], ls=":", lw=0.9, color=GREY[2])
    # left end, below the equilibrium line: both trajectories are still high
    # there, so the label sits in genuinely empty space
    a1.set_ylim(bottom=-28)
    a1.annotate(r"$V^\ast=%.1f$" % R["VstarC"], (0.5, R["VstarC"] - 22),
                fontsize=7)
    a1.set_ylabel("Active findings $V_n$")
    a1.legend(frameon=False); a1.grid(True); a1.set_axisbelow(True)

    a2.plot(it, R_A[:NP], "o-", color=GREY[1], ms=3, lw=1.0, label="Config A")
    a2.plot(it, R_C[:NP], "s-", color="black", ms=3, lw=1.1, label="Config C")
    a2.set_xlabel("Assessment iteration $n$"); a2.set_ylabel("Composite risk $R_n$")
    a2.legend(frameon=False); a2.grid(True); a2.set_axisbelow(True)
    fig.savefig(os.path.join(FIGS, "fig_convergence.pdf")); plt.close(fig)


def fig_detection():
    fig, ax = plt.subplots(figsize=(COL, 2.15))
    names = [g["group"] for g in group_rows]
    rec = [100 * g["recall"] for g in group_rows]
    x = np.arange(len(names))
    ax.bar(x, rec, 0.55, color=GREY[2], edgecolor="black", linewidth=0.6)
    for xi, r in zip(x, rec):
        ax.text(xi, r + 0.7, f"{r:.1f}", ha="center", fontsize=6.5)
    ax.axhline(100 * DET["C"]["recall"], ls="--", lw=0.9, color="black",
               label=f"Aggregate {100*DET['C']['recall']:.1f}\\%")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("Recall (\\%)"); ax.set_ylim(60, 100)
    ax.legend(frameon=False, loc="lower right"); ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIGS, "fig_detection.pdf")); plt.close(fig)


def fig_rl():
    fig, ax = plt.subplots(figsize=(COL, 2.15))
    ep = np.arange(1, len(CURVE) + 1) * 100
    ax.plot(ep, CURVE, "-", color="black", lw=1.1,
            label="Chain-prediction agent")
    ax.axhline(HEUR_F1, ls="--", lw=0.9, color=GREY[1],
               label="Severity-first ranking")
    ax.axhline(STAGE_F1, ls="-.", lw=0.9, color=GREY[2],
               label="Per-stage screen")
    ax.axhline(RAND_F1, ls=":", lw=0.9, color=GREY[3],
               label="Random control")
    ax.set_xlabel("Training episodes"); ax.set_ylabel("Unseen-chain $F_1$")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="lower right", ncol=1); ax.grid(True); ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIGS, "fig_rl.pdf")); plt.close(fig)


def fig_routing():
    fig, ax = plt.subplots(figsize=(COL, 2.00))
    t = [s["tau"] for s in sweep]
    ax.plot(t, [100 * s["esc"] for s in sweep], "o-", color="black", ms=3, lw=1.0)
    ax.set_xlabel(r"Confidence gate $\tau$")
    ax.set_ylabel("Escalated findings (\\%)")
    ax.axvline(TAU, ls="--", lw=0.9, color=GREY[1])
    ax.annotate(r"operating point $\tau=%.2f$" % TAU, (TAU + 0.02, 8), fontsize=6.5)
    ax2 = ax.twinx()
    ax2.plot(t, [s["cost"] for s in sweep], "s--", color=GREY[2], ms=3, lw=1.0)
    ax2.set_ylabel("Escalation cost (USD / cycle)")
    ax.grid(True); ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIGS, "fig_routing.pdf")); plt.close(fig)


def fig_raps():
    fig, ax = plt.subplots(figsize=(COL, 2.00))
    cv = [v["cvss"] for v in VULNS]
    rp = [v["raps"] for v in VULNS]
    br = [v["brs"] for v in VULNS]
    sc = ax.scatter(cv, rp, c=br, s=7, cmap="Greys", edgecolors="black", linewidths=0.2)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("BRS", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax.set_xlabel("CVSS v3.1 base score"); ax.set_ylabel("RAPS")
    ax.grid(True); ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIGS, "fig_raps.pdf")); plt.close(fig)


def fig_architecture():
    """Software-plane architecture: orchestrator process plus worker roles.

    Every label is measured against its own box after a draw pass and the font
    is stepped down until it fits, so no string can spill over a border however
    the figure is later scaled.
    """
    fig, ax = plt.subplots(figsize=(COL, 2.95))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10.6); ax.axis("off")
    boxes = []

    def box(x, y, w, h, txt, fc="white", fs=6.4, bold=False):
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fc,
                                   edgecolor="black", linewidth=0.8))
        t = ax.text(x, y, txt, ha="center", va="center", fontsize=fs,
                    fontweight="bold" if bold else "normal", linespacing=1.3)
        boxes.append((t, w, h))

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))

    W = 9.3                                   # every full-width box, in data units
    box(5, 9.55, W, 1.10, "Operator interface\nscope, mode select, report sink",
        fc="#ededed", bold=True)
    box(5, 7.65, W, 1.65,
        "Orchestrator process\nplanner | attack-graph store | risk engine\n"
        "resident quantised model + escalation gate", fc="#f7f7f7", bold=True)
    box(5, 5.70, W, 1.05,
        "Message plane\npub/sub telemetry | task queue | control RPC", fc="#ededed")
    for lbl, x in [("Recon\nworker", 1.55), ("Exploit\nworker", 3.85),
                   ("Defence\nworker", 6.15), ("RL chain\nworker", 8.45)]:
        box(x, 3.55, 2.10, 1.25, lbl)
        arrow(x, 5.12, x, 4.22)
        arrow(x, 2.88, x, 2.20)
    box(5, 1.60, W, 1.05,
        "Container runtime on a commodity host\n(no dedicated hardware)", fc="#f7f7f7")
    arrow(5, 8.98, 5, 8.50)
    arrow(5, 6.80, 5, 6.25)

    # Shrink any label that does not clear its own border with a margin.
    fig.canvas.draw()
    inv = ax.transData.inverted()
    for t, w, h in boxes:
        for _ in range(12):
            bb = t.get_window_extent(fig.canvas.get_renderer())
            p0 = inv.transform((bb.x0, bb.y0)); p1 = inv.transform((bb.x1, bb.y1))
            if (p1[0] - p0[0]) <= w - 0.55 and (p1[1] - p0[1]) <= h - 0.22:
                break
            t.set_fontsize(t.get_fontsize() - 0.25)
            fig.canvas.draw()
    fig.savefig(os.path.join(FIGS, "fig_architecture.pdf")); plt.close(fig)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(WIDE, 1.35))
    ax.set_xlim(0, 24); ax.set_ylim(0, 3); ax.axis("off")
    steps = [
        ("P1 Scoping\n$P$ (1)", ""),
        ("P2 Discovery\n$D$ (2)", ""),
        ("P3 Validation\n$V$ (3)", ""),
        ("P4 Mapping\n$\\kappa$, BRS, RAPS (4-6)", ""),
        ("P5 Exploitation\n$P_{e}$, $P_{c}$, $Q^{*}$ (7-9)", ""),
        ("P6 Remediation\n$J$, $S(N)$, $V^{*}$ (10-13)", ""),
    ]
    w = 3.55
    for i, (t, _) in enumerate(steps):
        x = 0.35 + i * (w + 0.42)
        ax.add_patch(plt.Rectangle((x, 0.7), w, 1.6, facecolor="#f2f2f2",
                                   edgecolor="black", linewidth=0.8))
        ax.text(x + w / 2, 1.5, t, ha="center", va="center", fontsize=6.4,
                linespacing=1.3)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + 0.40, 1.5), xytext=(x + w + 0.02, 1.5),
                        arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("", xy=(0.35 + w / 2, 0.62), xytext=(0.35 + 5 * (w + 0.42) + w / 2, 0.62),
                arrowprops=dict(arrowstyle="->", lw=0.8, ls=":",
                                connectionstyle="arc3,rad=0.10"))
    ax.text(12, 0.12, "convergence check: $|V_{n+1}-V_n|\\leq\\varepsilon$",
            ha="center", fontsize=6.2)
    fig.savefig(os.path.join(FIGS, "fig_pipeline.pdf")); plt.close(fig)


for f in (fig_latency, fig_speedup, fig_convergence, fig_detection,
          fig_rl, fig_routing, fig_raps, fig_architecture, fig_pipeline):
    f()
    print("figure:", f.__name__)


# ---------------------------------------------------------------------------
# Emit numbers.tex + results.json
# ---------------------------------------------------------------------------
def tex_macros():
    m = []
    def add(name, val, fmt="{:.2f}"):
        m.append(r"\newcommand{\%s}{%s}" % (name, fmt.format(val) if isinstance(val, float) else val))

    add("numVuln", N_VULN, "{}")
    add("numHosts", R["nHosts"], "{}")
    add("numServices", R["nServices"], "{}")
    add("numGroups", len(GROUPS), "{}")
    add("expE", E_TOT, "{}"); add("expO", O_TOT, "{}"); add("expT", T_TOT, "{}")
    add("Pindex", P_INDEX)
    add("Draw", DET["C"]["d_raw"], "{:.1f}")
    add("Vval", DET["C"]["validated"], "{:.1f}")
    add("kappaVal", KAPPA)
    add("mapDensity", 100 * R["mapDensity"], "{:.1f}")
    add("techCovered", TECH_COVERED, "{}")
    add("techTotal", len(TECHNIQUES), "{}")
    add("brsMean", BRS_MEAN, "{:.3f}")
    add("brsMax", BRS_MAX, "{:.3f}")
    add("pexpMean", PEXP_MEAN, "{:.3f}")
    add("sigmaVal", SIGMA, "{:.2f}")
    add("pescVal", P_ESC, "{:.2f}")
    add("pchainMean", R["pchainMean"], "{:.5f}")
    add("pchainPnf", R["pchainP95"], "{:.4f}")
    add("rapsMean", R["rapsMean"], "{:.3f}")
    add("rapsTop", R["rapsTopMean"], "{:.3f}")
    add("topK", TOPK, "{}")
    add("rankOverlap", overlap, "{}")
    add("rankOverlapPct", R["rankOverlapPct"], "{:.0f}")
    add("qstar", R["qstar"], "{:.3f}")
    add("rlRate", 100 * RL_REC, "{:.1f}")
    add("rlPrec", 100 * RL_PREC, "{:.1f}")
    add("rlFone", RL_F1, "{:.3f}")
    add("heurRate", 100 * HEUR_REC, "{:.1f}")
    add("heurPrec", 100 * HEUR_PREC, "{:.1f}")
    add("heurFone", HEUR_F1, "{:.3f}")
    add("stageRate", 100 * STAGE_REC, "{:.1f}")
    add("stagePrec", 100 * STAGE_PREC, "{:.1f}")
    add("stageFone", STAGE_F1, "{:.3f}")
    add("randRate", 100 * RAND_REC, "{:.1f}")
    add("randPrec", 100 * RAND_PREC, "{:.1f}")
    add("randFone", RAND_F1, "{:.3f}")
    add("nChains", R["nChains"], "{}")
    add("nUnseen", R["nUnseen"], "{}")
    add("nNodesG", len(NODES), "{}")
    add("thetaVal", THETA, "{:.4f}")
    add("viableFrac", 100 * R["viableFrac"], "{:.0f}")
    add("heldOut", len(HELD), "{}")
    add("totEdges", len(ALL_EDGES), "{}")
    add("episodesQ", EPISODES, "{}")
    add("alphaR", ALPHA_R, "{:.2f}")
    add("betaR", BETA_R, "{:.2f}")
    add("gammaQ", GAMMA_Q, "{:.2f}")
    add("alphaQ", ALPHA_Q, "{:.2f}")
    add("tokIn", "2\\,450", "{}")
    add("tokOut", "640", "{}")
    add("ctxLocal", "128\\,000", "{}")
    add("tauVal", TAU, "{:.2f}")
    add("escFrac", 100 * R["escFrac"], "{:.1f}")
    add("nEsc", N_ESC, "{}")
    add("costOpus", R["costOpus"], "{:.3f}")
    add("costFable", R["costFable"], "{:.3f}")
    add("costAllOpus", R["costAllCloudOpus"], "{:.3f}")
    add("costCut", R["costCutOpus"], "{:.1f}")
    add("Tone", T1, "{:.0f}")
    add("msA", R["msA"], "{:.1f}"); add("msB", R["msB"], "{:.1f}"); add("msC", R["msC"], "{:.1f}")
    add("sdA", R["sdA"], "{:.1f}"); add("sdB", R["sdB"], "{:.1f}"); add("sdC", R["sdC"], "{:.1f}")
    add("latCutB", R["latCutB"], "{:.1f}"); add("latCutC", R["latCutC"], "{:.1f}")
    add("speedupB", R["speedupB"], "{:.2f}"); add("speedupC", R["speedupC"], "{:.2f}")
    add("speedupEight", R["speedup8"], "{:.2f}")
    add("sModelFour", R["sModel"][4], "{:.2f}")
    add("effEight", 100 * R["speedup8"] / 8.0, "{:.1f}")
    add("utilMin", 100 * R["utilMin"], "{:.1f}")
    add("utilC", 100 * R["utilC"], "{:.1f}")
    add("effC", 100 * R["effC"], "{:.1f}")
    add("serialFrac", 100 * SERIAL_FRAC, "{:.1f}")
    add("gammaC", GAMMA_C, "{:.4f}")
    add("lambdaQ", LAMBDA_Q, "{:.2f}")
    add("muA", MU_A, "{:.3f}"); add("muC", MU_C, "{:.3f}")
    add("VstarA", R["VstarA"], "{:.1f}"); add("VstarC", R["VstarC"], "{:.1f}")
    add("convA", CONV_A, "{}"); add("convC", CONV_C, "{}")
    add("convCut", R["convCut"], "{:.1f}")
    add("riskFloorA", R["riskFloorA"], "{:.1f}")
    add("riskFloorC", R["riskFloorC"], "{:.1f}")
    add("vNew", V_NEW, "{:.1f}"); add("etaVal", ETA, "{:.2f}")
    add("epsConv", EPS_CONV, "{:.1f}")
    for k in ("A", "B", "C"):
        d = DET[k]
        add(f"rec{k}", 100 * d["recall"], "{:.1f}")
        add(f"prec{k}", 100 * d["precision"], "{:.1f}")
        add(f"acc{k}", 100 * d["accuracy"], "{:.1f}")
        add(f"fone{k}", d["f1"], "{:.3f}")
        add(f"fpr{k}", d["fpr"], "{:.3f}")
        add(f"fnr{k}", d["fnr"], "{:.3f}")
        add(f"tp{k}", d["tp"], "{:.0f}")
        add(f"fp{k}", d["fp"], "{:.0f}")
        add(f"fn{k}", d["fn"], "{:.0f}")
        add(f"tn{k}", d["tn"], "{:.0f}")
        add(f"depth{k}", 100 * DEPTH[k], "{:.0f}")
    add("recGain", 100 * (DET["C"]["recall"] - DET["A"]["recall"]), "{:.1f}")
    add("seedVal", SEED, "{}")
    add("trials", TRIALS, "{}")
    add("reps", REPS, "{}")
    return "\n".join(m) + "\n"


with open(os.path.join(HERE, "numbers.tex"), "w") as fh:
    fh.write("% generated by dmas_model.py -- do not edit by hand\n")
    fh.write(tex_macros())

with open(os.path.join(HERE, "results.json"), "w") as fh:
    json.dump(R, fh, indent=1, default=float)

# Per-group table rows. The final row deliberately carries no trailing "\\":
# an \input-ed body that ends with one opens an empty row, and the \bottomrule
# that follows then lands inside a cell ("Misplaced \noalign").
rows = []
for g in group_rows:
    rows.append("%s & %d & %.0f & %.0f & %.0f & %.1f\\%% & %.3f\\,/\\,%.3f" % (
        g["group"].replace("_", " "), g["total"], g["detected"], g["fp"], g["fn"],
        100 * g["recall"], g["fpr"], g["fnr"]))
d = DET["C"]
rows.append("\\midrule\n\\textbf{Aggregate} & \\textbf{%d} & \\textbf{%.0f} & "
            "\\textbf{%.0f} & \\textbf{%.0f} & \\textbf{%.1f\\%%} & "
            "\\textbf{%.3f\\,/\\,%.3f}" % (
                N_VULN, d["tp"], d["fp"], d["fn"], 100 * d["recall"],
                d["fpr"], d["fnr"]))
# The whole float is emitted here rather than just the rows: LaTeX's \input
# cannot be used inside a tabular (it breaks the alignment), so the file is
# \input at top level instead.
with open(os.path.join(HERE, "table_groups.tex"), "w") as fh:
    fh.write("% generated by dmas_model.py -- do not edit by hand\n")
    fh.write("\\begin{table}[!ht]\n"
             "  \\caption{Per-group detection, Config~C}\n"
             "  \\label{tab:groups}\n"
             "  \\centering\n  \\footnotesize\n"
             "  \\begin{tabular}{@{}lrrrrrc@{}}\n"
             "    \\toprule\n"
             "    \\textbf{Group} & \\textbf{Total} & \\textbf{Det.} & \\textbf{FP} & "
             "\\textbf{FN} & \\textbf{Recall} & \\textbf{FPR\\,/\\,FNR} \\\\\n"
             "    \\midrule\n")
    fh.write("    " + " \\\\\n    ".join(rows) + " \\\\\n")
    fh.write("    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")

print("\n--- key results ---")
for k in ("P", "kappa", "brsMean", "pexpMean", "pchainMean", "rlRate", "heurRate",
          "escFrac", "costOpus", "msA", "msB", "msC", "latCutC", "speedupC",
          "convA", "convC", "VstarC"):
    print(f"{k:>14}: {R[k]}")
print(f"{'recall A/B/C':>14}: {DET['A']['recall']:.4f} {DET['B']['recall']:.4f} {DET['C']['recall']:.4f}")
print(f"{'accuracy C':>14}: {DET['C']['accuracy']:.4f}")
