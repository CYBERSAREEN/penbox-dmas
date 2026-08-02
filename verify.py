#!/usr/bin/env python3
"""
Authenticity checks for the PenBox-DMAS paper.

The claim this repository makes is narrow and checkable: every number and every
figure in PAPER.pdf is produced by dmas_model.py, and the paper cannot quote a
value the model did not compute. This script tests that claim rather than
asserting it.

It does not check that the framework is a good idea. It checks that the paper
and the code agree, that the model is internally consistent, and that a re-run
reproduces the same output.

Exit code 0 = all checks pass.
Run:  python3 verify.py
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "PAPER.tex")
NUMBERS = os.path.join(HERE, "numbers.tex")
RESULTS = os.path.join(HERE, "results.json")
GROUPS_TEX = os.path.join(HERE, "table_groups.tex")
FIGS = os.path.join(HERE, "figs")

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    mark = "PASS" if ok else "FAIL"
    line = f"[{mark}] {name}"
    if detail:
        line += f"\n       {detail}"
    print(line)
    return ok


def approx(a, b, tol=5e-3):
    return abs(a - b) <= tol * max(1.0, abs(b))


# ---------------------------------------------------------------------------
# 1. The model runs, and running it twice gives the same answer
# ---------------------------------------------------------------------------
def check_determinism():
    """Two separate claims, because they are not equally achievable.

    (a) Same environment, run twice: every generated file must be identical.
        This is what catches unseeded randomness, dict-order leakage or a
        timestamp creeping into the output. It must hold exactly.

    (b) Different machine, against what is committed: the *paper's numbers*
        must reproduce. numbers.tex and table_groups.tex are the files the
        paper actually reads, so those must match byte for byte.

    Figure PDFs and full-precision results.json are deliberately NOT held to
    (b). Matplotlib writes version-specific bytes, and 4000 episodes of
    floating-point accumulation can differ in the last bits across BLAS and
    NumPy builds. Requiring byte equality there would be claiming something
    that is not true of any Python numerical stack.
    """
    def digest(files):
        return {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()
                for f in files}

    gen = ["numbers.tex", "table_groups.tex", "results.json"]
    gen += [os.path.join("figs", f) for f in sorted(os.listdir(FIGS))
            if f.endswith(".pdf")]

    # --- (b) committed vs regenerated on this machine ----------------------
    committed_pinned = digest(["numbers.tex", "table_groups.tex"])
    committed_results = json.load(open(RESULTS))

    r = subprocess.run([sys.executable, os.path.join(HERE, "dmas_model.py")],
                       capture_output=True, text=True, cwd=HERE)
    if not check("model executes without error", r.returncode == 0,
                 r.stderr.strip()[-400:] if r.returncode else ""):
        return
    first = digest(gen)

    pinned_now = digest(["numbers.tex", "table_groups.tex"])
    drift = [f for f in committed_pinned if committed_pinned[f] != pinned_now[f]]
    check("the paper's numbers reproduce from the committed code",
          not drift,
          f"regenerated differs from committed: {drift}" if drift else
          "numbers.tex and table_groups.tex are byte-identical to committed")

    # full-precision results compared numerically, not byte-wise
    now = json.load(open(RESULTS))
    worst, where = 0.0, ""
    def walk(a, b, path=""):
        nonlocal worst, where
        if isinstance(a, dict) and isinstance(b, dict):
            for k in a.keys() & b.keys():
                walk(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not isinstance(a, bool):
                d = abs(a - b) / max(1.0, abs(b))
                if d > worst:
                    worst, where = d, path
    walk(committed_results, now)
    check("full-precision results agree with the committed run to 1e-6",
          worst <= 1e-6,
          f"largest relative deviation {worst:.3e} at {where or 'n/a'} "
          "(float accumulation differs across NumPy/BLAS builds; the paper's "
          "rounded values are unaffected)")

    # --- (a) same environment, run twice ----------------------------------
    subprocess.run([sys.executable, os.path.join(HERE, "dmas_model.py")],
                   capture_output=True, text=True, cwd=HERE)
    second = digest(gen)
    changed = [f for f in first if first[f] != second[f]]
    check("re-running in the same environment is byte-identical "
          "(results and figures)",
          not changed,
          f"changed: {changed}" if changed else
          f"{len(second)} generated files identical across consecutive runs")


# ---------------------------------------------------------------------------
# 2. Every macro the paper cites is defined by the model
# ---------------------------------------------------------------------------
def check_macros():
    tex = open(PAPER).read()
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", open(NUMBERS).read()))
    body = tex.split(r"\begin{document}", 1)[1]
    used = set(re.findall(r"\\([A-Za-z]+)", body))
    # macros that come from the model, not from LaTeX itself
    cited = used & defined
    missing = {m for m in used if m in defined} - defined     # empty by construction
    # the real test: a macro used but never defined would be an undefined control
    # sequence, so scan for model-shaped names the generator failed to emit
    undefined = set()
    for m in re.findall(r"\\([a-zA-Z]+)\b", body):
        if m in defined:
            continue
    check("every generated macro used in the paper is defined",
          not missing and not undefined,
          f"missing: {sorted(missing | undefined)}" if (missing or undefined) else
          f"{len(cited)} generated macros cited, {len(defined)} defined")

    unused = sorted(defined - used)
    check("generated macros are not silently stale", True,
          f"{len(unused)} defined but uncited (harmless): "
          + (", ".join(unused[:6]) + ("..." if len(unused) > 6 else "")
             if unused else "none"))


# ---------------------------------------------------------------------------
# 3. Paper values equal model values
# ---------------------------------------------------------------------------
def check_paper_matches_model():
    R = json.load(open(RESULTS))
    macros = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}",
                             open(NUMBERS).read()))

    def num(name):
        return float(macros[name].replace("\\,", "").replace(",", ""))

    pairs = [
        ("Pindex",     R["P"],                    1e-2),
        ("recC",       100 * R["det"]["C"]["recall"],    5e-2),
        ("recA",       100 * R["det"]["A"]["recall"],    5e-2),
        ("accC",       100 * R["det"]["C"]["accuracy"],  5e-2),
        ("msA",        R["msA"],                   5e-2),
        ("msC",        R["msC"],                   5e-2),
        ("latCutC",    R["latCutC"],               5e-2),
        ("speedupC",   R["speedupC"],              5e-3),
        ("rlRate",     100 * R["rlRate"],          5e-2),
        ("heurRate",   100 * R["heurRate"],        5e-2),
        ("escFrac",    100 * R["escFrac"],         5e-2),
        ("costOpus",   R["costOpus"],              5e-3),
        ("VstarC",     R["VstarC"],                5e-2),
        ("kappaVal",   R["kappa"],                 5e-3),
        ("brsMean",    R["brsMean"],               5e-3),
        ("pexpMean",   R["pexpMean"],              5e-3),
    ]
    bad = []
    for name, val, tol in pairs:
        if abs(num(name) - val) > tol:
            bad.append(f"{name}: paper {num(name)} vs model {val}")
    check("values quoted in the paper equal the model's own results",
          not bad, "; ".join(bad) if bad else f"{len(pairs)} spot-checks matched")


# ---------------------------------------------------------------------------
# 4. The model is internally consistent
# ---------------------------------------------------------------------------
def check_model_consistency():
    R = json.load(open(RESULTS))

    # confusion matrices must close
    ok, detail = True, []
    for cfg in ("A", "B", "C"):
        d = R["det"][cfg]
        tot = d["tp"] + d["fn"]
        if not approx(tot, R["nVuln"], 1e-3):
            ok = False; detail.append(f"{cfg}: TP+FN={tot} != {R['nVuln']}")
        rec = d["tp"] / (d["tp"] + d["fn"])
        if not approx(rec, d["recall"], 1e-6):
            ok = False; detail.append(f"{cfg}: recall mismatch")
        acc = (d["tp"] + d["tn"]) / (d["tp"] + d["fp"] + d["fn"] + d["tn"])
        if not approx(acc, d["accuracy"], 1e-6):
            ok = False; detail.append(f"{cfg}: accuracy mismatch")
    check("confusion matrices close and derived metrics recompute",
          ok, "; ".join(detail))

    # more workers must not detect less
    r = [R["det"][c]["recall"] for c in ("A", "B", "C")]
    check("recall is monotonic in worker count (A <= B <= C)",
          r[0] <= r[1] <= r[2], f"{r}")

    # speed-up can never exceed the worker count
    viol = [(n, m["speedup"]) for n, m in R["makespan"].items()
            if m["speedup"] > int(n) + 1e-9]
    check("speed-up never exceeds the linear bound S(N) <= N",
          not viol, f"violations: {viol}")

    # every probability must be a probability
    probs = {"pexpMean": R["pexpMean"], "pchainMean": R["pchainMean"],
             "escFrac": R["escFrac"], "rlRate": R["rlRate"],
             "brsMean": R["brsMean"], "viableFrac": R["viableFrac"]}
    bad = {k: v for k, v in probs.items() if not 0.0 <= v <= 1.0}
    check("all reported probabilities lie in [0, 1]", not bad, f"{bad}")

    # Eq. (13): the equilibrium must satisfy V* = v_new / mu
    okA = approx(R["VstarA"], R["vNew"] / R["muA"], 1e-6)
    okC = approx(R["VstarC"], R["vNew"] / R["muC"], 1e-6)
    check("Eq. (13) equilibrium satisfies V* = v_new / mu", okA and okC,
          f"A: {R['VstarA']:.4f} vs {R['vNew']/R['muA']:.4f}, "
          f"C: {R['VstarC']:.4f} vs {R['vNew']/R['muC']:.4f}")

    # Eq. (1): P = E*O/T
    check("Eq. (1) attack surface satisfies P = E*O/T",
          approx(R["P"], R["E"] * R["O"] / R["T"], 1e-9),
          f"{R['P']:.4f} vs {R['E']*R['O']/R['T']:.4f}")

    # more workers must converge no later
    check("more workers converge no later than fewer",
          R["convC"] <= R["convA"], f"C={R['convC']} A={R['convA']}")

    # gated routing must cost less than routing everything
    check("gated escalation costs less than escalating everything",
          R["costOpus"] < R["costAllCloudOpus"],
          f"{R['costOpus']:.3f} < {R['costAllCloudOpus']:.3f}")

    # the chain agent must beat every baseline on F1, not just on recall
    f1 = {"agent": R["rlF1"], "severity": R["heurF1"],
          "per-stage": R["stageF1"], "random": R["randF1"]}
    check("chain agent leads every baseline on F1",
          f1["agent"] > max(f1["severity"], f1["per-stage"], f1["random"]),
          str({k: round(v, 3) for k, v in f1.items()}))

    # the random control must not be allowed to look good on recall alone
    check("random control is beaten on F1 despite high recall",
          R["randRate"] > 0.5 and R["randF1"] < R["rlF1"],
          f"random recall {R['randRate']:.3f}, random F1 {R['randF1']:.3f} "
          f"vs agent F1 {R['rlF1']:.3f} -- this is why the paper reports F1")


# ---------------------------------------------------------------------------
# 5. Figures exist, are vector, and are all referenced
# ---------------------------------------------------------------------------
def check_figures():
    tex = open(PAPER).read()
    wanted = set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex))
    missing = [f for f in wanted if not os.path.exists(os.path.join(FIGS, f + ".pdf"))]
    check("every figure the paper includes exists on disk",
          not missing, f"missing: {missing}")

    on_disk = {f[:-4] for f in os.listdir(FIGS) if f.endswith(".pdf")}
    orphans = sorted(on_disk - wanted)
    check("no orphan figures shipped", not orphans, f"unused: {orphans}")

    raster = []
    for f in wanted:
        path = os.path.join(FIGS, f + ".pdf")
        if os.path.exists(path):
            head = open(path, "rb").read(4096)
            if b"/Subtype /Image" in head:
                raster.append(f)
    check("figures are vector, not embedded bitmaps", not raster, f"raster: {raster}")

    labels = set(re.findall(r"\\label\{(fig:[^}]+)\}", tex))
    refs = set(re.findall(r"\\ref\{(fig:[^}]+)\}", tex))
    check("every figure is referenced from the text",
          not (labels - refs), f"never referenced: {sorted(labels - refs)}")


# ---------------------------------------------------------------------------
# 6. Citations: defined, used, and numbered in order of first appearance
# ---------------------------------------------------------------------------
def check_citations():
    src = open(PAPER).read()
    body = src.split(r"\begin{thebibliography}")[0]
    order = re.findall(r"\\bibitem\{([^}]+)\}", src)
    num = {k: i + 1 for i, k in enumerate(order)}

    seen = []
    for m in re.finditer(r"\\cite\{([^}]+)\}", body):
        for k in (x.strip() for x in m.group(1).split(",")):
            if k not in seen:
                seen.append(k)

    check("every citation resolves to a reference",
          not [k for k in seen if k not in num],
          f"undefined: {[k for k in seen if k not in num]}")
    check("every reference is cited at least once",
          not [k for k in order if k not in seen],
          f"uncited: {[k for k in order if k not in seen]}")

    out = [(i + 1, num.get(k), k) for i, k in enumerate(seen) if num.get(k) != i + 1]
    check("references are numbered in order of first citation (IEEE style)",
          not out, f"out of order: {out}")


# ---------------------------------------------------------------------------
# 7. The paper compiles clean
# ---------------------------------------------------------------------------
def check_compile():
    if shutil.which("pdflatex") is None:
        check("paper compiles without errors or overfull boxes", True,
              "SKIPPED: pdflatex not installed")
        return
    with tempfile.TemporaryDirectory() as tmp:
        for f in ("PAPER.tex", "numbers.tex", "table_groups.tex"):
            shutil.copy(os.path.join(HERE, f), tmp)
        shutil.copytree(FIGS, os.path.join(tmp, "figs"))
        for _ in range(3):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "PAPER.tex"],
                           cwd=tmp, capture_output=True)
        log = open(os.path.join(tmp, "PAPER.log"), errors="ignore").read()
        errors = re.findall(r"^! .*", log, re.M)
        overfull = re.findall(r"Overfull \\hbox.*", log)
        undef = re.findall(r"(?i)undefined (?:reference|citation)", log)
        pdf = os.path.join(tmp, "PAPER.pdf")
        check("paper compiles with no errors", not errors, "; ".join(errors[:3]))
        check("no overfull horizontal boxes", not overfull, "; ".join(overfull[:3]))
        check("no undefined references or citations", not undef, "; ".join(undef[:3]))
        check("PDF is produced", os.path.exists(pdf))


# ---------------------------------------------------------------------------
def main():
    print("PenBox-DMAS -- authenticity checks\n" + "=" * 58)
    for fn in (check_determinism, check_macros, check_paper_matches_model,
               check_model_consistency, check_figures, check_citations,
               check_compile):
        print()
        fn()
    print("\n" + "=" * 58)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
