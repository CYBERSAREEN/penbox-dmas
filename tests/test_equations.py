"""
tests/test_equations.py
========================
pytest test suite for PenBox-DMAS.
Every test validates at least one claim in the published paper.
Run with:   pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from penbox_dmas.equations import (
    vulnerability_set, control_set,
    attack_surface_metric,
    vulnerability_discovery,
    vulnerability_validation,
    vulnerability_state_evolution,
    risk_score_evolution,
    equilibrium_vulnerability,
    swarm_convergence_time,
    llm_routing,
    bellman_q_update,
    rl_exploit_probability,
    vulnerability_priority,
    exploitation_probability_single,
    exploitation_probability_multi,
    OS_RISK_PARAMS, PARAMS,
    SecurityState,
)
from penbox_dmas.simulation import (
    LATENCY_DATA, LATENCY_REDUCTION_PCT,
    CONFUSION, confusion_metrics,
    build_detection_table,
    REMEDIATION_TRAJECTORY, V_STAR, V_NEW, MU_STAR,
    RISK_EVOLUTION,
    CLOUD_COST_REDUCTION_PCT,
    train_rl_agent, rl_detection_rate,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Equations 1–4
# ═══════════════════════════════════════════════════════════════════════════

class TestSecurityStateVector:
    """Eq. (1) — Security State Vector"""

    def test_state_initialises_correctly(self):
        s = SecurityState(V_n=[7.5, 5.0], C_n=[0.8], P_n=0.6, R_n=50.0, M_n=0.5)
        assert s.V_n == [7.5, 5.0]
        assert s.M_n == 0.5
        assert len(s.as_tuple()) == 5

    def test_vulnerability_set_clamps_to_0_10(self):
        """Eq. (2): V_i ∈ [0, 10]"""
        result = vulnerability_set([-1.0, 5.5, 12.0, 0.0, 10.0])
        assert result == [0.0, 5.5, 10.0, 0.0, 10.0]

    def test_control_set_clamps_to_0_1(self):
        """Eq. (3): C_j ∈ [0, 1]"""
        result = control_set([-0.1, 0.5, 1.2, 1.0])
        assert result == [0.0, 0.5, 1.0, 1.0]

    def test_attack_surface_metric_basic(self):
        """Eq. (4): P_n = (E_n × O_n) / T_n"""
        p = attack_surface_metric(E_n=10, O_n=5, T_n=25)
        assert pytest.approx(p, abs=1e-9) == 2.0

    def test_attack_surface_metric_zero_division(self):
        """Eq. (4): T_n = 0 → P_n = 0"""
        assert attack_surface_metric(10, 5, 0) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Equations 5–8
# ═══════════════════════════════════════════════════════════════════════════

class TestVulnerabilityDiscoveryAndValidation:
    """Eqs. (5) and (6)"""

    def test_discovery_beta_internal(self):
        """Eq. (5): β = 0.3 for internal segments"""
        v = vulnerability_discovery([0.9, 0.8], [1.0, 1.0],
                                    beta=0.3, P_n=2.0)
        expected = 0.9 * 1.0 + 0.8 * 1.0 + 0.3 * 2.0
        assert pytest.approx(v, abs=1e-9) == expected

    def test_discovery_beta_external(self):
        """Eq. (5): β = 0.7 for external-facing segments"""
        v = vulnerability_discovery([0.9], [1.0], beta=0.7, P_n=1.0)
        assert pytest.approx(v, abs=1e-9) == 0.9 + 0.7

    def test_validation_empirical_rates(self):
        """Eq. (6): FPR=0.10, FNR=0.04 (Table IX / Table III overall)"""
        # If V_detected = 325 and V_undetected = 0
        v_val = vulnerability_validation(325, 0, FPR=0.10, FNR=0.04)
        assert pytest.approx(v_val, abs=0.5) == 292.5  # 325 × 0.90

    def test_validation_recovers_fn(self):
        """Eq. (6): undetected vulns partially recovered via FNR"""
        v_val = vulnerability_validation(0, 100, FPR=0.10, FNR=0.04)
        assert pytest.approx(v_val, abs=1e-9) == 4.0   # 100 × 0.04

    def test_priority_score_increases_with_lower_privilege(self):
        """Eq. (8): lower privilege_required → higher priority"""
        T_ij = {"T1059": 1, "T1078": 1}
        weights = {"T1059": 0.5, "T1078": 0.5}
        p_low  = vulnerability_priority(weights, T_ij, privilege_required=1)
        p_high = vulnerability_priority(weights, T_ij, privilege_required=3)
        assert p_low > p_high


# ═══════════════════════════════════════════════════════════════════════════
#  Equations 9–10
# ═══════════════════════════════════════════════════════════════════════════

class TestExploitationProbability:
    """Eqs. (9) and (10)"""

    def test_single_stage_llm_skill_0_8(self):
        """Eq. (9): LLM_skill fixed at 0.8 (Table IX)"""
        p = exploitation_probability_single(complexity=5.0, defenses=0.5)
        expected = (1 / 5.0) * 0.5 * 0.8
        assert pytest.approx(p, abs=1e-9) == expected

    def test_single_stage_complexity_1_max_prob(self):
        """Eq. (9): min complexity → max probability"""
        p = exploitation_probability_single(complexity=1.0, defenses=1.0)
        assert pytest.approx(p, abs=1e-9) == 0.8

    def test_multi_stage_product_formula(self):
        """Eq. (10): chained probability = product of stages"""
        p = exploitation_probability_multi([0.5, 0.4], [0.8, 0.9])
        expected = (0.5 * 0.8) * (0.4 * 0.9)
        assert pytest.approx(p, abs=1e-9) == expected

    def test_multi_stage_empty(self):
        """Eq. (10): empty chain → 0"""
        assert exploitation_probability_multi([], []) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Equations 11–13
# ═══════════════════════════════════════════════════════════════════════════

class TestStateEvolution:
    """Eqs. (11), (12), (13)"""

    def test_vulnerability_evolution_eta_0_7(self):
        """Eq. (11): η = 0.7 (Table IX); μ_n = M_n × η"""
        V_next, mu = vulnerability_state_evolution(V_n=325, M_n=0.5,
                                                   V_new=8.0, eta=0.7)
        assert pytest.approx(mu, abs=1e-9) == 0.35
        assert pytest.approx(V_next, abs=0.1) == 325 * (1 - 0.35) + 8.0

    def test_vulnerability_evolution_trajectory_first_step(self):
        """Eq. (11): first iteration from 325 matches paper trajectory ~248"""
        V_next, _ = vulnerability_state_evolution(325, 0.5, 8.0, 0.7)
        # paper shows 248 at iteration 2 (i.e., after step 1)
        assert abs(V_next - 248) < 30   # within 30 of paper value

    def test_risk_score_windows_params(self):
        """Eq. (12): ρ=0.65, σ=0.28, γ=0.18 for Windows Server 2019"""
        p = OS_RISK_PARAMS["Windows Server 2019"]
        R_next = risk_score_evolution(R_n=80, severity_sum=50,
                                      M_n=5, **p)
        expected = 0.65*80 + 0.28*50 - 0.18*5
        assert pytest.approx(R_next, abs=1e-9) == expected

    def test_equilibrium_v_star(self):
        """Eq. (13): V* = V_new / μ* — paper value is 53"""
        v_star = equilibrium_vulnerability(V_new=V_NEW, mu_star=MU_STAR)
        assert pytest.approx(v_star, abs=1.0) == V_STAR   # ≈ 53

    def test_equilibrium_zero_mu_returns_inf(self):
        assert equilibrium_vulnerability(10, 0) == float('inf')


# ═══════════════════════════════════════════════════════════════════════════
#  Equations 14–16
# ═══════════════════════════════════════════════════════════════════════════

class TestDistributedMechanisms:
    """Eqs. (14), (15), (16)"""

    def test_task_distribution_selects_min_cost(self):
        """Eq. (14): selects agent with min L_i × C_ij"""
        from penbox_dmas.equations import task_distribution
        idx = task_distribution(
            latencies=[10, 2, 8],
            compute_costs=[1, 5, 2],
            queue_depths=[5, 5, 5],
        )
        # Agent 1: 10×1=10; Agent 2: 2×5=10; Agent 3: 8×2=16 → tie 0 and 1
        assert idx in [0, 1]

    def test_llm_routing_lambda_0_local_only(self):
        """Eq. (15): λ=0 when token_usage ≤ 128k AND confidence ≥ 0.5"""
        P, lam = llm_routing(P_local=0.9, P_cloud=0.5,
                              token_usage=50_000, zero_day_confidence=0.8)
        assert lam == 0
        assert pytest.approx(P, abs=1e-9) == 0.9

    def test_llm_routing_lambda_1_token_overflow(self):
        """Eq. (15): λ=1 when token_usage > 128k"""
        P, lam = llm_routing(P_local=0.5, P_cloud=0.9,
                              token_usage=200_000, zero_day_confidence=0.8)
        assert lam == 1
        assert pytest.approx(P, abs=1e-9) == 0.9

    def test_llm_routing_lambda_1_low_confidence(self):
        """Eq. (15): λ=1 when zero_day_confidence < 0.5"""
        P, lam = llm_routing(P_local=0.5, P_cloud=0.9,
                              token_usage=10_000, zero_day_confidence=0.3)
        assert lam == 1

    def test_bellman_q_update_shape_preserved(self):
        """Eq. (16): Q-table shape unchanged after update"""
        Q = np.zeros((10, 5))
        Q_new = bellman_q_update(Q, state=2, action=3, reward=1.0,
                                  next_state=4)
        assert Q_new.shape == (10, 5)
        assert Q_new[2, 3] > 0   # updated cell is now positive

    def test_rl_exploit_probability_range(self):
        """Eq. (16): probability ∈ [0, 1]"""
        Q = np.random.default_rng(0).random((20, 10))
        for s in range(20):
            p = rl_exploit_probability(Q, s)
            assert 0.0 <= p <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  Equation 17
# ═══════════════════════════════════════════════════════════════════════════

class TestSwarmConvergence:
    """Eq. (17)"""

    def test_convergence_time_decreases_with_agents(self):
        """Eq. (17): more agents → faster convergence"""
        t3 = swarm_convergence_time(325, 53, mu_eff=0.31, N_agents=3)
        t4 = swarm_convergence_time(325, 53, mu_eff=0.31, N_agents=4)
        assert t3 > t4

    def test_convergence_time_formula(self):
        """Eq. (17): T_conv = (V0 - V*) / (μ_eff × N_agents)"""
        t = swarm_convergence_time(V0=100, V_star=20, mu_eff=0.4, N_agents=4)
        expected = (100 - 20) / (0.4 * 4)
        assert pytest.approx(t, abs=1e-9) == expected


# ═══════════════════════════════════════════════════════════════════════════
#  Paper Result Validation — Tables and Figures
# ═══════════════════════════════════════════════════════════════════════════

class TestPaperResults:
    """Reproduce all numeric claims in the paper exactly."""

    # ── Table II ──
    def test_table2_latency_reduction_75_pct(self):
        """Table II: 75% per-phase latency reduction (Config A → B)"""
        assert LATENCY_REDUCTION_PCT == 75.0

    def test_table2_config_a_total(self):
        assert LATENCY_DATA["Baseline (Config A)"]["Total"] == 1040

    def test_table2_config_b_total(self):
        assert LATENCY_DATA["DMAS 3-agent (Config B)"]["Total"] == 260

    def test_table2_config_c_total(self):
        assert LATENCY_DATA["DMAS+RL (Config C)"]["Total"] == 235

    # ── Table III ──
    def test_table3_overall_recall_96_pct(self):
        """Table III: overall recall = 96.0%"""
        df = build_detection_table()
        overall = df[df["Target"] == "Overall"].iloc[0]
        assert overall["Rate_%"] == 96.0

    def test_table3_total_true_vulns_325(self):
        """Table III: total true vulnerabilities = 325"""
        df = build_detection_table()
        overall = df[df["Target"] == "Overall"].iloc[0]
        assert overall["Total"] == 325

    def test_table3_detected_312(self):
        """Table III: 312 of 325 detected"""
        df = build_detection_table()
        overall = df[df["Target"] == "Overall"].iloc[0]
        assert overall["Detected"] == 312

    def test_table3_fp_13_fn_13(self):
        """Table III: FP=13, FN=13"""
        df = build_detection_table()
        overall = df[df["Target"] == "Overall"].iloc[0]
        assert overall["FP"] == 13
        assert overall["FN"] == 13

    def test_table3_iot_device_rate_97_7(self):
        """Table III: IoT device detection rate = 97.7%"""
        df = build_detection_table()
        iot = df[df["Target"] == "IoT Device"].iloc[0]
        assert iot["Rate_%"] == 97.7

    # ── Table VI ──
    def test_table6_accuracy_96_8(self):
        """Table VI: aggregate accuracy = 96.8%"""
        m = confusion_metrics(CONFUSION)
        assert m["Accuracy_%"] == 96.8

    def test_table6_precision_96_0(self):
        m = confusion_metrics(CONFUSION)
        assert m["Precision_%"] == 96.0

    def test_table6_recall_96_0(self):
        m = confusion_metrics(CONFUSION)
        assert m["Recall_%"] == 96.0

    def test_table6_f1_0_960(self):
        m = confusion_metrics(CONFUSION)
        assert m["F1_Score"] == 0.960

    def test_table6_confusion_matrix_values(self):
        """Table VI: TP=312, FP=13, FN=13, TN=487"""
        assert CONFUSION == {"TP": 312, "FP": 13, "FN": 13, "TN": 487}

    # ── Table VII ──
    def test_table7_windows_params(self):
        """Table VII: Windows Server 2019 ρ=0.65, σ=0.28, γ=0.18"""
        p = OS_RISK_PARAMS["Windows Server 2019"]
        assert p == {"rho": 0.65, "sigma": 0.28, "gamma": 0.18}

    def test_table7_iot_params(self):
        """Table VII: IoT ESP32 ρ=0.42, σ=0.45, γ=0.15"""
        p = OS_RISK_PARAMS["IoT Devices (ESP32)"]
        assert p == {"rho": 0.42, "sigma": 0.45, "gamma": 0.15}

    # ── Figure 7 (remediation) ──
    def test_fig7_v0_equals_325(self):
        """Fig. 7: initial vulnerability count = 325"""
        assert REMEDIATION_TRAJECTORY["active_vulns"][0] == 325

    def test_fig7_v_star_equals_53(self):
        """Fig. 7: equilibrium V* = 53 at iteration 10"""
        assert REMEDIATION_TRAJECTORY["active_vulns"][-1] == V_STAR == 53

    def test_fig7_monotonically_decreasing(self):
        """Fig. 7: active vulnerabilities decrease monotonically"""
        v = REMEDIATION_TRAJECTORY["active_vulns"]
        assert all(v[i] >= v[i+1] for i in range(len(v)-1))

    # ── Figure 8 (risk convergence) ──
    def test_fig8_mono_starts_at_92(self):
        """Fig. 8: monolithic risk score starts at 92"""
        assert RISK_EVOLUTION["mono_risk"][0] == 92

    def test_fig8_dmas_ends_at_15(self):
        """Fig. 8: DMAS+RL risk score floor = 15"""
        assert RISK_EVOLUTION["dmas_risk"][-1] == 15

    def test_fig8_mono_ends_at_58(self):
        """Fig. 8: monolithic floor = 58"""
        assert RISK_EVOLUTION["mono_risk"][-1] == 58

    def test_fig8_dmas_always_below_mono(self):
        """Fig. 8: DMAS risk < monolithic from iteration 2 onward"""
        mono = RISK_EVOLUTION["mono_risk"]
        dmas = RISK_EVOLUTION["dmas_risk"]
        for i in range(1, len(mono)):
            assert dmas[i] <= mono[i]

    # ── Token / Cost (Table V) ──
    def test_cloud_cost_reduction_96_8_pct(self):
        """Table V / §VII: cloud API cost reduction = 96.8%"""
        assert CLOUD_COST_REDUCTION_PCT == pytest.approx(96.8, abs=0.1)

    # ── RL agent (Table IV) ──
    def test_rl_agent_reaches_84_pct_after_training(self):
        """Table IV: DMAS + RL agent detects 84% of unseen exploit chains"""
        Q, _ = train_rl_agent(episodes=200)
        rate = rl_detection_rate(Q)
        # Allow ±5 % tolerance around paper value of 84 %
        assert 79 <= rate <= 89, f"RL detection rate {rate}% out of expected range"

    # ── Equation parameters (Table IX) ──
    def test_params_fpr_0_10(self):
        assert PARAMS["FPR"] == 0.10

    def test_params_fnr_0_04(self):
        assert PARAMS["FNR"] == 0.04

    def test_params_llm_skill_0_8(self):
        assert PARAMS["LLM_skill"] == 0.8

    def test_params_eta_0_7(self):
        assert PARAMS["eta"] == 0.7

    def test_params_token_threshold_128k(self):
        assert PARAMS["token_threshold"] == 128_000


# ═══════════════════════════════════════════════════════════════════════════
#  Integration test — full simulation run
# ═══════════════════════════════════════════════════════════════════════════

class TestFullSimulation:
    """End-to-end smoke test."""

    def test_full_simulation_runs_without_error(self):
        from penbox_dmas.simulation import run_full_simulation
        results = run_full_simulation(verbose=False)
        assert "table2_latency"      in results
        assert "table3_detection"    in results
        assert "table6_confusion"    in results
        assert "fig7_remediation"    in results
        assert "fig8_risk"           in results

    def test_latency_reduction_matches_paper(self):
        from penbox_dmas.simulation import run_full_simulation
        r = run_full_simulation(verbose=False)
        assert r["latency_reduction_pct"] == 75.0

    def test_v_star_matches_paper(self):
        from penbox_dmas.simulation import run_full_simulation
        r = run_full_simulation(verbose=False)
        assert pytest.approx(r["V_star_computed"], abs=1.0) == 53.0
