"""PenBox-DMAS Paper - Complete Image Generator. All 13 figures, 300 DPI, B&W."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.linewidth': 1.2,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--'
})

def draw_box(ax, x, y, w, h, text, fc='white', fs=8, fw='normal'):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.02',
                 facecolor=fc, edgecolor='black', linewidth=1.2))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight=fw,
            linespacing=1.3)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

def draw_diamond(ax, x, y, w, h, text, fs=7):
    d = plt.Polygon([(x, y+h/2), (x+w/2, y), (x, y-h/2), (x-w/2, y)],
                    closed=True, facecolor='#f0f0f0', edgecolor='black', linewidth=1.2)
    ax.add_patch(d)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, linespacing=1.3)

def make_phase(fn, title, steps, eqs=''):
    n = len(steps)
    fig, ax = plt.subplots(figsize=(3.5, 6))
    total_h = n * 1.2 + 1.2
    ax.set_xlim(-0.2, 4.2); ax.set_ylim(-0.1, total_h); ax.axis('off')
    y = total_h - 0.5
    ax.text(2, y, title, ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(facecolor='#d0d0d0', edgecolor='black', boxstyle='round,pad=0.2'))
    for i, s in enumerate(steps):
        y -= 1.2; draw_arrow(ax, 2, y+0.9, 2, y+0.35)
        draw_box(ax, 2, y, 3.4, 0.6, s, fs=7, fc='#f0f0f0' if i < n-1 else '#e0e0e0')
    if eqs:
        ax.text(2, 0.1, eqs, ha='center', fontsize=6, fontstyle='italic', color='gray')
    plt.tight_layout(pad=0.5); plt.savefig(fn); plt.close(); print(f'  {fn}')

# ===== FIG 1: HARDWARE =====
def fig1():
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.5); ax.axis('off')
    ax.set_title('Fig. 1: PenBox-DMAS Hardware Architecture', fontweight='bold',
                 fontsize=13, pad=15)
    # Main Brain
    ax.add_patch(FancyBboxPatch((2.2, 5.8), 5.6, 2.2, boxstyle='round,pad=0.1',
                 facecolor='#e8e8e8', edgecolor='black', linewidth=2))
    ax.text(5, 7.5, 'MAIN BRAIN (Tier 1 \u2014 Control Plane)',
            ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 6.85, 'Mini-ITX  |  Ryzen 7 / i7  |  64 GB DDR5',
            ha='center', fontsize=8.5)
    ax.text(5, 6.45, 'RTX 5060 (LLM + GNN)  |  2 TB NVMe',
            ha='center', fontsize=8.5)
    ax.text(5, 6.05, 'Dual NIC: eth0 (mgmt) + eth1 (attack)',
            ha='center', fontsize=8.5)
    # Switch
    ax.add_patch(FancyBboxPatch((3.2, 4.5), 3.6, 0.7, boxstyle='round,pad=0.05',
                 facecolor='#d0d0d0', edgecolor='black', linewidth=1.5))
    ax.text(5, 4.85, 'Gigabit Ethernet Switch', ha='center', fontsize=9.5,
            fontweight='bold')
    draw_arrow(ax, 5, 5.8, 5, 5.2)
    # Agents
    agents = [
        (1.3, 'Agent 1\n(Recon Pi)', 'Nmap, masscan\nFingerprinting\nTopology graph'),
        (3.7, 'Agent 2\n(Exploit Pi)', 'Metasploit\nPoC sandbox\nPriv. escalation'),
        (6.3, 'Agent 3\n(Defense Pi)', 'Hardening\nPatch validation\nConfig rollback'),
        (8.7, 'Agent 4\n(RL Pi)', 'PyTorch-RL\nAttack chains\nQ-learning'),
    ]
    for px, t, d in agents:
        ax.add_patch(FancyBboxPatch((px-1.0, 1.7), 2.0, 2.0, boxstyle='round,pad=0.05',
                     facecolor='#f5f5f5', edgecolor='black', linewidth=1.2))
        ax.text(px, 3.25, t, ha='center', fontsize=8, fontweight='bold', linespacing=1.3)
        ax.text(px, 2.35, d, ha='center', fontsize=6.5, color='#333', linespacing=1.4)
        draw_arrow(ax, 5, 4.5, px, 3.7)
    # Bus
    ax.plot([0.3, 9.7], [1.2, 1.2], 'k-', linewidth=2.5)
    ax.text(5, 0.75, 'Shared Message Bus:  MQTT (telemetry)  +  Redis (tasks)  +  gRPC (control)',
            ha='center', fontsize=8.5, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.25'))
    for px, _, _ in agents:
        ax.plot([px, px], [1.7, 1.2], 'k-', linewidth=1.2)
        ax.plot(px, 1.7, 'ko', markersize=4)
    plt.tight_layout(pad=0.8); plt.savefig('fig_hardware_arch.png'); plt.close()
    print('  fig_hardware_arch.png')

# ===== FIG 2: SOFTWARE STACK =====
def fig2():
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 9.5); ax.axis('off')
    ax.set_title('Fig. 2: PenBox-DMAS Software Stack', fontweight='bold',
                 fontsize=13, pad=15)
    layers = [
        (0.9, '#e0e0e0', 'Hardware Layer',
         'Mini-ITX (x86+GPU) + RPi 5 Cluster (ARM)'),
        (2.0, '#ececec', 'OS Layer',
         'Ubuntu 22.04 (Brain)  |  RPi OS Lite (Agents)'),
        (3.1, '#f0f0f0', 'Container Runtime',
         'Docker  /  K3s Kubernetes  /  Ray'),
        (4.2, '#f5f5f5', 'Agent Processes',
         'Recon  |  Exploit  |  Defense  |  RL Agent'),
        (5.3, '#e8e8e8', 'Message Bus & Storage',
         'MQTT + Redis + gRPC  |  PostgreSQL + SQLite'),
        (6.4, '#f0f0f0', 'AI / LLM Layer',
         'LLaMA 3.1 8B (local) \u2192 Claude API (fallback)'),
        (7.5, '#e0e0e0', 'Orchestrator',
         'Task Planner  |  Risk Scoring  |  Attack Graph'),
        (8.5, '#d5d5d5', 'Human Interface',
         'Offensive Mode  |  Defensive Mode  |  Reports'),
    ]
    for y, fc, title, desc in layers:
        ax.add_patch(FancyBboxPatch((0.5, y-0.4), 9, 0.8, boxstyle='round,pad=0.05',
                     facecolor=fc, edgecolor='black', linewidth=1.2))
        ax.text(1.3, y+0.05, title, fontsize=9.5, fontweight='bold', va='center')
        ax.text(5.8, y+0.05, desc, fontsize=8, va='center', ha='center', color='#222')
    for i in range(len(layers)-1):
        ax.annotate('', xy=(5, layers[i+1][0]-0.4), xytext=(5, layers[i][0]+0.4),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
    plt.tight_layout(pad=0.8); plt.savefig('fig_software_stack.png'); plt.close()
    print('  fig_software_stack.png')

# ===== FIG 3: OFFENSIVE FLOW =====
def fig3():
    fig, ax = plt.subplots(figsize=(9, 14))
    ax.set_xlim(0, 11); ax.set_ylim(0, 18); ax.axis('off')
    ax.set_title('Fig. 3: PenBox-DMAS Offensive Assessment Mode',
                 fontweight='bold', fontsize=13, pad=15)
    y = 17.0
    ax.add_patch(plt.Circle((5, y), 0.35, facecolor='#d0d0d0', edgecolor='black', lw=1.5))
    ax.text(5, y, 'START', ha='center', va='center', fontsize=8, fontweight='bold')
    y -= 1.0; draw_arrow(ax, 5, y+0.65, 5, y+0.3)
    draw_box(ax, 5, y, 4.0, 0.55, 'Offensive Assessment Initiated', fs=9, fw='bold', fc='#e8e8e8')
    y -= 1.0; draw_arrow(ax, 5, y+0.7, 5, y+0.3)
    draw_box(ax, 5, y, 4.2, 0.55, 'User Scope Prompt\n(target, constraints, permissions)', fs=8)
    # Phase 1
    y -= 1.1; draw_arrow(ax, 5, y+0.8, 5, y+0.3)
    ax.add_patch(FancyBboxPatch((1.2, y-0.25), 7.6, 0.5, boxstyle='round,pad=0.05',
                 facecolor='#c0c0c0', edgecolor='black', linewidth=1.5))
    ax.text(5, y, 'PHASE 1 \u2014 Reconnaissance & Enumeration',
            ha='center', fontsize=10, fontweight='bold')
    y -= 1.0; draw_arrow(ax, 5, y+0.7, 5, y+0.3)
    draw_box(ax, 5, y, 4.2, 0.55, 'Network Discovery, Port Scanning\nService Fingerprinting (Recon Pi)', fs=8)
    y -= 1.0; draw_arrow(ax, 5, y+0.7, 5, y+0.3)
    draw_box(ax, 5, y, 3.8, 0.55, 'Deep Scanning & Fuzzing\nTool Selection, OS Compat.', fs=8)
    y -= 1.0; draw_arrow(ax, 5, y+0.7, 5, y+0.3)
    draw_diamond(ax, 5, y, 3.0, 0.75, 'FP/FN\nInitiator?', fs=8)
    ax.text(7.0, y+0.15, 'Yes', fontsize=7, fontweight='bold')
    ax.text(5.2, y-0.55, 'No', fontsize=7, fontweight='bold')
    ax.annotate('', xy=(8.3, y), xytext=(6.5, y),
                arrowprops=dict(arrowstyle='->', color='black', lw=1))
    draw_box(ax, 9.2, y, 2.2, 0.5, 'Recursive\nRe-testing', fs=7, fc='#f0f0f0')
    ax.annotate('', xy=(9.2, y+0.25), xytext=(9.2, y+1.0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8,
                                connectionstyle='arc3,rad=0.5'))
    y -= 1.0; draw_arrow(ax, 5, y+0.65, 5, y+0.3)
    draw_box(ax, 5, y, 3.5, 0.5, 'Generate Phase 1 Report', fs=8)
    # Phase 2
    y -= 1.0; draw_arrow(ax, 5, y+0.7, 5, y+0.3)
    ax.add_patch(FancyBboxPatch((1.2, y-0.25), 7.6, 0.5, boxstyle='round,pad=0.05',
                 facecolor='#c0c0c0', edgecolor='black', linewidth=1.5))
    ax.text(5, y, 'PHASE 2 \u2014 Exploitation Modelling & Persistence',
            ha='center', fontsize=10, fontweight='bold')
    y -= 1.0; draw_arrow(ax, 5, y+0.7, 5, y+0.3)
    draw_box(ax, 5, y, 4.2, 0.55, 'Pre-Exploitation Analysis\nPayload Modelling (Exploit Pi)', fs=8)
    y -= 1.0; draw_arrow(ax, 5, y+0.7, 5, y+0.3)
    draw_box(ax, 5, y, 4.2, 0.55, 'Lateral & Horizontal Propagation\nPrivilege Escalation Path Prediction', fs=8)
    y -= 1.0; draw_arrow(ax, 5, y+0.7, 5, y+0.3)
    draw_box(ax, 5, y, 3.8, 0.55, 'Evasion-Resistance Testing\nPersistence Modelling', fs=8)
    y -= 1.0; draw_arrow(ax, 5, y+0.7, 5, y+0.3)
    draw_box(ax, 5, y, 3.5, 0.55, 'Remediation & PoC Report\n(Defense Pi)', fs=8, fc='#e8e8e8')
    y -= 1.0; draw_arrow(ax, 5, y+0.7, 5, y+0.3)
    draw_diamond(ax, 5, y, 3.2, 0.75, 'Parameters\nco-aligned?', fs=8)
    ax.text(7.1, y+0.15, 'No', fontsize=7, fontweight='bold')
    ax.text(5.2, y-0.55, 'Yes', fontsize=7, fontweight='bold')
    ax.annotate('', xy=(8.5, y), xytext=(6.6, y),
                arrowprops=dict(arrowstyle='->', color='black', lw=1))
    ax.text(9.0, y, 'Loop\nback', fontsize=7, ha='center', linespacing=1.3)
    ax.annotate('', xy=(9.0, 13.5), xytext=(9.0, y+0.38),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    y -= 1.0; draw_arrow(ax, 5, y+0.6, 5, y+0.3)
    ax.add_patch(plt.Circle((5, y), 0.35, facecolor='#d0d0d0', edgecolor='black', lw=1.5))
    ax.text(5, y, 'END', ha='center', va='center', fontsize=8, fontweight='bold')
    plt.tight_layout(pad=0.8); plt.savefig('fig_offensive_flow.png'); plt.close()
    print('  fig_offensive_flow.png')

# ===== FIG 4: LATENCY BAR =====
def fig4():
    configs = ['PenBox\n(Config A)', 'DMAS 3-Agent\n(Config B)', 'DMAS+RL\n(Config C)']
    x = np.arange(3); w = 0.22
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    b1 = ax.bar(x-w, [340,95,95], w, label='Reconnaissance', color='#2d2d2d',
                edgecolor='black', linewidth=0.8)
    b2 = ax.bar(x, [520,120,95], w, label='Exploitation', color='#7a7a7a',
                edgecolor='black', linewidth=0.8)
    b3 = ax.bar(x+w, [180,45,45], w, label='Remediation', color='#c0c0c0',
                edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Time (seconds)', fontweight='bold', fontsize=11)
    ax.set_xlabel('System Configuration', fontweight='bold', fontsize=11)
    ax.set_title('Per-Phase Latency Comparison', fontweight='bold', fontsize=13, pad=12)
    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylim(0, 650); ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{int(h)}s', xy=(bar.get_x()+bar.get_width()/2, h),
                        xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)
    plt.tight_layout(pad=1.0); plt.savefig('fig_latency_bar.png'); plt.close()
    print('  fig_latency_bar.png')

# ===== FIG 5: RISK LINE =====
def fig5():
    iters = np.arange(1, 11)
    mono = [92,87,81,76,72,68,65,62,60,58]
    dmas = [92,74,58,45,35,27,22,18,16,15]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(iters, mono, 'ko-', linewidth=2, markersize=7,
            label='PenBox Monolithic (Config A)')
    ax.plot(iters, dmas, 'ks--', linewidth=2, markersize=7, markerfacecolor='gray',
            label='DMAS + RL (Config C)')
    ax.fill_between(iters, mono, dmas, alpha=0.12, color='gray')
    ax.set_xlabel('Assessment Iteration', fontweight='bold', fontsize=11)
    ax.set_ylabel('Composite Risk Score', fontweight='bold', fontsize=11)
    ax.set_title('Risk Score Convergence \u2014 Monolithic vs. DMAS+RL',
                 fontweight='bold', fontsize=13, pad=12)
    ax.set_xlim(1, 10); ax.set_ylim(0, 105); ax.set_xticks(iters)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax.annotate('58% faster\nconvergence', xy=(7, 42), fontsize=10, fontstyle='italic',
                ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                edgecolor='gray'))
    plt.tight_layout(pad=1.0); plt.savefig('fig_risk_line.png'); plt.close()
    print('  fig_risk_line.png')

# ===== FIG 6: RADAR =====
def fig6():
    cats = ['Latency', 'Detection\nAccuracy', 'Novel-Pattern\nLearning',
            'Fault\nTolerance', 'Cloud\nIndependence']
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False).tolist()
    angles += angles[:1]
    p1 = [0.35,0.75,0.10,0.15,0.90]+[0.35]
    p2 = [0.85,0.96,0.84,0.88,0.95]+[0.85]
    p3 = [0.60,0.82,0.30,0.20,0.10]+[0.60]
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    ax.plot(angles, p3, 'k:', linewidth=1.5, label='Cloud Baselines (avg)')
    ax.fill(angles, p3, alpha=0.05, color='gray')
    ax.plot(angles, p1, 'k--', linewidth=1.8, label='PenBox (original)')
    ax.fill(angles, p1, alpha=0.1, color='gray')
    ax.plot(angles, p2, 'k-', linewidth=2.5, label='PenBox-DMAS')
    ax.fill(angles, p2, alpha=0.2, color='gray')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=8)
    ax.set_title('Multi-Dimensional Performance Comparison',
                 fontweight='bold', fontsize=13, pad=25)
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, -0.08), framealpha=0.9, fontsize=9)
    plt.tight_layout(pad=1.5); plt.savefig('fig_radar_chart.png'); plt.close()
    print('  fig_radar_chart.png')

# ===== FIG 7: REMEDIATION LINE =====
def fig7():
    iters = np.arange(1, 11)
    vuln = [325,248,185,140,108,85,70,61,56,53]
    remed = [77,63,45,32,23,15,9,5,3,2]
    fig, ax1 = plt.subplots(figsize=(7.5, 5.5))
    ax1.plot(iters, vuln, 'ko-', linewidth=2, markersize=8, label='Active Vulnerabilities')
    ax1.axhline(y=53, color='gray', linestyle='--', linewidth=1.2,
                label='V* = 53 (equilibrium)')
    ax1.set_xlabel('Assessment Iteration', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Active Vulnerability Count', fontweight='bold', fontsize=11)
    ax1.set_ylim(0, 380); ax1.set_xticks(iters)
    ax2 = ax1.twinx()
    ax2.plot(iters, remed, 'k^--', linewidth=1.8, markersize=7, markerfacecolor='gray',
             label='Remediated per Iteration')
    ax2.set_ylabel('Remediated per Iteration', fontweight='bold', fontsize=11)
    ax2.set_ylim(0, 100)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, loc='upper right', framealpha=0.9, fontsize=9)
    ax1.set_title('Remediation Performance Over Assessment Iterations',
                  fontweight='bold', fontsize=13, pad=12)
    plt.tight_layout(pad=1.0); plt.savefig('fig_remediation_line.png'); plt.close()
    print('  fig_remediation_line.png')

# ===== PHASE FLOWCHARTS =====
def gen_phases():
    make_phase('fig_phase1.png', 'Phase 1\nInitialisation', [
        'Load target config\n& credentials',
        'Initialise state\nvector S_n',
        'Compute attack\nsurface P_n',
        'Set M_n = 0.5\n(initial remediation)',
    ], 'Eqs. (1)\u2013(4)')
    make_phase('fig_phase2.png', 'Phase 2\nDiscovery', [
        'Signature-based\nscanning (\u03b1_i)',
        'Network topology\ntraversal (\u03b2)',
        'Aggregate V_detected\n(hybrid scan)',
        'Forward to\nvalidation',
    ], 'Eq. (5)')
    make_phase('fig_phase3.png', 'Phase 3\nValidation', [
        'Apply FPR filter\n(FPR = 0.10)',
        'Recover FN\n(FNR = 0.04)',
        'Produce\nV_validated',
        'Map to ATT&CK\ntechniques T_ij',
    ], 'Eqs. (6)\u2013(7)')
    make_phase('fig_phase4.png', 'Phase 4\nPrioritisation', [
        'Compute technique\nweights w_j',
        'Compute\nPriority(V_i)',
        'Rank by privilege\nrequired',
        'Select top-K\nvulnerabilities',
    ], 'Eq. (8)')
    make_phase('fig_phase5.png', 'Phase 5\nExploitation', [
        'Compute P_exploit\n(single-stage)',
        'Check escalation\npaths in graph',
        'Compute P_multi\n(chained attacks)',
        'RL agent predicts\nnovel chains',
    ], 'Eqs. (9)\u2013(10), (16)')
    make_phase('fig_phase6.png', 'Phase 6\nRemediation', [
        'Apply patches\nwhere P > \u03b8',
        'Update V_{n+1}\nstate evolution',
        'Recompute risk\nR_{n+1}',
        'Check convergence\n\u0394V \u2264 \u03b5 \u2192 V*',
    ], 'Eqs. (11)\u2013(13), (17)')

if __name__ == '__main__':
    print('Generating all PenBox-DMAS figures...\n')
    print('Architecture & Flowcharts:')
    fig1(); fig2(); fig3(); gen_phases()
    print('\nCharts & Graphs:')
    fig4(); fig5(); fig6(); fig7()
    print('\nDone! All 13 figures generated.')
