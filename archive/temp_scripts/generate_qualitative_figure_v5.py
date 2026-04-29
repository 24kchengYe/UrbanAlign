"""
Qualitative Figure v5 — Merged concept scores + LWRR in one compound panel.

Changes from v4:
  - Concept scores and LWRR merged into adjacent sub-axes (one visual column)
  - 3 thin bars per dimension: Score A (green), Score B (blue), LWRR contribution (green/orange)
  - Removed GT:A badge
  - Increased gap between images and charts to prevent overlap
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

sys.stdout.reconfigure(encoding='utf-8')

# ── Paths ──
OUT = 'urbanalign_outputs'
FIG_DIR = os.path.join(OUT, 'qualitative_figure')
DATA_FILE = os.path.join(FIG_DIR, 'fig3_data.json')

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

CATEGORIES = ['safety', 'lively', 'wealthy', 'beautiful', 'boring', 'depressing']
CAT_DISPLAY = {
    'safety': 'Safety', 'lively': 'Lively', 'wealthy': 'Wealthy',
    'beautiful': 'Beautiful', 'boring': 'Boring', 'depressing': 'Depressing'
}

# ── Color Palette ──
PAL = {
    'A':         '#2E7D32',
    'A_bar':     '#43A047',
    'A_light':   '#E8F5E9',
    'B':         '#1565C0',
    'B_bar':     '#42A5F5',
    'B_light':   '#E3F2FD',
    'pos':       '#2E7D32',
    'neg':       '#E65100',
    'pos_bar':   '#66BB6A',
    'neg_bar':   '#FF8A65',
    'correct':   '#2E7D32',
    'wrong':     '#C62828',
    'header':    '#212121',
    'subtext':   '#616161',
    'muted':     '#9E9E9E',
    'border':    '#E0E0E0',
    'grid':      '#EEEEEE',
    'equal_bar': '#BDBDBD',
    # Baseline confidence (distinct from concept A/B)
    'conf_a':    '#FF9800',   # orange
    'conf_b':    '#7B1FA2',   # purple
    'conf_eq':   '#BDBDBD',   # grey
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica Neue'],
    'axes.linewidth': 0.4,
    'axes.edgecolor': PAL['border'],
})


def norm10(val):
    return val / 10.0 if val > 10 else float(val)


def shorten(name, max_len=999):
    """Return full dimension name (no abbreviation)."""
    return name


def remap_pred(original_pred, human_winner):
    if original_pred == 'equal':
        return 'E'
    return 'A' if original_pred == human_winner else 'B'


def remap_probs(probs_dict, human_winner):
    if not probs_dict:
        return None
    if human_winner == 'left':
        return {'A': probs_dict.get('left', 0), 'B': probs_dict.get('right', 0),
                'E': probs_dict.get('equal', 0)}
    return {'A': probs_dict.get('right', 0), 'B': probs_dict.get('left', 0),
            'E': probs_dict.get('equal', 0)}


# ══════════════════════════════════════
# Figure layout
# ══════════════════════════════════════
n_rows = 6
fig_w = 16
row_h = 2.6
fig_h = row_h * n_rows + 0.3

fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

# Outer: ImgA | ImgB | Gap | MergedChart | Methods
gs = GridSpec(n_rows, 5, figure=fig,
             width_ratios=[1.2, 1.2, 1.0, 2.8, 1.8],
             hspace=0.55, wspace=0.06,
             left=0.005, right=0.995, top=0.97, bottom=0.05)

export_data = {}
img_a_axes = []  # for row separators

for row_i, cat in enumerate(CATEGORIES):
    p = data['pairs'].get(cat)
    if not p:
        continue

    human = p['human_winner']
    corrected = p['lwrr_corrected']
    dims = p['dimension_names']
    weights = np.array(p['lwrr_weights'])
    fitted_delta = p['fitted_ts_delta']
    n_d = len(dims)

    if human == 'left':
        a_scores_raw = p['dimension_scores_left']
        b_scores_raw = p['dimension_scores_right']
        a_img_file = f"{cat}_left_{p['left_id']}.jpg"
        b_img_file = f"{cat}_right_{p['right_id']}.jpg"
        sign = 1
    else:
        a_scores_raw = p['dimension_scores_right']
        b_scores_raw = p['dimension_scores_left']
        a_img_file = f"{cat}_right_{p['right_id']}.jpg"
        b_img_file = f"{cat}_left_{p['left_id']}.jpg"
        sign = -1

    a_vals = [norm10(a_scores_raw[d]) for d in dims]
    b_vals = [norm10(b_scores_raw[d]) for d in dims]
    a_raw = [float(a_scores_raw[d]) for d in dims]
    b_raw = [float(b_scores_raw[d]) for d in dims]

    raw_sum_a = sum(a_vals)
    raw_sum_b = sum(b_vals)
    raw_pred_ab = remap_pred(p['raw_vlm_pred'], human)
    ua_pred_ab = remap_pred(p['urbanalign_pred'], human)

    s_diff = np.array([
        float(p['dimension_scores_left'][d]) - float(p['dimension_scores_right'][d])
        for d in dims
    ])
    intercept_raw = fitted_delta - np.dot(weights, s_diff)
    delta_towards_a = sign * fitted_delta
    intercept_towards_a = sign * intercept_raw
    contrib_towards_a = sign * (weights * s_diff)

    # ════════════════════════════════════════
    # Col 0: Image A
    # ════════════════════════════════════════
    ax_a = fig.add_subplot(gs[row_i, 0])
    img_a_axes.append(ax_a)
    try:
        ax_a.imshow(mpimg.imread(os.path.join(FIG_DIR, a_img_file)))
    except Exception:
        ax_a.text(0.5, 0.5, 'N/A', ha='center', va='center',
                  transform=ax_a.transAxes, fontsize=14)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    for sp in ax_a.spines.values():
        sp.set_edgecolor(PAL['A']); sp.set_linewidth(3)

    ax_a.set_xlabel('A', fontsize=10, color=PAL['A'],
                    fontweight='bold', labelpad=3)
    if row_i == 0:
        ax_a.set_title('Image A', fontsize=13, fontweight='bold',
                       color=PAL['header'], pad=8)

    # ════════════════════════════════════════
    # Col 1: Image B
    # ════════════════════════════════════════
    ax_b = fig.add_subplot(gs[row_i, 1])
    try:
        ax_b.imshow(mpimg.imread(os.path.join(FIG_DIR, b_img_file)))
    except Exception:
        ax_b.text(0.5, 0.5, 'N/A', ha='center', va='center',
                  transform=ax_b.transAxes, fontsize=14)
    ax_b.set_xticks([]); ax_b.set_yticks([])
    for sp in ax_b.spines.values():
        sp.set_edgecolor(PAL['border']); sp.set_linewidth(1)

    ax_b.set_xlabel('B', fontsize=10, color=PAL['subtext'],
                    fontweight='bold', labelpad=3)
    if row_i == 0:
        ax_b.set_title('Image B', fontsize=13, fontweight='bold',
                       color=PAL['header'], pad=8)

    # Category name centered below the image pair
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    cat_x = (pos_a.x0 + pos_b.x1) / 2
    cat_y = pos_a.y0 - 0.025
    fig.text(cat_x, cat_y, CAT_DISPLAY[cat], ha='center', va='top',
             fontsize=14, fontweight='bold', color=PAL['header'])

    # ════════════════════════════════════════
    # Col 2: Gap (empty)
    # ════════════════════════════════════════
    fig.add_subplot(gs[row_i, 2]).axis('off')

    # ════════════════════════════════════════
    # Col 3: Merged Chart — Concept Scores (left) + LWRR (right)
    # ════════════════════════════════════════
    gs_inner = gs[row_i, 3].subgridspec(1, 2, width_ratios=[1.4, 2.0], wspace=0.0)
    ax_cs = fig.add_subplot(gs_inner[0, 0])
    ax_lw = fig.add_subplot(gs_inner[0, 1], sharey=ax_cs)

    n_total = n_d + 1  # dims + intercept
    y_pos = np.arange(n_total)

    # ── Concept Scores (paired A/B thin bars) ──
    bar_h_cs = 0.16
    gap_ab = 0.02

    for i in range(n_d):
        ya = y_pos[i] - gap_ab - bar_h_cs / 2
        yb = y_pos[i] + gap_ab + bar_h_cs / 2

        ax_cs.barh(ya, a_vals[i], bar_h_cs, color=PAL['A_bar'], alpha=0.85,
                   edgecolor='white', linewidth=0.3, zorder=3)
        ax_cs.barh(yb, b_vals[i], bar_h_cs, color=PAL['B_bar'], alpha=0.70,
                   edgecolor='white', linewidth=0.3, zorder=3)

        # Score annotations at bar end (show normalized 0-10 values)
        max_v = max(max(a_vals), max(b_vals))
        ax_cs.text(a_vals[i] + max_v * 0.04, ya, f'{a_vals[i]:.0f}',
                   fontsize=8.5, color=PAL['A'], va='center', ha='left')
        ax_cs.text(b_vals[i] + max_v * 0.04, yb, f'{b_vals[i]:.0f}',
                   fontsize=8.5, color=PAL['B'], va='center', ha='left')

    # Separator before intercept
    ax_cs.axhline(y=n_d - 0.5, color=PAL['border'], linewidth=0.6, linestyle='--')

    max_score = max(max(a_vals), max(b_vals))
    ax_cs.set_xlim(0, max_score * 1.45)
    ax_cs.set_yticks(y_pos)
    labels_y = [shorten(d, 14) for d in dims] + ['Intercept']
    ax_cs.set_yticklabels(labels_y, fontsize=10)
    ax_cs.invert_yaxis()

    for sp in ['top', 'right']:
        ax_cs.spines[sp].set_visible(False)
    ax_cs.spines['left'].set_linewidth(0.3)
    ax_cs.spines['bottom'].set_linewidth(0.3)
    ax_cs.tick_params(axis='y', length=0, pad=4)
    ax_cs.tick_params(axis='x', labelsize=9, colors=PAL['muted'])

    # Vertical separator between concept and LWRR
    ax_cs.axvline(x=ax_cs.get_xlim()[1], color=PAL['border'],
                  linewidth=0.6, linestyle=':', clip_on=False)

    if row_i == 0:
        ax_cs.set_title('Concept Scores', fontsize=12, fontweight='bold',
                        color=PAL['header'], pad=6)

    # ── LWRR Contribution (diverging bars) ──
    contrib_full = list(contrib_towards_a) + [intercept_towards_a]
    bar_h_lw = 0.28
    colors_lw = [PAL['pos_bar'] if v >= 0 else PAL['neg_bar'] for v in contrib_full]

    ax_lw.barh(y_pos, contrib_full, bar_h_lw, color=colors_lw, alpha=0.85,
               edgecolor='white', linewidth=0.3, zorder=3)

    # Value annotations
    x_range = max(max(contrib_full) - min(contrib_full), 1)
    for i, val in enumerate(contrib_full):
        ha = 'left' if val >= 0 else 'right'
        off = x_range * 0.03 if val >= 0 else -x_range * 0.03
        c = PAL['pos'] if val >= 0 else PAL['neg']
        ax_lw.text(val + off, y_pos[i], f'{val:+.1f}',
                   va='center', ha=ha, fontsize=10, color=c, fontweight='bold')

    ax_lw.axhline(y=n_d - 0.5, color=PAL['border'], linewidth=0.6,
                  linestyle='--', zorder=1)
    ax_lw.axvline(x=0, color=PAL['subtext'], linewidth=0.6, zorder=2)

    xmin_lw, xmax_lw = min(contrib_full), max(contrib_full)
    pad_lw = max(abs(xmin_lw), abs(xmax_lw)) * 0.45
    ax_lw.set_xlim(min(xmin_lw - pad_lw, -0.5), max(xmax_lw + pad_lw, 0.5))

    plt.setp(ax_lw.get_yticklabels(), visible=False)
    ax_lw.tick_params(axis='y', length=0)
    for sp in ['top', 'right', 'left']:
        ax_lw.spines[sp].set_visible(False)
    ax_lw.spines['bottom'].set_linewidth(0.3)
    ax_lw.tick_params(axis='x', labelsize=9, colors=PAL['muted'])

    if row_i == 0:
        ax_lw.set_title('LWRR \u0394\u2192A', fontsize=12, fontweight='bold',
                        color=PAL['header'], pad=6)

    # Direction labels at bottom
    xleft, xright = ax_lw.get_xlim()
    ax_lw.text(xleft + (xright - xleft) * 0.02, n_total - 0.4,
               '\u2190B', fontsize=8, color=PAL['neg'], va='center', ha='left')
    ax_lw.text(xright - (xright - xleft) * 0.02, n_total - 0.4,
               'A\u2192', fontsize=8, color=PAL['pos'], va='center', ha='right')

    # Summary box below LWRR — single line
    raw_ok = (raw_pred_ab == 'A')
    ua_ok = (ua_pred_ab == 'A')
    ua_c = PAL['correct'] if ua_ok else PAL['wrong']
    raw_m = '\u2713' if raw_ok else '\u2717'
    ua_m = '\u2713' if ua_ok else '\u2717'
    corr = ' [Corrected]' if corrected else ''

    summary = (f'Raw: A={raw_sum_a:.0f} vs B={raw_sum_b:.0f} \u2192 '
               f'{raw_pred_ab} {raw_m}  |  '
               f'LWRR: \u0394={delta_towards_a:+.1f} \u2192 '
               f'{ua_pred_ab} {ua_m}{corr}')

    # Place summary centered between concept scores and LWRR sub-axes
    pos_cs = ax_cs.get_position()
    pos_lw = ax_lw.get_position()
    center_x = (pos_cs.x0 + pos_lw.x1) / 2
    fig.text(center_x, pos_lw.y0 - 0.018, summary,
             ha='center', va='top', fontsize=8.5, color=ua_c, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.25',
                       fc=PAL['A_light'] if ua_ok else '#FFEBEE',
                       ec=ua_c, lw=0.7, alpha=0.95))

    # ════════════════════════════════════════
    # Col 4: Methods + Probability Bars (no GT badge)
    # ════════════════════════════════════════
    ax_m = fig.add_subplot(gs[row_i, 4])

    bp = p.get('baseline_predictions', {})
    bprob = p.get('baseline_probabilities', {})

    methods = [
        ('Concept \u03a3',      p['raw_vlm_pred'],        None,                   False),
        ('ResNet Siamese',      bp.get('C0_ResNet', '?'),  bprob.get('C0_ResNet'), False),
        ('CLIP Siamese',        bp.get('C1_CLIP', '?'),    bprob.get('C1_CLIP'),   False),
        ('Seg. Regression',     bp.get('C2_SegReg', '?'),  bprob.get('C2_SegReg'), False),
        ('Zero-shot VLM',       bp.get('C3_VLM', '?'),     None,                   False),
        ('UrbanAlign',          p['urbanalign_pred'],       None,                   True),
    ]

    n_m = len(methods)
    ax_m.set_xlim(0, 10)
    ax_m.set_ylim(n_m - 0.5, -0.5)
    ax_m.axis('off')

    if row_i == 0:
        ax_m.set_title('Baselines', fontsize=12, fontweight='bold',
                       color=PAL['header'], pad=6)

    for mi, (mname, mpred_raw, mprob_raw, is_ours) in enumerate(methods):
        y_m = mi
        mpred_ab = remap_pred(mpred_raw, human) if mpred_raw != '?' else '?'
        is_correct = (mpred_ab == 'A')
        mc = PAL['correct'] if is_correct else PAL['wrong']
        mark = '\u2713' if is_correct else '\u2717'
        bg_c = PAL['A_light'] if is_correct else '#FFEBEE'

        # Method name
        nw = 'bold' if is_ours else 'normal'
        ns = 13 if is_ours else 12
        ax_m.text(0, y_m, mname, ha='left', va='center',
                  fontsize=ns, color=PAL['header'], fontweight=nw)

        # Stacked probability bar
        if mprob_raw and isinstance(mprob_raw, dict):
            remapped = remap_probs(mprob_raw, human)
            if remapped:
                pa = remapped.get('A', 0)
                pb = remapped.get('B', 0)
                pe = remapped.get('E', 0)
                bar_left = 4.0
                bar_w = 2.5
                bh = 0.32

                ax_m.barh(y_m, pa * bar_w, bh, left=bar_left,
                          color=PAL['conf_a'], alpha=0.8,
                          edgecolor='white', linewidth=0.3, zorder=3)
                ax_m.barh(y_m, pb * bar_w, bh, left=bar_left + pa * bar_w,
                          color=PAL['conf_b'], alpha=0.7,
                          edgecolor='white', linewidth=0.3, zorder=3)
                if pe > 0.01:
                    ax_m.barh(y_m, pe * bar_w, bh,
                              left=bar_left + (pa + pb) * bar_w,
                              color=PAL['conf_eq'], alpha=0.5,
                              edgecolor='white', linewidth=0.3, zorder=3)

                for seg_name, seg_val in [('A', pa), ('B', pb), ('E', pe)]:
                    if seg_val >= 0.15:
                        if seg_name == 'A':
                            sx = bar_left + pa * bar_w / 2
                        elif seg_name == 'B':
                            sx = bar_left + pa * bar_w + pb * bar_w / 2
                        else:
                            sx = bar_left + (pa + pb) * bar_w + pe * bar_w / 2
                        ax_m.text(sx, y_m, f'{seg_val:.0%}',
                                  ha='center', va='center', fontsize=8,
                                  color='white', fontweight='bold', zorder=4)

        # Prediction badge
        lw = 1.2 if is_ours else 0.7
        fs = 12.5 if is_ours else 11.5
        ax_m.text(8.5, y_m, f' {mpred_ab} {mark} ', ha='center', va='center',
                  fontsize=fs, color=mc, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.2', fc=bg_c, ec=mc,
                            lw=lw, alpha=0.95))

    # (legends moved to bottom of figure)

    # ── Export data ──
    cal_a = [a_raw[i] * weights[i] for i in range(n_d)]
    cal_b = [b_raw[i] * weights[i] for i in range(n_d)]
    export_data[cat] = {
        'image_a_is': human,
        'a_img': a_img_file, 'b_img': b_img_file,
        'raw_sum_a': float(raw_sum_a), 'raw_sum_b': float(raw_sum_b),
        'raw_vlm_pred': raw_pred_ab, 'urbanalign_pred': ua_pred_ab,
        'fitted_delta_towards_a': float(delta_towards_a),
        'intercept_towards_a': float(intercept_towards_a),
        'lwrr_corrected': corrected,
        'contributions_towards_a': {d: float(contrib_towards_a[i])
                                    for i, d in enumerate(dims)},
        'method_predictions': {
            m[0]: remap_pred(m[1], human) if m[1] != '?' else '?'
            for m in methods
        },
    }

# ── Bottom legend (shared for Concept A/B + Baseline conf.) ──
legend_y = 0.005
fs_leg = 11.5
items = [
    (PAL['A_bar'], 'Score A (Concept)'),
    (PAL['B_bar'], 'Score B (Concept)'),
    (PAL['conf_a'], 'conf. A (Baseline)'),
    (PAL['conf_b'], 'conf. B (Baseline)'),
    (PAL['conf_eq'], 'equal (Baseline)'),
]
n_items = len(items)
x_start, x_end = 0.15, 0.92
x_step = (x_end - x_start) / n_items
for idx, (col, label) in enumerate(items):
    x = x_start + idx * x_step
    fig.text(x, legend_y, '\u25a0 ', fontsize=fs_leg, color=col,
             ha='left', va='bottom', fontweight='bold')
    fig.text(x + 0.018, legend_y, label, fontsize=fs_leg,
             color=PAL['subtext'], ha='left', va='bottom')

# ── Row separators ──
for sep_i in range(1, n_rows):
    if sep_i >= len(img_a_axes) or sep_i - 1 >= len(img_a_axes):
        continue
    pos_cur = img_a_axes[sep_i].get_position()
    pos_prev = img_a_axes[sep_i - 1].get_position()
    y_mid = (pos_cur.y1 + pos_prev.y0) / 2
    fig.add_artist(plt.Line2D(
        [0.005, 0.995], [y_mid, y_mid],
        transform=fig.transFigure, color=PAL['grid'],
        linewidth=0.7, linestyle='-', zorder=0))

# ── Save ──
for ext, dpi in [('.pdf', 300), ('.png', 200)]:
    path = os.path.join(FIG_DIR, f'fig_qualitative_v5{ext}')
    fig.savefig(path, bbox_inches='tight', pad_inches=0.05, dpi=dpi,
                facecolor='white')
    print(f"Saved: {path}")

plt.close()

# ── Merge back to JSON ──
for cat in CATEGORIES:
    if cat not in export_data or cat not in data['pairs']:
        continue
    e = export_data[cat]
    p = data['pairs'][cat]
    p['intercept_towards_a'] = e['intercept_towards_a']

with open(DATA_FILE, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f"Updated: {DATA_FILE}")

# ── Summary ──
print("\n=== SUMMARY ===")
for cat in CATEGORIES:
    d = export_data.get(cat)
    if not d:
        continue
    tag = ' [CORRECTED]' if d['lwrr_corrected'] else ''
    print(f"  {CAT_DISPLAY[cat]:12s}{tag}")
    print(f"    Raw: A={d['raw_sum_a']:.0f} B={d['raw_sum_b']:.0f}"
          f" \u2192 {d['raw_vlm_pred']}")
    print(f"    LWRR: \u0394={d['fitted_delta_towards_a']:+.1f}"
          f" \u2192 {d['urbanalign_pred']}")
    for mn, mp in d['method_predictions'].items():
        ok = '\u2713' if mp == 'A' else '\u2717'
        print(f"    {mn:15s}: {mp} {ok}")
