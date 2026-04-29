"""
Qualitative Figure v3 — Publication-ready for ECCV 2026.

Key design changes from v2:
  - All fonts ~50-80% larger for readability at print scale
  - LWRR column: contribution bars (diverging from 0) instead of paired A/B bars
  - Method column: colored badges instead of plain text
  - Professional color palette matching fig1/fig2
  - Better spacing, subtle gridlines, visual hierarchy
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

# ── Publication Color Palette (harmonized with fig1/fig2) ──
PAL = {
    'A':          '#2E7D32',   # Green for A (selected / GT)
    'A_bar':      '#43A047',   # Green bar fill
    'A_light':    '#E8F5E9',   # Light green background
    'B':          '#1565C0',   # Blue for B
    'B_bar':      '#42A5F5',   # Blue bar fill
    'B_light':    '#E3F2FD',   # Light blue background
    'pos':        '#2E7D32',   # Green: supports A
    'neg':        '#E65100',   # Orange: supports B
    'pos_bar':    '#66BB6A',   # Green bar
    'neg_bar':    '#FF8A65',   # Orange bar
    'correct':    '#2E7D32',   # Green checkmark
    'wrong':      '#C62828',   # Red cross
    'header':     '#212121',   # Main text
    'subtext':    '#616161',   # Secondary text
    'muted':      '#9E9E9E',   # Muted axis text
    'border':     '#E0E0E0',   # Border
    'grid':       '#EEEEEE',   # Grid lines
}

# ── Style ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica Neue'],  # DejaVu first for unicode ✓✗
    'axes.linewidth': 0.4,
    'axes.edgecolor': PAL['border'],
})

# ── Helpers ──
def norm10(val):
    return val / 10.0 if val > 10 else float(val)


def shorten(name, max_len=18):
    # Always apply these abbreviations for consistency
    abbrevs = [
        ('Maintenance', 'Maint.'), ('Infrastructure', 'Infra.'),
        ('Condition', 'Cond.'), ('Quality', 'Ql.'),
        ('Presence', 'Pres.'), ('Adequacy', 'Adeq.'),
        ('Pedestrian', 'Ped.'), ('Complexity', 'Cplx.'),
        ('Architectural', 'Arch.'), ('Diversity', 'Div.'),
        ('Vibrancy', 'Vibr.'), ('Visibility', 'Visib.'),
        ('Landscaping', 'Land.'), ('Furniture', 'Furn.'),
        ('Enclosure', 'Encl.'), ('Variety', 'Var.'),
        ('Activity', 'Act.'), ('Vegetation', 'Veg.'),
        ('Cleanliness', 'Cln.'), ('Modernity', 'Mod.'),
        ('and ', '& '), ('Greenery', 'Green.'),
        ('Coordination', 'Coord.'), ('Coherence', 'Coher.'),
        ('Elements', 'Elem.'), ('Natural', 'Nat.'),
        ('Lighting', 'Light.'), ('Street', 'St.'),
        ('Building', 'Bldg.'), ('Color', 'Col.'),
        ('Visual', 'Vis.'), ('Spatial', 'Sp.'),
        ('Public', 'Pub.'), ('Commercial', 'Comm.'),
        ('Human', 'Hum.'), ('Pavement', 'Pvmt.'),
        ('Amenities', 'Amen.'), ('Signage', 'Sign.'),
        ('Integrity', 'Integ.'),
    ]
    for old, new in abbrevs:
        name = name.replace(old, new)
    # Final truncation if still too long
    if len(name) > max_len:
        name = name[:max_len - 1] + '.'
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


# ══════════════════════════════════════════════════════════════
# Figure layout
# ══════════════════════════════════════════════════════════════
n_rows = 6
fig_w = 16
row_h = 2.3
fig_h = row_h * n_rows + 0.7

fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

# Columns: ImgA | ImgB | gap | ConceptScores | gap | LWRR | gap | Methods
gs = GridSpec(n_rows, 8, figure=fig,
             width_ratios=[0.80, 0.80, 0.20, 2.5, 0.06, 2.1, 0.04, 1.4],
             hspace=0.55, wspace=0.06,
             left=0.068, right=0.995, top=0.965, bottom=0.042)

export_data = {}

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
    y = np.arange(n_d)

    # ── A/B mapping ──
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

    # LWRR math
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
    try:
        ax_a.imshow(mpimg.imread(os.path.join(FIG_DIR, a_img_file)))
    except Exception:
        ax_a.text(0.5, 0.5, 'N/A', ha='center', va='center',
                  transform=ax_a.transAxes, fontsize=13)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    for sp in ax_a.spines.values():
        sp.set_edgecolor(PAL['A']); sp.set_linewidth(3)

    ax_a.set_ylabel(CAT_DISPLAY[cat], fontsize=14, fontweight='bold',
                    rotation=0, labelpad=52, va='center', color=PAL['header'])
    ax_a.set_xlabel('A (Selected)', fontsize=10, color=PAL['A'],
                    fontweight='bold', labelpad=3)
    if row_i == 0:
        ax_a.set_title('Image A', fontsize=12.5, fontweight='bold',
                       color=PAL['header'], pad=8)

    # ════════════════════════════════════════
    # Col 1: Image B
    # ════════════════════════════════════════
    ax_b = fig.add_subplot(gs[row_i, 1])
    try:
        ax_b.imshow(mpimg.imread(os.path.join(FIG_DIR, b_img_file)))
    except Exception:
        ax_b.text(0.5, 0.5, 'N/A', ha='center', va='center',
                  transform=ax_b.transAxes, fontsize=13)
    ax_b.set_xticks([]); ax_b.set_yticks([])
    for sp in ax_b.spines.values():
        sp.set_edgecolor(PAL['border']); sp.set_linewidth(1)

    ax_b.set_xlabel('B', fontsize=10, color=PAL['subtext'],
                    fontweight='bold', labelpad=3)
    if row_i == 0:
        ax_b.set_title('Image B', fontsize=12.5, fontweight='bold',
                       color=PAL['header'], pad=8)

    # Gap
    fig.add_subplot(gs[row_i, 2]).axis('off')

    # ════════════════════════════════════════
    # Col 3: VLM Concept Scores (paired bars)
    # ════════════════════════════════════════
    ax_sc = fig.add_subplot(gs[row_i, 3])
    bh = 0.25

    ax_sc.barh(y - bh / 2, a_vals, bh, color=PAL['A_bar'], alpha=0.85,
               edgecolor='white', linewidth=0.5, label='A', zorder=3)
    ax_sc.barh(y + bh / 2, b_vals, bh, color=PAL['B_bar'], alpha=0.60,
               edgecolor='white', linewidth=0.5, label='B', zorder=3)

    for i in range(n_d):
        ax_sc.text(a_vals[i] + 0.15, y[i] - bh / 2, f'{a_vals[i]:.0f}',
                   va='center', ha='left', fontsize=8.5, color=PAL['A'], fontweight='bold')
        ax_sc.text(b_vals[i] + 0.15, y[i] + bh / 2, f'{b_vals[i]:.0f}',
                   va='center', ha='left', fontsize=8.5, color=PAL['B'], fontweight='bold')

    labels_sc = [shorten(d, 16) for d in dims]
    ax_sc.set_yticks(y)
    ax_sc.set_yticklabels(labels_sc, fontsize=9)
    ax_sc.set_xlim(0, 11)
    ax_sc.set_xticks([0, 2, 4, 6, 8, 10])
    ax_sc.invert_yaxis()
    for sp in ['top', 'right']:
        ax_sc.spines[sp].set_visible(False)
    ax_sc.spines['left'].set_linewidth(0.3)
    ax_sc.spines['bottom'].set_linewidth(0.3)
    ax_sc.tick_params(axis='y', length=0, pad=4)
    ax_sc.tick_params(axis='x', labelsize=8, colors=PAL['muted'])
    ax_sc.xaxis.grid(True, linewidth=0.3, color=PAL['grid'], zorder=0)

    if row_i == 0:
        ax_sc.set_title('VLM Concept Scores (0\u201310)', fontsize=12.5,
                        fontweight='bold', color=PAL['header'], pad=8)
        ax_sc.legend(fontsize=9.5, loc='lower right', frameon=True,
                     framealpha=0.95, edgecolor=PAL['border'], ncol=2,
                     handlelength=1.2, handleheight=0.8)

    # Summary badge
    raw_ok = (raw_pred_ab == 'A')
    rc = PAL['correct'] if raw_ok else PAL['wrong']
    rm = '\u2713' if raw_ok else '\u2717'
    raw_text = f'\u03a3: A={raw_sum_a:.0f}  B={raw_sum_b:.0f}  \u2192  {raw_pred_ab} {rm}'
    ax_sc.text(0.5, -0.13, raw_text, transform=ax_sc.transAxes,
               ha='center', va='top', fontsize=10.5, color=rc, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3',
                         fc=PAL['A_light'] if raw_ok else '#FFEBEE',
                         ec=rc, lw=0.8, alpha=0.95))

    # Gap
    fig.add_subplot(gs[row_i, 4]).axis('off')

    # ════════════════════════════════════════
    # Col 5: LWRR Contribution (diverging bars)
    # ════════════════════════════════════════
    ax_lw = fig.add_subplot(gs[row_i, 5])

    contrib_full = list(contrib_towards_a) + [intercept_towards_a]
    y_full = np.arange(n_d + 1)
    labels_lw = [shorten(d, 14) for d in dims] + ['Intercept']

    colors_lw = [PAL['pos_bar'] if v >= 0 else PAL['neg_bar'] for v in contrib_full]
    bar_h = 0.45
    ax_lw.barh(y_full, contrib_full, bar_h, color=colors_lw, alpha=0.85,
               edgecolor='white', linewidth=0.5, zorder=3)

    # Value annotations
    x_range = max(max(contrib_full) - min(contrib_full), 1)
    for i, val in enumerate(contrib_full):
        ha = 'left' if val >= 0 else 'right'
        off = x_range * 0.03 if val >= 0 else -x_range * 0.03
        c = PAL['pos'] if val >= 0 else PAL['neg']
        ax_lw.text(val + off, y_full[i], f'{val:+.1f}',
                   va='center', ha=ha, fontsize=8.5, color=c, fontweight='bold')

    # Separator before intercept
    ax_lw.axhline(y=n_d - 0.5, color=PAL['border'], linewidth=0.6,
                  linestyle='--', zorder=1)
    # Zero line
    ax_lw.axvline(x=0, color=PAL['subtext'], linewidth=0.6, zorder=2)

    ax_lw.set_yticks(y_full)
    ax_lw.set_yticklabels(labels_lw, fontsize=9)
    ax_lw.invert_yaxis()

    xmin, xmax = min(contrib_full), max(contrib_full)
    pad = max(abs(xmin), abs(xmax)) * 0.35
    ax_lw.set_xlim(min(xmin - pad, -0.5), max(xmax + pad, 0.5))

    for sp in ['top', 'right']:
        ax_lw.spines[sp].set_visible(False)
    ax_lw.spines['left'].set_linewidth(0.3)
    ax_lw.spines['bottom'].set_linewidth(0.3)
    ax_lw.tick_params(axis='y', length=0, pad=4)
    ax_lw.tick_params(axis='x', labelsize=8, colors=PAL['muted'])

    # Axis direction labels (first row only)
    if row_i == 0:
        ax_lw.set_title('LWRR Contribution (\u0394 towards A)',
                        fontsize=12.5, fontweight='bold', color=PAL['header'], pad=8)

    # Direction indicators at x-axis level
    xleft, xright = ax_lw.get_xlim()
    ax_lw.text(xleft + (xright - xleft) * 0.02, n_d + 0.95,
               '\u2190 B', fontsize=8, color=PAL['neg'], va='center', ha='left')
    ax_lw.text(xright - (xright - xleft) * 0.02, n_d + 0.95,
               'A \u2192', fontsize=8, color=PAL['pos'], va='center', ha='right')

    # Summary badge
    ua_ok = (ua_pred_ab == 'A')
    uc = PAL['correct'] if ua_ok else PAL['wrong']
    um = '\u2713' if ua_ok else '\u2717'
    corr = '  [Corrected]' if corrected else ''
    ua_text = f'\u0394={delta_towards_a:+.1f}  \u2192  {ua_pred_ab} {um}{corr}'
    ax_lw.text(0.5, -0.13, ua_text, transform=ax_lw.transAxes,
               ha='center', va='top', fontsize=10.5, color=uc, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3',
                         fc=PAL['A_light'] if ua_ok else '#FFEBEE',
                         ec=uc, lw=0.8, alpha=0.95))

    # Gap
    fig.add_subplot(gs[row_i, 6]).axis('off')

    # ════════════════════════════════════════
    # Col 7: Method Comparison
    # ════════════════════════════════════════
    ax_m = fig.add_subplot(gs[row_i, 7])
    ax_m.axis('off')
    ax_m.set_xlim(0, 1)
    ax_m.set_ylim(0, 1)

    if row_i == 0:
        ax_m.set_title('Method Comparison', fontsize=12.5,
                       fontweight='bold', color=PAL['header'], pad=8)

    bp = p.get('baseline_predictions', {})
    methods = [
        ('Concept \u03a3',  p['raw_vlm_pred'],         False),
        ('C0 ResNet',       bp.get('C0_ResNet', '?'),   False),
        ('C1 CLIP',         bp.get('C1_CLIP', '?'),     False),
        ('C2 SegReg',       bp.get('C2_SegReg', '?'),   False),
        ('C3 VLM',          bp.get('C3_VLM', '?'),      False),
        ('UrbanAlign',      p['urbanalign_pred'],        True),
    ]

    n_m = len(methods)
    y_sp = 0.78 / n_m
    y_start = 0.88

    # GT badge
    ax_m.text(0.5, 0.99, 'GT: A', ha='center', va='top',
              fontsize=10.5, color=PAL['A'], fontweight='bold',
              transform=ax_m.transAxes,
              bbox=dict(boxstyle='round,pad=0.3', fc=PAL['A_light'],
                        ec=PAL['A'], lw=0.8))

    for mi, (mname, mpred_raw, is_ours) in enumerate(methods):
        yp = y_start - mi * y_sp
        mpred_ab = remap_pred(mpred_raw, human) if mpred_raw != '?' else '?'
        is_correct = (mpred_ab == 'A')
        mark = '\u2713' if is_correct else '\u2717'
        mc = PAL['correct'] if is_correct else PAL['wrong']
        bg = PAL['A_light'] if is_correct else '#FFEBEE'

        nw = 'bold' if is_ours else 'normal'
        ns = 11.5 if is_ours else 10
        bs = 11 if is_ours else 10

        # Method name (left-aligned)
        ax_m.text(0.0, yp, mname, ha='left', va='center',
                  fontsize=ns, color=PAL['header'], fontweight=nw,
                  transform=ax_m.transAxes)

        # Prediction badge (right-aligned, colored pill)
        lw = 1.3 if is_ours else 0.7
        ax_m.text(0.88, yp, f' {mpred_ab} {mark} ', ha='center', va='center',
                  fontsize=bs, color=mc, fontweight='bold',
                  transform=ax_m.transAxes,
                  bbox=dict(boxstyle='round,pad=0.22', fc=bg, ec=mc, lw=lw,
                            alpha=0.95))

    # ── Export data ──
    cal_a = [a_raw[i] * weights[i] for i in range(n_d)]
    cal_b = [b_raw[i] * weights[i] for i in range(n_d)]
    export_data[cat] = {
        'image_a_is': human,
        'a_img': a_img_file,
        'b_img': b_img_file,
        'raw_sum_a': float(raw_sum_a),
        'raw_sum_b': float(raw_sum_b),
        'raw_vlm_pred': raw_pred_ab,
        'urbanalign_pred': ua_pred_ab,
        'fitted_delta_towards_a': float(delta_towards_a),
        'intercept_towards_a': float(intercept_towards_a),
        'lwrr_corrected': corrected,
        'dimension_scores_a': {d: norm10(a_scores_raw[d]) for d in dims},
        'dimension_scores_b': {d: norm10(b_scores_raw[d]) for d in dims},
        'lwrr_weights': p['lwrr_weights'],
        'calibrated_scores_a': {d: float(cal_a[i]) for i, d in enumerate(dims)},
        'calibrated_scores_b': {d: float(cal_b[i]) for i, d in enumerate(dims)},
        'calibrated_sum_a': float(sum(cal_a) + intercept_towards_a),
        'calibrated_sum_b': float(sum(cal_b)),
        'contributions_towards_a': {d: float(contrib_towards_a[i])
                                    for i, d in enumerate(dims)},
        'raw_scores_a': {d: float(a_raw[i]) for i, d in enumerate(dims)},
        'raw_scores_b': {d: float(b_raw[i]) for i, d in enumerate(dims)},
        'method_predictions': {
            m[0]: remap_pred(m[1], human) if m[1] != '?' else '?'
            for m in methods
        },
    }

# ── Row separators (thin horizontal lines between categories) ──
for row_i in range(1, n_rows):
    # Get the top of current row and bottom of previous row from axes positions
    ax_cur = fig.axes[row_i * 7]  # first axis in each row (Image A) = every 7 axes
    ax_prev_sc = fig.axes[(row_i - 1) * 7 + 2]  # concept scores of prev row
    pos_cur = ax_cur.get_position()
    pos_prev = ax_prev_sc.get_position()
    y_mid = (pos_cur.y1 + pos_prev.y0) / 2
    fig.add_artist(plt.Line2D(
        [0.068, 0.995], [y_mid, y_mid],
        transform=fig.transFigure, color=PAL['grid'],
        linewidth=0.6, linestyle='-', zorder=0))

# ── Footer ──
fig.text(0.5, 0.006,
         'A = human-selected (green border).  B = not selected.  '
         '\u03a3 = concept score sum.  '
         'LWRR \u0394: green \u2192 supports A, orange \u2192 supports B.  '
         '[Corrected] = LWRR flipped an incorrect raw prediction.',
         ha='center', va='bottom', fontsize=9, color=PAL['subtext'],
         style='italic')

# ── Save ──
for ext, dpi in [('.pdf', 300), ('.png', 200)]:
    path = os.path.join(FIG_DIR, f'fig_qualitative_v3{ext}')
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=dpi,
                facecolor='white')
    print(f"Saved: {path}")

plt.close()

# ── Merge computed fields back into fig3_data.json ──
for cat in CATEGORIES:
    if cat not in export_data or cat not in data['pairs']:
        continue
    e = export_data[cat]
    p = data['pairs'][cat]
    p['intercept_towards_a'] = e['intercept_towards_a']
    p['calibrated_sum_a'] = e['calibrated_sum_a']
    p['calibrated_sum_b'] = e['calibrated_sum_b']

with open(DATA_FILE, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f"Updated: {DATA_FILE}")

# ── Summary ──
print("\n=== SUMMARY (A = Selected, B = Not Selected) ===")
for cat in CATEGORIES:
    d = export_data.get(cat)
    if not d:
        continue
    tag = ' [CORRECTED]' if d['lwrr_corrected'] else ''
    print(f"\n  {CAT_DISPLAY[cat]:12s}{tag}")
    print(f"    Concept: A={d['raw_sum_a']:.0f}  B={d['raw_sum_b']:.0f}  "
          f"\u2192 {d['raw_vlm_pred']}")
    print(f"    LWRR:    \u0394={d['fitted_delta_towards_a']:+.1f}  "
          f"\u2192 {d['urbanalign_pred']}")
    for mname, mpred in d['method_predictions'].items():
        ok = '\u2713' if mpred == 'A' else '\u2717'
        print(f"    {mname:15s}: {mpred} {ok}")
