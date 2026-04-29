"""
Qualitative Figure v4 — Compact ECCV layout.

Changes from v3:
  - Removed VLM concept scores column (raw sum shown as annotation)
  - Methods column now includes stacked probability bars for baselines
  - All fonts ≥ 11pt for print readability
  - No footer text (goes to LaTeX caption)
  - Wider gaps to prevent label/image overlap
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
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica Neue'],
    'axes.linewidth': 0.4,
    'axes.edgecolor': PAL['border'],
})


def norm10(val):
    return val / 10.0 if val > 10 else float(val)


def shorten(name, max_len=18):
    for old, new in [
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
    ]:
        name = name.replace(old, new)
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


# ══════════════════════════════════════
# Figure
# ══════════════════════════════════════
n_rows = 6
fig_w = 13
row_h = 2.5
fig_h = row_h * n_rows + 0.3

fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

# Cols: ImgA | ImgB | gap | LWRR | gap | Methods+Probs
gs = GridSpec(n_rows, 6, figure=fig,
             width_ratios=[0.85, 0.85, 0.18, 2.8, 0.10, 3.0],
             hspace=0.55, wspace=0.05,
             left=0.075, right=0.995, top=0.97, bottom=0.015)

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
    try:
        ax_a.imshow(mpimg.imread(os.path.join(FIG_DIR, a_img_file)))
    except Exception:
        ax_a.text(0.5, 0.5, 'N/A', ha='center', va='center',
                  transform=ax_a.transAxes, fontsize=14)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    for sp in ax_a.spines.values():
        sp.set_edgecolor(PAL['A']); sp.set_linewidth(3)

    ax_a.set_ylabel(CAT_DISPLAY[cat], fontsize=15, fontweight='bold',
                    rotation=0, labelpad=55, va='center', color=PAL['header'])
    ax_a.set_xlabel('A (Selected)', fontsize=11, color=PAL['A'],
                    fontweight='bold', labelpad=3)
    if row_i == 0:
        ax_a.set_title('Image A', fontsize=14, fontweight='bold',
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

    ax_b.set_xlabel('B', fontsize=11, color=PAL['subtext'],
                    fontweight='bold', labelpad=3)
    if row_i == 0:
        ax_b.set_title('Image B', fontsize=14, fontweight='bold',
                       color=PAL['header'], pad=8)

    # Gap
    fig.add_subplot(gs[row_i, 2]).axis('off')

    # ════════════════════════════════════════
    # Col 3: LWRR Contribution (diverging bars)
    # ════════════════════════════════════════
    ax_lw = fig.add_subplot(gs[row_i, 3])

    contrib_full = list(contrib_towards_a) + [intercept_towards_a]
    y_full = np.arange(n_d + 1)
    labels_lw = [shorten(d, 16) for d in dims] + ['Intercept']

    colors_lw = [PAL['pos_bar'] if v >= 0 else PAL['neg_bar'] for v in contrib_full]
    bar_h = 0.50
    ax_lw.barh(y_full, contrib_full, bar_h, color=colors_lw, alpha=0.85,
               edgecolor='white', linewidth=0.5, zorder=3)

    x_range = max(max(contrib_full) - min(contrib_full), 1)
    for i, val in enumerate(contrib_full):
        ha = 'left' if val >= 0 else 'right'
        off = x_range * 0.03 if val >= 0 else -x_range * 0.03
        c = PAL['pos'] if val >= 0 else PAL['neg']
        ax_lw.text(val + off, y_full[i], f'{val:+.1f}',
                   va='center', ha=ha, fontsize=10, color=c, fontweight='bold')

    ax_lw.axhline(y=n_d - 0.5, color=PAL['border'], linewidth=0.6,
                  linestyle='--', zorder=1)
    ax_lw.axvline(x=0, color=PAL['subtext'], linewidth=0.6, zorder=2)

    ax_lw.set_yticks(y_full)
    ax_lw.set_yticklabels(labels_lw, fontsize=11)
    ax_lw.invert_yaxis()

    xmin, xmax = min(contrib_full), max(contrib_full)
    pad = max(abs(xmin), abs(xmax)) * 0.35
    ax_lw.set_xlim(min(xmin - pad, -0.5), max(xmax + pad, 0.5))

    for sp in ['top', 'right']:
        ax_lw.spines[sp].set_visible(False)
    ax_lw.spines['left'].set_linewidth(0.3)
    ax_lw.spines['bottom'].set_linewidth(0.3)
    ax_lw.tick_params(axis='y', length=0, pad=5)
    ax_lw.tick_params(axis='x', labelsize=9, colors=PAL['muted'])

    if row_i == 0:
        ax_lw.set_title('LWRR Contribution (\u0394 towards A)',
                        fontsize=14, fontweight='bold', color=PAL['header'], pad=8)

    # Direction labels
    xleft, xright = ax_lw.get_xlim()
    ax_lw.text(xleft + (xright - xleft) * 0.02, n_d + 0.95,
               '\u2190 B', fontsize=9, color=PAL['neg'], va='center', ha='left')
    ax_lw.text(xright - (xright - xleft) * 0.02, n_d + 0.95,
               'A \u2192', fontsize=9, color=PAL['pos'], va='center', ha='right')

    # Summary: Raw VLM + LWRR prediction (two-line badge)
    raw_ok = (raw_pred_ab == 'A')
    ua_ok = (ua_pred_ab == 'A')
    raw_c = PAL['correct'] if raw_ok else PAL['wrong']
    ua_c = PAL['correct'] if ua_ok else PAL['wrong']
    raw_m = '\u2713' if raw_ok else '\u2717'
    ua_m = '\u2713' if ua_ok else '\u2717'
    corr = '  [Corrected]' if corrected else ''

    line1 = f'Raw: \u03a3A={raw_sum_a:.0f}, B={raw_sum_b:.0f} \u2192 {raw_pred_ab} {raw_m}'
    line2 = f'LWRR: \u0394={delta_towards_a:+.1f} \u2192 {ua_pred_ab} {ua_m}{corr}'
    summary = f'{line1}\n{line2}'

    ax_lw.text(0.5, -0.10, summary, transform=ax_lw.transAxes,
               ha='center', va='top', fontsize=10.5, color=ua_c, fontweight='bold',
               linespacing=1.4,
               bbox=dict(boxstyle='round,pad=0.35',
                         fc=PAL['A_light'] if ua_ok else '#FFEBEE',
                         ec=ua_c, lw=0.8, alpha=0.95))

    # Gap
    fig.add_subplot(gs[row_i, 4]).axis('off')

    # ════════════════════════════════════════
    # Col 5: Methods + Probability Bars
    # ════════════════════════════════════════
    ax_m = fig.add_subplot(gs[row_i, 5])

    bp = p.get('baseline_predictions', {})
    bprob = p.get('baseline_probabilities', {})

    # (name, pred_raw, prob_dict_raw, is_ours)
    methods = [
        ('Concept \u03a3',  p['raw_vlm_pred'],        None,                      False),
        ('C0 ResNet',       bp.get('C0_ResNet', '?'),  bprob.get('C0_ResNet'),    False),
        ('C1 CLIP',         bp.get('C1_CLIP', '?'),    bprob.get('C1_CLIP'),      False),
        ('C2 SegReg',       bp.get('C2_SegReg', '?'),  bprob.get('C2_SegReg'),    False),
        ('C3 VLM',          bp.get('C3_VLM', '?'),     None,                      False),
        ('UrbanAlign',      p['urbanalign_pred'],       None,                      True),
    ]

    n_m = len(methods)

    # Use data coordinates for precise layout
    ax_m.set_xlim(0, 10)
    ax_m.set_ylim(n_m - 0.5, -0.8)  # inverted, extra space at top for GT badge
    ax_m.axis('off')

    if row_i == 0:
        ax_m.set_title('Method Comparison', fontsize=14,
                       fontweight='bold', color=PAL['header'], pad=8)

    # GT badge
    ax_m.text(5, -0.45, 'GT: A', ha='center', va='center',
              fontsize=12, color=PAL['A'], fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.3', fc=PAL['A_light'],
                        ec=PAL['A'], lw=0.8))

    for mi, (mname, mpred_raw, mprob_raw, is_ours) in enumerate(methods):
        y_m = mi
        mpred_ab = remap_pred(mpred_raw, human) if mpred_raw != '?' else '?'
        is_correct = (mpred_ab == 'A')
        mc = PAL['correct'] if is_correct else PAL['wrong']
        mark = '\u2713' if is_correct else '\u2717'
        bg_c = PAL['A_light'] if is_correct else '#FFEBEE'

        # Method name (left, x=0..3.2)
        nw = 'bold' if is_ours else 'normal'
        ns = 12.5 if is_ours else 11.5
        ax_m.text(0, y_m, mname, ha='left', va='center',
                  fontsize=ns, color=PAL['header'], fontweight=nw)

        # Stacked probability bar (middle, x=3.5..7.0)
        if mprob_raw and isinstance(mprob_raw, dict):
            remapped = remap_probs(mprob_raw, human)
            if remapped:
                pa = remapped.get('A', 0)
                pb = remapped.get('B', 0)
                pe = remapped.get('E', 0)
                bar_left = 3.5
                bar_w = 3.5  # total bar width in data coords
                bh = 0.38

                # Stacked segments
                ax_m.barh(y_m, pa * bar_w, bh, left=bar_left,
                          color=PAL['A_bar'], alpha=0.8,
                          edgecolor='white', linewidth=0.3, zorder=3)
                ax_m.barh(y_m, pb * bar_w, bh, left=bar_left + pa * bar_w,
                          color=PAL['B_bar'], alpha=0.65,
                          edgecolor='white', linewidth=0.3, zorder=3)
                if pe > 0.01:
                    ax_m.barh(y_m, pe * bar_w, bh,
                              left=bar_left + (pa + pb) * bar_w,
                              color=PAL['equal_bar'], alpha=0.5,
                              edgecolor='white', linewidth=0.3, zorder=3)

                # Probability text on largest segment
                segments = [('A', pa), ('B', pb), ('E', pe)]
                for seg_name, seg_val in segments:
                    if seg_val >= 0.15:  # only label segments ≥ 15%
                        if seg_name == 'A':
                            seg_x = bar_left + pa * bar_w / 2
                            seg_c = 'white' if pa > 0.3 else PAL['A']
                        elif seg_name == 'B':
                            seg_x = bar_left + pa * bar_w + pb * bar_w / 2
                            seg_c = 'white' if pb > 0.3 else PAL['B']
                        else:
                            seg_x = bar_left + (pa + pb) * bar_w + pe * bar_w / 2
                            seg_c = PAL['subtext']
                        ax_m.text(seg_x, y_m, f'{seg_val:.0%}',
                                  ha='center', va='center', fontsize=9,
                                  color=seg_c, fontweight='bold', zorder=4)

        # Prediction badge (right, x=8..9.5)
        lw = 1.3 if is_ours else 0.7
        fs = 12 if is_ours else 11
        ax_m.text(8.8, y_m, f' {mpred_ab} {mark} ', ha='center', va='center',
                  fontsize=fs, color=mc, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.22', fc=bg_c, ec=mc,
                            lw=lw, alpha=0.95))

    # Probability legend (first row only)
    if row_i == 0:
        for lx, lc, lt in [(4.0, PAL['A_bar'], 'P(A)'),
                            (5.5, PAL['B_bar'], 'P(B)'),
                            (6.8, PAL['equal_bar'], 'P(E)')]:
            ax_m.plot(lx - 0.3, n_m + 0.0, 's', color=lc, markersize=5,
                      alpha=0.7)
            ax_m.text(lx, n_m + 0.0, lt, fontsize=9, color=PAL['subtext'],
                      va='center')

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

# ── Row separators ──
all_axes = fig.get_axes()
# Each row has 5 drawn axes (ImgA, ImgB, gap, LWRR, gap, Methods) = 6 axes
axes_per_row = 6
for sep_i in range(1, n_rows):
    ax_cur = all_axes[sep_i * axes_per_row]       # ImgA of current row
    ax_prev = all_axes[(sep_i - 1) * axes_per_row]  # ImgA of prev row
    pos_cur = ax_cur.get_position()
    pos_prev = ax_prev.get_position()
    y_mid = (pos_cur.y1 + pos_prev.y0) / 2
    fig.add_artist(plt.Line2D(
        [0.075, 0.995], [y_mid, y_mid],
        transform=fig.transFigure, color=PAL['grid'],
        linewidth=0.7, linestyle='-', zorder=0))

# ── Save ──
for ext, dpi in [('.pdf', 300), ('.png', 200)]:
    path = os.path.join(FIG_DIR, f'fig_qualitative_v4{ext}')
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=dpi,
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
