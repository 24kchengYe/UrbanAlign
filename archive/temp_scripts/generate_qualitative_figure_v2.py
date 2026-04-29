"""
Qualitative Figure v5.

Key design:
  - A = Human Selected (always first), B = Not Selected (always second)
  - GT is always A, so correct predictions = "A", wrong = "B"
  - Raw VLM total = mean of shown dimension scores (self-consistent)
  - LWRR contribution towards A (= towards GT)
  - C3 VLM labelled "(direct)" — no semantic concepts
  - Baseline probabilities shown
  - Intercept saved to JSON only
"""

import json
import os
import sys
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

# ── Colors ──
C_A = '#2e7d32'
C_B = '#1565c0'
C_GT = '#1a9850'
C_POS = '#2e7d32'
C_NEG = '#c62828'
C_HDR = '#212121'
C_OK = '#2e7d32'
C_WRONG = '#c62828'


def norm10(val):
    return val / 10.0 if val > 10 else float(val)


def shorten(name):
    for old, new in [
        ('Maintenance', 'Mnt.'), ('Infrastructure', 'Infra.'),
        ('Condition', 'Cond.'), ('Quality', 'Ql.'),
        ('Presence', 'Pres.'), ('Adequacy', 'Adeq.'),
        ('Pedestrian', 'Ped.'), ('Complexity', 'Cpx.'),
        ('Architectural', 'Arch.'), ('Diversity', 'Div.'),
        ('Vibrancy', 'Vibr.'), ('Visibility', 'Vis.'),
        ('Landscaping', 'Land.'), ('Furniture', 'Furn.'),
        ('Enclosure', 'Encl.'), ('Variety', 'Var.'),
        ('Activity', 'Act.'), ('Vegetation', 'Veg.'),
        ('Cleanliness', 'Cln.'), ('Modernity', 'Mod.'),
        ('and ', '& '), ('Greenery', 'Green.'),
        ('Coordination', 'Coord.'), ('Coherence', 'Coher.'),
        ('Elements', 'Elem.'), ('Natural', 'Nat.'),
        ('Lighting', 'Light.'), ('Street', 'St.'),
        ('Building', 'Bld.'), ('Color', 'Col.'),
        ('Visual', 'Vis.'), ('Spatial', 'Sp.'),
        ('Public', 'Pub.'), ('Commercial', 'Comm.'),
        ('Human', 'Hum.'),
    ]:
        name = name.replace(old, new)
    return name


def remap_pred(original_pred, human_winner):
    """Remap left/right prediction to A/B in our convention.
    A = human_winner side, B = the other side.
    So if human='right' and pred='right' → 'A' (correct).
       if human='right' and pred='left'  → 'B' (wrong).
    """
    if original_pred == 'equal':
        return 'E'
    if original_pred == human_winner:
        return 'A'   # predicted the selected side → A
    else:
        return 'B'   # predicted the other side → B


def remap_probs(probs_dict, human_winner):
    """Remap prob_left/prob_right to prob_A/prob_B.
    A = human_winner side."""
    if not probs_dict:
        return None
    if human_winner == 'left':
        return {'A': probs_dict.get('left', 0), 'B': probs_dict.get('right', 0),
                'E': probs_dict.get('equal', 0)}
    else:
        return {'A': probs_dict.get('right', 0), 'B': probs_dict.get('left', 0),
                'E': probs_dict.get('equal', 0)}


# ── Figure ──
n_rows = 6
fig_w = 24
row_h = 2.8
fig_h = row_h * n_rows + 1.5

fig = plt.figure(figsize=(fig_w, fig_h))

gs = GridSpec(n_rows, 8, figure=fig,
             width_ratios=[0.85, 0.85, 0.03, 2.2, 0.06, 2.0, 0.06, 2.0],
             hspace=0.65, wspace=0.08,
             left=0.065, right=0.995, top=0.955, bottom=0.035)

export_data = {}

for row_i, cat in enumerate(CATEGORIES):
    p = data['pairs'].get(cat)
    if not p:
        continue

    human = p['human_winner']  # 'left' or 'right'
    corrected = p['lwrr_corrected']
    dims = p['dimension_names']
    weights = np.array(p['lwrr_weights'])
    fitted_delta = p['fitted_ts_delta']
    n_d = len(dims)
    y = np.arange(n_d)

    # ── A = Selected (human winner), B = Not Selected ──
    if human == 'left':
        a_scores_raw = p['dimension_scores_left']
        b_scores_raw = p['dimension_scores_right']
        a_img_file = f"{cat}_left_{p['left_id']}.jpg"
        b_img_file = f"{cat}_right_{p['right_id']}.jpg"
        sign = 1   # s_diff = left - right = A - B
    else:
        a_scores_raw = p['dimension_scores_right']
        b_scores_raw = p['dimension_scores_left']
        a_img_file = f"{cat}_right_{p['right_id']}.jpg"
        b_img_file = f"{cat}_left_{p['left_id']}.jpg"
        sign = -1  # s_diff = left - right = B - A → flip

    # Normalized scores (0-10) for display in Raw column
    a_vals = [norm10(a_scores_raw[d]) for d in dims]
    b_vals = [norm10(b_scores_raw[d]) for d in dims]
    # Raw scores (original scale) for LWRR column — matches what Ridge was trained on
    a_raw = [float(a_scores_raw[d]) for d in dims]
    b_raw = [float(b_scores_raw[d]) for d in dims]

    # Raw VLM total = sum of normalized dimension scores (all on 1-10 scale)
    raw_sum_a = sum(a_vals)
    raw_sum_b = sum(b_vals)
    # Raw VLM prediction remapped to A/B
    raw_pred_ab = remap_pred(p['raw_vlm_pred'], human)

    # UrbanAlign prediction
    ua_pred_ab = remap_pred(p['urbanalign_pred'], human)

    # ── Col 0: Image A (Selected) ──
    ax_a = fig.add_subplot(gs[row_i, 0])
    try:
        ax_a.imshow(mpimg.imread(os.path.join(FIG_DIR, a_img_file)))
    except Exception:
        ax_a.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_a.transAxes)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    for sp in ax_a.spines.values():
        sp.set_edgecolor(C_GT); sp.set_linewidth(3.5)
    ax_a.set_xlabel('A  (Selected \u2713)', fontsize=9, color=C_GT, fontweight='bold')
    ax_a.set_ylabel(CAT_DISPLAY[cat], fontsize=12.5, fontweight='bold',
                    rotation=0, labelpad=48, va='center', color=C_HDR)
    if row_i == 0:
        ax_a.set_title('Image A\n(Human Selected)', fontsize=10, fontweight='bold', color=C_HDR, pad=6)

    # ── Col 1: Image B (Not Selected) ──
    ax_b = fig.add_subplot(gs[row_i, 1])
    try:
        ax_b.imshow(mpimg.imread(os.path.join(FIG_DIR, b_img_file)))
    except Exception:
        ax_b.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_b.transAxes)
    ax_b.set_xticks([]); ax_b.set_yticks([])
    for sp in ax_b.spines.values():
        sp.set_edgecolor('#bbb'); sp.set_linewidth(1)
    ax_b.set_xlabel('B', fontsize=9, color='#666', fontweight='bold')
    if row_i == 0:
        ax_b.set_title('Image B\n(Not Selected)', fontsize=10, fontweight='bold', color=C_HDR, pad=6)

    # ── Col 2: Gap ──
    fig.add_subplot(gs[row_i, 2]).axis('off')

    # ── Col 3: Dimension Scores + Raw VLM aggregate ──
    ax_sc = fig.add_subplot(gs[row_i, 3])

    bh = 0.22
    ax_sc.barh(y - bh / 2, a_vals, bh, color=C_A, alpha=0.80,
               edgecolor='white', linewidth=0.4, label='A (Selected)', zorder=3)
    ax_sc.barh(y + bh / 2, b_vals, bh, color=C_B, alpha=0.55,
               edgecolor='white', linewidth=0.4, label='B', zorder=3)

    for i in range(n_d):
        ax_sc.text(a_vals[i] + 0.1, y[i] - bh / 2, f'{a_vals[i]:.0f}',
                   va='center', ha='left', fontsize=6.5, color=C_A, fontweight='bold')
        ax_sc.text(b_vals[i] + 0.1, y[i] + bh / 2, f'{b_vals[i]:.0f}',
                   va='center', ha='left', fontsize=6.5, color=C_B, fontweight='bold')

    labels = [shorten(d) for d in dims]
    ax_sc.set_yticks(y)
    ax_sc.set_yticklabels(labels, fontsize=7.5)
    ax_sc.set_xlim(0, 11)
    ax_sc.set_xticks([0, 2, 4, 6, 8, 10])
    ax_sc.invert_yaxis()
    for sp in ['top', 'right']:
        ax_sc.spines[sp].set_visible(False)
    ax_sc.spines['left'].set_linewidth(0.3)
    ax_sc.spines['bottom'].set_linewidth(0.3)
    ax_sc.tick_params(axis='y', length=0, pad=3)
    ax_sc.tick_params(axis='x', labelsize=6)

    if row_i == 0:
        ax_sc.set_title('VLM Concept Scores (pre-LWRR)', fontsize=10,
                        fontweight='bold', color=C_HDR, pad=6)
        ax_sc.legend(fontsize=7, loc='lower right', frameon=True,
                     framealpha=0.9, edgecolor='#ccc', ncol=2)

    # Raw VLM: overall_intensity + prediction
    raw_ok = (raw_pred_ab == 'A')
    raw_mark = '\u2713' if raw_ok else '\u2717'
    raw_color = C_OK if raw_ok else C_WRONG
    raw_text = f'Concept Sum: A={raw_sum_a:.1f}  B={raw_sum_b:.1f} \u2192 {raw_pred_ab} {raw_mark}'
    ax_sc.text(0.5, -0.12, raw_text, transform=ax_sc.transAxes,
               ha='center', va='top', fontsize=7.5, color=raw_color, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.25', fc='#f5f5f5', ec=raw_color, lw=0.7, alpha=0.9))

    # ── Col 4: Gap ──
    fig.add_subplot(gs[row_i, 4]).axis('off')

    # ── Col 5: LWRR Calibrated Scores (paired A/B bars) ──
    ax_ct = fig.add_subplot(gs[row_i, 5])

    # LWRR calibrated: score_raw × original_weight (NO sign flip)
    # Math: fitted_delta = dot(w, left-right) + intercept
    #   When A=left:  cal_sum_a - cal_sum_b = dot(w, A-B) → add intercept_raw to A → total = delta
    #   When A=right: cal_sum_a - cal_sum_b = -dot(w, left-right) → add -intercept_raw to A → total = -delta = delta_towards_a
    cal_a = [a_raw[i] * weights[i] for i in range(n_d)]
    cal_b = [b_raw[i] * weights[i] for i in range(n_d)]

    # Intercept and delta
    s_diff = np.array([
        float(p['dimension_scores_left'][d]) - float(p['dimension_scores_right'][d])
        for d in dims
    ])
    intercept_raw = fitted_delta - np.dot(weights, s_diff)
    delta_towards_a = sign * fitted_delta
    intercept_towards_a = sign * intercept_raw
    contrib_towards_a = sign * (weights * s_diff)

    # Add intercept as extra row (assigned to A side)
    # Total A = sum(cal_a) + intercept_towards_a, Total B = sum(cal_b)
    # Total A - Total B = delta_towards_a (determines prediction: >0 → A wins)
    y_int = np.append(y, n_d)  # extra position for intercept
    cal_a_full = cal_a + [intercept_towards_a]
    cal_b_full = cal_b + [0.0]  # intercept only on A side

    # Draw paired bars (allow negative values)
    ax_ct.barh(y_int - bh / 2, cal_a_full, bh, color=C_A, alpha=0.80,
               edgecolor='white', linewidth=0.4, label='A (Selected)', zorder=3)
    ax_ct.barh(y_int[:n_d] + bh / 2, cal_b, bh, color=C_B, alpha=0.55,
               edgecolor='white', linewidth=0.4, label='B', zorder=3)

    # Value annotations
    all_cal = cal_a_full + cal_b_full
    x_min_val = min(all_cal)
    x_max_val = max(all_cal)
    x_range = max(x_max_val - x_min_val, 1)
    for i in range(n_d):
        for val, y_off, c in [(cal_a[i], -bh/2, C_A), (cal_b[i], bh/2, C_B)]:
            if val >= 0:
                ax_ct.text(val + x_range * 0.02, y[i] + y_off, f'{val:.1f}',
                           va='center', ha='left', fontsize=6, color=c, fontweight='bold')
            else:
                ax_ct.text(val - x_range * 0.02, y[i] + y_off, f'{val:.1f}',
                           va='center', ha='right', fontsize=6, color=c, fontweight='bold')
    # Intercept annotation
    iv = intercept_towards_a
    if iv >= 0:
        ax_ct.text(iv + x_range * 0.02, n_d - bh/2, f'{iv:+.1f}',
                   va='center', ha='left', fontsize=6, color=C_A, fontweight='bold')
    else:
        ax_ct.text(iv - x_range * 0.02, n_d - bh/2, f'{iv:+.1f}',
                   va='center', ha='right', fontsize=6, color=C_A, fontweight='bold')

    ax_ct.axvline(x=0, color='#888', linewidth=0.5, zorder=2, linestyle='-')
    # Separator line before intercept row
    ax_ct.axhline(y=n_d - 0.5, color='#ccc', linewidth=0.5, linestyle='--', zorder=1)

    labels_ct = [shorten(d) for d in dims] + ['Intercept']
    ax_ct.set_yticks(y_int)
    ax_ct.set_yticklabels(labels_ct, fontsize=7.5)
    pad_l = abs(x_min_val) * 0.2 if x_min_val < 0 else 0
    pad_r = x_max_val * 0.2 if x_max_val > 0 else 0
    ax_ct.set_xlim(x_min_val - pad_l - x_range * 0.08, x_max_val + pad_r + x_range * 0.08)
    ax_ct.invert_yaxis()
    for sp in ['top', 'right']:
        ax_ct.spines[sp].set_visible(False)
    ax_ct.spines['left'].set_linewidth(0.3)
    ax_ct.spines['bottom'].set_linewidth(0.3)
    ax_ct.tick_params(axis='y', length=0, pad=3)
    ax_ct.tick_params(axis='x', labelsize=6)

    if row_i == 0:
        ax_ct.set_title('LWRR Calibrated Scores\n(score \u00d7 weight + intercept)', fontsize=10,
                        fontweight='bold', color=C_HDR, pad=6)
        ax_ct.legend(fontsize=7, loc='lower right', frameon=True,
                     framealpha=0.9, edgecolor='#ccc', ncol=2)

    # Calibrated sum + prediction at bottom (A includes intercept)
    cal_sum_a = sum(cal_a) + intercept_towards_a
    cal_sum_b = sum(cal_b)
    ua_ok = (ua_pred_ab == 'A')
    ua_mark = '\u2713' if ua_ok else '\u2717'
    ua_color = C_OK if ua_ok else C_WRONG
    corr_tag = '  [Corrected]' if corrected else ''
    lwrr_text = f'LWRR: A={cal_sum_a:.1f}  B={cal_sum_b:.1f} \u2192 {ua_pred_ab} {ua_mark}{corr_tag}'
    ax_ct.text(0.5, -0.12, lwrr_text, transform=ax_ct.transAxes,
               ha='center', va='top', fontsize=7.5, color=ua_color, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.25', fc='#e8f5e9' if ua_ok else '#ffebee',
                         ec=ua_color, lw=0.7, alpha=0.9))

    # ── Col 6: Gap ──
    fig.add_subplot(gs[row_i, 6]).axis('off')

    # ── Col 7: All Method Predictions ──
    ax_m = fig.add_subplot(gs[row_i, 7])
    ax_m.axis('off')
    ax_m.set_xlim(0, 1)
    ax_m.set_ylim(0, 1)

    if row_i == 0:
        ax_m.set_title('All Methods', fontsize=10,
                       fontweight='bold', color=C_HDR, pad=6)

    bp = p.get('baseline_predictions', {})
    bprob = p.get('baseline_probabilities', {})

    methods = [
        ('Concept Sum',          p['raw_vlm_pred'],           None),
        ('C0 Siamese',           bp.get('C0_ResNet', '?'),    bprob.get('C0_ResNet')),
        ('C1 CLIP',              bp.get('C1_CLIP', '?'),      bprob.get('C1_CLIP')),
        ('C2 SegReg',            bp.get('C2_SegReg', '?'),    bprob.get('C2_SegReg')),
        ('C3 VLM (no concepts)', bp.get('C3_VLM', '?'),       None),
        ('UrbanAlign',           p['urbanalign_pred'],         None),
    ]

    n_m = len(methods)
    y_spacing = 0.80 / n_m
    y_start = 0.88

    # GT badge — always A
    ax_m.text(0.5, 0.99, 'GT: A', ha='center', va='top',
              fontsize=8, color=C_GT, fontweight='bold', transform=ax_m.transAxes,
              bbox=dict(boxstyle='round,pad=0.2', fc='#e8f5e9', ec=C_GT, lw=0.5))

    for mi, (mname, mpred_raw, mprob_raw) in enumerate(methods):
        y_pos = y_start - mi * y_spacing
        mpred_ab = remap_pred(mpred_raw, human) if mpred_raw not in ('?',) else '?'
        is_correct = (mpred_ab == 'A')
        mark = '\u2713' if is_correct else '\u2717'
        mc = C_OK if is_correct else C_WRONG
        is_ours = (mname == 'UrbanAlign')
        fw = 'bold' if is_ours else 'normal'
        fs = 8.5 if is_ours else 7.5

        # Method name (left)
        ax_m.text(0.0, y_pos, mname, ha='left', va='center',
                  fontsize=fs, color=C_HDR, fontweight=fw, transform=ax_m.transAxes)

        # Prediction result (middle)
        pred_text = f'{mpred_ab} {mark}'
        ax_m.text(0.58, y_pos, pred_text, ha='center', va='center',
                  fontsize=fs, color=mc, fontweight='bold', transform=ax_m.transAxes)

        # Probabilities A/B/E (right) — show all three if available
        if mprob_raw and isinstance(mprob_raw, dict):
            remapped = remap_probs(mprob_raw, human)
            if remapped:
                pa, pb, pe = remapped.get('A', 0), remapped.get('B', 0), remapped.get('E', 0)
                prob_text = f'{pa:.0%}/{pb:.0%}/{pe:.0%}'
                ax_m.text(1.0, y_pos, prob_text, ha='right', va='center',
                          fontsize=6.5, color='#555', fontweight='normal',
                          transform=ax_m.transAxes)

        ax_m.text(1.0, y_pos, '', transform=ax_m.transAxes)  # placeholder

    # Column header for probabilities
    if row_i == 0:
        ax_m.text(1.0, 1.0, 'A / B / E', ha='right', va='bottom',
                  fontsize=6.5, color='#888', fontstyle='italic',
                  transform=ax_m.transAxes)

    # ── Export data ──
    export_data[cat] = {
        'image_a_is': human,  # which original side is A
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
        'contributions_towards_a': {d: float(contrib_towards_a[i]) for i, d in enumerate(dims)},
        'raw_scores_a': {d: float(a_raw[i]) for i, d in enumerate(dims)},
        'raw_scores_b': {d: float(b_raw[i]) for i, d in enumerate(dims)},
        'method_predictions': {m[0]: remap_pred(m[1], human) if m[1] != '?' else '?' for m in methods},
        'baseline_probabilities_remapped': {
            bn: remap_probs(bprob.get(bn), human)
            for bn in ['C0_ResNet', 'C1_CLIP', 'C2_SegReg']
            if bprob.get(bn)
        },
    }

# ── Bottom annotation ──
fig.text(0.5, 0.003,
         'A = human-selected image (green border).  B = not selected.  '
         'Concept Sum = sum of multi-agent VLM concept scores (pre-calibration).  '
         'Calibrated = score \u00d7 LWRR weight + intercept; negative bars indicate inverted dimensions.  '
         'C3 VLM (no concepts) = zero-shot VLM prompt without semantic concepts.  '
         'Probabilities: P(A) / P(B) / P(Equal).',
         ha='center', va='bottom', fontsize=7.5, color='#666', style='italic')

# ── Save figure ──
for ext, dpi in [('.pdf', 300), ('.png', 200)]:
    path = os.path.join(FIG_DIR, f'fig_qualitative_v2{ext}')
    fig.savefig(path, bbox_inches='tight', pad_inches=0.12, dpi=dpi)
    print(f"Saved: {path}")

plt.close()

# ── Write computed fields back into fig3_data.json (merged: input + computed) ──
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
print(f"Updated: {DATA_FILE} (merged input + computed)")

# ── Summary ──
print("\n=== SUMMARY (A = Selected, B = Not Selected) ===")
for cat in CATEGORIES:
    d = export_data.get(cat)
    if not d:
        continue
    tag = ' [CORRECTED]' if d['lwrr_corrected'] else ''
    print(f"\n  {CAT_DISPLAY[cat]:12s}{tag}")
    print(f"    Sum(norm):  A(sel)={d['raw_sum_a']:.1f}  B={d['raw_sum_b']:.1f}")
    print(f"    Raw VLM: {d['raw_vlm_pred']}   UrbanAlign: {d['urbanalign_pred']}")
    print(f"    LWRR Δ={d['fitted_delta_towards_a']:+.1f}  (intercept={d['intercept_towards_a']:+.1f})")
    for mname, mpred in d['method_predictions'].items():
        ok = '\u2713' if mpred == 'A' else '\u2717'
        print(f"    {mname:20s}: {mpred} {ok}")
