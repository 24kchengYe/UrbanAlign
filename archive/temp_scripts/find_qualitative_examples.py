"""Find best qualitative examples for paper figure."""
import pandas as pd
import json
import ast
import os
import shutil

OUT = 'urbanalign_outputs'
IMAGE_DIR = r'H:\RawData13-全球街景\mit place pulse\01 Place Pluse2.0数据集\01 Place Pulse 2.0论文数据集\final_photo_dataset'
CATEGORIES = ['safety', 'beautiful', 'lively', 'wealthy', 'boring', 'depressing']

print('=' * 80)
print('SEARCHING: UrbanAlign correct + ALL 3 baselines wrong (across all categories)')
print('=' * 80)

all_examples = []

for cat in CATEGORIES:
    st3 = pd.read_csv(f'{OUT}/stage3_mode4_aligned_{cat}.csv')
    st3['pair_key'] = st3['left_id'] + '|' + st3['right_id']

    st2 = pd.read_csv(f'{OUT}/stage2_mode4_all_scored_{cat}.csv')
    st2['pair_key'] = st2['left_id'] + '|' + st2['right_id']

    baselines = {}
    for bname, bfile in [
        ('C1_CLIP', f'stage7_baseline_c1_siamese_clip_{cat}.csv'),
        ('C2_Seg', f'stage7_baseline_c2_segmentation_regression_{cat}.csv'),
        ('C3_VLM', f'stage7_baseline_c3_zeroshot_vlm_{cat}.csv'),
    ]:
        bdf = pd.read_csv(f'{OUT}/{bfile}')
        bdf['pair_key'] = bdf['left_id'] + '|' + bdf['right_id']
        baselines[bname] = bdf

    common_keys = set(st3['pair_key'])
    for bdf in baselines.values():
        common_keys &= set(bdf['pair_key'])

    for key in common_keys:
        row3 = st3[st3['pair_key'] == key].iloc[0]
        human = row3['human_winner']
        ua_pred = row3['synthetic_winner']
        if ua_pred != human:
            continue

        # Check Stage 2 raw prediction
        raw_rows = st2[st2['pair_key'] == key]
        raw_pred = raw_rows.iloc[0]['synthetic_winner'] if len(raw_rows) > 0 else 'N/A'
        raw_correct = (raw_pred == human)

        all_wrong = True
        baseline_preds = {}
        for bname, bdf in baselines.items():
            brow = bdf[bdf['pair_key'] == key].iloc[0]
            baseline_preds[bname] = brow['synthetic_winner']
            if brow['synthetic_winner'] == human:
                all_wrong = False

        if not all_wrong:
            continue

        sa = json.loads(row3['image_a_scores'].replace("'", '"'))
        sb = json.loads(row3['image_b_scores'].replace("'", '"'))
        max_diff = max(abs(sa[d] - sb[d]) for d in sa)

        all_examples.append({
            'cat': cat,
            'left_id': row3['left_id'],
            'right_id': row3['right_id'],
            'human': human,
            'ua_pred': ua_pred,
            'raw_pred': raw_pred,
            'raw_correct': raw_correct,
            'lwrr_corrected': (not raw_correct and ua_pred == human),
            'confidence': row3['confidence_score'],
            'max_dim_diff': max_diff,
            'scores_a': sa,
            'scores_b': sb,
            'local_weights': row3['local_weights'],
            'intensity_a': row3['overall_intensity_a'],
            'intensity_b': row3['overall_intensity_b'],
            'baseline_preds': baseline_preds,
        })

print(f'Total examples (UA correct + all baselines wrong): {len(all_examples)}')

corrected = [e for e in all_examples if e['lwrr_corrected']]
already_ok = [e for e in all_examples if not e['lwrr_corrected']]
print(f'  LWRR corrected (raw wrong -> aligned right): {len(corrected)}')
print(f'  Already correct (raw right, still right after LWRR): {len(already_ok)}')

# Print all sorted by confidence
for e in sorted(all_examples, key=lambda x: x['confidence'], reverse=True):
    corr_tag = ' [LWRR-CORRECTED]' if e['lwrr_corrected'] else ''
    print(f"\n  [{e['cat']}] conf={e['confidence']:.2f} | "
          f"left={e['left_id'][:12]}... right={e['right_id'][:12]}... | "
          f"human={e['human']}, raw={e['raw_pred']}, aligned={e['ua_pred']}{corr_tag}")
    for bname, bpred in e['baseline_preds'].items():
        print(f"    {bname}: {bpred} (WRONG)")

print('\n' + '=' * 80)
print('TOP 3 RECOMMENDED FOR FIGURE (diverse categories, highest confidence)')
print('=' * 80)

# Pick best from different categories
used_cats = set()
top_picks = []

# Prefer LWRR-corrected examples first (shows calibration value)
for e in sorted(corrected, key=lambda x: x['confidence'], reverse=True):
    if e['cat'] not in used_cats and len(top_picks) < 3:
        top_picks.append(e)
        used_cats.add(e['cat'])

# Fill remaining with already-correct (shows overall accuracy)
for e in sorted(already_ok, key=lambda x: x['confidence'], reverse=True):
    if e['cat'] not in used_cats and len(top_picks) < 3:
        top_picks.append(e)
        used_cats.add(e['cat'])

# If still less than 3, allow same category
for e in sorted(all_examples, key=lambda x: x['confidence'], reverse=True):
    if e not in top_picks and len(top_picks) < 3:
        top_picks.append(e)

# Copy images and print detailed info
fig_dir = os.path.join(OUT, 'qualitative_figure')
os.makedirs(fig_dir, exist_ok=True)

for idx, e in enumerate(top_picks):
    print(f'\n{"="*60}')
    print(f'EXAMPLE {idx+1}: [{e["cat"].upper()}] {"LWRR-CORRECTED" if e["lwrr_corrected"] else "CONSISTENTLY CORRECT"}')
    print(f'{"="*60}')
    print(f'  Confidence: {e["confidence"]:.2f}')
    print(f'  Left  ID: {e["left_id"]}')
    print(f'  Right ID: {e["right_id"]}')
    print(f'  Human winner: {e["human"]}')
    print(f'  Raw VLM (Stage 2): {e["raw_pred"]} ({"CORRECT" if e["raw_correct"] else "WRONG"})')
    print(f'  UrbanAlign (Stage 3): {e["ua_pred"]} (CORRECT)')
    for bname, bpred in e['baseline_preds'].items():
        print(f'  {bname}: {bpred} (WRONG)')

    print(f'\n  Dimension scores:')
    print(f'  {"Dimension":<30} {"Left":>6} {"Right":>6} {"Diff":>6}')
    print(f'  {"-"*50}')
    for dim in e['scores_a']:
        a, b = e['scores_a'][dim], e['scores_b'][dim]
        print(f'  {dim:<30} {a:>6} {b:>6} {a-b:>+6}')
    print(f'  {"Overall intensity":<30} {e["intensity_a"]:>6.0f} {e["intensity_b"]:>6.0f} {e["intensity_a"]-e["intensity_b"]:>+6.0f}')

    # LWRR weights
    try:
        lw = ast.literal_eval(e['local_weights'])
        dims = list(e['scores_a'].keys())
        print(f'\n  LWRR local weights (sorted by |w|):')
        for dim, w in sorted(zip(dims, lw), key=lambda x: abs(x[1]), reverse=True):
            bar = '#' * int(abs(w) * 3)
            sign = '+' if w > 0 else '-'
            print(f'  {dim:<30} {w:>+7.2f}  {sign}{bar}')
    except:
        pass

    # Copy images
    for side, img_id in [('left', e['left_id']), ('right', e['right_id'])]:
        src = os.path.join(IMAGE_DIR, f'{img_id}.jpg')
        dst = os.path.join(fig_dir, f'example{idx+1}_{e["cat"]}_{side}_{img_id}.jpg')
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f'\n  Copied: {dst}')
        else:
            print(f'\n  WARNING: Image not found: {src}')

print(f'\n{"="*80}')
print(f'All images copied to: {os.path.abspath(fig_dir)}')
print(f'{"="*80}')

# Print summary table for LaTeX
print('\n\nFIGURE CAPTION SUGGESTION:')
print('Three pairs from the Pool set where UrbanAlign predicts correctly while')
print('all supervised/zero-shot baselines fail. For each pair, the radar/bar chart')
print('shows VLM-extracted dimension scores (left image in blue, right in orange).')
print('LWRR local weights (bottom) reveal which dimensions drive the calibrated')
print('prediction, providing per-sample interpretability.')
