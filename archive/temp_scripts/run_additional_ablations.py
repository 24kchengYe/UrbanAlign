"""
Additional ablation experiments addressing reviewer concerns:
1. Fair baselines on same features (Bradley-Terry, Global Ridge, kNN, Global Logistic)
2. ε/θ sensitivity sweep
3. CLIP-in-regression ablation (LWRR on Δ_sem vs [Δ_sem, Δ_CLIP])
All CPU-only, no API calls needed.
"""

import pandas as pd
import numpy as np
import json
import ast
import os
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from collections import defaultdict

OUT = 'urbanalign_outputs'
CATEGORIES = ['safety', 'beautiful', 'lively', 'wealthy', 'boring', 'depressing']

# ============================================================
# Utility: parse scored data into numeric arrays
# ============================================================

def parse_scores(score_str):
    """Parse JSON score string into dict."""
    if pd.isna(score_str):
        return {}
    try:
        return json.loads(score_str.replace("'", '"'))
    except:
        try:
            return ast.literal_eval(score_str)
        except:
            return {}

def get_dim_names(cat):
    """Get dimension names for a category."""
    path = f'{OUT}/stage1_semantic_dimensions_{cat}.json'
    with open(path, 'r') as f:
        data = json.load(f)
    return [d['name'] for d in data[cat]['dimensions']]

def build_feature_matrices(cat, mode=4):
    """Build Δ_sem, Δ_CLIP, and labels for ref and pool splits."""
    # Load scored data
    scored = pd.read_csv(f'{OUT}/stage2_mode{mode}_all_scored_{cat}.csv')
    ref_split = pd.read_csv(f'{OUT}/data_split_reference_{cat}.csv')
    pool_split = pd.read_csv(f'{OUT}/data_split_pool_{cat}.csv')

    # Load CLIP embeddings
    clip_data = np.load(f'{OUT}/clip_embeddings.npz', allow_pickle=True)
    clip_paths = clip_data['paths']
    clip_embs = clip_data['embeddings']

    # Build CLIP lookup (extract image_id from path)
    clip_lookup = {}
    for i, p in enumerate(clip_paths):
        img_id = os.path.splitext(os.path.basename(str(p)))[0]
        clip_lookup[img_id] = clip_embs[i]

    # Load TrueSkill
    ts = pd.read_csv(f'{OUT}/trueskill_ratings_{cat}.csv')
    ts_lookup = dict(zip(ts['image_id'], ts['mu']))

    dim_names = get_dim_names(cat)

    def process_split(split_df, scored_df):
        """Process a split into feature matrices."""
        split_keys = set(zip(split_df['left_id'].astype(str), split_df['right_id'].astype(str)))
        winner_col = 'human_winner' if 'human_winner' in split_df.columns else 'winner'
        split_winners = dict(zip(
            zip(split_df['left_id'].astype(str), split_df['right_id'].astype(str)),
            split_df[winner_col]
        ))

        X_sem = []
        X_clip = []
        y_labels = []
        y_ts = []
        valid_pairs = []

        for _, row in scored_df.iterrows():
            lid = str(row['left_id'])
            rid = str(row['right_id'])
            key = (lid, rid)
            if key not in split_keys:
                continue

            scores_a = parse_scores(row['image_a_scores'])
            scores_b = parse_scores(row['image_b_scores'])
            if not scores_a or not scores_b:
                continue

            # Semantic difference
            delta_sem = []
            for dim in dim_names:
                sa = scores_a.get(dim, 5)
                sb = scores_b.get(dim, 5)
                delta_sem.append((sa - sb) / 10.0)  # normalize to [-1, 1]

            # CLIP difference
            if lid in clip_lookup and rid in clip_lookup:
                ca = clip_lookup[lid]
                cb = clip_lookup[rid]
                ca_norm = ca / (np.linalg.norm(ca) + 1e-8)
                cb_norm = cb / (np.linalg.norm(cb) + 1e-8)
                delta_clip = ca_norm - cb_norm
            else:
                continue

            # TrueSkill difference
            ts_a = ts_lookup.get(lid, 25.0)
            ts_b = ts_lookup.get(rid, 25.0)

            human_winner = split_winners.get(key, 'equal')

            X_sem.append(delta_sem)
            X_clip.append(delta_clip)
            y_labels.append(human_winner)
            y_ts.append(ts_a - ts_b)
            valid_pairs.append(key)

        return (np.array(X_sem), np.array(X_clip), y_labels,
                np.array(y_ts), valid_pairs)

    ref_data = process_split(ref_split, scored)
    pool_data = process_split(pool_split, scored)

    return ref_data, pool_data, dim_names


def mirror_augment(X, y_labels, y_ts):
    """Mirror augmentation: add (B,A) with flipped labels."""
    X_aug = np.vstack([X, -X])
    y_ts_aug = np.concatenate([y_ts, -y_ts])

    label_flip = {'left': 'right', 'right': 'left', 'equal': 'equal'}
    y_aug = y_labels + [label_flip.get(l, l) for l in y_labels]

    return X_aug, y_aug, y_ts_aug


def encode_labels(labels, exclude_equal=True):
    """Convert string labels to numeric, optionally filtering equal."""
    mapping = {'left': 0, 'right': 1, 'equal': 2}
    indices = []
    encoded = []
    for i, l in enumerate(labels):
        if exclude_equal and l == 'equal':
            continue
        indices.append(i)
        encoded.append(mapping.get(l, 2))
    return indices, encoded


# ============================================================
# Experiment 1: Fair baselines on same features
# ============================================================

def run_fair_baselines():
    """Run Bradley-Terry, Global Ridge, kNN, Global Logistic on same features."""
    print("=" * 70)
    print("EXPERIMENT 1: Fair Baselines on Same Features")
    print("=" * 70)

    results = []

    for cat in CATEGORIES:
        print(f"\n--- {cat} ---")
        ref_data, pool_data, dim_names = build_feature_matrices(cat)
        X_ref_sem, X_ref_clip, y_ref_labels, y_ref_ts, _ = ref_data
        X_pool_sem, X_pool_clip, y_pool_labels, y_pool_ts, _ = pool_data

        if len(X_ref_sem) == 0 or len(X_pool_sem) == 0:
            print(f"  Skipping {cat}: no data")
            continue

        # Mirror augmentation on ref
        X_ref_sem_aug, y_ref_aug, y_ref_ts_aug = mirror_augment(
            X_ref_sem, y_ref_labels, y_ref_ts)
        X_ref_clip_aug = np.vstack([X_ref_clip, -X_ref_clip])

        # Hybrid features
        alpha = 0.3
        X_ref_hybrid_aug = np.hstack([alpha * X_ref_clip_aug, (1-alpha) * X_ref_sem_aug])
        X_pool_hybrid = np.hstack([alpha * X_pool_clip, (1-alpha) * X_pool_sem])

        # Encode labels (excl equal for training)
        ref_idx_excl, y_ref_enc_excl = encode_labels(y_ref_aug, exclude_equal=True)
        pool_idx_excl, y_pool_enc_excl = encode_labels(y_pool_labels, exclude_equal=True)
        pool_idx_incl, y_pool_enc_incl = encode_labels(y_pool_labels, exclude_equal=False)

        if len(ref_idx_excl) < 5 or len(pool_idx_excl) < 5:
            print(f"  Skipping {cat}: insufficient data")
            continue

        baselines = {
            'BT / Logistic (Δ_sem)': {
                'model': LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial'),
                'X_train': X_ref_sem_aug[ref_idx_excl],
                'X_test_excl': X_pool_sem[pool_idx_excl],
                'X_test_incl': X_pool_sem[pool_idx_incl],
            },
            'Global Ridge (Δ_sem)': {
                'model': RidgeClassifier(alpha=1.0),
                'X_train': X_ref_sem_aug[ref_idx_excl],
                'X_test_excl': X_pool_sem[pool_idx_excl],
                'X_test_incl': X_pool_sem[pool_idx_incl],
            },
            'Global Ridge (hybrid)': {
                'model': RidgeClassifier(alpha=1.0),
                'X_train': X_ref_hybrid_aug[ref_idx_excl],
                'X_test_excl': X_pool_hybrid[pool_idx_excl],
                'X_test_incl': X_pool_hybrid[pool_idx_incl],
            },
            'kNN (hybrid, K=20)': {
                'model': KNeighborsClassifier(n_neighbors=min(20, len(ref_idx_excl))),
                'X_train': X_ref_hybrid_aug[ref_idx_excl],
                'X_test_excl': X_pool_hybrid[pool_idx_excl],
                'X_test_incl': X_pool_hybrid[pool_idx_incl],
            },
            'Global Logistic (hybrid)': {
                'model': LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial'),
                'X_train': X_ref_hybrid_aug[ref_idx_excl],
                'X_test_excl': X_pool_hybrid[pool_idx_excl],
                'X_test_incl': X_pool_hybrid[pool_idx_incl],
            },
        }

        y_train = np.array(y_ref_enc_excl)
        y_test_excl = np.array(y_pool_enc_excl)
        y_test_incl = np.array(y_pool_enc_incl)

        for bname, bconf in baselines.items():
            try:
                model = bconf['model']
                model.fit(bconf['X_train'], y_train)

                # Excl equal
                pred_excl = model.predict(bconf['X_test_excl'])
                acc_excl = accuracy_score(y_test_excl, pred_excl)
                kappa_excl = cohen_kappa_score(y_test_excl, pred_excl)

                # Incl equal (test includes equal, model predicts 2-class)
                pred_incl = model.predict(bconf['X_test_incl'])
                acc_incl = accuracy_score(y_test_incl, pred_incl)
                kappa_incl = cohen_kappa_score(y_test_incl, pred_incl)

                print(f"  {bname}: Acc(excl)={acc_excl:.1%}, κ={kappa_excl:.3f} | "
                      f"Acc(incl)={acc_incl:.1%}, κ={kappa_incl:.3f} | "
                      f"n_excl={len(y_test_excl)}, n_incl={len(y_test_incl)}")

                results.append({
                    'category': cat,
                    'baseline': bname,
                    'acc_excl': acc_excl,
                    'kappa_excl': kappa_excl,
                    'acc_incl': acc_incl,
                    'kappa_incl': kappa_incl,
                    'n_excl': len(y_test_excl),
                    'n_incl': len(y_test_incl),
                })
            except Exception as e:
                print(f"  {bname}: ERROR - {e}")

    df = pd.DataFrame(results)
    df.to_csv(f'{OUT}/ablation_fair_baselines.csv', index=False)

    # Print summary
    print("\n\n=== SUMMARY: Fair Baselines (excl-equal) ===")
    pivot = df.pivot_table(values='acc_excl', index='baseline', columns='category', aggfunc='first')
    pivot['avg'] = pivot.mean(axis=1)
    print((pivot * 100).round(1).to_string())

    print("\n=== SUMMARY: Fair Baselines (κ, excl-equal) ===")
    pivot_k = df.pivot_table(values='kappa_excl', index='baseline', columns='category', aggfunc='first')
    pivot_k['avg'] = pivot_k.mean(axis=1)
    print(pivot_k.round(3).to_string())

    return df


# ============================================================
# Experiment 2: ε/θ sensitivity sweep
# ============================================================

def run_eps_theta_sensitivity():
    """Sweep ε (score-diff threshold) and θ (equal-consensus threshold)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: ε/θ Sensitivity Sweep")
    print("=" * 70)

    results = []

    eps_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    theta_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for cat in CATEGORIES:
        aligned = pd.read_csv(f'{OUT}/stage3_mode4_aligned_{cat}.csv')

        for eps in eps_values:
            for theta in theta_values:
                # Re-infer predictions with new ε/θ
                correct_excl = 0
                total_excl = 0
                correct_incl = 0
                total_incl = 0

                for _, row in aligned.iterrows():
                    human = row['human_winner']
                    delta = row['fitted_ts_delta']
                    eq_cons = row.get('equal_consensus', 0.0)
                    if pd.isna(eq_cons):
                        eq_cons = 0.0

                    # Re-infer with new thresholds
                    if abs(delta) < eps or eq_cons > theta:
                        pred = 'equal'
                    elif delta > 0:
                        pred = 'left'
                    else:
                        pred = 'right'

                    # Incl equal
                    total_incl += 1
                    if pred == human:
                        correct_incl += 1

                    # Excl equal
                    if human != 'equal':
                        total_excl += 1
                        if pred == human:
                            correct_excl += 1

                acc_excl = correct_excl / total_excl if total_excl > 0 else 0
                acc_incl = correct_incl / total_incl if total_incl > 0 else 0

                results.append({
                    'category': cat,
                    'epsilon': eps,
                    'theta': theta,
                    'acc_excl': acc_excl,
                    'acc_incl': acc_incl,
                    'n_excl': total_excl,
                    'n_incl': total_incl,
                })

    df = pd.DataFrame(results)
    df.to_csv(f'{OUT}/ablation_eps_theta_sensitivity.csv', index=False)

    # Print summary: best ε for each category (with θ=0.6 default)
    print("\n=== ε sensitivity (θ=0.6 fixed, excl-equal) ===")
    for cat in CATEGORIES:
        sub = df[(df['category'] == cat) & (df['theta'] == 0.6)]
        print(f"\n{cat}:")
        for _, r in sub.sort_values('epsilon').iterrows():
            marker = " ← default" if r['epsilon'] == 0.8 else ""
            print(f"  ε={r['epsilon']:.1f}: Acc={r['acc_excl']:.1%}{marker}")

    # Print summary: best θ for each category (with ε=0.8 default)
    print("\n=== θ sensitivity (ε=0.8 fixed, excl-equal) ===")
    for cat in CATEGORIES:
        sub = df[(df['category'] == cat) & (df['epsilon'] == 0.8)]
        print(f"\n{cat}:")
        for _, r in sub.sort_values('theta').iterrows():
            marker = " ← default" if r['theta'] == 0.6 else ""
            print(f"  θ={r['theta']:.1f}: Acc={r['acc_excl']:.1%}{marker}")

    # Overall range
    print("\n=== ε/θ Combined Range (excl-equal) ===")
    for cat in CATEGORIES:
        sub = df[df['category'] == cat]
        best = sub.loc[sub['acc_excl'].idxmax()]
        worst = sub.loc[sub['acc_excl'].idxmin()]
        default = sub[(sub['epsilon'] == 0.8) & (sub['theta'] == 0.6)].iloc[0]
        print(f"  {cat}: default={default['acc_excl']:.1%}, "
              f"best={best['acc_excl']:.1%} (ε={best['epsilon']}, θ={best['theta']}), "
              f"worst={worst['acc_excl']:.1%}, range={best['acc_excl']-worst['acc_excl']:.1%}")

    return df


# ============================================================
# Experiment 3: CLIP-in-regression ablation
# ============================================================

def run_clip_regression_ablation():
    """Compare LWRR on Δ_sem only vs LWRR on [Δ_sem, Δ_CLIP]."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: CLIP-in-Regression Ablation")
    print("=" * 70)

    results = []

    for cat in CATEGORIES:
        print(f"\n--- {cat} ---")
        ref_data, pool_data, dim_names = build_feature_matrices(cat)
        X_ref_sem, X_ref_clip, y_ref_labels, y_ref_ts, _ = ref_data
        X_pool_sem, X_pool_clip, y_pool_labels, y_pool_ts, _ = pool_data

        if len(X_ref_sem) == 0 or len(X_pool_sem) == 0:
            continue

        # Mirror augment ref
        X_ref_sem_aug, y_ref_aug, y_ref_ts_aug = mirror_augment(
            X_ref_sem, y_ref_labels, y_ref_ts)
        X_ref_clip_aug = np.vstack([X_ref_clip, -X_ref_clip])

        alpha = 0.3
        K = 20
        tau = 1.0
        lam = 1.0
        eps = 0.8
        theta_eq = 0.6

        # Build hybrid for neighbourhood search (always)
        X_ref_hybrid_aug = np.hstack([alpha * X_ref_clip_aug, (1-alpha) * X_ref_sem_aug])
        X_pool_hybrid = np.hstack([alpha * X_pool_clip, (1-alpha) * X_pool_sem])

        configs = {
            'LWRR on Δ_sem only (current)': X_ref_sem_aug,
            'LWRR on [Δ_sem, Δ_CLIP]': np.hstack([X_ref_sem_aug, X_ref_clip_aug]),
        }

        pool_configs = {
            'LWRR on Δ_sem only (current)': X_pool_sem,
            'LWRR on [Δ_sem, Δ_CLIP]': np.hstack([X_pool_sem, X_pool_clip]),
        }

        for config_name, X_ref_reg in configs.items():
            X_pool_reg = pool_configs[config_name]

            correct_excl = 0
            total_excl = 0
            correct_incl = 0
            total_incl = 0

            for i in range(len(X_pool_sem)):
                q_hybrid = X_pool_hybrid[i]
                q_reg = X_pool_reg[i]
                human = y_pool_labels[i]

                # Cosine similarity for neighbourhood (always hybrid)
                sims = X_ref_hybrid_aug @ q_hybrid / (
                    np.linalg.norm(X_ref_hybrid_aug, axis=1) * np.linalg.norm(q_hybrid) + 1e-8)

                k = min(K, len(sims))
                top_k = np.argsort(sims)[-k:]

                # Kernel weights
                w = np.exp(sims[top_k] / tau)

                # Local ridge regression
                X_local = X_ref_reg[top_k]
                y_local = y_ref_ts_aug[top_k]
                W = np.diag(w)

                try:
                    H = X_local.T @ W @ X_local + lam * np.eye(X_local.shape[1])
                    w_hat = np.linalg.solve(H, X_local.T @ W @ y_local)
                    delta = w_hat @ q_reg
                except:
                    delta = 0.0

                # Equal consensus
                eq_count = sum(1 for j in top_k if y_ref_aug[j] == 'equal')
                eq_cons = eq_count / k

                if abs(delta) < eps or eq_cons > theta_eq:
                    pred = 'equal'
                elif delta > 0:
                    pred = 'left'
                else:
                    pred = 'right'

                total_incl += 1
                if pred == human:
                    correct_incl += 1
                if human != 'equal':
                    total_excl += 1
                    if pred == human:
                        correct_excl += 1

            acc_excl = correct_excl / total_excl if total_excl > 0 else 0
            acc_incl = correct_incl / total_incl if total_incl > 0 else 0

            # Compute kappa
            preds = []
            trues = []
            for i in range(len(X_pool_sem)):
                human = y_pool_labels[i]
                if human == 'equal':
                    continue
                q_hybrid = X_pool_hybrid[i]
                q_reg = X_pool_reg[i]
                sims = X_ref_hybrid_aug @ q_hybrid / (
                    np.linalg.norm(X_ref_hybrid_aug, axis=1) * np.linalg.norm(q_hybrid) + 1e-8)
                k = min(K, len(sims))
                top_k = np.argsort(sims)[-k:]
                w = np.exp(sims[top_k] / tau)
                X_local = X_ref_reg[top_k]
                y_local = y_ref_ts_aug[top_k]
                W = np.diag(w)
                try:
                    H = X_local.T @ W @ X_local + lam * np.eye(X_local.shape[1])
                    w_hat = np.linalg.solve(H, X_local.T @ W @ y_local)
                    delta = w_hat @ q_reg
                except:
                    delta = 0.0
                eq_count = sum(1 for j in top_k if y_ref_aug[j] == 'equal')
                eq_cons = eq_count / k
                if abs(delta) < eps or eq_cons > theta_eq:
                    pred = 'equal'
                elif delta > 0:
                    pred = 'left'
                else:
                    pred = 'right'
                label_map = {'left': 0, 'right': 1, 'equal': 2}
                preds.append(label_map[pred])
                trues.append(label_map[human])

            kappa_excl = cohen_kappa_score(trues, preds) if len(trues) > 0 else 0

            print(f"  {config_name}: Acc(excl)={acc_excl:.1%}, κ={kappa_excl:.3f}, "
                  f"n_excl={total_excl}")

            results.append({
                'category': cat,
                'config': config_name,
                'acc_excl': acc_excl,
                'kappa_excl': kappa_excl,
                'acc_incl': acc_incl,
                'n_excl': total_excl,
            })

    df = pd.DataFrame(results)
    df.to_csv(f'{OUT}/ablation_clip_in_regression.csv', index=False)

    print("\n=== SUMMARY ===")
    pivot = df.pivot_table(values='acc_excl', index='config', columns='category', aggfunc='first')
    pivot['avg'] = pivot.mean(axis=1)
    print((pivot * 100).round(1).to_string())

    return df


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("Running additional ablation experiments...\n")

    df1 = run_fair_baselines()
    df2 = run_eps_theta_sensitivity()
    df3 = run_clip_regression_ablation()

    print("\n\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Results saved to:")
    print(f"  {OUT}/ablation_fair_baselines.csv")
    print(f"  {OUT}/ablation_eps_theta_sensitivity.csv")
    print(f"  {OUT}/ablation_clip_in_regression.csv")
