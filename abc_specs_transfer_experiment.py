"""
SPECS Cross-Dataset Transfer Experiment
========================================
Tests UrbanAlign dimension transferability from Place Pulse 2.0 to SPECS.

Strategy:
  1. Preprocess SPECS pairwise data → TrueSkill ratings
  2. Extract CLIP features for SPECS images
  3. Transfer Place Pulse Stage 1 dimensions (no re-extraction)
  4. Run Stage 2 (Mode 2, cheapest) on sampled SPECS pairs
  5. Run Stage 3 (LWRR) calibration
  6. Evaluate against human labels
  7. Also run zero-shot VLM baseline for comparison

Output: specs_transfer_results.csv with accuracy/kappa per category
"""

import os
import sys
import json
import time
import base64
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score, accuracy_score

# ── Configuration ──────────────────────────────────────────────
SPECS_DIR = os.getenv("SPECS_DIR", r"H:\RawData13-全球街景\SPECS")
SPECS_LABELS = os.path.join(SPECS_DIR, "labels", "processed", "global_mapped_cleaned.csv")
SPECS_SVI_DIR = os.path.join(SPECS_DIR, "svi")
SPECS_METADATA = os.path.join(SPECS_SVI_DIR, "metadata.csv")

# Place Pulse outputs (for dimension transfer)
PP_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urbanalign_outputs")

# SPECS experiment output
SPECS_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specs_outputs")
os.makedirs(SPECS_OUTPUT_DIR, exist_ok=True)

# API config (import from config.py to avoid duplication)
from config import API_KEY, BASE_URL, MODEL_NAME

# Experiment params
CATEGORIES = ['safety', 'beautiful', 'lively', 'wealthy', 'boring', 'depressing']
# Map SPECS question names to our category names
SPECS_CAT_MAP = {'safe': 'safety', 'beautiful': 'beautiful', 'lively': 'lively',
                 'wealthy': 'wealthy', 'boring': 'boring', 'depressing': 'depressing'}

MAX_PAIRS_PER_CAT = 300  # Sample size per category (Mode 4 is ~56s/pair)
LABELED_SET_RATIO = 0.4  # 40% ref, 10% val, 50% test
VAL_RATIO = 0.1
RANDOM_SEED = 42

# LWRR defaults
K_MAX_DEFAULT = 20
ALPHA_DEFAULT = 0.0  # Semantic-only (no CLIP needed, tests dimension transferability)
TAU_KERNEL_DEFAULT = 1.0
RIDGE_ALPHA_DEFAULT = 1.0
EQUAL_EPS_DEFAULT = 0.8

# TrueSkill
import trueskill
TS_ENV = trueskill.TrueSkill(mu=25.0, sigma=8.333, draw_probability=0.10)


def load_specs_data():
    """Load and preprocess SPECS pairwise comparison data."""
    print("=" * 60)
    print("Step 1: Loading SPECS data")
    print("=" * 60)

    df = pd.read_csv(SPECS_LABELS)
    meta = pd.read_csv(SPECS_METADATA)

    # Build image number → uuid → city → path mapping
    img_map = {}
    for _, row in meta.iterrows():
        img_num = row['Image number']
        uuid = row['uuid']
        city = row['city']
        img_map[img_num] = {
            'uuid': uuid,
            'city': city,
            'path': os.path.join(SPECS_SVI_DIR, city, f"{uuid}.jpeg")
        }

    # Filter to our 6 categories
    df_filtered = df[df['Question'].isin(SPECS_CAT_MAP.keys())].copy()
    df_filtered['category'] = df_filtered['Question'].map(SPECS_CAT_MAP)

    print(f"Total comparisons (6 cats): {len(df_filtered)}")
    for cat in CATEGORIES:
        n = len(df_filtered[df_filtered['category'] == cat])
        print(f"  {cat}: {n} comparisons")

    return df_filtered, img_map


def compute_trueskill(df_filtered):
    """Compute TrueSkill ratings per image per category."""
    print("\n" + "=" * 60)
    print("Step 2: Computing TrueSkill ratings")
    print("=" * 60)

    cache_path = os.path.join(SPECS_OUTPUT_DIR, "specs_trueskill.csv")
    if os.path.exists(cache_path):
        print(f"  Loading cached TrueSkill from {cache_path}")
        return pd.read_csv(cache_path)

    all_ratings = []
    for cat in CATEGORIES:
        cat_data = df_filtered[df_filtered['category'] == cat]
        specs_cat = [k for k, v in SPECS_CAT_MAP.items() if v == cat][0]

        # Initialize TrueSkill ratings for all images
        ratings = {}
        for img_num in pd.concat([cat_data['Left_image'], cat_data['Right_image']]).unique():
            ratings[img_num] = TS_ENV.create_rating()

        # Process each comparison
        for _, row in cat_data.iterrows():
            left_id = row['Left_image']
            right_id = row['Right_image']
            score = row['Score']

            r_left = ratings[left_id]
            r_right = ratings[right_id]

            if score == 'left':
                new_left, new_right = TS_ENV.rate_1vs1(r_left, r_right)
            elif score == 'right':
                new_right, new_left = TS_ENV.rate_1vs1(r_right, r_left)
            else:  # equal
                new_left, new_right = TS_ENV.rate_1vs1(r_left, r_right, drawn=True)

            ratings[left_id] = new_left
            ratings[right_id] = new_right

        for img_num, rating in ratings.items():
            all_ratings.append({
                'image_number': img_num,
                'category': cat,
                'mu': rating.mu,
                'sigma': rating.sigma
            })

        print(f"  {cat}: {len(ratings)} images rated from {len(cat_data)} comparisons")

    ts_df = pd.DataFrame(all_ratings)
    ts_df.to_csv(cache_path, index=False)
    print(f"  Saved to {cache_path}")
    return ts_df


def aggregate_pairs(df_filtered):
    """Aggregate single-vote pairs to get majority-vote labels."""
    print("\n" + "=" * 60)
    print("Step 3: Aggregating pairwise labels (majority vote)")
    print("=" * 60)

    results = {}
    for cat in CATEGORIES:
        cat_data = df_filtered[df_filtered['category'] == cat]

        # Group by (Left_image, Right_image) and take majority vote
        grouped = cat_data.groupby(['Left_image', 'Right_image'])['Score'].agg(
            lambda x: x.value_counts().index[0]  # Most common label
        ).reset_index()
        grouped.columns = ['left_id', 'right_id', 'winner']
        grouped['category'] = cat

        results[cat] = grouped
        print(f"  {cat}: {len(grouped)} unique pairs")

    return results


def sample_and_split(agg_pairs, ts_df):
    """Sample pairs and split into ref/val/test."""
    print("\n" + "=" * 60)
    print("Step 4: Sampling and splitting data")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)
    splits = {}

    for cat in CATEGORIES:
        pairs = agg_pairs[cat].copy()

        # Sample if needed
        if len(pairs) > MAX_PAIRS_PER_CAT:
            pairs = pairs.sample(n=MAX_PAIRS_PER_CAT, random_state=RANDOM_SEED)

        # Shuffle
        pairs = pairs.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

        # Split: 60% ref, 20% val, 20% test
        n = len(pairs)
        n_ref = int(n * LABELED_SET_RATIO)
        n_val = int(n * VAL_RATIO)

        ref = pairs.iloc[:n_ref]
        val = pairs.iloc[n_ref:n_ref + n_val]
        test = pairs.iloc[n_ref + n_val:]

        # Add TrueSkill scores
        ts_cat = ts_df[ts_df['category'] == cat].set_index('image_number')

        for split_name, split_df in [('ref', ref), ('val', val), ('test', test)]:
            split_df = split_df.copy()
            split_df['left_mu'] = split_df['left_id'].map(ts_cat['mu'])
            split_df['right_mu'] = split_df['right_id'].map(ts_cat['mu'])
            split_df['ts_diff'] = split_df['left_mu'] - split_df['right_mu']
            if split_name == 'ref':
                ref = split_df
            elif split_name == 'val':
                val = split_df
            else:
                test = split_df

        splits[cat] = {'ref': ref, 'val': val, 'test': test}
        print(f"  {cat}: ref={len(ref)}, val={len(val)}, test={len(test)}")

    return splits


def extract_clip_features(img_map):
    """Extract CLIP features for all SPECS images."""
    print("\n" + "=" * 60)
    print("Step 5: Extracting CLIP features")
    print("=" * 60)

    cache_path = os.path.join(SPECS_OUTPUT_DIR, "specs_clip_embeddings.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        print(f"  Loaded cached CLIP embeddings: {len(data['image_numbers'])} images")
        return dict(zip(data['image_numbers'].tolist(), data['embeddings']))

    try:
        import torch
        import clip
        from PIL import Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-L/14", device=device)
        print(f"  CLIP loaded on {device}")

        embeddings = {}
        image_numbers = []
        embed_list = []

        for img_num, info in sorted(img_map.items()):
            img_path = info['path']
            if not os.path.exists(img_path):
                print(f"  WARNING: Missing image {img_path}")
                continue

            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(image)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                feat = feat.cpu().numpy().flatten()

            embeddings[img_num] = feat
            image_numbers.append(img_num)
            embed_list.append(feat)

            if len(embeddings) % 50 == 0:
                print(f"  Processed {len(embeddings)}/{len(img_map)} images")

        np.savez(cache_path,
                 image_numbers=np.array(image_numbers),
                 embeddings=np.array(embed_list))
        print(f"  Saved {len(embeddings)} CLIP embeddings to {cache_path}")
        return embeddings

    except ImportError:
        print("  WARNING: CLIP not available. Using random embeddings for testing.")
        embeddings = {}
        for img_num in img_map:
            embeddings[img_num] = np.random.randn(768).astype(np.float32)
            embeddings[img_num] /= np.linalg.norm(embeddings[img_num])
        return embeddings


def load_pp_dimensions():
    """Load Place Pulse Stage 1 dimensions for transfer."""
    print("\n" + "=" * 60)
    print("Step 6: Loading Place Pulse dimensions (transfer)")
    print("=" * 60)

    dims = {}
    for cat in CATEGORIES:
        dim_file = os.path.join(PP_OUTPUT_DIR, f"stage1_semantic_dimensions_{cat}.json")
        if not os.path.exists(dim_file):
            # Try unified file
            dim_file = os.path.join(PP_OUTPUT_DIR, "stage1_semantic_dimensions.json")

        if os.path.exists(dim_file):
            with open(dim_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if cat in data:
                dim_list = data[cat].get('dimensions', data[cat].get('semantic_dimensions', []))
            else:
                dim_list = data.get('dimensions', [])

            dims[cat] = dim_list
            dim_names = [d['name'] for d in dim_list]
            print(f"  {cat}: {len(dim_list)} dimensions — {', '.join(dim_names[:4])}...")
        else:
            print(f"  WARNING: No dimensions found for {cat}")
            dims[cat] = []

    return dims


def encode_image_base64(img_path, max_size=512, quality=70):
    """Encode image to base64 for VLM API, resizing to fit API limits."""
    from PIL import Image
    from io import BytesIO

    img = Image.open(img_path)
    # Resize keeping aspect ratio
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


_SESSION = None

def _get_session():
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        })
    return _SESSION


def call_vlm_api(messages, temperature=0.0, max_retries=5):
    """Call VLM API with retry and SSL resilience."""
    session = _get_session()
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2000
    }

    for attempt in range(max_retries):
        try:
            resp = session.post(BASE_URL, json=payload, timeout=60, verify=False,
                                proxies={'http': None, 'https': None})
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
            elif resp.status_code == 400 and 'TooBig' in resp.text:
                print(f"  RequestDataTooBig — skipping this pair")
                return None
            else:
                print(f"  API error {resp.status_code}: {resp.text[:200]}")
                time.sleep(2 ** attempt)
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            wait = min(2 ** attempt + 1, 16)
            print(f"  SSL/Connection error (attempt {attempt+1}/{max_retries}), retry in {wait}s...")
            time.sleep(wait)
            # Reset session on SSL errors
            global _SESSION
            _SESSION = None
        except Exception as e:
            print(f"  Request error: {e}")
            time.sleep(2 ** attempt)

    return None


def parse_json_response(response):
    """Parse JSON from VLM response, handling markdown code blocks."""
    if response is None:
        return None
    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        print(f"  Failed to parse JSON: {response[:200]}")
        return None


def score_pair_mode4(left_path, right_path, dimensions, category):
    """Score a pair using Mode 4 (Observer→Debater→Judge, 3 API calls)."""
    dim_names = [d['name'] for d in dimensions]
    dim_desc = "\n".join([f"- {d['name']}: {d.get('description', d['name'])}" for d in dimensions])

    left_b64 = encode_image_base64(left_path)
    right_b64 = encode_image_base64(right_path)

    img_content = [
        {"type": "text", "text": "Image A:"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{left_b64}"}},
        {"type": "text", "text": "Image B:"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{right_b64}"}}
    ]

    # Step 1: Observer
    obs_prompt = f"""Compare VISUAL DIFFERENCES between A and B for "{category}".

[DIMENSIONS]
{dim_desc}

Describe contrasts (A has X, B has Y) in 3-5 sentences.
"""
    observation = call_vlm_api([
        {"role": "system", "content": "Observer focusing on differences."},
        {"role": "user", "content": [{"type": "text", "text": obs_prompt}] + img_content}
    ], temperature=0.3)

    if not observation:
        return None
    time.sleep(0.5)

    # Step 2: Debater
    deb_prompt = f"""Argue for BOTH A and B on each dimension.

[COMPARISON]
{observation}

[DIMENSIONS]
{dim_desc}

For each dimension:
- Why A scores HIGH? Why A scores LOW?
- Why B scores HIGH? Why B scores LOW?

Be concise (2-3 sentences per dimension).
"""
    debate = call_vlm_api([
        {"role": "system", "content": "Debater arguing both sides."},
        {"role": "user", "content": [{"type": "text", "text": deb_prompt}] + img_content}
    ], temperature=0.5)

    if not debate:
        return None
    time.sleep(0.5)

    # Step 3: Judge
    judge_prompt = f"""Final comparative judgment for "{category}".

[OBSERVATION]
{observation}

[DEBATE]
{debate}

[DIMENSIONS]
{dim_desc}

Rate BOTH images, and determine winner.

Output JSON:
{{
  "scores_A": {{{', '.join([f'"{d}": score' for d in dim_names])}}},
  "scores_B": {{{', '.join([f'"{d}": score' for d in dim_names])}}},
  "winner": "left"/"right"/"equal",
  "intensity": <float, sum of absolute score differences>
}}"""
    response = call_vlm_api([
        {"role": "system", "content": "Impartial judge. JSON only."},
        {"role": "user", "content": [{"type": "text", "text": judge_prompt}] + img_content}
    ], temperature=0.1)

    return parse_json_response(response)


def run_stage2_scoring(splits, dims, img_map):
    """Run Stage 2 VLM scoring on all splits with checkpoint resume."""
    print("\n" + "=" * 60)
    print("Step 7: Running Stage 2 VLM scoring (Mode 4: Observer→Debater→Judge)")
    print("=" * 60)

    cache_path = os.path.join(SPECS_OUTPUT_DIR, "specs_stage2_scored.csv")
    checkpoint_path = os.path.join(SPECS_OUTPUT_DIR, "specs_stage2_checkpoint.csv")

    # Check if final cache exists (complete)
    total_pairs = sum(len(splits[cat]['ref']) + len(splits[cat]['val']) + len(splits[cat]['test'])
                      for cat in CATEGORIES)
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        if len(cached) >= total_pairs * 0.95:  # Allow 5% failed pairs
            print(f"  Loading complete scored data from {cache_path} ({len(cached)} pairs)")
            return cached

    # Resume from checkpoint OR scored cache (whichever has more data)
    all_results = []
    scored_keys = set()

    # Check both checkpoint and scored cache for resume data
    resume_sources = []
    if os.path.exists(checkpoint_path):
        resume_sources.append(('checkpoint', checkpoint_path))
    if os.path.exists(cache_path):
        resume_sources.append(('scored_cache', cache_path))

    for source_name, source_path in resume_sources:
        try:
            prev = pd.read_csv(source_path)
            prev_keys = {(r['category'], r['left_id'], r['right_id']) for r in prev.to_dict('records')}
            if len(prev_keys) > len(scored_keys):
                all_results = prev.to_dict('records')
                scored_keys = prev_keys
                print(f"  Resuming from {source_name}: {len(all_results)} pairs already scored")
        except Exception as e:
            print(f"  Warning: could not read {source_name}: {e}")

    processed = len(all_results)
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 10  # Stop after 10 consecutive API failures

    for cat in CATEGORIES:
        cat_dims = dims[cat]
        if not cat_dims:
            print(f"  Skipping {cat}: no dimensions")
            continue

        dim_names = [d['name'] for d in cat_dims]

        for split_name in ['ref', 'val', 'test']:
            split_df = splits[cat][split_name]

            for idx, row in split_df.iterrows():
                left_id = row['left_id']
                right_id = row['right_id']

                # Skip already scored pairs
                if (cat, left_id, right_id) in scored_keys:
                    continue

                left_info = img_map.get(left_id, {})
                right_info = img_map.get(right_id, {})

                left_path = left_info.get('path', '')
                right_path = right_info.get('path', '')

                if not os.path.exists(left_path) or not os.path.exists(right_path):
                    print(f"  Missing images for pair ({left_id}, {right_id})")
                    continue

                result = score_pair_mode4(left_path, right_path, cat_dims, cat)
                processed += 1

                if result is None:
                    consecutive_failures += 1
                    print(f"  Failed pair ({left_id}, {right_id}) for {cat} [consecutive failures: {consecutive_failures}]")
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"\n  *** ABORTING: {MAX_CONSECUTIVE_FAILURES} consecutive failures (likely quota exhausted) ***")
                        print(f"  Saving checkpoint with {len(all_results)} scored pairs...")
                        pd.DataFrame(all_results).to_csv(checkpoint_path, index=False)
                        # Also save to scored cache for safety
                        pd.DataFrame(all_results).to_csv(cache_path, index=False)
                        print(f"  Checkpoint saved. Re-run script after recharging API quota.")
                        return pd.DataFrame(all_results) if all_results else pd.DataFrame()
                    continue

                # Reset consecutive failure counter on success
                consecutive_failures = 0

                # Extract scores
                scores_a = result.get('scores_A', {})
                scores_b = result.get('scores_B', {})

                record = {
                    'category': cat,
                    'split': split_name,
                    'left_id': left_id,
                    'right_id': right_id,
                    'human_winner': row['winner'],
                    'synthetic_winner': result.get('winner', 'unknown'),
                    'intensity': result.get('intensity', 0),
                    'left_mu': row.get('left_mu', 0),
                    'right_mu': row.get('right_mu', 0),
                    'ts_diff': row.get('ts_diff', 0),
                }

                # Add per-dimension scores
                for i, dname in enumerate(dim_names):
                    record[f'score_A_{i}'] = scores_a.get(dname, 5.0)
                    record[f'score_B_{i}'] = scores_b.get(dname, 5.0)
                    record[f'dim_name_{i}'] = dname
                record['n_dims'] = len(dim_names)

                all_results.append(record)

                if processed % 5 == 0:
                    print(f"  Progress: {processed}/{total_pairs} pairs ({100*processed/total_pairs:.0f}%)")
                    # Save checkpoint (NOT the final cache path)
                    pd.DataFrame(all_results).to_csv(checkpoint_path, index=False)

                time.sleep(0.5)  # Rate limiting

    scored_df = pd.DataFrame(all_results)
    # Save scored data
    scored_df.to_csv(cache_path, index=False)
    # Only remove checkpoint if we scored enough pairs (>90% of total)
    if len(scored_df) >= total_pairs * 0.90:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        print(f"\n  COMPLETE: Saved {len(scored_df)} scored pairs to {cache_path}")
    else:
        # Keep checkpoint for resume
        scored_df.to_csv(checkpoint_path, index=False)
        print(f"\n  PARTIAL: Saved {len(scored_df)}/{total_pairs} scored pairs. Re-run to continue.")
    return scored_df


def run_lwrr(scored_df, clip_embeddings):
    """Run Stage 3 LWRR calibration."""
    print("\n" + "=" * 60)
    print("Step 8: Running LWRR calibration")
    print("=" * 60)

    from scipy.spatial.distance import cosine

    results = []

    for cat in CATEGORIES:
        cat_data = scored_df[scored_df['category'] == cat]
        if len(cat_data) == 0:
            continue

        ref_data = cat_data[cat_data['split'] == 'ref']
        test_data = cat_data[cat_data['split'] == 'test']

        if len(ref_data) == 0 or len(test_data) == 0:
            print(f"  {cat}: insufficient data (ref={len(ref_data)}, test={len(test_data)})")
            continue

        n_dims = int(cat_data.iloc[0]['n_dims'])

        # Build reference manifold
        ref_vectors = []
        ref_sem_diffs = []
        ref_ts_diffs = []
        ref_winners = []

        for _, row in ref_data.iterrows():
            left_id = int(row['left_id'])
            right_id = int(row['right_id'])

            # CLIP differential
            clip_left = clip_embeddings.get(left_id)
            clip_right = clip_embeddings.get(right_id)
            if clip_left is None or clip_right is None:
                continue

            clip_diff = clip_left - clip_right
            clip_diff = clip_diff / (np.linalg.norm(clip_diff) + 1e-8)

            # Semantic differential
            sem_a = np.array([row.get(f'score_A_{i}', 5.0) for i in range(n_dims)])
            sem_b = np.array([row.get(f'score_B_{i}', 5.0) for i in range(n_dims)])
            sem_diff = (sem_a - sem_b) / 10.0  # Normalize to [0,1]

            # Hybrid vector
            alpha = ALPHA_DEFAULT
            hybrid = np.concatenate([alpha * clip_diff, (1 - alpha) * sem_diff])

            ref_vectors.append(hybrid)
            ref_sem_diffs.append(sem_diff)
            ref_ts_diffs.append(row['ts_diff'])
            ref_winners.append(row['human_winner'])

            # Mirror augmentation
            ref_vectors.append(-hybrid)
            ref_sem_diffs.append(-sem_diff)
            ref_ts_diffs.append(-row['ts_diff'])
            mirror_winner = 'right' if row['human_winner'] == 'left' else ('left' if row['human_winner'] == 'right' else 'equal')
            ref_winners.append(mirror_winner)

        if len(ref_vectors) < 10:
            print(f"  {cat}: too few reference vectors ({len(ref_vectors)})")
            continue

        ref_vectors = np.array(ref_vectors)
        ref_sem_diffs = np.array(ref_sem_diffs)
        ref_ts_diffs = np.array(ref_ts_diffs)

        # Process test pairs
        test_preds = []
        test_true = []
        raw_preds = []

        for _, row in test_data.iterrows():
            left_id = int(row['left_id'])
            right_id = int(row['right_id'])

            clip_left = clip_embeddings.get(left_id)
            clip_right = clip_embeddings.get(right_id)
            if clip_left is None or clip_right is None:
                continue

            clip_diff = clip_left - clip_right
            clip_diff = clip_diff / (np.linalg.norm(clip_diff) + 1e-8)

            sem_a = np.array([row.get(f'score_A_{i}', 5.0) for i in range(n_dims)])
            sem_b = np.array([row.get(f'score_B_{i}', 5.0) for i in range(n_dims)])
            sem_diff = (sem_a - sem_b) / 10.0

            alpha = ALPHA_DEFAULT
            query = np.concatenate([alpha * clip_diff, (1 - alpha) * sem_diff])

            # Find K nearest neighbors
            K = min(K_MAX_DEFAULT, len(ref_vectors))
            sims = ref_vectors @ query / (np.linalg.norm(ref_vectors, axis=1) * np.linalg.norm(query) + 1e-8)
            top_k = np.argsort(sims)[-K:]

            # Kernel weighting
            weights = np.exp(sims[top_k] / TAU_KERNEL_DEFAULT)

            # Local ridge regression
            X_local = ref_sem_diffs[top_k]
            y_local = ref_ts_diffs[top_k]
            W = np.diag(weights)

            try:
                XtWX = X_local.T @ W @ X_local + RIDGE_ALPHA_DEFAULT * np.eye(n_dims)
                XtWy = X_local.T @ W @ y_local
                w_hat = np.linalg.solve(XtWX, XtWy)

                delta = w_hat @ sem_diff

                # Equal consensus
                n_equal = sum(1 for i in top_k if ref_winners[i] == 'equal')
                eq_ratio = n_equal / K

                if abs(delta) < EQUAL_EPS_DEFAULT or eq_ratio > 0.6:
                    pred = 'equal'
                elif delta > 0:
                    pred = 'left'
                else:
                    pred = 'right'

            except np.linalg.LinAlgError:
                pred = row['synthetic_winner']

            test_preds.append(pred)
            test_true.append(row['human_winner'])
            raw_preds.append(row['synthetic_winner'])

        if len(test_preds) == 0:
            continue

        # Compute metrics (excluding equal)
        mask_excl = [t != 'equal' for t in test_true]
        if sum(mask_excl) > 0:
            true_excl = [t for t, m in zip(test_true, mask_excl) if m]
            pred_excl = [p for p, m in zip(test_preds, mask_excl) if m]
            raw_excl = [r for r, m in zip(raw_preds, mask_excl) if m]

            acc_aligned = accuracy_score(true_excl, pred_excl)
            kappa_aligned = cohen_kappa_score(true_excl, pred_excl)
            acc_raw = accuracy_score(true_excl, raw_excl)
            kappa_raw = cohen_kappa_score(true_excl, raw_excl)
        else:
            acc_aligned = acc_raw = kappa_aligned = kappa_raw = 0

        # Including equal
        acc_incl = accuracy_score(test_true, test_preds)
        kappa_incl = cohen_kappa_score(test_true, test_preds)

        results.append({
            'category': cat,
            'n_ref': len(ref_data),
            'n_test': len(test_data),
            'n_test_excl': sum(mask_excl),
            'raw_acc_excl': acc_raw,
            'raw_kappa_excl': kappa_raw,
            'aligned_acc_excl': acc_aligned,
            'aligned_kappa_excl': kappa_aligned,
            'aligned_acc_incl': acc_incl,
            'aligned_kappa_incl': kappa_incl,
            'delta_pp': (acc_aligned - acc_raw) * 100,
        })

        print(f"  {cat}: Raw={acc_raw:.1%} → Aligned={acc_aligned:.1%} "
              f"(Δ={100*(acc_aligned-acc_raw):+.1f}pp), κ={kappa_aligned:.3f}, n_test={sum(mask_excl)}")

    return pd.DataFrame(results)


def run_zero_shot_baseline(splits, img_map):
    """Run zero-shot VLM baseline on test split."""
    print("\n" + "=" * 60)
    print("Step 9: Zero-shot VLM baseline")
    print("=" * 60)

    cache_path = os.path.join(SPECS_OUTPUT_DIR, "specs_zeroshot_results.csv")
    if os.path.exists(cache_path):
        print(f"  Loading cached results from {cache_path}")
        return pd.read_csv(cache_path)

    results = []

    for cat in CATEGORIES:
        test_df = splits[cat]['test']
        preds = []
        trues = []

        for _, row in test_df.iterrows():
            left_id = row['left_id']
            right_id = row['right_id']

            left_info = img_map.get(left_id, {})
            right_info = img_map.get(right_id, {})
            left_path = left_info.get('path', '')
            right_path = right_info.get('path', '')

            if not os.path.exists(left_path) or not os.path.exists(right_path):
                continue

            prompt = f'Which image looks more {cat}? Answer with ONLY "left" or "right".'

            left_b64 = encode_image_base64(left_path)
            right_b64 = encode_image_base64(right_path)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{left_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{right_b64}"}}
                ]
            }]

            response = call_vlm_api(messages, temperature=0.0)
            if response:
                resp_lower = response.strip().lower()
                if 'left' in resp_lower:
                    pred = 'left'
                elif 'right' in resp_lower:
                    pred = 'right'
                else:
                    pred = 'equal'
                preds.append(pred)
                trues.append(row['winner'])

            time.sleep(0.5)

        if preds:
            mask_excl = [t != 'equal' for t in trues]
            if sum(mask_excl) > 0:
                true_excl = [t for t, m in zip(trues, mask_excl) if m]
                pred_excl = [p for p, m in zip(preds, mask_excl) if m]
                acc = accuracy_score(true_excl, pred_excl)
                kappa = cohen_kappa_score(true_excl, pred_excl)
            else:
                acc = kappa = 0

            results.append({
                'category': cat,
                'acc_excl': acc,
                'kappa_excl': kappa,
                'n_test': len(preds)
            })
            print(f"  {cat}: Zero-shot Acc={acc:.1%}, κ={kappa:.3f}, n={sum(mask_excl)}")

    zs_df = pd.DataFrame(results)
    zs_df.to_csv(cache_path, index=False)
    return zs_df


def main():
    print("=" * 60)
    print("SPECS Cross-Dataset Transfer Experiment")
    print("=" * 60)

    # Step 1: Load data
    df_filtered, img_map = load_specs_data()

    # Step 2: TrueSkill
    ts_df = compute_trueskill(df_filtered)

    # Step 3: Aggregate pairs
    agg_pairs = aggregate_pairs(df_filtered)

    # Step 4: Sample and split
    splits = sample_and_split(agg_pairs, ts_df)

    # Step 5: CLIP features
    clip_embeddings = extract_clip_features(img_map)

    # Step 6: Load PP dimensions
    dims = load_pp_dimensions()

    # Step 7: Stage 2 scoring
    scored_df = run_stage2_scoring(splits, dims, img_map)

    # Step 8: LWRR
    lwrr_results = run_lwrr(scored_df, clip_embeddings)

    # Step 9: Zero-shot baseline (on test only)
    # Uncomment to also run zero-shot:
    # zs_results = run_zero_shot_baseline(splits, img_map)

    # Save final results
    output_path = os.path.join(SPECS_OUTPUT_DIR, "specs_transfer_results.csv")
    lwrr_results.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("FINAL RESULTS: SPECS Transfer Experiment")
    print("=" * 60)
    print(lwrr_results.to_string(index=False))

    # Summary
    avg_raw = lwrr_results['raw_acc_excl'].mean()
    avg_aligned = lwrr_results['aligned_acc_excl'].mean()
    avg_kappa = lwrr_results['aligned_kappa_excl'].mean()
    print(f"\nAverage: Raw={avg_raw:.1%} → Aligned={avg_aligned:.1%} "
          f"(Δ={100*(avg_aligned-avg_raw):+.1f}pp), κ={avg_kappa:.3f}")


if __name__ == '__main__':
    main()
