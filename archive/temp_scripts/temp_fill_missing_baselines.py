"""
Temporary script: Fill missing ResNet50 and Segmentation features for safety,
then re-run C0 (ResNet Siamese) and C2 (SegReg) baselines for safety ONLY.
"""
import os, sys, json
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

# ── Config ──
from config import IMAGE_DIR, PCA_DIMS
from abc_stage2_multi_mode_synthesis import get_split_data

OUTPUT_DIR = 'urbanalign_outputs'
RESNET_CACHE = os.path.join(OUTPUT_DIR, "resnet50_features.npz")
SEGMENTATION_CACHE = os.path.join(OUTPUT_DIR, "segmentation_features.npz")
CAT = 'safety'

# Cityscapes groups (same as stage7)
URBAN_ELEMENT_GROUPS = {
    'Flat':         [0, 1],
    'Construction': [2, 3, 4],
    'Object':       [5, 6, 7],
    'Nature':       [8, 9],
    'Sky':          [10],
    'Human':        [11, 12],
    'Vehicle':      [13, 14, 15, 16, 17, 18]
}


def find_image(image_id):
    for ext in ['.jpg', '.jpeg', '.png']:
        path = os.path.join(IMAGE_DIR, f"{image_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def get_missing_ids():
    """Find image IDs needed for safety but missing from caches."""
    df_ref, df_pool = get_split_data(category=CAT)
    needed = set()
    for df in [df_ref, df_pool]:
        needed.update(df['left_id'].astype(str).tolist())
        needed.update(df['right_id'].astype(str).tolist())

    # Load existing caches
    existing_resnet = set()
    if os.path.exists(RESNET_CACHE):
        data = np.load(RESNET_CACHE, allow_pickle=True)
        existing_resnet = set(data['image_ids'].tolist())

    existing_seg = set()
    if os.path.exists(SEGMENTATION_CACHE):
        data = np.load(SEGMENTATION_CACHE, allow_pickle=True)
        existing_seg = set(data['image_ids'].tolist())

    missing_resnet = needed - existing_resnet
    missing_seg = needed - existing_seg
    print(f"Safety needs {len(needed)} unique images")
    print(f"  Missing from ResNet cache:  {len(missing_resnet)}")
    print(f"  Missing from Seg cache:     {len(missing_seg)}")
    return missing_resnet, missing_seg


def fill_resnet_features(missing_ids):
    """Extract ResNet50 features for missing images and append to cache."""
    if not missing_ids:
        print("ResNet: nothing to fill")
        return

    import torch
    from torchvision import models, transforms
    from PIL import Image

    print(f"\n[ResNet50] Extracting features for {len(missing_ids)} images...")

    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    except:
        model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1])).eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    new_features = {}
    for img_id in tqdm(missing_ids, desc="ResNet50"):
        img_path = find_image(img_id)
        if not img_path:
            continue
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(tensor).flatten().cpu().numpy()
            new_features[img_id] = feat
        except Exception as e:
            print(f"  Skip {img_id}: {e}")

    print(f"  Extracted {len(new_features)} new ResNet features")

    # Merge with existing cache
    if os.path.exists(RESNET_CACHE):
        data = np.load(RESNET_CACHE, allow_pickle=True)
        old_ids = data['image_ids'].tolist()
        old_feats = data['features']
    else:
        old_ids, old_feats = [], np.empty((0, 2048))

    all_ids = old_ids + list(new_features.keys())
    all_feats = np.vstack([old_feats] + [new_features[k].reshape(1, -1) for k in new_features])
    np.savez(RESNET_CACHE, image_ids=np.array(all_ids), features=all_feats)
    print(f"  Updated cache: {len(all_ids)} total images")


def fill_segmentation_features(missing_ids):
    """Extract segmentation features for missing images and append to cache."""
    if not missing_ids:
        print("Segmentation: nothing to fill")
        return

    import torch
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    from PIL import Image

    print(f"\n[Segmentation] Extracting features for {len(missing_ids)} images...")

    model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    group_names = sorted(URBAN_ELEMENT_GROUPS.keys())
    new_features = {}

    for img_id in tqdm(missing_ids, desc="Segmentation"):
        img_path = find_image(img_id)
        if not img_path:
            continue
        try:
            img = Image.open(img_path).convert('RGB')
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                upsampled = torch.nn.functional.interpolate(
                    logits, size=img.size[::-1], mode='bilinear', align_corners=False
                )
                pred = upsampled.argmax(1)[0].cpu().numpy()

            total_pixels = pred.size
            proportions = []
            for g_name in group_names:
                g_indices = URBAN_ELEMENT_GROUPS[g_name]
                mask = np.isin(pred, g_indices)
                proportions.append(mask.sum() / total_pixels)
            new_features[img_id] = np.array(proportions, dtype=np.float32)
        except Exception as e:
            print(f"  Skip {img_id}: {e}")

    print(f"  Extracted {len(new_features)} new segmentation features")

    # Merge with existing cache
    if os.path.exists(SEGMENTATION_CACHE):
        data = np.load(SEGMENTATION_CACHE, allow_pickle=True)
        old_ids = data['image_ids'].tolist()
        old_feats = data['features']
    else:
        n_groups = len(URBAN_ELEMENT_GROUPS)
        old_ids, old_feats = [], np.empty((0, n_groups))

    all_ids = old_ids + list(new_features.keys())
    all_feats = np.vstack([old_feats] + [new_features[k].reshape(1, -1) for k in new_features])
    np.savez(SEGMENTATION_CACHE, image_ids=np.array(all_ids), features=all_feats)
    print(f"  Updated cache: {len(all_ids)} total images")


def rerun_c0_safety():
    """Re-run C0 ResNet Siamese for safety only."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print(f"\n{'='*60}\nRe-running C0 (ResNet Siamese) for safety\n{'='*60}")

    data = np.load(RESNET_CACHE, allow_pickle=True)
    resnet_map = dict(zip(data['image_ids'].tolist(), data['features']))

    class SiameseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.tower = nn.Sequential(
                nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 128), nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 3, 64), nn.ReLU(),
                nn.Linear(64, 3),
            )
        def forward(self, x_l, x_r):
            z_l, z_r = self.tower(x_l), self.tower(x_r)
            combined = torch.cat([z_l, z_r, torch.abs(z_l - z_r)], dim=1)
            return self.classifier(combined)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_map = {'left': 0, 'right': 1, 'equal': 2}
    inv_map = {0: 'left', 1: 'right', 2: 'equal'}

    df_ref, df_pool = get_split_data(category=CAT)

    def get_winner(row):
        return row.get('human_winner') or row.get('winner')

    def prepare_batch(df):
        Ls, Rs, Ys, raw = [], [], [], []
        for _, r in df.iterrows():
            fl = resnet_map.get(str(r['left_id']))
            fr = resnet_map.get(str(r['right_id']))
            if fl is not None and fr is not None:
                Ls.append(fl); Rs.append(fr)
                Ys.append(label_map.get(str(get_winner(r)).lower(), 2))
                raw.append(r)
        return (np.array(Ls), np.array(Rs), np.array(Ys), raw) if Ls else (None, None, None, [])

    ref_L, ref_R, ref_Y, _ = prepare_batch(df_ref)
    pool_L, pool_R, pool_Y, pool_raw = prepare_batch(df_pool)
    if ref_L is None or pool_L is None:
        print("  ERROR: Not enough data")
        return

    print(f"  Ref: {len(ref_L)} pairs, Pool: {len(pool_L)} pairs")

    # Mirror augmentation
    flip_map = {0: 1, 1: 0, 2: 2}
    ref_L_all = np.concatenate([ref_L, ref_R])
    ref_R_all = np.concatenate([ref_R, ref_L])
    ref_Y_all = np.concatenate([ref_Y, np.array([flip_map[y] for y in ref_Y])])

    # Train
    model = SiameseNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    ds = TensorDataset(
        torch.FloatTensor(ref_L_all), torch.FloatTensor(ref_R_all), torch.LongTensor(ref_Y_all)
    )
    loader = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=True)

    model.train()
    for epoch in range(100):
        for bL, bR, bY in loader:
            bL, bR, bY = bL.to(device), bR.to(device), bY.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bL, bR), bY)
            loss.backward()
            optimizer.step()

    # Predict on pool
    model.eval()
    results = []
    with torch.no_grad():
        pL = torch.FloatTensor(pool_L).to(device)
        pR = torch.FloatTensor(pool_R).to(device)
        logits = model(pL, pR)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    for i, r in enumerate(pool_raw):
        results.append({
            'left_id': str(r['left_id']),
            'right_id': str(r['right_id']),
            'human_winner': str(get_winner(r)).lower(),
            'synthetic_winner': inv_map[preds[i]],
            'prob_left': float(probs[i][0]),
            'prob_right': float(probs[i][1]),
            'prob_equal': float(probs[i][2]),
            'category': CAT,
        })

    out_path = os.path.join(OUTPUT_DIR, f'stage7_baseline_c0_siamese_resnet_{CAT}.csv')
    pd.DataFrame(results).to_csv(out_path, index=False)
    correct = sum(1 for r in results if r['synthetic_winner'] == r['human_winner'])
    print(f"  Saved {len(results)} rows → {out_path}")
    print(f"  Accuracy: {correct}/{len(results)} = {correct/len(results):.1%}")


def rerun_c2_safety():
    """Re-run C2 SegReg for safety only."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingClassifier

    print(f"\n{'='*60}\nRe-running C2 (SegReg) for safety\n{'='*60}")

    # Load features
    seg_data = np.load(SEGMENTATION_CACHE, allow_pickle=True)
    seg_map = dict(zip(seg_data['image_ids'].tolist(), seg_data['features']))

    clip_map = {}
    clip_path = os.path.join(OUTPUT_DIR, 'clip_embeddings.npz')
    if os.path.exists(clip_path):
        cdata = np.load(clip_path, allow_pickle=True)
        # CLIP npz uses keys 'paths' (full file paths) and 'embeddings', not 'image_ids'/'features'
        clip_ids = [os.path.splitext(os.path.basename(p))[0] for p in cdata['paths'].tolist()]
        clip_map = dict(zip(clip_ids, cdata['embeddings']))

    clip_pca = None
    if clip_map:
        all_clip_ids = list(clip_map.keys())
        all_clip_feats = np.array([clip_map[k] for k in all_clip_ids])
        clip_pca = PCA(n_components=PCA_DIMS, random_state=42)
        clip_pca.fit(all_clip_feats)
        print(f"  CLIP PCA: {all_clip_feats.shape[1]}d → {PCA_DIMS}d")

    df_ref, df_pool = get_split_data(category=CAT)
    print(f"  Ref: {len(df_ref)} pairs, Pool: {len(df_pool)} pairs")

    def get_winner(row):
        return row.get('human_winner') or row.get('winner')

    def build_features(df):
        X, y, rows = [], [], []
        for _, r in df.iterrows():
            l_id, r_id = str(r['left_id']), str(r['right_id'])
            s_l, s_r = seg_map.get(l_id), seg_map.get(r_id)
            if s_l is None or s_r is None:
                continue
            seg_diff = s_l - s_r
            if clip_pca is not None and l_id in clip_map and r_id in clip_map:
                c_l = clip_pca.transform(clip_map[l_id].reshape(1, -1))[0]
                c_r = clip_pca.transform(clip_map[r_id].reshape(1, -1))[0]
                feat = np.concatenate([seg_diff, c_l - c_r])
            else:
                feat = seg_diff
            X.append(feat)
            y.append(str(get_winner(r)).lower())
            rows.append(r)
        return np.array(X) if X else None, y, rows

    X_ref, y_ref, _ = build_features(df_ref)
    X_pool, y_pool, pool_rows = build_features(df_pool)
    if X_ref is None or X_pool is None:
        print("  ERROR: Not enough data")
        return

    print(f"  Ref features: {X_ref.shape}, Pool features: {X_pool.shape}")

    # Mirror augmentation
    X_ref_aug = np.vstack([X_ref, -X_ref])
    flip = {'left': 'right', 'right': 'left', 'equal': 'equal'}
    y_ref_aug = y_ref + [flip[y] for y in y_ref]

    # Train
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
    clf.fit(X_ref_aug, y_ref_aug)
    preds = clf.predict(X_pool)
    probs = clf.predict_proba(X_pool)

    results = []
    for i, r in enumerate(pool_rows):
        prob_dict = dict(zip(clf.classes_, probs[i]))
        results.append({
            'left_id': str(r['left_id']),
            'right_id': str(r['right_id']),
            'human_winner': str(get_winner(r)).lower(),
            'synthetic_winner': preds[i],
            'prob_left': float(prob_dict.get('left', 0)),
            'prob_right': float(prob_dict.get('right', 0)),
            'prob_equal': float(prob_dict.get('equal', 0)),
            'category': CAT,
        })

    out_path = os.path.join(OUTPUT_DIR, f'stage7_baseline_c2_segmentation_regression_{CAT}.csv')
    pd.DataFrame(results).to_csv(out_path, index=False)
    correct = sum(1 for r in results if r['synthetic_winner'] == r['human_winner'])
    print(f"  Saved {len(results)} rows → {out_path}")
    print(f"  Accuracy: {correct}/{len(results)} = {correct/len(results):.1%}")


if __name__ == '__main__':
    # Step 1: Find missing images
    missing_resnet, missing_seg = get_missing_ids()

    # Step 2: Extract missing features
    fill_resnet_features(missing_resnet)
    fill_segmentation_features(missing_seg)

    # Step 3: Re-run baselines for safety only
    rerun_c0_safety()
    rerun_c2_safety()

    print(f"\n{'='*60}")
    print("Done! Safety baselines C0 and C2 are now complete.")
    print(f"{'='*60}")
