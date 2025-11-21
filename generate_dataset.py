import json
import numpy as np
import random
from collections import defaultdict

# -----------------------------
# Load data
# -----------------------------

features = np.load("train_features.npy")     # shape (N, D)
with open("data/train_data.json", "r") as f:
    train_data = json.load(f)               # list of dicts, length N

N, D = features.shape
D_metric = 768
D_other = D - D_metric

# -----------------------------
# Build metric -> score list & indices
# -----------------------------
metric_to_scores = defaultdict(list)
metric_to_indices = defaultdict(list)

for i, item in enumerate(train_data):
    m = item["metric_name"]
    s = float(item["score"])
    metric_to_scores[m].append(s)
    metric_to_indices[m].append(i)

# -----------------------------
# Helper functions
# -----------------------------

def pick_random_metric_embedding(exclude_index=None):
    """Pick metric-meaning embedding from a random other sample."""
    while True:
        idx = random.randint(0, N - 1)
        if idx != exclude_index:
            return features[idx, :D_metric].copy()

def pick_random_index_with_score_below_5(metric):
    idxs = [i for i in metric_to_indices[metric] if float(train_data[i]["score"]) < 5]
    return random.choice(idxs)

def pick_random_index_with_score_above_or_equal_5(metric):
    idxs = [i for i in metric_to_indices[metric] if float(train_data[i]["score"]) >= 5]
    return random.choice(idxs)

# ----------- PATCHED FUNCTION -----------
def pr_mix_and_score(idx_low, idx_high, low_weight_range=(0.5, 0.9)):
    """
    Mix PR embeddings, not metric embeddings.
    PR_low and PR_high live in features[:, D_metric:].
    """
    l = random.uniform(*low_weight_range)

    pr_low = features[idx_low, D_metric:]
    pr_high = features[idx_high, D_metric:]

    pr_mix = l * pr_low + (1 - l) * pr_high

    s_low = float(train_data[idx_low]["score"])
    s_high = float(train_data[idx_high]["score"])
    s_mix = l * s_low + (1 - l) * s_high

    norm_1 = np.linalg.norm(pr_mix[:768])
    if norm_1 > 0:
        pr_mix[:768] = pr_mix[:768] / norm_1

    norm_2 = np.linalg.norm(pr_mix[768:])
    if norm_2 > 0:
        pr_mix[768:] = pr_mix[768:] / norm_2


    return pr_mix, s_mix
# ----------------------------------------


# -----------------------------
# Generate augmented dataset
# -----------------------------

aug_features = []
aug_scores = []

for idx in range(N):

    item = train_data[idx]
    metric = item["metric_name"]

    scores_for_metric = metric_to_scores[metric]
    only_high = all(s >= 8 for s in scores_for_metric)
    has_low = any(s < 5 for s in scores_for_metric)

    base_feature = features[idx].copy()

    # -----------------------------------------------------
    # CASE A: Metric has only high (>= 7) scores
    # -----------------------------------------------------
    if only_high:
        new_feat = base_feature.copy()
        new_feat[:D_metric] = pick_random_metric_embedding(exclude_index=idx)
        aug_features.append(new_feat)
        aug_scores.append(0.0)

    # -----------------------------------------------------
    # CASE B (PATCHED): PR Mixup for metrics with low scores
    # -----------------------------------------------------
    if has_low:
        try:
            idx_low = pick_random_index_with_score_below_5(metric)
            idx_high = pick_random_index_with_score_above_or_equal_5(metric)
        except IndexError:
            continue

        for _ in range(3):
            pr_mix, s_mix = pr_mix_and_score(idx_low, idx_high)
            new_feat = base_feature.copy()
            new_feat[D_metric:] = pr_mix    # PATCH: mix PR, not metric
            aug_features.append(new_feat)
            aug_scores.append(s_mix)

# -----------------------------
# Convert arrays / save results
# -----------------------------

aug_features = np.array(aug_features) if len(aug_features) else np.empty((0, D))
aug_scores = np.clip(np.round(np.array(aug_scores, float)), 0, 10)

# np.save("augmented_features.npy", aug_features)
# np.save("augmented_scores.npy", aug_scores)

# print("Saved augmented_features.npy and augmented_scores.npy")
# print("Augmented samples:", len(aug_features))

orig_scores = np.clip(np.round(np.array([float(item["score"]) for item in train_data])), 0, 10)
full_features = np.vstack([features, aug_features]) if len(aug_features) else features.copy()
full_scores = np.concatenate([orig_scores, aug_scores]) if len(aug_scores) else orig_scores.copy()

np.save("full_features.npy", full_features)
np.save("full_scores.npy", full_scores)

print("Saved full_features.npy and full_scores.npy")
print("Full dataset size:", full_features.shape[0])
