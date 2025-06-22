from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from histobench.evaluation.compute_embeddings_tcga_ut import load_embeddings
from histobench.evaluation.evaluate_lunghist700 import custom_balanced_group_kfold

# ---- User variables ----
EMBEDDINGS_PATH = "data/embeddings/lunghist700/UNI2_LungHist700_10x_whole_roi.h5"
CSV_METADATA = "data/LungHist700/metadata.csv"
N_SPLITS = 4
LABEL_COLUMN = "superclass"  # or "label" depending on your metadata
GROUP_COLUMN = "patient_id"
# ------------------------

# Load embeddings and metadata
embeddings, image_paths = load_embeddings(EMBEDDINGS_PATH)
filenames = [Path(path).stem for path in image_paths]
metadata_df = pd.read_csv(CSV_METADATA).set_index("filename")

# Get group and label arrays
groups = metadata_df.loc[filenames, GROUP_COLUMN].values
labels = metadata_df.loc[filenames, LABEL_COLUMN].values

# Convert labels to numeric if needed
unique_labels = sorted(set(labels))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y = np.array([label_map[label] for label in labels])

# Run the CV split and evaluate
cv = custom_balanced_group_kfold(embeddings, y, groups, n_splits=N_SPLITS, random_state=42)
all_test_groups = []

print(f"Evaluating {N_SPLITS}-fold CV splits...\n")
for fold_idx, (train_idx, test_idx) in enumerate(cv):
    train_labels = y[train_idx]
    test_labels = y[test_idx]
    train_groups = groups[train_idx]
    test_groups = groups[test_idx]
    all_test_groups.extend(test_groups)

    train_dist = dict(Counter(train_labels))
    test_dist = dict(Counter(test_labels))
    overlap = set(train_groups) & set(test_groups)

    print(f"Fold {fold_idx + 1}/{N_SPLITS}:")
    print(f"  Train class dist: {train_dist}")
    print(f"  Test class dist:  {test_dist}")
    print(f"  Train groups: {len(set(train_groups))}, Test groups: {len(set(test_groups))}")
    print(f"  Overlap groups: {overlap if overlap else 'None'}")
    print("")

# Check that all groups are used exactly once in test sets
test_group_counts = Counter(all_test_groups)
print(test_group_counts)

# Compute number of images per group in the whole dataset
images_per_group = Counter(groups)

# Now check if each group's test count matches its total image count
all_ok = True
for group, total_images in images_per_group.items():
    test_count = test_group_counts[group]
    if test_count != total_images:
        print(f"Group {group}: test_count={test_count}, total_images={total_images} [MISMATCH]")
        all_ok = False

if all_ok:
    print("\nAll groups are used exactly once as test set (no leakage, no missing).")
else:
    print("\nSome groups are missing or duplicated in test sets! Check your CV split.")


# fold_class_counts = []
# cv = custom_balanced_group_kfold(embeddings, y, groups, n_splits=N_SPLITS, random_state=42)
# for fold_idx, (train_idx, test_idx) in enumerate(cv):
#     test_labels = y[test_idx]
#     counts = np.bincount(test_labels, minlength=len(unique_labels))
#     fold_class_counts.append(counts)

# fold_class_counts = np.array(fold_class_counts)
# plt.figure(figsize=(8, 4))
# for i, label in enumerate(unique_labels):
#     plt.plot(range(1, N_SPLITS + 1), fold_class_counts[:, i], marker="o", label=str(label))
# plt.xlabel("Fold")
# plt.ylabel("Test set count")
# plt.title("Class distribution in test sets across folds")
# plt.legend(title="Class")
# plt.tight_layout()
# plt.show()
