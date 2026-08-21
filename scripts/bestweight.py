import numpy as np

path = "ckpt/ALMT_MOSI_Dual_C4_Intensity/best_validation_predictions.npz"
data = np.load(path)

reg = data["regression_predictions"].reshape(-1)
ord_pred = data["ordinal_predictions"].reshape(-1)
labels = data["labels"].reshape(-1)

true_class = np.round(np.clip(labels, -3, 3)).astype(int)
results = []

for rho in np.arange(0.0, 1.001, 0.01):
    fused = (1.0 - rho) * reg + rho * ord_pred
    pred_class = np.round(np.clip(fused, -3, 3)).astype(int)

    acc7 = np.mean(pred_class == true_class)
    mae = np.mean(np.abs(fused - labels))

    extreme_recalls = []
    for cls in (-3, 3):
        mask = true_class == cls
        recall = np.mean(pred_class[mask] == cls) if mask.any() else 0.0
        extreme_recalls.append(recall)

    extreme_recall = np.mean(extreme_recalls)
    results.append((acc7, extreme_recall, -mae, rho))

best = max(results)
print("best rho:", best[3])
print("validation Acc-7:", best[0])
print("extreme macro recall:", best[1])
print("validation MAE:", -best[2])

print("\nTop 10:")
for acc7, extreme_recall, negative_mae, rho in sorted(
    results, reverse=True
)[:10]:
    print(
        f"rho={rho:.2f}, Acc7={acc7:.4f}, "
        f"extreme_recall={extreme_recall:.4f}, MAE={-negative_mae:.4f}"
    )