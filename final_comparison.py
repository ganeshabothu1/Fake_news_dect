
import pandas as pd
import matplotlib.pyplot as plt
import os

# Results folder
os.makedirs("results", exist_ok=True)

# Load existing CSV files
lr = pd.read_csv("results/logistic_regression_results.csv")
image = pd.read_csv("results/image_model_comparison.csv")

# Handle image CSV model names
if "Model" not in image.columns:
    image = image.rename(
        columns={image.columns[0]: "Model"}
    )

# Combine all three models
final_df = pd.concat(
    [lr, image],
    ignore_index=True
)

# Keep required evaluation columns
columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC AUC"
]

final_df = final_df[columns]

# Save final comparison CSV
final_df.to_csv(
    "results/final_model_comparison.csv",
    index=False
)

print("\nFINAL MODEL COMPARISON")
print("=" * 65)
print(final_df.to_string(index=False))

# Generate comparison graph
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC AUC"
]

plot_df = final_df.set_index("Model")[metrics]

plot_df.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Final Comparison: Logistic Regression vs CNN vs MobileNetV2")
plt.ylabel("Performance Score")
plt.xlabel("Models")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/final_model_comparison.png",
    dpi=300
)

plt.show()

print("\nFinal comparison CSV and graph saved successfully!")
