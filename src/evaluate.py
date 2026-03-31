import matplotlib.pyplot as plt

models = ["Baseline", "Hybrid"]
f1_scores = [0.5154, 0.51]

plt.figure()
plt.bar(models, f1_scores)

plt.xlabel("Model")
plt.ylabel("Macro F1 Score")
plt.title("Baseline vs Hybrid Emotion Classification")

for i, v in enumerate(f1_scores):
    plt.text(i, v + 0.001, f"{v:.4f}", ha='center')

plt.ylim(0.48, 0.53)

plt.savefig("comparison_chart.png")
plt.show()