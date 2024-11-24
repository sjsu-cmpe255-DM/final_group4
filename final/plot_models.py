

import matplotlib.pyplot as plt


models = ['KNN', 'Naive Bayes', 'Logistic Regression', 'Random Forest', 'Ensemble', 'Linear SVC']
accuracies = [52, 59, 62, 64, 65, 66]


plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])


plt.title('Comparison of Model Accuracies', fontsize=18, fontweight='bold')
plt.xlabel('Models', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.ylim(0, 100)  


for spine in plt.gca().spines.values():
    spine.set_visible(False)


for i, acc in enumerate(accuracies):
    plt.text(i, acc + 2, f"{acc}%", ha='center', fontsize=12, fontweight='bold', color='black')


plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.tight_layout()
plt.show()
