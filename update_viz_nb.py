import json
import os

NOTEBOOK_PATH = "c:/Users/REHAN SHAIK/Downloads/MLProjectHousePricePrediction/notebooks/datavisualization.ipynb"

with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # 1. Update load_models cell
        if "    metrics['SGD Adaptive'] = json.load(f)\n" in cell['source']:
            idx = cell['source'].index("    metrics['SGD Adaptive'] = json.load(f)\n")
            if "models['Decision Tree'] = joblib.load(RESULTS_DIR / 'decision_tree.joblib')\n" not in cell['source']:
                cell['source'].insert(idx + 1, "\n")
                cell['source'].insert(idx + 2, "# Decision Tree\n")
                cell['source'].insert(idx + 3, "models['Decision Tree'] = joblib.load(RESULTS_DIR / 'decision_tree.joblib')\n")
                cell['source'].insert(idx + 4, "predictions['Decision Tree'] = np.load(RESULTS_DIR / 'pred_decision_tree.npy')\n")
                cell['source'].insert(idx + 5, "with open(RESULTS_DIR / 'metrics_decision_tree.json') as f:\n")
                cell['source'].insert(idx + 6, "    metrics['Decision Tree'] = json.load(f)\n")
                cell['source'].insert(idx + 7, "\n")
                cell['source'].insert(idx + 8, "# Neural Network\n")
                cell['source'].insert(idx + 9, "models['Neural Network'] = joblib.load(RESULTS_DIR / 'neural_network.joblib')\n")
                cell['source'].insert(idx + 10, "predictions['Neural Network'] = np.load(RESULTS_DIR / 'pred_neural_network.npy')\n")
                cell['source'].insert(idx + 11, "with open(RESULTS_DIR / 'metrics_neural_network.json') as f:\n")
                cell['source'].insert(idx + 12, "    metrics['Neural Network'] = json.load(f)\n")

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Notebook updated.")
