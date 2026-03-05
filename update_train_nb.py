import json
import os

NOTEBOOK_PATH = "c:/Users/REHAN SHAIK/Downloads/MLProjectHousePricePrediction/notebooks/train.ipynb"

with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # 1. Add imports
        if "from sklearn.linear_model import LinearRegression, SGDRegressor\n" in cell['source']:
            idx = cell['source'].index("from sklearn.linear_model import LinearRegression, SGDRegressor\n")
            if "from sklearn.tree import DecisionTreeRegressor\n" not in cell['source']:
                cell['source'].insert(idx + 1, "from sklearn.tree import DecisionTreeRegressor\n")
                cell['source'].insert(idx + 2, "from sklearn.neural_network import MLPRegressor\n")
        
        # 2. Add models to cross validation
        if "'Linear (FS)': (lr_fs, X_train_fs),\n" in cell['source']:
            idx = cell['source'].index("    'Linear (FS)': (lr_fs, X_train_fs),\n")
            if "    'Decision Tree': (dt_reg, X_train),\n" not in cell['source']:
                cell['source'].insert(idx + 1, "    'Decision Tree': (dt_reg, X_train),\n")
                cell['source'].insert(idx + 2, "    'Neural Network': (nn_reg, X_train_scaled),\n")
                
        # 3. Add models to all_metrics comparison list
        if "    results_poly[3]['metrics'],\n" in cell['source']:
            idx = cell['source'].index("    results_poly[3]['metrics'],\n")
            if "    metrics_dt,\n" not in cell['source']:
                cell['source'].insert(idx + 1, "    results_sgd['SGD (constant)']['metrics'],\n")
                cell['source'].insert(idx + 2, "    results_sgd['SGD (adaptive)']['metrics'],\n")
                cell['source'].insert(idx + 3, "    metrics_dt,\n")
                cell['source'].insert(idx + 4, "    metrics_nn\n")
                # Need to update the trailing comma of results_poly[3]
                cell['source'][idx] = "    results_poly[3]['metrics'],\n"
                
                # Check lines above to make sure comma logic is right.
                # Actually, in JSON source arrays, each element is typically a line with trailing \n.

# Insert the new cells after the SGD gradient descent one
sgd_cell_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any(("5. GRADIENT DESCENT OPTIMIZATION" in line for line in cell['source'])):
        sgd_cell_idx = i
        break

if sgd_cell_idx != -1:
    dt_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "decision_tree_val",
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5b. DECISION TREE REGRESSION\n",
            "print(\"=\"*60)\n",
            "print(\"5b. DECISION TREE REGRESSION\")\n",
            "print(\"=\"*60)\n",
            "dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42)\n",
            "dt_reg.fit(X_train, y_train)\n",
            "metrics_dt, pred_dt = evaluate_model(dt_reg, X_train, X_test, y_train, y_test, \"Decision Tree Regression\")\n",
            "print_metrics(metrics_dt)\n",
            "joblib.dump(dt_reg, str(RESULTS_DIR) + '/decision_tree.joblib')\n",
            "np.save(str(RESULTS_DIR) + '/pred_decision_tree.npy', pred_dt)\n",
            "with open(str(RESULTS_DIR) + '/metrics_decision_tree.json', 'w') as f:\n",
            "    json.dump(metrics_dt, f, indent=2)\n",
            "print(\"\\n✅ Decision Tree model saved!\")\n"
        ]
    }
    
    nn_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "neural_network_val",
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5c. NEURAL NETWORK REGRESSION (MLP)\n",
            "print(\"=\"*60)\n",
            "print(\"5c. NEURAL NETWORK REGRESSION (MLP)\")\n",
            "print(\"=\"*60)\n",
            "nn_reg = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, early_stopping=True)\n",
            "nn_reg.fit(X_train_scaled, y_train)\n",
            "metrics_nn, pred_nn = evaluate_model(nn_reg, X_train_scaled, X_test_scaled, y_train, y_test, \"Neural Network Regression\")\n",
            "print_metrics(metrics_nn)\n",
            "joblib.dump(nn_reg, str(RESULTS_DIR) + '/neural_network.joblib')\n",
            "np.save(str(RESULTS_DIR) + '/pred_neural_network.npy', pred_nn)\n",
            "with open(str(RESULTS_DIR) + '/metrics_neural_network.json', 'w') as f:\n",
            "    json.dump(metrics_nn, f, indent=2)\n",
            "print(\"\\n✅ Neural Network model saved!\")\n"
        ]
    }
    
    # Check to prevent duplicate injection
    has_dt = any("5b. DECISION TREE REGRESSION" in "".join(c['source']) for c in nb['cells'])
    if not has_dt:
        nb['cells'].insert(sgd_cell_idx + 1, nn_cell)
        nb['cells'].insert(sgd_cell_idx + 1, dt_cell)

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Notebook updated.")
