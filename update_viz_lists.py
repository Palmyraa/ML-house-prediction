import json

NOTEBOOK_PATH = "c:/Users/REHAN SHAIK/Downloads/MLProjectHousePricePrediction/notebooks/datavisualization.ipynb"

with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for idx, line in enumerate(source):
            if "models_to_plot = [" in line and "SGD Adaptive" in line:
                # Replace the hardcoded list to include the two new models
                new_list = "models_to_plot = ['Linear Univariate', 'Linear Multivariate', 'Linear Feature Selection', 'Polynomial Degree 2', 'Polynomial Degree 3', 'SGD Constant', 'SGD Adaptive', 'Decision Tree', 'Neural Network']\n"
                source[idx] = new_list

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Visualization notebook patch completed.")
