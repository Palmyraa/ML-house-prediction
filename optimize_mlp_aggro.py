import json

path = "c:/Users/REHAN SHAIK/Downloads/MLProjectHousePricePrediction/notebooks/train.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Super aggressive MLP reduction
        if "MLPRegressor" in source:
            source = source.replace("hidden_layer_sizes=(100, 50)", "hidden_layer_sizes=(10,)")
            source = source.replace("hidden_layer_sizes=(50,)", "hidden_layer_sizes=(10,)")
            source = source.replace("max_iter=1000", "max_iter=50")
            source = source.replace("max_iter=200", "max_iter=50")
        
        # Convert back safely
        lines = []
        for line in source.splitlines(True):
            lines.append(line)
        cell['source'] = lines

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Aggressive MLP optimization applied.")
