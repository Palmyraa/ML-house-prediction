import json
import io

path = "c:/Users/REHAN SHAIK/Downloads/MLProjectHousePricePrediction/notebooks/datavisualization.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Break long lines into two lines via HTML <br>
        source = source.replace("Mean: {np.mean(y_test - pred):.2f}, Std:", "Mean: {np.mean(y_test - pred):.2f}<br>Std:")
        source = source.replace('f"{name} (R² = ', 'f"{name}<br>(R² = ')
        
        # 2. Add an explicit loop to reduce annotation font sizes BEFORE updating layout
        if "make_subplots" in source and "fig.update_layout(" in source:
            if "annotation['font']" not in source:
                inject = "for annotation in fig['layout']['annotations']:\n    annotation['font'] = dict(size=11)\n\nfig.update_layout("
                source = source.replace("fig.update_layout(", inject)
                
        # Safely convert the single string back into the Jupyter array of strings format
        lines = list(io.StringIO(source))
        cell['source'] = lines

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Overlay fix applied.")
