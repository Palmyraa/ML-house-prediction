import json
import io

path = "c:/Users/REHAN SHAIK/Downloads/MLProjectHousePricePrediction/notebooks/datavisualization.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Fix vertical spacing string match
        source = source.replace("vertical_spacing=0.12/n_rows", "vertical_spacing=0.25, horizontal_spacing=0.08")
        
        # Give more height per row
        source = source.replace("height=400*n_rows", "height=500*n_rows")
        
        # Break the Mean and Std text into multiple lines
        source = source.replace("}, Std:", "},<br>Std:")
        
        # Make font size even smaller for annotations
        source = source.replace("dict(size=11)", "dict(size=10)")
                
        # Safely convert the single string back into the Jupyter array of strings format
        lines = []
        for line in source.splitlines(True):
            lines.append(line)
        cell['source'] = lines

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Overlay layout fix applied.")
