import json
import os

notebook_path = "Candlestick-based-prediction-dashboard/cvdl-final.ipynb"
output_path = "Candlestick-based-prediction-dashboard/extracted_code.py"

if not os.path.exists(notebook_path):
    # Try direct path
    notebook_path = "cvdl-final.ipynb"
    output_path = "extracted_code.py"

with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

code_cells = []
for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
        code_lines = cell.get("source", [])
        code_cells.append("".join(code_lines))
        code_cells.append("\n" + "#" * 80 + "\n")

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(code_cells))

print(f"Extracted code to {output_path}")
