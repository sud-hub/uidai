import json

# Load the notebook
notebook_path = r"d:\Sudarshan Khot\Coding\UIDAI\Child_MBU_Predictive_Dropout_Model_v8.1.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find and update cells
for i, cell in enumerate(notebook['cells']):
    # 1. Add district risk scoring note BEFORE the code cell
    if cell.get('cell_type') == 'markdown' and any('Deployment Prioritization' in line for line in cell.get('source', [])):
        # Insert new markdown cell after this one
        new_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Note on District Risk Scores:**\n",
                "\n",
                "District-level risk scores are illustrative and derived from modeling samples. Full-scale deployment would compute these metrics on complete administrative datasets."
            ]
        }
        notebook['cells'].insert(i + 1, new_cell)
        print("✓ Added district risk scoring note")
        break

# 2. Add compliance interpretation note AFTER the compliance interpretation markdown
for i, cell in enumerate(notebook['cells']):
    if cell.get('cell_type') == 'markdown' and any('Compliance Interpretation Note' in line for line in cell.get('source', [])):
        # Check if the note already contains the biometric update explanation
        source_text = ''.join(cell.get('source', []))
        if 'multiple update attempts' not in source_text:
            # Update the cell to include the additional explanation
            cell['source'] = [
                "### Compliance Interpretation Note\n",
                "\n",
                "**Biometric Update Counts:**\n",
                "\n",
                "Biometric update counts may exceed unique enrolments due to multiple update attempts per child across time; compliance values are therefore interpreted as system throughput indicators rather than one-time completion ratios.\n",
                "\n",
                "**Operational Context:**\n",
                "\n",
                "Monthly compliance rates may reflect operational constraints (camp availability, staffing gaps, data ingestion delays) rather than beneficiary intent. These metrics are used as contextual indicators and not as standalone performance judgments."
            ]
            print("✓ Enhanced compliance interpretation note")
        break

# Save the updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4, ensure_ascii=False)

print(f"\n✓ Notebook updated successfully: {notebook_path}")
print("\nChanges made:")
print("1. Added district risk scoring clarification note")
print("2. Enhanced compliance interpretation with biometric update explanation")
