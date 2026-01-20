import json

# Load the notebook
with open(r'd:\Sudarshan Khot\Coding\UIDAI\Child_MBU_Predictive_Dropout_Model_v8.2_notebook.txt', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Monthly Trend Visualization cell
monthly_viz_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Visualization: Monthly Trend\n",
        "fig, ax = plt.subplots(figsize=(12, 6))\n",
        "months_str = [str(m) for m in monthly_analysis.index]\n",
        "ax.plot(months_str, monthly_analysis['enrolments'], marker='o', linewidth=2, label='Enrolments', color='#2E86AB')\n",
        "ax.plot(months_str, monthly_analysis['updates']/10, marker='s', linewidth=2, label='Updates (scaled /10)', color='#A23B72')\n",
        "ax.set_xlabel('Month', fontsize=12, fontweight='bold')\n",
        "ax.set_ylabel('Count', fontsize=12, fontweight='bold')\n",
        "ax.set_title('Temporal Pattern: Enrolments vs Biometric Updates', fontsize=14, fontweight='bold', pad=20)\n",
        "ax.legend(fontsize=10)\n",
        "ax.grid(True, alpha=0.3)\n",
        "plt.xticks(rotation=45)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
}

# Find the index to insert after Section 3 (Temporal Trend Analysis)
# Looking for the cell with "## 4. Predictive Model: Dropout Risk Classifier"
insert_index = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown' and any('## 4. Predictive Model' in line for line in cell.get('source', [])):
        insert_index = i
        break

if insert_index is not None:
    # Insert the visualization cell before Section 4
    notebook['cells'].insert(insert_index, monthly_viz_cell)
    print(f"Inserted Monthly Trend visualization at index {insert_index}")
else:
    print("Could not find Section 4 to insert visualization")

# Save the updated notebook
with open(r'd:\Sudarshan Khot\Coding\UIDAI\Child_MBU_Predictive_Dropout_Model_v8.2_notebook.txt', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4)

print(f"Total cells now: {len(notebook['cells'])}")
print("Notebook updated successfully!")
