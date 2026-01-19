
import json
import os
import sys

log_file = "refine_log.txt"

def log(msg):
    with open(log_file, "a") as f:
        f.write(msg + "\n")

try:
    log("Script started")
    notebook_path = 'notebooks/Child_MBU_Predictive_Dropout_Model.ipynb'
    new_notebook_path = 'notebooks/Refined_Child_MBU_Model.ipynb'

    if not os.path.exists(notebook_path):
        log(f"Error: {notebook_path} not found.")
        sys.exit(1)

    log("Reading notebook...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    log("Notebook read successfully.")

    # --- 1. Executive Summary ---
    log("Updating Exec Summary...")
    new_exec_summary = [
        "# Child MBU Predictive Dropout & Outreach Model\n",
        "## Executive Summary: Mission Mode Intervention\n",
        "\n",
        "**Objective**: Prevent child Aadhaar deactivation and scholarship denials through predictive intervention.\n",
        "\n",
        "### Actionable Outcome\n",
        "This model prevents ~X% of children from losing access to scholarships by predicting update failures before they happen.\n",
        "Focusing on **Ease of Living**, we identify the Top 50 Intervention Zones for the next 3 months of outreach.\n",
        "\n",
        "**Policy Context**: Supports UIDAI's Mission Mode for child biometric updates, leveraging the MBU fee waiver (valid until Oct 2026).\n",
        "\n",
        "---"
    ]
    nb['cells'][0]['source'] = new_exec_summary

    # --- 2. Fix BASE_PATH (Cell 4) ---
    log("Fixing BASE_PATH...")
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            for i, line in enumerate(source):
                if "BASE_PATH =" in line:
                    source[i] = "BASE_PATH = \"..\" # Relative path for portability\n"
                    break

    # --- 3. Refinement A: Risk Score (Cell 7) ---
    log("Adding Risk Score...")
    risk_cell = None
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and any("child_analysis['compliance_ratio'] =" in line for line in cell['source']):
            risk_cell = cell
            break

    if risk_cell:
        risk_logic = [
            "\n",
            "# --- Refinement A: Child Exclusion Risk Score ---\n",
            "print(\"Calculating Operational Volatility and Risk Scores...\")\n",
            "\n",
            "# Calculate Operational Volatility (CV of daily updates)\n",
            "daily_volatility = df_bio.groupby(['pincode', 'date'])['bio_age_5_17'].sum().groupby('pincode').std().fillna(0)\n",
            "child_analysis['volatility'] = daily_volatility\n",
            "\n",
            "# Normalize Volatility\n",
            "child_analysis['volatility_score'] = (child_analysis['volatility'] - child_analysis['volatility'].min()) / \\\n",
            "                                     (child_analysis['volatility'].max() - child_analysis['volatility'].min())\n",
            "\n",
            "# Calculate Risk Score\n",
            "norm_compliance = (child_analysis['compliance_ratio'] - child_analysis['compliance_ratio'].min()) / \\\n",
            "                  (child_analysis['compliance_ratio'].max() - child_analysis['compliance_ratio'].min())\n",
            "\n",
            "# Weighted Child Exclusion Risk Score\n",
            "child_analysis['risk_score'] = (1 - norm_compliance) * 0.6 + child_analysis['volatility_score'] * 0.4\n",
            "\n",
            "risk_zones = len(child_analysis[child_analysis['risk_score'] > 0.7])\n",
            "print(f\"Top Risk Pincodes identified (Crisis Zones): {risk_zones}\")\n"
        ]
        risk_cell['source'].extend(risk_logic)

    # --- 4. Refinement B: Enhance Migration Impact (Section 3) ---
    log("Enhancing Migration...")
    mig_cell = None
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and any("Migration Impact Analysis" in line for line in cell['source']):
            mig_cell = cell
            break

    if mig_cell:
        new_mig_code = [
            "# --- Refinement B: Migration Impact (Net-Importing vs Net-Exporting) ---\n",
            "\n",
            "print(\"Analyzing Migration Patterns (Import vs Export)...\")\n",
            "\n",
            "# Ensure state info is linked\n",
            "if 'state' not in df_demo.columns:\n",
            "    if 'state' in df_enrol.columns:\n",
            "        # Map pincode to state from enrolment data\n",
            "        pin_state_map = df_enrol[['pincode', 'state']].drop_duplicates().set_index('pincode')\n",
            "        pin_state_map = pin_state_map[~pin_state_map.index.duplicated()]\n",
            "        df_demo['state'] = df_demo['pincode'].map(pin_state_map['state'])\n",
            "\n",
            "if 'state' in df_demo.columns:\n",
            "    # Demographic Updates (Address Change) Volume\n",
            "    state_demo = df_demo.groupby('state')['demo_age_5_17'].sum()\n",
            "    state_enrol = df_enrol.groupby('state')['age_5_17'].sum()\n",
            "\n",
            "    migration_impact = pd.DataFrame({'demo_updates': state_demo, 'enrolments': state_enrol}).fillna(0)\n",
            "    # High Demo Updates relative to population -> Net-Importing/Churn\n",
            "    migration_impact['churn_rate'] = migration_impact['demo_updates'] / (migration_impact['enrolments'] + 1)\n",
            "\n",
            "    # Top Net-Importing States\n",
            "    top_importing = migration_impact.sort_values('churn_rate', ascending=False).head(5)\n",
            "    print(\"\\nTop 5 Net-Importing States (High Migration Churn):\")\n",
            "    print(top_importing[['churn_rate']])\n",
            "    print(\"\\nAction: Recommendation to set up 'Migrant Help Desks' in these high-import zones.\")\n",
            "else:\n",
            "    print(\"State column not available for Migration Analysis.\")\n"
        ]
        mig_cell['source'] = new_mig_code

    # --- 5 & 6. Visualization & Social Impact (New Cells) ---
    log("Adding Heatmap and Social Impact...")
    heatmap_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Refinement: Deployment Recommendations Heatmap ---\n",
            "import seaborn as sns\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "\n",
            "# Select Top 50 Risk Zones\n",
            "top_50_pins = child_analysis.sort_values('risk_score', ascending=False).head(50)\n",
            "\n",
            "plt.figure(figsize=(10, 6))\n",
            "# Create a stylized heatmap (using reshape for grid visualization)\n",
            "heatmap_data = top_50_pins['risk_score'].values\n",
            "# Pad if less than 50\n",
            "if len(heatmap_data) < 50:\n",
            "    heatmap_data = np.pad(heatmap_data, (0, 50 - len(heatmap_data)), 'constant')\n",
            "    \n",
            "sns.heatmap(heatmap_data.reshape(5, 10), cmap='RdYlGn_r', annot=True, fmt='.2f', \n",
            "            xticklabels=False, yticklabels=False, cbar_kws={'label': 'Risk Score'})\n",
            "\n",
            "plt.title('Deployment Recommendation: Top 50 Intervention Zones Heatmap')\n",
            "plt.show()"
        ]
    }

    social_impact_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Refinement C: Social Impact (Benefits Saved) ---\n",
            "\n",
            "# Estimations based on policy impact\n",
            "SCHOLARSHIP_VAL_YR = 12000 # INR\n",
            "pre_matric_students = top_50_pins['update_gap'].sum()\n",
            "\n",
            "# If we intervene in Top 50 zones\n",
            "benefits_protected = pre_matric_students * SCHOLARSHIP_VAL_YR / 10**7 # Convert to Crores\n",
            "\n",
            "print(\"Final Social Impact Metrics:\")\n",
            "print(f\"1. Target Intervention: Top 50 Pincodes (e.g., {top_50_pins.index[0]}...)\")\n",
            "print(f\"2. Students Protected: {int(pre_matric_students):,} students\")\n",
            "print(f\"3. Benefits Saved: ~₹{benefits_protected:.2f} Crores in Scholarship Value\")\n"
        ]
    }

    nb['cells'].append(heatmap_cell)
    nb['cells'].append(social_impact_cell)

    log(f"Writing to {new_notebook_path}...")
    with open(new_notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    log("Write complete.")

except Exception as e:
    log(f"Exception: {str(e)}")
    print(e)
