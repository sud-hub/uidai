import json
import sys

print("Creating Child_MBU_Predictive_Dropout_Model_v8.2.ipynb with FULL dataset...")

with open(r"d:\Sudarshan Khot\Coding\UIDAI\Child_MBU_Predictive_Dropout_Model_v8.1.ipynb", 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'source' in cell:
        source = ''.join(cell['source'])
        
        if 'enrol_sample = df_enrol.sample(min(100000, len(df_enrol)), random_state=42).copy()' in source:
            print("Found sampling code - replacing with full dataset...")
            
            new_source = []
            for line in cell['source']:
                if 'enrol_sample = df_enrol.sample(min(100000, len(df_enrol)), random_state=42).copy()' in line:
                    new_source.append('enrol_sample = df_enrol.copy()\n')
                elif 'bio_sample = df_bio.sample(min(100000, len(df_bio)), random_state=42).copy()' in line:
                    new_source.append('bio_sample = df_bio.copy()\n')
                else:
                    new_source.append(line)
            
            cell['source'] = new_source
            cell['outputs'] = []
            cell['execution_count'] = None
        
        elif 'Note: District rankings based on sampled data' in source:
            print("Updating district ranking note...")
            
            new_source = []
            for line in cell['source']:
                if 'Note: District rankings based on sampled data' in line:
                    new_source.append('print(f"\\nNote: District rankings based on FULL dataset ({len(merged):,} records).")\n')
                elif 'Full population analysis recommended for operational deployment.' in line:
                    new_source.append('print("Analysis uses complete population data for accurate operational deployment.")\n')
                else:
                    new_source.append(line)
            
            cell['source'] = new_source
            cell['outputs'] = []
            cell['execution_count'] = None

for cell in notebook['cells']:
    if cell['cell_type'] == 'markdown' and 'source' in cell:
        source = ''.join(cell['source'])
        
        if '# Child MBU Predictive Dropout & Outreach Model' in source:
            print("Updating title to v8.2...")
            
            new_source = []
            for line in cell['source']:
                if '# Child MBU Predictive Dropout & Outreach Model' in line:
                    new_source.append('# Child MBU Predictive Dropout & Outreach Model v8.2\n')
                else:
                    new_source.append(line)
            
            cell['source'] = new_source

output_path = r"d:\Sudarshan Khot\Coding\UIDAI\Child_MBU_Predictive_Dropout_Model_v8.2.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4, ensure_ascii=False)

print(f"\nSuccessfully created: {output_path}")
print("\nChanges made:")
print("1. Removed sampling - now uses FULL dataset (1M+ records)")
print("2. Updated district ranking notes to reflect full population analysis")
print("3. Updated title to v8.2")
print("\nDistrict child counts will now show accurate numbers from entire dataset!")
