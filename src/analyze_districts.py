import pandas as pd
import numpy as np
from collections import Counter
import re

BASE_PATH = r"d:/Sudarshan Khot/Coding/UIDAI"

print("Loading datasets to extract district names...\n")

# Load all datasets
enrol_chunks = []
for file in ['api_data_aadhar_enrolment_0_500000.csv',
             'api_data_aadhar_enrolment_500000_1000000.csv',
             'api_data_aadhar_enrolment_1000000_1006029.csv']:
    df = pd.read_csv(f"{BASE_PATH}/api_data_aadhar_enrolment/api_data_aadhar_enrolment/{file}")
    enrol_chunks.append(df)
df_enrol = pd.concat(enrol_chunks, ignore_index=True)

bio_chunks = []
for file in ['api_data_aadhar_biometric_0_500000.csv', 
             'api_data_aadhar_biometric_500000_1000000.csv']:
    df = pd.read_csv(f"{BASE_PATH}/api_data_aadhar_biometric/api_data_aadhar_biometric/{file}")
    bio_chunks.append(df)
df_bio = pd.concat(bio_chunks, ignore_index=True)

print(f"Loaded {len(df_enrol):,} enrolment records")
print(f"Loaded {len(df_bio):,} biometric records\n")

# Extract all unique districts (raw)
all_districts_raw = set()
if 'district' in df_enrol.columns:
    all_districts_raw.update(df_enrol['district'].dropna().unique())
if 'district' in df_bio.columns:
    all_districts_raw.update(df_bio['district'].dropna().unique())

print(f"Total unique district names (raw): {len(all_districts_raw)}\n")

# Analyze district names
print("="*80)
print("DISTRICT NAME ANALYSIS")
print("="*80)

# 1. Find numeric or invalid entries
invalid_districts = []
for district in all_districts_raw:
    district_str = str(district).strip()
    if district_str.isdigit() or len(district_str) < 2:
        invalid_districts.append(district)

print(f"\n1. INVALID/NUMERIC DISTRICTS ({len(invalid_districts)}):")
print("-"*80)
for d in sorted(invalid_districts):
    print(f"   - {d}")

# 2. Find districts with special characters or extra spaces
special_char_districts = []
for district in all_districts_raw:
    district_str = str(district).strip()
    if '  ' in district_str or re.search(r'[^a-zA-Z\s\-\&\(\)]', district_str):
        special_char_districts.append(district)

print(f"\n2. DISTRICTS WITH SPECIAL CHARACTERS/EXTRA SPACES ({len(special_char_districts)}):")
print("-"*80)
for d in sorted(special_char_districts)[:20]:  # Show first 20
    print(f"   - '{d}'")
if len(special_char_districts) > 20:
    print(f"   ... and {len(special_char_districts) - 20} more")

# 3. Find potential duplicates (case-insensitive)
district_lower_map = {}
for district in all_districts_raw:
    district_str = str(district).strip()
    if district_str not in invalid_districts:
        lower = district_str.lower()
        if lower not in district_lower_map:
            district_lower_map[lower] = []
        district_lower_map[lower].append(district_str)

duplicates = {k: v for k, v in district_lower_map.items() if len(v) > 1}

print(f"\n3. POTENTIAL CASE DUPLICATES ({len(duplicates)}):")
print("-"*80)
for lower, variants in sorted(duplicates.items())[:20]:  # Show first 20
    print(f"   '{lower}' -> {variants}")
if len(duplicates) > 20:
    print(f"   ... and {len(duplicates) - 20} more")

# 4. Group by state to find inconsistencies
print(f"\n4. DISTRICTS BY STATE:")
print("-"*80)
state_district_map = {}
for idx, row in df_enrol.iterrows():
    state = str(row.get('state', '')).strip()
    district = str(row.get('district', '')).strip()
    if state and district and not district.isdigit():
        if state not in state_district_map:
            state_district_map[state] = set()
        state_district_map[state].add(district)

# Show states with most districts (potential issues)
state_counts = [(state, len(districts)) for state, districts in state_district_map.items()]
state_counts.sort(key=lambda x: x[1], reverse=True)

print("\nTop 10 states by district count:")
for state, count in state_counts[:10]:
    print(f"   {state}: {count} districts")

# Save all unique districts to file
output_file = f"{BASE_PATH}/unique_districts.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("ALL UNIQUE DISTRICT NAMES (SORTED)\n")
    f.write("="*80 + "\n\n")
    
    valid_districts = [d for d in all_districts_raw if str(d).strip() not in [str(x) for x in invalid_districts]]
    
    for district in sorted(valid_districts):
        f.write(f"{district}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("INVALID/NUMERIC DISTRICTS\n")
    f.write("="*80 + "\n\n")
    
    for district in sorted(invalid_districts):
        f.write(f"{district}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("CASE DUPLICATES\n")
    f.write("="*80 + "\n\n")
    
    for lower, variants in sorted(duplicates.items()):
        f.write(f"{lower} -> {variants}\n")

print(f"\n\n✓ Full district list saved to: {output_file}")

# Save state-district mapping
state_district_file = f"{BASE_PATH}/state_district_mapping.txt"
with open(state_district_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("STATE-DISTRICT MAPPING\n")
    f.write("="*80 + "\n\n")
    
    for state in sorted(state_district_map.keys()):
        f.write(f"\n{state} ({len(state_district_map[state])} districts):\n")
        f.write("-"*80 + "\n")
        for district in sorted(state_district_map[state]):
            f.write(f"  - {district}\n")

print(f"✓ State-district mapping saved to: {state_district_file}")

# Generate cleaning recommendations
recommendations_file = f"{BASE_PATH}/district_cleaning_recommendations.txt"
with open(recommendations_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("DISTRICT CLEANING RECOMMENDATIONS\n")
    f.write("="*80 + "\n\n")
    
    f.write("1. REMOVE INVALID ENTRIES:\n")
    f.write("-"*80 + "\n")
    for d in sorted(invalid_districts):
        f.write(f"   Remove: '{d}'\n")
    
    f.write("\n2. STANDARDIZE CASE VARIANTS:\n")
    f.write("-"*80 + "\n")
    for lower, variants in sorted(duplicates.items()):
        if len(variants) > 1:
            # Recommend the title case version
            recommended = max(variants, key=lambda x: sum(1 for c in x if c.isupper()))
            f.write(f"   Standardize to: '{recommended}'\n")
            for variant in variants:
                if variant != recommended:
                    f.write(f"      - '{variant}' -> '{recommended}'\n")
    
    f.write("\n3. CLEAN EXTRA SPACES:\n")
    f.write("-"*80 + "\n")
    for d in sorted(special_char_districts)[:50]:
        cleaned = ' '.join(str(d).split())
        if cleaned != str(d):
            f.write(f"   '{d}' -> '{cleaned}'\n")

print(f"✓ Cleaning recommendations saved to: {recommendations_file}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total unique districts (raw):     {len(all_districts_raw)}")
print(f"Invalid/numeric entries:          {len(invalid_districts)}")
print(f"Special character issues:         {len(special_char_districts)}")
print(f"Case duplicates:                  {len(duplicates)}")
print(f"Valid unique districts (approx):  {len(all_districts_raw) - len(invalid_districts)}")
print("="*80)
