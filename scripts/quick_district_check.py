import pandas as pd

BASE_PATH = r"d:/Sudarshan Khot/Coding/UIDAI"

print("Extracting district names from enrolment data...")

# Load only enrolment data (faster)
df = pd.read_csv(f"{BASE_PATH}/api_data_aadhar_enrolment/api_data_aadhar_enrolment/api_data_aadhar_enrolment_0_500000.csv")

# Get unique districts
unique_districts = sorted(df['district'].dropna().unique())

print(f"\nFound {len(unique_districts)} unique districts in first 500K records\n")

# Save to file
output_file = f"{BASE_PATH}/unique_districts_sample.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(f"UNIQUE DISTRICTS ({len(unique_districts)} total)\n")
    f.write("="*80 + "\n\n")
    for i, district in enumerate(unique_districts, 1):
        f.write(f"{i:4d}. {district}\n")

print(f"✓ Saved to: {output_file}")

# Show first 50
print("\nFirst 50 districts:")
print("-"*80)
for i, district in enumerate(unique_districts[:50], 1):
    print(f"{i:4d}. {district}")

if len(unique_districts) > 50:
    print(f"\n... and {len(unique_districts) - 50} more (see file)")
