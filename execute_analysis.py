import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = r"d:/Sudarshan Khot/Coding/UIDAI"

print("="*80)
print("CHILD MBU PREDICTIVE DROPOUT MODEL - QUANTIFIED ANALYSIS REPORT")
print("="*80)

print("\n[1] LOADING DATASETS...")
bio_chunks = []
for file in ['api_data_aadhar_biometric_0_500000.csv', 
             'api_data_aadhar_biometric_500000_1000000.csv']:
    df = pd.read_csv(f"{BASE_PATH}/api_data_aadhar_biometric/api_data_aadhar_biometric/{file}")
    bio_chunks.append(df)
df_bio = pd.concat(bio_chunks, ignore_index=True)

demo_chunks = []
for file in ['api_data_aadhar_demographic_0_500000.csv',
             'api_data_aadhar_demographic_500000_1000000.csv']:
    df = pd.read_csv(f"{BASE_PATH}/api_data_aadhar_demographic/api_data_aadhar_demographic/{file}")
    demo_chunks.append(df)
df_demo = pd.concat(demo_chunks, ignore_index=True)

enrol_chunks = []
for file in ['api_data_aadhar_enrolment_0_500000.csv',
             'api_data_aadhar_enrolment_500000_1000000.csv',
             'api_data_aadhar_enrolment_1000000_1006029.csv']:
    df = pd.read_csv(f"{BASE_PATH}/api_data_aadhar_enrolment/api_data_aadhar_enrolment/{file}")
    enrol_chunks.append(df)
df_enrol = pd.concat(enrol_chunks, ignore_index=True)

print(f"✓ Biometric Records: {len(df_bio):,}")
print(f"✓ Demographic Records: {len(df_demo):,}")
print(f"✓ Enrolment Records: {len(df_enrol):,}")

print("\n[2] DATA CLEANING...")
df_bio.replace([np.inf, -np.inf], np.nan, inplace=True)
df_demo.replace([np.inf, -np.inf], np.nan, inplace=True)
df_enrol.replace([np.inf, -np.inf], np.nan, inplace=True)

if 'date' in df_enrol.columns:
    df_enrol['date'] = pd.to_datetime(df_enrol['date'], dayfirst=True, errors='coerce')
if 'date' in df_bio.columns:
    df_bio['date'] = pd.to_datetime(df_bio['date'], dayfirst=True, errors='coerce')
if 'date' in df_demo.columns:
    df_demo['date'] = pd.to_datetime(df_demo['date'], dayfirst=True, errors='coerce')

print(f"✓ Date range: {df_enrol['date'].min()} to {df_enrol['date'].max()}")

print("\n[3] FEATURE ENGINEERING - COMPLIANCE ANALYSIS...")
bio_child_by_pin = df_bio.groupby('pincode')['bio_age_5_17'].sum()
enrol_child_by_pin = df_enrol.groupby('pincode')['age_5_17'].sum()

child_analysis = pd.DataFrame({
    'bio_updates': bio_child_by_pin,
    'enrolments': enrol_child_by_pin
}).fillna(0)

child_analysis['compliance_ratio'] = child_analysis['bio_updates'] / (child_analysis['enrolments'] + 1)
child_analysis['update_gap'] = child_analysis['enrolments'] - child_analysis['bio_updates']

print(f"✓ Total pincodes analyzed: {len(child_analysis):,}")
print(f"✓ Average compliance ratio: {child_analysis['compliance_ratio'].mean():.4f}")
print(f"✓ Median compliance ratio: {child_analysis['compliance_ratio'].median():.4f}")
print(f"✓ Std deviation: {child_analysis['compliance_ratio'].std():.4f}")
print(f"✓ Total child enrolments: {child_analysis['enrolments'].sum():,.0f}")
print(f"✓ Total biometric updates: {child_analysis['bio_updates'].sum():,.0f}")
print(f"✓ Total update gap: {child_analysis['update_gap'].sum():,.0f}")

print("\n[4] MIGRATION IMPACT ANALYSIS...")
demo_by_pin = df_demo.groupby('pincode')[['demo_age_5_17', 'demo_age_17_']].sum()
demo_by_pin['total_demo'] = demo_by_pin.sum(axis=1)

bio_by_pin = df_bio.groupby('pincode')[['bio_age_5_17', 'bio_age_17_']].sum()
bio_by_pin['total_bio'] = bio_by_pin.sum(axis=1)

pincode_profile = child_analysis.join(demo_by_pin[['total_demo']], how='left')
pincode_profile = pincode_profile.join(bio_by_pin[['total_bio']], how='left')
pincode_profile.fillna(0, inplace=True)

pincode_profile['demo_churn_ratio'] = (pincode_profile['total_demo'] / 
                                        (pincode_profile['enrolments'] + 1))

migrant_threshold = pincode_profile['demo_churn_ratio'].quantile(0.80)
pincode_profile['migrant_indicator'] = pincode_profile['demo_churn_ratio'] >= migrant_threshold

significant = pincode_profile[pincode_profile['enrolments'] >= 50]

migrant_compliance = significant[significant['migrant_indicator']]['compliance_ratio'].mean()
standard_compliance = significant[~significant['migrant_indicator']]['compliance_ratio'].mean()

compliance_diff = ((migrant_compliance - standard_compliance) / standard_compliance) * 100

print(f"✓ Migrant threshold (80th percentile churn): {migrant_threshold:.4f}")
print(f"✓ Significant pincodes (>=50 enrolments): {len(significant):,}")
print(f"✓ High-churn zones: {significant['migrant_indicator'].sum():,}")
print(f"✓ Standard zones: {(~significant['migrant_indicator']).sum():,}")
print(f"✓ High-churn zones compliance: {migrant_compliance:.4f}")
print(f"✓ Standard zones compliance: {standard_compliance:.4f}")
print(f"✓ Compliance difference: {compliance_diff:+.2f}%")

print("\n[5] RISK PRIORITIZATION FRAMEWORK...")
scaler = MinMaxScaler(feature_range=(0, 100))

at_risk_pins = pincode_profile[
    (pincode_profile['enrolments'] >= 30) &
    (pincode_profile['update_gap'] > 0)
].copy()

at_risk_pins['volume_score'] = scaler.fit_transform(
    at_risk_pins[['update_gap']]
)

at_risk_pins['urgency_score'] = scaler.fit_transform(
    -at_risk_pins[['compliance_ratio']]
)

at_risk_pins['access_score'] = at_risk_pins['migrant_indicator'].astype(int) * 40

at_risk_pins['priority_score'] = (
    at_risk_pins['volume_score'] * 0.40 +
    at_risk_pins['urgency_score'] * 0.35 +
    at_risk_pins['access_score'] * 0.25
)

priority_list = at_risk_pins.nlargest(50, 'priority_score')

print(f"✓ Total pincodes requiring intervention: {len(at_risk_pins):,}")
print(f"✓ Top 50 priority pincodes identified")
print(f"✓ Children in top 50 pincodes: {priority_list['enrolments'].sum():,.0f}")
print(f"✓ Update gap in top 50: {priority_list['update_gap'].sum():,.0f}")
print(f"✓ High-churn zones in top 50: {priority_list['migrant_indicator'].sum()}")
print(f"✓ Average priority score (top 50): {priority_list['priority_score'].mean():.2f}")
print(f"✓ Min priority score (top 50): {priority_list['priority_score'].min():.2f}")
print(f"✓ Max priority score (top 50): {priority_list['priority_score'].max():.2f}")

print("\n[6] OVERALL COMPLIANCE METRICS...")
child_enrolments = df_enrol['age_5_17'].sum()
child_bio_updates = df_bio['bio_age_5_17'].sum()
overall_ratio = child_enrolments / (child_bio_updates + 1)

print(f"✓ Total child enrolments (age 5-17): {child_enrolments:,}")
print(f"✓ Total child biometric updates: {child_bio_updates:,}")
print(f"✓ Enrolment-to-Update Ratio: {overall_ratio:.2f}:1")

print("\n[7] TEMPORAL PATTERN ANALYSIS...")
if 'date' in df_enrol.columns:
    df_enrol['day_of_week'] = df_enrol['date'].dt.day_name()
    
    daily_volumes = df_enrol.groupby('day_of_week')[['age_0_5', 'age_5_17', 'age_18_greater']].sum().sum(axis=1)
    daily_volumes = daily_volumes.reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    
    mean_daily = daily_volumes.mean()
    std_daily = daily_volumes.std()
    cv = std_daily / mean_daily
    
    print(f"✓ Coefficient of Variation: {cv:.4f}")
    print(f"✓ Peak day: {daily_volumes.idxmax()}")
    print(f"✓ Peak volume: {daily_volumes.max():,.0f}")
    print(f"✓ Lowest day: {daily_volumes.idxmin()}")
    print(f"✓ Lowest volume: {daily_volumes.min():,.0f}")
    print(f"✓ Average daily volume: {mean_daily:,.0f}")
    print(f"✓ Peak multiplier: {daily_volumes.max() / mean_daily:.2f}x")
    print(f"✓ Daily volumes by day:")
    for day, vol in daily_volumes.items():
        print(f"   {day}: {vol:,.0f}")

print("\n[8] TOP 20 PRIORITY PINCODES DETAILED...")
print("-" * 100)
print(f"{'Rank':<6} {'Pincode':<10} {'Enrolments':<12} {'Gap':<10} {'Compliance':<12} {'Churn':<8} {'Priority':<10}")
print("-" * 100)
for idx, (pin, row) in enumerate(priority_list.head(20).iterrows(), 1):
    zone = "High" if row['migrant_indicator'] else "Std"
    print(f"{idx:<6} {pin:<10} {int(row['enrolments']):<12,} {int(row['update_gap']):<10,} "
          f"{row['compliance_ratio']:<12.4f} {zone:<8} {row['priority_score']:<10.2f}")

print("\n[9] PHASED DEPLOYMENT BREAKDOWN...")
phase1 = priority_list.head(10)
phase2 = priority_list.iloc[10:30]
phase3 = priority_list.iloc[30:50]

print(f"✓ Phase 1 (Top 10):")
print(f"  - Total enrolments: {phase1['enrolments'].sum():,.0f}")
print(f"  - Update gap: {phase1['update_gap'].sum():,.0f}")
print(f"  - High-churn zones: {phase1['migrant_indicator'].sum()}")

print(f"✓ Phase 2 (11-30):")
print(f"  - Total enrolments: {phase2['enrolments'].sum():,.0f}")
print(f"  - Update gap: {phase2['update_gap'].sum():,.0f}")
print(f"  - High-churn zones: {phase2['migrant_indicator'].sum()}")

print(f"✓ Phase 3 (31-50):")
print(f"  - Total enrolments: {phase3['enrolments'].sum():,.0f}")
print(f"  - Update gap: {phase3['update_gap'].sum():,.0f}")
print(f"  - High-churn zones: {phase3['migrant_indicator'].sum()}")

print("\n[10] PROJECTED IMPACT...")
total_gap = priority_list.head(50)['update_gap'].sum()
print(f"✓ Total children requiring updates (top 50): {total_gap:,.0f}")
print(f"✓ Scholarship applications protected (65%): {total_gap * 0.65:,.0f}")
print(f"✓ Exam registrations enabled (45%): {total_gap * 0.45:,.0f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
