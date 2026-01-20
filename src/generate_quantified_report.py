import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
import json
warnings.filterwarnings('ignore')

BASE_PATH = r"d:/Sudarshan Khot/Coding/UIDAI"

report_data = {}

print("Loading datasets...")
bio_chunks = []
for file in ['api_data_aadhar_biometric_0_500000.csv', 'api_data_aadhar_biometric_500000_1000000.csv']:
    bio_chunks.append(pd.read_csv(f"{BASE_PATH}/api_data_aadhar_biometric/api_data_aadhar_biometric/{file}"))
df_bio = pd.concat(bio_chunks, ignore_index=True)

demo_chunks = []
for file in ['api_data_aadhar_demographic_0_500000.csv', 'api_data_aadhar_demographic_500000_1000000.csv']:
    demo_chunks.append(pd.read_csv(f"{BASE_PATH}/api_data_aadhar_demographic/api_data_aadhar_demographic/{file}"))
df_demo = pd.concat(demo_chunks, ignore_index=True)

enrol_chunks = []
for file in ['api_data_aadhar_enrolment_0_500000.csv', 'api_data_aadhar_enrolment_500000_1000000.csv', 'api_data_aadhar_enrolment_1000000_1006029.csv']:
    enrol_chunks.append(pd.read_csv(f"{BASE_PATH}/api_data_aadhar_enrolment/api_data_aadhar_enrolment/{file}"))
df_enrol = pd.concat(enrol_chunks, ignore_index=True)

report_data['dataset_sizes'] = {
    'biometric_records': len(df_bio),
    'demographic_records': len(df_demo),
    'enrolment_records': len(df_enrol)
}

print("Cleaning data...")
for df in [df_bio, df_demo, df_enrol]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

report_data['date_range'] = {
    'start': str(df_enrol['date'].min()),
    'end': str(df_enrol['date'].max())
}

print("Calculating compliance metrics...")
bio_child = df_bio.groupby('pincode')['bio_age_5_17'].sum()
enrol_child = df_enrol.groupby('pincode')['age_5_17'].sum()

child_analysis = pd.DataFrame({'bio_updates': bio_child, 'enrolments': enrol_child}).fillna(0)
child_analysis['compliance_ratio'] = child_analysis['bio_updates'] / (child_analysis['enrolments'] + 1)
child_analysis['update_gap'] = child_analysis['enrolments'] - child_analysis['bio_updates']

report_data['compliance_metrics'] = {
    'total_pincodes': len(child_analysis),
    'avg_compliance': float(child_analysis['compliance_ratio'].mean()),
    'median_compliance': float(child_analysis['compliance_ratio'].median()),
    'std_compliance': float(child_analysis['compliance_ratio'].std()),
    'total_enrolments': int(child_analysis['enrolments'].sum()),
    'total_updates': int(child_analysis['bio_updates'].sum()),
    'total_gap': int(child_analysis['update_gap'].sum()),
    'overall_ratio': float(child_analysis['enrolments'].sum() / (child_analysis['bio_updates'].sum() + 1))
}

print("Migration impact analysis...")
demo_by_pin = df_demo.groupby('pincode')[['demo_age_5_17', 'demo_age_17_']].sum()
demo_by_pin['total_demo'] = demo_by_pin.sum(axis=1)

pincode_profile = child_analysis.join(demo_by_pin[['total_demo']], how='left').fillna(0)
pincode_profile['demo_churn_ratio'] = pincode_profile['total_demo'] / (pincode_profile['enrolments'] + 1)

migrant_threshold = pincode_profile['demo_churn_ratio'].quantile(0.80)
pincode_profile['migrant_indicator'] = pincode_profile['demo_churn_ratio'] >= migrant_threshold

significant = pincode_profile[pincode_profile['enrolments'] >= 50]
migrant_compliance = significant[significant['migrant_indicator']]['compliance_ratio'].mean()
standard_compliance = significant[~significant['migrant_indicator']]['compliance_ratio'].mean()

report_data['migration_analysis'] = {
    'migrant_threshold': float(migrant_threshold),
    'significant_pincodes': len(significant),
    'high_churn_count': int(significant['migrant_indicator'].sum()),
    'standard_count': int((~significant['migrant_indicator']).sum()),
    'migrant_compliance': float(migrant_compliance),
    'standard_compliance': float(standard_compliance),
    'compliance_difference_pct': float(((migrant_compliance - standard_compliance) / standard_compliance) * 100)
}

print("Building priority framework...")
scaler = MinMaxScaler(feature_range=(0, 100))
at_risk = pincode_profile[(pincode_profile['enrolments'] >= 30) & (pincode_profile['update_gap'] > 0)].copy()

at_risk['volume_score'] = scaler.fit_transform(at_risk[['update_gap']])
at_risk['urgency_score'] = scaler.fit_transform(-at_risk[['compliance_ratio']])
at_risk['access_score'] = at_risk['migrant_indicator'].astype(int) * 40
at_risk['priority_score'] = (at_risk['volume_score'] * 0.40 + at_risk['urgency_score'] * 0.35 + at_risk['access_score'] * 0.25)

priority_list = at_risk.nlargest(50, 'priority_score')

report_data['priority_framework'] = {
    'total_at_risk_pincodes': len(at_risk),
    'top50_enrolments': int(priority_list['enrolments'].sum()),
    'top50_gap': int(priority_list['update_gap'].sum()),
    'top50_migrant_zones': int(priority_list['migrant_indicator'].sum()),
    'avg_priority_score': float(priority_list['priority_score'].mean()),
    'min_priority_score': float(priority_list['priority_score'].min()),
    'max_priority_score': float(priority_list['priority_score'].max())
}

phase1 = priority_list.head(10)
phase2 = priority_list.iloc[10:30]
phase3 = priority_list.iloc[30:50]

report_data['deployment_phases'] = {
    'phase1': {'enrolments': int(phase1['enrolments'].sum()), 'gap': int(phase1['update_gap'].sum()), 'migrant_zones': int(phase1['migrant_indicator'].sum())},
    'phase2': {'enrolments': int(phase2['enrolments'].sum()), 'gap': int(phase2['update_gap'].sum()), 'migrant_zones': int(phase2['migrant_indicator'].sum())},
    'phase3': {'enrolments': int(phase3['enrolments'].sum()), 'gap': int(phase3['update_gap'].sum()), 'migrant_zones': int(phase3['migrant_indicator'].sum())}
}

total_gap = priority_list['update_gap'].sum()
report_data['social_impact'] = {
    'children_protected': int(total_gap),
    'scholarships_saved': int(total_gap * 0.65),
    'exams_enabled': int(total_gap * 0.45),
    'economic_value_cr': round((total_gap * 0.65 * 12000) / 10000000, 2)
}

top20_list = []
for idx, (pin, row) in enumerate(priority_list.head(20).iterrows(), 1):
    top20_list.append({
        'rank': idx,
        'pincode': int(pin),
        'enrolments': int(row['enrolments']),
        'gap': int(row['update_gap']),
        'compliance': float(row['compliance_ratio']),
        'zone_type': 'High-Churn' if row['migrant_indicator'] else 'Standard',
        'priority_score': float(row['priority_score'])
    })
report_data['top20_pincodes'] = top20_list

if 'date' in df_enrol.columns:
    df_enrol['day_of_week'] = df_enrol['date'].dt.day_name()
    daily = df_enrol.groupby('day_of_week')[['age_0_5', 'age_5_17', 'age_18_greater']].sum().sum(axis=1)
    daily = daily.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    report_data['temporal_patterns'] = {
        'cv': float(daily.std() / daily.mean()),
        'peak_day': daily.idxmax(),
        'peak_volume': int(daily.max()),
        'lowest_day': daily.idxmin(),
        'lowest_volume': int(daily.min()),
        'avg_daily': int(daily.mean()),
        'peak_multiplier': float(daily.max() / daily.mean()),
        'daily_volumes': {day: int(vol) for day, vol in daily.items()}
    }

with open('quantified_report_data.json', 'w') as f:
    json.dump(report_data, f, indent=2)

print("\n" + "="*80)
print("QUANTIFIED ANALYSIS REPORT - KEY METRICS")
print("="*80)
print(json.dumps(report_data, indent=2))
print("\nJSON saved to: quantified_report_data.json")
