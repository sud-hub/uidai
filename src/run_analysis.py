
import pandas as pd
import numpy as np
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Redirect output
sys.stdout = open('analysis_results.txt', 'w', encoding='utf-8')

print("=== STARTING ANALYSIS ===")

BASE_PATH = os.getcwd()
print(f"Working Directory: {BASE_PATH}")

# 1. Data Loading
print("\nLoading datasets...")
try:
    bio_chunks = []
    for file in ['api_data_aadhar_biometric_0_500000.csv', 'api_data_aadhar_biometric_500000_1000000.csv']:
        path = os.path.join(BASE_PATH, 'api_data_aadhar_biometric', 'api_data_aadhar_biometric', file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            bio_chunks.append(df)
        else:
            print(f"Warning: File not found {path}")
    
    if bio_chunks:
        df_bio = pd.concat(bio_chunks, ignore_index=True)
    else:
        print("Error: No biometric data loaded")
        sys.exit(1)

    demo_chunks = []
    for file in ['api_data_aadhar_demographic_0_500000.csv', 'api_data_aadhar_demographic_500000_1000000.csv']:
        path = os.path.join(BASE_PATH, 'api_data_aadhar_demographic', 'api_data_aadhar_demographic', file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            demo_chunks.append(df)
    df_demo = pd.concat(demo_chunks, ignore_index=True)

    enrol_chunks = []
    for file in ['api_data_aadhar_enrolment_0_500000.csv', 'api_data_aadhar_enrolment_500000_1000000.csv', 'api_data_aadhar_enrolment_1000000_1006029.csv']:
        path = os.path.join(BASE_PATH, 'api_data_aadhar_enrolment', 'api_data_aadhar_enrolment', file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            enrol_chunks.append(df)
    df_enrol = pd.concat(enrol_chunks, ignore_index=True)

    print(f"Biometric Records: {len(df_bio):,}")
    print(f"Demographic Records: {len(df_demo):,}")
    print(f"Enrolment Records: {len(df_enrol):,}")

    # Handling Inf and Dates
    df_bio.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_demo.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_enrol.replace([np.inf, -np.inf], np.nan, inplace=True)

    if 'date' in df_enrol.columns:
        df_enrol['date'] = pd.to_datetime(df_enrol['date'], dayfirst=True, errors='coerce')
    if 'date' in df_bio.columns:
        df_bio['date'] = pd.to_datetime(df_bio['date'], dayfirst=True, errors='coerce')
    
    # 2. Risk Score
    print("\nCalculating Risk Scores...")
    bio_child_by_pin = df_bio.groupby('pincode')['bio_age_5_17'].sum()
    enrol_child_by_pin = df_enrol.groupby('pincode')['age_5_17'].sum()

    child_analysis = pd.DataFrame({
        'bio_updates': bio_child_by_pin,
        'enrolments': enrol_child_by_pin
    }).fillna(0)

    child_analysis['compliance_ratio'] = child_analysis['bio_updates'] / (child_analysis['enrolments'] + 1)
    child_analysis['update_gap'] = child_analysis['enrolments'] - child_analysis['bio_updates']

    daily_bio = df_bio.groupby(['pincode', 'date'])['bio_age_5_17'].sum().reset_index()
    volatility = daily_bio.groupby('pincode')['bio_age_5_17'].std().fillna(0)
    child_analysis['operational_volatility'] = volatility

    child_analysis['inverse_compliance'] = 1 - child_analysis['compliance_ratio'].clip(0, 1)
    child_analysis['volume_weight'] = np.log1p(child_analysis['enrolments'])

    child_analysis['risk_score'] = (
        child_analysis['inverse_compliance'] * 
        child_analysis['operational_volatility'] * 
        child_analysis['volume_weight']
    )

    max_risk = child_analysis['risk_score'].max()
    if max_risk > 0:
        child_analysis['risk_score_normalized'] = (child_analysis['risk_score'] / max_risk) * 100
    else:
        child_analysis['risk_score_normalized'] = 0

    print(f"Total pincodes: {len(child_analysis)}")
    print(f"Avg Risk: {child_analysis['risk_score_normalized'].mean():.2f}")

    # 3. Migration
    print("\nMigration Analysis...")
    demo_by_pin = df_demo.groupby('pincode')[['demo_age_5_17', 'demo_age_17_']].sum()
    demo_by_pin['total_demo'] = demo_by_pin.sum(axis=1)
    bio_by_pin = df_bio.groupby('pincode')[['bio_age_5_17', 'bio_age_17_']].sum()
    bio_by_pin['total_bio'] = bio_by_pin.sum(axis=1)

    pincode_profile = child_analysis.join(demo_by_pin[['total_demo']], how='left')
    pincode_profile = pincode_profile.join(bio_by_pin[['total_bio']], how='left')
    pincode_profile.fillna(0, inplace=True)

    pincode_profile['demo_churn_ratio'] = (pincode_profile['total_demo'] / (pincode_profile['enrolments'] + 1))
    
    # State mapping (approx first 2 digits)
    pincode_profile['state_code'] = pincode_profile.index.astype(str).str[:2]
    migrant_threshold = pincode_profile['demo_churn_ratio'].quantile(0.80)
    pincode_profile['migrant_indicator'] = pincode_profile['demo_churn_ratio'] >= migrant_threshold

    # 4. Deployment
    print("\nDeployment Intelligence...")
    pincode_profile['priority_score'] = (
        0.40 * pincode_profile['risk_score_normalized'] +
        0.30 * (pincode_profile['volume_weight'] / pincode_profile['volume_weight'].max() * 100) +
        0.20 * (pincode_profile['migrant_indicator'].astype(int) * 100) +
        0.10 * (pincode_profile['operational_volatility'] / pincode_profile['operational_volatility'].max() * 100)
    )

    top_50_zones = pincode_profile.nlargest(50, 'priority_score')
    deployment_schedule = top_50_zones.copy()
    deployment_schedule['estimated_children_at_risk'] = deployment_schedule['update_gap'].clip(lower=0)
    deployment_schedule['recommended_vans'] = np.ceil(deployment_schedule['estimated_children_at_risk'] / 500)

    # 5. Social Impact
    print("\nCalculating Social Impact...")
    SCHOLARSHIP_ELIGIBILITY_RATE = 0.60
    AVG_SCHOLARSHIP_AMOUNT = 3000
    INTERVENTION_SUCCESS_RATE = 0.75

    tier_zones = deployment_schedule.head(5)
    top_5_risk = tier_zones['estimated_children_at_risk'].sum()
    top_5_saved = top_5_risk * SCHOLARSHIP_ELIGIBILITY_RATE * INTERVENTION_SUCCESS_RATE
    benefits_saved = top_5_saved * AVG_SCHOLARSHIP_AMOUNT

    total_risk_50 = deployment_schedule['estimated_children_at_risk'].sum()
    total_saved_50 = total_risk_50 * SCHOLARSHIP_ELIGIBILITY_RATE * INTERVENTION_SUCCESS_RATE
    
    prevention_rate = (top_5_saved / child_analysis['enrolments'].sum() * 100)

    # OUTPUT REPORT DATA
    print("\n--- REPORT DATA START ---")
    print(f"METRIC_TOTAL_PINCODES:{len(child_analysis)}")
    print(f"METRIC_HIGH_RISK_ZONES:{len(child_analysis[child_analysis['risk_score_normalized'] > 75])}")
    print(f"METRIC_CHILDREN_AT_RISK_50:{int(total_risk_50)}")
    print(f"METRIC_TOP_5_SAVED_CHILDREN:{int(top_5_saved)}")
    print(f"METRIC_TOP_50_SAVED_CHILDREN:{int(total_saved_50)}")
    print(f"METRIC_PREVENTION_RATE:{prevention_rate:.2f}")
    print(f"METRIC_BENEFITS_SAVED_CRORES:{(benefits_saved/10000000):.2f}")
    print(f"METRIC_VANS_NEEDED:{int(deployment_schedule['recommended_vans'].sum())}")
    print("--- REPORT DATA END ---")

    # CSV Export
    deployment_schedule.to_csv('deployment_top50.csv')
    print("CSV exported.")

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")
    sys.exit(1)
