
import pandas as pd
import numpy as np
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Define paths
base_path = r"d:/Sudarshan Khot/Coding/UIDAI"
bio_path = os.path.join(base_path, "api_data_aadhar_biometric", "api_data_aadhar_biometric", "api_data_aadhar_biometric_0_500000.csv")
demo_path = os.path.join(base_path, "api_data_aadhar_demographic", "api_data_aadhar_demographic", "api_data_aadhar_demographic_0_500000.csv")
enrol_path = os.path.join(base_path, "api_data_aadhar_enrolment", "api_data_aadhar_enrolment", "api_data_aadhar_enrolment_0_500000.csv")

def load_data():
    print("Loading datasets...")
    df_bio = pd.read_csv(bio_path)
    df_demo = pd.read_csv(demo_path)
    df_enrol = pd.read_csv(enrol_path)
    
    # Standardize Dates
    for df in [df_bio, df_demo, df_enrol]:
        if 'date' in df.columns:
            # Flexible date parsing
            df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    
    return df_bio, df_demo, df_enrol


def normalize_state_names(df):
    if 'state' in df.columns:
        df['state'] = df['state'].astype(str).str.strip().str.title()
        # Common corrections
        replacements = {
            'Westbengal': 'West Bengal',
            'West Bengli': 'West Bengal',
            'Wb': 'West Bengal',
            'Odisha': 'Odisha',
            'Orissa': 'Odisha',
            'Chattisgarh': 'Chhattisgarh',
            'Chhatisgarh': 'Chhattisgarh',
            'Andhra Pradesh': 'Andhra Pradesh'
        }
        df['state'] = df['state'].replace(replacements)
        # Fix mixed case variations like 'West bengal' -> 'West Bengal'
        # The title() call handles most, but manual map helps specific typos
    return df

def get_pincode_context(df_enrol, top_pincodes):
    # Get the most frequent district for each pincode
    # Filter df to only these pincodes
    subset = df_enrol[df_enrol['pincode'].isin(top_pincodes)]
    if subset.empty:
        return {}
    
    # Mode aggregation is tricky in pandas groupby, use first appearing or max count
    pin_dist = subset.groupby('pincode')['district'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
    return pin_dist.to_dict()

def analyze_winning_patterns_script(df_bio, df_demo, df_enrol):
    results = []
    print("Analyzing Winning Patterns...")
    
    # 0. Data Cleaning
    print("Cleaning State Names...")
    df_bio = normalize_state_names(df_bio)
    df_demo = normalize_state_names(df_demo)
    df_enrol = normalize_state_names(df_enrol)

    # Define numeric columns
    bio_cols = ['bio_age_5_17', 'bio_age_17_']
    demo_cols = ['demo_age_5_17', 'demo_age_17_']
    enrol_cols = ['age_0_5', 'age_5_17', 'age_18_greater']

    # =========================================================================================
    # 🏆 PATTERN 2: Child Biometric Compliance & Predictive Dropout Model
    # =========================================================================================
    print("Processing Pattern 2...")
    bio_pin = df_bio.groupby('pincode')['bio_age_5_17'].sum()
    enrol_pin = df_enrol.groupby('pincode')['age_5_17'].sum()
    
    child_risk = pd.concat([bio_pin, enrol_pin], axis=1).fillna(0)
    child_risk['Compliance_Ratio'] = child_risk['bio_age_5_17'] / (child_risk['age_5_17'] + 1)
    
    # Filter for meaningful volume (top 50 percentile of enrollment volume) to avoid noise
    vol_thresh = child_risk['age_5_17'].quantile(0.5)
    high_vol_districts = child_risk[child_risk['age_5_17'] > vol_thresh]
    
    # Lowest compliance ratio = Highest Risk
    dropout_zones = high_vol_districts.sort_values('Compliance_Ratio').head(5)
    
    # Get Context (Districts)
    pin_context = get_pincode_context(df_enrol, dropout_zones.index)
    
    # Format Evidence
    evidence = {}
    for pin, row in dropout_zones.iterrows():
        dist = pin_context.get(pin, 'Unknown')
        evidence[f"{pin} ({dist})"] = round(row['Compliance_Ratio'], 3)

    results.append({
        "Name": "Child Biometric Predictive Dropout Model",
        "Problem": "High risk of scholarship/exam denial due to outdated child biometrics.",
        "Insight": f"Identified top 5 Pincodes (e.g., in {list(pin_context.values())[0] if pin_context else 'Target Districts'}) with high school-age entry but critical Failure-to-Update rates.",
        "Action": "Deploy 'MBU Camp' vans to schools in these specific Pincodes next month.",
        "Data": evidence
    })

    # =========================================================================================
    # ⚠️ PATTERN 19: Operational Volatility & Smart Staffing
    # =========================================================================================
    print("Processing Pattern 19...")
    daily_vol = df_enrol.groupby('date')[enrol_cols].sum().sum(axis=1)
    
    cv = daily_vol.std() / daily_vol.mean()
    
    daily_df = daily_vol.reset_index(name='volume')
    daily_df['dow'] = daily_df['date'].dt.day_name()
    dow_avg = daily_df.groupby('dow')['volume'].mean().sort_values(ascending=False)
    peak_day = dow_avg.index[0]
    peak_multiplier = dow_avg.iloc[0] / daily_vol.mean()

    results.append({
        "Name": "Smart Staffing Forecast (Volatility Index)",
        "Problem": f"High operational volatility (CV: {cv:.2f}) causes long queues and staff burnout.",
        "Insight": f"{peak_day}s see {peak_multiplier:.1f}x the average volume.",
        "Action": f"Implement 'Dynamic Rostering': Shift 30% of admin staff to front-desk roles on {peak_day}s. Pre-scale server capacity for peak seasons.",
        "Data": {"Volatility_CV": round(cv, 2), "Peak_Day": peak_day, "Peak_Multiplier": round(peak_multiplier, 2)}
    })

    # =========================================================================================
    # 🔍 PATTERN 18: Inclusion Shadow Zones (Pincode Depth)
    # =========================================================================================
    print("Processing Pattern 18...")
    total_txns = df_enrol.groupby('pincode')[enrol_cols].sum().sum(axis=1)
    avg_txn = total_txns.mean()
    median_txn = total_txns.median()
    
    shadow_zone_count = total_txns[total_txns < (median_txn * 0.1)].count()
    
    results.append({
        "Name": "Inclusion Shadow Zones",
        "Problem": f"Uneven service distribution leaves {shadow_zone_count} Pincodes effectively 'offline'.",
        "Insight": "These 'Thin Zones' represent last-mile exclusion risks for elderly/disabled residents.",
        "Action": "Cross-reference these Pincodes with Census data. If population > 5000, mandate a permanent 'Mini-ASK' center.",
        "Data": {"Avg_Txn_Per_Pin": int(avg_txn), "Shadow_Pincodes_Detected": int(shadow_zone_count)}
    })

    # =========================================================================================
    # 🏙️ PATTERN 1: State Maturity & Self-Service Push
    # =========================================================================================
    print("Processing Pattern 1...")
    state_bio = df_bio.groupby('state')[bio_cols].sum().sum(axis=1)
    state_demo = df_demo.groupby('state')[demo_cols].sum().sum(axis=1)
    state_new = df_enrol.groupby('state')[enrol_cols].sum().sum(axis=1)
    
    # Align all series into a single DataFrame to handle index mismatches safely
    state_stats = pd.DataFrame({
        'Bio': state_bio,
        'Demo': state_demo,
        'New': state_new
    }).fillna(0)
    
    state_stats['Maintenance_Index'] = (state_stats['Bio'] + state_stats['Demo']) / (state_stats['New'] + 1)
    
    # Filter for Major States (> 1000 enrolments)
    major_states = state_stats[state_stats['New'] > 1000]['Maintenance_Index'].sort_values(ascending=False)
    
    if not major_states.empty:
        top_state = major_states.index[0]
        score = major_states.iloc[0]
    else:
        # Fallback to absolute top
        sorted_stats = state_stats['Maintenance_Index'].sort_values(ascending=False)
        top_state = sorted_stats.index[0]
        score = sorted_stats.iloc[0]
    
    results.append({
        "Name": "Digital Literacy Campaign Map",
        "Problem": "Mature states wasting resources on enrolment camps instead of update efficiency.",
        "Insight": f"Top Major State '{top_state}' has a Maintenance Index of {score:.1f}, indicating total saturation.",
        "Action": "Launch 'SSUP Sunday' campaigns here to migrate 40% of footfall to online channels.",
        "Data": {"Top_Mature_State": top_state, "Index_Score": round(score, 2)}
    })

    # =========================================================================================
    # 🆕 PATTERN: Hyper-Active "Update Loop" Zones
    # =========================================================================================
    print("Processing New Pattern: Update Loops...")
    demo_load = df_demo.groupby('pincode')['demo_age_17_'].sum()
    new_load = df_enrol.groupby('pincode')['age_18_greater'].sum()
    
    error_proxy = pd.concat([demo_load, new_load], axis=1).fillna(0)
    error_proxy = error_proxy[error_proxy['age_18_greater'] > 50] 
    
    error_proxy['Churn_Rate'] = error_proxy['demo_age_17_'] / error_proxy['age_18_greater']
    hotspots = error_proxy.sort_values('Churn_Rate', ascending=False).head(5)
    
    pin_context_loop = get_pincode_context(df_enrol, hotspots.index)
    
    evidence_loop = {}
    for pin, row in hotspots.iterrows():
        dist = pin_context_loop.get(pin, 'Unknown')
        evidence_loop[f"{pin} ({dist})"] = round(row['Churn_Rate'], 1)

    results.append({
        "Name": "Operator Error Hotspot Map",
        "Problem": "High-frequency update zones suggest operator errors or confusing local documentation requirements.",
        "Insight": f"Top 5 zones show {hotspots['Churn_Rate'].mean():.1f}x demographic churn vs new enrolments.",
        "Action": "Audit Enrolment Agencies (EA) in these Pincodes. Stop the 'Update-Reject-Update' cycle.",
        "Data": evidence_loop
    })

    # =========================================================================================
    # 🆕 PATTERN: Newborn Catchment Speed Index
    # =========================================================================================
    print("Processing New Pattern: Newborn Integration...")
    infant_df = df_enrol.groupby('date')['age_0_5'].sum()
    infant_df_reset = infant_df.reset_index()
    infant_df_reset['month'] = infant_df_reset['date'].dt.month_name()
    monthly_infant = infant_df_reset.groupby('month')['age_0_5'].mean().sort_values()
    
    if not monthly_infant.empty:
        lowest_month = monthly_infant.index[0]
        peak_month = monthly_infant.index[-1]
        peak_val = monthly_infant.iloc[-1]
        low_val = monthly_infant.iloc[0]
        drop_pct = (peak_val - low_val) / peak_val * 100
    else:
        lowest_month = "N/A"
        drop_pct = 0
    
    results.append({
        "Name": "Seamless Birth Integration Framework",
        "Problem": "Lag between birth and ID generation causes delays in benefits.",
        "Insight": f"In {lowest_month}, infant enrolment drops by {drop_pct:.0f}% compared to {peak_month}.",
        "Action": "Integrate Aadhaar Kits into Maternity Wards during low-enrolment months.",
        "Data": {"Lowest_Month": lowest_month, "Drop_From_Peak": f"{drop_pct:.1f}%"}
    })

    # =========================================================================================
    # 🆕 PATTERN: The "Ageing Biometric" Risk (Adults)
    # =========================================================================================
    print("Processing New Pattern: Ageing Biometrics...")
    adult_bio = df_bio.groupby('state')['bio_age_17_'].sum()
    adult_demo = df_demo.groupby('state')['demo_age_17_'].sum()
    
    risk_df = pd.concat([adult_bio, adult_demo], axis=1).fillna(0)
    risk_df['Bio_Neglect_Ratio'] = risk_df['bio_age_17_'] / (risk_df['demo_age_17_'] + 1)
    
    # Filter for states with significant activity (> 5000 demo updates) to ignore noisy small regions
    risk_df = risk_df[risk_df['demo_age_17_'] > 5000]
    
    neglect_states = risk_df.sort_values('Bio_Neglect_Ratio').head(5)
    
    results.append({
        "Name": "Authentication Assurance Risk (Biometric Neglect)",
        "Problem": "Adults updating demographics but skipping biometric refreshes will face AEBAS failures.",
        "Insight": f"States like {neglect_states.index[0]} have a nearly zero biometric refresh rate.",
        "Action": "Send SMS alerts: 'Please update fingerprints to avoid banking failures.'",
        "Data": neglect_states['Bio_Neglect_Ratio'].apply(lambda x: round(x, 4)).to_dict()
    })

    return results


def save_report(results):
    filename = "winning_hackathon_report.txt"
    with open(filename, "w", encoding='utf-8') as f:
        f.write("UIDAI DATA HACKATHON 2026: WINNING SUBMISSION\n")
        f.write("===============================================\n\n")
        f.write("Executive Summary: This report moves beyond statistics to 'Systemic Solutions'.\n")
        f.write("Focus: Preventing Denial of Service, Inclusion, and Predictive Resource Allocation.\n\n")
        
        for i, res in enumerate(results, 1):
            f.write(f"🚀 STRATEGIC PILLAR {i}: {res['Name']}\n")
            f.write(f"🛑 Problem: {res['Problem']}\n")
            f.write(f"💡 Insight: {res['Insight']}\n")
            f.write(f"⚡ ACTION:  {res['Action']}\n")
            f.write(f"📊 Evidence: {res['Data']}\n")
            f.write("-" * 60 + "\n\n")
            
    print(f"Report saved to {filename}")

if __name__ == "__main__":
    try:
        df_bio, df_demo, df_enrol = load_data()
        winning_insights = analyze_winning_patterns_script(df_bio, df_demo, df_enrol)
        save_report(winning_insights)
    except Exception as e:
        with open("error_log_winning.txt", "w") as f:
            f.write(f"Error: {e}\n")
            import traceback
            f.write(traceback.format_exc())
        print("Error encountered. Check error_log_winning.txt")
