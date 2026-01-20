import pandas as pd
import os

base_path = r"d:/Sudarshan Khot/Coding/UIDAI"

# Paths to the first chunk of each dataset
biometric_path = os.path.join(base_path, "api_data_aadhar_biometric", "api_data_aadhar_biometric", "api_data_aadhar_biometric_0_500000.csv")
demographic_path = os.path.join(base_path, "api_data_aadhar_demographic", "api_data_aadhar_demographic", "api_data_aadhar_demographic_0_500000.csv")
enrolment_path = os.path.join(base_path, "api_data_aadhar_enrolment", "api_data_aadhar_enrolment", "api_data_aadhar_enrolment_0_500000.csv")


# Analyze each
with open("inspection_results.txt", "w") as f:
    def log(msg):
        print(msg)
        f.write(str(msg) + "\n")

    def analyze_file(path, name):
        log(f"--- Analyzing {name} ---")
        if not os.path.exists(path):
            log(f"File not found: {path}")
            return
        
        try:
            df = pd.read_csv(path, nrows=1000) # Read only 1000 rows for initial inspection
            log("Columns: " + str(df.columns.tolist()))
            log("\nHead:\n" + str(df.head(3)))
            log("\nInfo:")
            # Capture info() output which prints to buffer
            import io
            buf = io.StringIO()
            df.info(buf=buf)
            log(buf.getvalue())
            log("\nDescribe:\n" + str(df.describe(include='all')))
            return df
        except Exception as e:
            log(f"Error reading {name}: {e}")
            return None

    df_bio = analyze_file(biometric_path, "Biometric Data")
    log("\n" + "="*50 + "\n")
    df_demo = analyze_file(demographic_path, "Demographic Data")
    log("\n" + "="*50 + "\n")
    df_enrol = analyze_file(enrolment_path, "Enrolment Data")

