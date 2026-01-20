# Child MBU Predictive Dropout & Outreach Model

## Overview
The Child MBU (Mandatory Biometric Update) Predictive Dropout & Outreach Model is a data-driven framework designed to identify children at risk of exclusion from government subsidies and services due to outdated biometric records. By analyzing demographic, biometric, and enrolment patterns across Aadhar datasets, this model predicts high-risk zones and prioritizes intervention strategies.

The system utilizes a proprietary **Child Exclusion Risk Score** to quantify urgency and facilitates targeted deployment of mobile outreach units to areas with the highest "update gaps."

## Key Features

### 1. Child Exclusion Risk Score
A composite metric that combines:
- **Compliance Ratio**: The gap between child enrolments and successful biometric updates.
- **Operational Volatility**: Using the Coefficient of Variation (CV) to measure demand stability.
- **Urgency Metrics**: Time-sensitive risk factors for children aged 5-17.

### 2. Deployment Intelligence
- **Prioritized Intervention Zones**: Identifies the Top 50 high-risk pincodes requiring immediate attention.
- **Phased Rollout Plan**: structured deployment across three phases (Phase 1: Critical, Phase 2: High, Phase 3: Moderate) to optimize resource allocation.
- **Cost-Benefit Analysis**: Balances the cost of mobile units against the social impact of reinstated services.

### 3. Migration Impact Analysis
- **Churn Detection**: Distinguishes between stable residential areas and high-churn migrant zones using demographic shifts.
- **Compliance Variance**: Quantifies the specific impact of migration on biometric update compliance (e.g., differential compliance rates in high-churn vs. standard zones).

### 4. Social Impact Quantification
Estimates the tangible human impact of interventions:
- **Scholarship Protection**: Number of students whose financial aid access is preserved.
- **Exam Access**: Number of students enabled to register for board exams.
- **Identity Security**: Reduction in potential exclusion errors.

## Repository Structure

### `data/`
Contains the core datasets and mapping files required for the analysis:
- **`raw/`**: Unzipped CSV chunks for biometric, demographic, and enrolment data.
- **`zips/`**: Original compressed archives.
- **`IDs/`**: Reference images and identification documents.
- **`state_district_mapping.txt`**: Mapping file for geographic aggregation.

### `notebooks/`
Jupyter notebooks for interactive analysis and visualization:
- **`Child_MBU_Predictive_Dropout_Model_v8.1.ipynb`**: The latest stable version of the predictive model, including full text analysis and visualizations.
- **`EDA_*.ipynb`**: Exploratory Data Analysis notebooks for specific data domains (Biometric, Demographic, Enrolment).

### `src/`
Core Python scripts for data processing and analysis execution:
- **`execute_analysis.py`**: The main driver script that runs the end-to-end analysis pipeline, generating the Quantified Analysis Report.
- **`run_analysis.py`**: Helper script for parameterized execution.
- **`analyze_*.py`**: Specialized modules for pattern detection and district-level aggregation.

### `scripts/`
Utility scripts for maintenance and file handling:
- **`convert_to_notebook.py`**: Tools to convert raw Python scripts or text files into structured Jupyter notebooks.
- **`refine_notebook.py`**: Automatic text refinement and cell organization utilities.

### `archive/`
Storage for previous model versions (`v7.1`, `v8.0`, `v8.2`) and backup files.

## Installation

1. Clone the repository and navigate to the project root.
2. Create and activate a virtual environment (optional but recommended).
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Requirements include:**
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.5.2

## Usage

### Running the Analysis Report
To generate the full text-based investigation report including risk scores and priority lists:

```bash
python src/execute_analysis.py
```

### Interactive Modeling
To explore the data, visualize trends, and modify model parameters, open the primary notebook:

```bash
jupyter notebook notebooks/Child_MBU_Predictive_Dropout_Model_v8.1.ipynb
```

## Methodology
The model follows a rigorous data pipeline:
1.  **Data Ingestion**: Loading stratified chunks of 3M+ records across three dimensions (Biometric, Demographic, Enrolment).
2.  **Cleaning & Normalization**: Handling infinite values, normalizing date formats, and mapping geographic identifiers.
3.  **Feature Engineering**: Calculating compliance ratios, update gaps, and churn indicators at the pincode level.
4.  **Risk Scoring**: Applying the weighted algorithm (Volume 40%, Urgency 35%, Access 25%) to rank intervention targets.
5.  **Output Generation**: Producing the prioritized list of 50 pincodes and the aggregate district-level risk profile.

## Contact
For queries regarding the model logic or dataset specifications, please refer to the documentation within the `docs/` folder or contact the development team.
