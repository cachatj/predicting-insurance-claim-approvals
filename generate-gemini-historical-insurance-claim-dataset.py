import polars as pl
import random
from faker import Faker
from datetime import datetime, timedelta
import numpy as np
import os
import time

# --- Configurable Parameters ---

# Data Volume & Timeframe
CLAIMS_PER_YEAR = 500000
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2024, 12, 31)

# Unique Entities
NUM_UNIQUE_PATIENTS = 2000000
NUM_UNIQUE_PROVIDERS = 65000  # Midpoint of 50k-75k
NUM_UNIQUE_PAYERS = 20

# Denial Rate & Status Format
BASE_DENIAL_RATE_RANGE = (0.18, 0.20) # Overall target range
# Claim Status: 0 = Approved, 1 = Denied (most common for ML binary classification)
CLAIM_STATUS_DENIED = 1
CLAIM_STATUS_APPROVED = 0

# Output Files
OUTPUT_DIR = "synthetic_claims_data"
CSV_FILENAME = "synthetic_claims.csv"
PARQUET_FILENAME = "synthetic_claims.parquet"
# Set to True to also generate separate patient and provider files (memory intensive for large numbers)
GENERATE_SEPARATE_PATIENT_PROVIDER_FILES = False # Default to False, change if needed

# --- Setup ---
fake = Faker()
Faker.seed(0) # for reproducibility
random.seed(0)
np.random.seed(0)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Calculate total claims and years
num_years = END_DATE.year - START_DATE.year + 1
total_claims = CLAIMS_PER_YEAR * num_years
total_days = (END_DATE - START_DATE).days

print(f"Starting data generation...")
print(f"Target Claims: {total_claims}")
print(f"Years: {num_years} ({START_DATE.year}-{END_DATE.year})")
print(f"Unique Patients: {NUM_UNIQUE_PATIENTS}")
print(f"Unique Providers: {NUM_UNIQUE_PROVIDERS}")
print(f"Unique Payers: {NUM_UNIQUE_PAYERS}")
# Corrected print format for the range tuple
print(f"Target Denial Rate: {BASE_DENIAL_RATE_RANGE[0]*100:.1f}% - {BASE_DENIAL_RATE_RANGE[1]*100:.1f}%")
print(f"Claim Status: {CLAIM_STATUS_APPROVED}=Approved, {CLAIM_STATUS_DENIED}=Denied")
print("-" * 30)

start_time = time.time()

# --- Generate Unique Entities ---
print("Generating unique entity IDs...")
# Using simple sequential IDs prefixed for clarity and uniqueness
# Ensure the padding (e.g., :08d) is sufficient for the max number of entities
patient_ids = [f"PAT_{i+1:08d}" for i in range(NUM_UNIQUE_PATIENTS)]
provider_ids = [f"PROV_{i+1:05d}" for i in range(NUM_UNIQUE_PROVIDERS)]
payer_ids = [f"PAYER_{i+1:02d}" for i in range(NUM_UNIQUE_PAYERS)]
print("  Unique entity IDs generated.")

# --- Define Realistic Codes & Distributions ---
print("Defining codes and distributions...")
# Sample CPT Codes (with rough frequency weights)
cpt_codes = ['99213', '99214', '99203', '99204', '99395', '99396', '85025', '80053', '71046', '36415', '99283', '99284', '99285', '90686', '90716']
cpt_weights = [20, 18, 10, 8, 5, 5, 7, 6, 4, 3, 5, 4, 2, 2, 1]
cpt_weights = np.array(cpt_weights) / sum(cpt_weights) # Normalize

# Sample ICD-10 Codes (with rough frequency weights)
# !! Consider replacing placeholders with a more comprehensive list if needed !!
icd10_codes = ['E11.9', 'I10', 'J45.909', 'R07.9', 'M54.5', 'G89.29', 'N39.0', 'Z00.00', 'F41.9', 'K21.9', 'L20.9', 'R51', 'S06.0X0A', 'T63.01XA', 'Z12.39'] # Placeholder ICD-10 codes
icd10_weights = [15, 12, 8, 10, 7, 6, 5, 5, 4, 6, 4, 3, 3, 1, 1]
icd10_weights = np.array(icd10_weights) / sum(icd10_weights) # Normalize

# Denial Reason Codes (based on user input distribution) [2]
# Added a few more specific codes based on common issues [1, 3, 4]
denial_weights = {'OTHER': 34, 'ADMIN': 18, 'EXCLUDED': 16, 'NO_AUTH': 9, 'MED_NEC': 6, 'CODING_MISMATCH': 5, 'INCOMPLETE_DOCS': 5, 'TIMELY_FILING': 3, 'DUPLICATE': 2, 'ELIGIBILITY': 2}
denial_reason_list = list(denial_weights.keys())
denial_weight_list = np.array([denial_weights[reason] for reason in denial_reason_list]) / sum(denial_weights.values())

# Provider Specialties (assign some as higher risk for denials)
# !! Consider replacing placeholders with a more comprehensive list if needed !!
specialties = ['Internal Medicine', 'Pediatrics', 'Family Practice', 'Cardiology', 'Dermatology', 'Radiology', 'Emergency Medicine', 'Psychiatry', 'General Surgery', 'Orthopedics'] # Placeholder specialties
high_denial_specialties = {'Radiology', 'Emergency Medicine', 'Psychiatry', 'General Surgery'} # Example set
provider_specialty_map = {prov_id: random.choice(specialties) for prov_id in provider_ids}

# Payer Base Denial Rates (assign some higher rates)
payer_denial_rates = {}
num_high_rate_payers = int(NUM_UNIQUE_PAYERS * 0.25) # ~25% of payers have higher base rates
high_rate_payers = random.sample(payer_ids, num_high_rate_payers)
for payer in payer_ids:
    if payer in high_rate_payers:
        # Assign higher base denial rate for these payers
        payer_denial_rates[payer] = random.uniform(BASE_DENIAL_RATE_RANGE[1] * 1.1, BASE_DENIAL_RATE_RANGE[1] * 1.5) # Slightly above target max
    else:
        # Assign denial rate around the target range for others
        payer_denial_rates[payer] = random.uniform(BASE_DENIAL_RATE_RANGE[0] * 0.8, BASE_DENIAL_RATE_RANGE[1] * 1.2) # Around target range

print("  Codes and distributions defined.")

# --- Generate Claims Data ---
print(f"Generating {total_claims} claims records (this may take a while)...")
# FIX: Initialize claims_data as an empty list before the loop
claims_data = []
denial_count = 0

# Pre-sample choices to speed up the loop
print("  Pre-sampling random choices...")
patient_choices = random.choices(patient_ids, k=total_claims)
provider_choices = random.choices(provider_ids, k=total_claims)
payer_choices = random.choices(payer_ids, k=total_claims)
cpt_choices = np.random.choice(cpt_codes, size=total_claims, p=cpt_weights)
icd10_choices = np.random.choice(icd10_codes, size=total_claims, p=icd10_weights)
auth_obtained_choices = np.random.rand(total_claims) > 0.15 # 85% True
coding_mismatch_choices = np.random.rand(total_claims) < 0.08 # 8% True
docs_incomplete_choices = np.random.rand(total_claims) < 0.10 # 10% True
denial_reason_choices = np.random.choice(denial_reason_list, size=total_claims, p=denial_weight_list)
random_determiners = np.random.rand(total_claims) # For final denial decision
random_reason_overrides = np.random.rand(total_claims, 2) # For reason override logic
random_paid_percentages = np.random.uniform(0.70, 0.95, size=total_claims) # For approved claims
print("  Pre-sampling complete.")

# Loop through and generate each claim
for i in range(total_claims):
    if (i + 1) % (total_claims // 20) == 0 or i == 0: # Print progress every 5% and at the start
        progress_percent = (i + 1) / total_claims * 100
        current_time = time.time()
        elapsed = current_time - start_time
        print(f"  Generating claim {i+1}/{total_claims} ({progress_percent:.1f}%) - Elapsed: {elapsed:.1f}s")

    # Basic Claim Info from pre-sampled choices
    claim_id = f"CLAIM_{i+1:09d}" # Start claim ID from 1 for consistency
    patient_id = patient_choices[i]
    provider_id = provider_choices[i]
    payer_id = payer_choices[i]
    provider_specialty = provider_specialty_map[provider_id] # Look up specialty

    # Dates
    date_of_service = START_DATE + timedelta(days=random.randint(0, total_days))
    submission_lag_days = random.randint(1, 30)
    claim_submission_date = date_of_service + timedelta(days=submission_lag_days)
    # Ensure submission date doesn't exceed the overall end date
    claim_submission_date = min(claim_submission_date, END_DATE)

    # Codes and Charges from pre-sampled choices
    cpt_code = cpt_choices[i]
    icd10_code = icd10_choices[i]
    # Keep some randomness here, maybe base range on CPT code in a future version
    claim_charge_amount = round(random.uniform(50.0, 5000.0), 2)

    # Factors influencing denial from pre-sampled choices
    prior_authorization_obtained = auth_obtained_choices[i]
    coding_mismatch = coding_mismatch_choices[i]
    documentation_incomplete = docs_incomplete_choices[i]

    # Determine Claim Status (Apply correlations)
    current_denial_prob = payer_denial_rates[payer_id] # Start with payer base rate

    # Adjust probability based on factors
    denial_reason_override = None
    # Factor: Prior Authorization
    if not prior_authorization_obtained:
        current_denial_prob = min(current_denial_prob * 3.5, 0.90) # Significantly increase, cap at 90%
        denial_reason_override = 'NO_AUTH' # Likely reason
    # Factor: Provider Specialty
    if provider_specialty in high_denial_specialties:
        current_denial_prob *= 1.4 # Increase by 40%
    # Factor: Coding Mismatch
    if coding_mismatch:
        current_denial_prob *= 1.8 # Increase by 80%
        # High chance this is the reason if no other override
        if denial_reason_override is None and random_reason_overrides[i, 0] < 0.7:
             denial_reason_override = 'CODING_MISMATCH'
    # Factor: Documentation Incomplete
    if documentation_incomplete:
        current_denial_prob *= 1.7 # Increase by 70%
        # High chance this is the reason if no other override
        if denial_reason_override is None and random_reason_overrides[i, 1] < 0.6:
             denial_reason_override = 'INCOMPLETE_DOCS'

    # Factor: Submission Lag (minor effect)
    if submission_lag_days > 20:
         current_denial_prob *= 1.05
    elif submission_lag_days > 10:
         current_denial_prob *= 1.02

    # Clamp probability to ensure it stays within reasonable bounds
    current_denial_prob = max(0.01, min(current_denial_prob, 0.95)) # Keep within 1-95% bounds

    # Final status determination using pre-sampled random number
    is_denied = random_determiners[i] < current_denial_prob
    claim_status = CLAIM_STATUS_DENIED if is_denied else CLAIM_STATUS_APPROVED

    # Assign Denial Reason and Paid Amount
    denial_reason_code = None
    paid_amount = 0.0

    if claim_status == CLAIM_STATUS_DENIED:
        denial_count += 1
        if denial_reason_override:
            # If a specific factor strongly suggests a reason, use it with high probability
            if random.random() < 0.85: # 85% chance to use the override reason
                 denial_reason_code = denial_reason_override
            else: # Otherwise, pick from pre-sampled weighted list
                 denial_reason_code = denial_reason_choices[i]
        else:
            # Pick from pre-sampled weighted list if no strong factor identified
            denial_reason_code = denial_reason_choices[i]
        paid_amount = 0.0
    else: # Approved
        # Simulate partial payment based on charge amount using pre-sampled percentage
        paid_amount = round(claim_charge_amount * random_paid_percentages[i], 2)
        denial_reason_code = None # Ensure no denial reason for approved claims

    # Append claim record as a dictionary
    claims_data.append({
        "claim_id": claim_id,
        "patient_id": patient_id,
        "provider_id": provider_id,
        "payer_id": payer_id,
        "provider_specialty": provider_specialty,
        "date_of_service": date_of_service.date(),
        "claim_submission_date": claim_submission_date.date(),
        "submission_lag_days": submission_lag_days,
        "cpt_code": cpt_code,
        "icd10_code": icd10_code,
        "claim_charge_amount": claim_charge_amount,
        "prior_authorization_obtained": prior_authorization_obtained,
        "coding_mismatch_flag": coding_mismatch,
        "documentation_incomplete_flag": documentation_incomplete,
        "claim_status": claim_status,
        "denial_reason_code": denial_reason_code, # Will be None if approved
        "paid_amount": paid_amount,
    })

# --- Create Polars DataFrame ---
print("Converting generated list to Polars DataFrame...")
claims_df = pl.DataFrame(claims_data)
print(f"  Initial DataFrame memory usage: {claims_df.estimated_size('mb'):.2f} MB")
del claims_data # Free up memory from the list
print("  List deleted to free memory.")

# Define schema for clarity and potential performance/memory benefits
schema = {
    "claim_id": pl.Utf8,
    "patient_id": pl.Utf8,
    "provider_id": pl.Utf8,
    "payer_id": pl.Utf8,
    "provider_specialty": pl.Categorical, # Use Categorical for efficiency with repeated strings
    "date_of_service": pl.Date,
    "claim_submission_date": pl.Date,
    "submission_lag_days": pl.Int16, # Small int type is sufficient
    "cpt_code": pl.Categorical, # Use Categorical for efficiency
    "icd10_code": pl.Categorical, # Use Categorical for efficiency
    "claim_charge_amount": pl.Float64,
    "prior_authorization_obtained": pl.Boolean,
    "coding_mismatch_flag": pl.Boolean,
    "documentation_incomplete_flag": pl.Boolean,
    "claim_status": pl.Int8, # Use Int8 for 0/1 status
    "denial_reason_code": pl.Categorical, # Use Categorical for efficiency
    "paid_amount": pl.Float64,
}
print("Casting DataFrame to defined schema...")
claims_df = claims_df.cast(schema, strict=True) # Use strict=True to catch casting errors
print(f"  DataFrame memory usage after casting: {claims_df.estimated_size('mb'):.2f} MB")
print("  DataFrame created and cast to schema.")

# --- Generate Separate Patient/Provider Files (Optional) ---
if GENERATE_SEPARATE_PATIENT_PROVIDER_FILES:
    print("Generating separate patient file...")
    # FIX: Initialize patient_data as an empty list
    patient_data = []
    # Generate patient details - this can be slow for millions
    # Consider generating only a subset or simplifying if performance is critical
    # Limit generation for performance if needed, e.g., first 50k
    num_patients_to_generate = min(len(patient_ids), 50000)
    print(f"  Generating details for {num_patients_to_generate} patients...")
    for i, pat_id in enumerate(patient_ids[:num_patients_to_generate]):
         if (i + 1) % 5000 == 0:
             print(f"    Generated details for {i+1}/{num_patients_to_generate} patients...")
         patient_data.append({
             "patient_id": pat_id,
             "age": random.randint(0, 95),
             "gender": random.choice(['Male', 'Female', 'Other']),
             "zip_code": fake.zipcode(),
             # Add more fields as needed (e.g., medical history flags)
         })
    patients_df = pl.DataFrame(patient_data)
    # Define schema for patient data
    patient_schema = {
        "patient_id": pl.Utf8,
        "age": pl.Int8,
        "gender": pl.Categorical,
        "zip_code": pl.Utf8
    }
    patients_df = patients_df.cast(patient_schema, strict=True)
    patients_csv_path = os.path.join(OUTPUT_DIR, "synthetic_patients.csv")
    patients_parquet_path = os.path.join(OUTPUT_DIR, "synthetic_patients.parquet")
    print(f"  Saving patient data to CSV: {patients_csv_path}")
    patients_df.write_csv(patients_csv_path)
    print(f"  Saving patient data to Parquet: {patients_parquet_path}")
    patients_df.write_parquet(patients_parquet_path)
    print("  Patient file saved.")
    del patient_data # Free memory
    del patients_df

    print("Generating separate provider file...")
    # FIX: Initialize provider_data as an empty list
    provider_data = []
    print(f"  Generating details for {len(provider_ids)} providers...")
    for i, prov_id in enumerate(provider_ids):
         if (i + 1) % 5000 == 0:
             print(f"    Generated details for {i+1}/{len(provider_ids)} providers...")
         provider_data.append({
             "provider_id": prov_id,
             "provider_specialty": provider_specialty_map[prov_id],
             "provider_npi": fake.unique.numerify(text='##########'), # Fake NPI
             "provider_zip_code": fake.zipcode(),
             "clinic_name": "ANEC", # As requested
             # Add more fields as needed (e.g., years_experience)
         })
    providers_df = pl.DataFrame(provider_data)
    # Define schema for provider data
    provider_schema = {
        "provider_id": pl.Utf8,
        "provider_specialty": pl.Categorical,
        "provider_npi": pl.Utf8,
        "provider_zip_code": pl.Utf8,
        "clinic_name": pl.Utf8
    }
    # Cast provider specialty to Categorical for consistency and efficiency
    providers_df = providers_df.cast(provider_schema, strict=True)
    providers_csv_path = os.path.join(OUTPUT_DIR, "synthetic_providers.csv")
    providers_parquet_path = os.path.join(OUTPUT_DIR, "synthetic_providers.parquet")
    print(f"  Saving provider data to CSV: {providers_csv_path}")
    providers_df.write_csv(providers_csv_path)
    print(f"  Saving provider data to Parquet: {providers_parquet_path}")
    providers_df.write_parquet(providers_parquet_path)
    print("  Provider file saved.")
    del provider_data # Free memory
    del providers_df
else:
    print("Skipping generation of separate patient/provider files as per configuration.")


# --- Save Main Claims Output ---
csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
parquet_path = os.path.join(OUTPUT_DIR, PARQUET_FILENAME)

print(f"Saving main claims data to CSV: {csv_path}")
claims_df.write_csv(csv_path)
print("  CSV saved.")

print(f"Saving main claims data to Parquet: {parquet_path}")
claims_df.write_parquet(parquet_path)
print("  Parquet saved.")

# --- Final Report ---
end_time = time.time()
elapsed_time = end_time - start_time
actual_denial_rate = (denial_count / total_claims) * 100 if total_claims > 0 else 0
final_df_rows = len(claims_df)

print("-" * 30)
print("Data Generation Complete!")
print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print(f"Target claims: {total_claims}")
print(f"Actual claims generated: {final_df_rows}")
print(f"Actual denial rate: {actual_denial_rate:.2f}% (Target: {BASE_DENIAL_RATE_RANGE[0]*100:.1f}% - {BASE_DENIAL_RATE_RANGE[1]*100:.1f}%)")
print(f"Final DataFrame memory usage: {claims_df.estimated_size('mb'):.2f} MB")
print(f"Output files saved in directory: {OUTPUT_DIR}")
print("-" * 30)

# Display sample data and schema
print("DataFrame Schema:")
print(claims_df.schema)
print("\nSample Data (first 5 rows):")
print(claims_df.head())

# Clear the large DataFrame from memory if no longer needed immediately
# del claims_df
# print("Claims DataFrame cleared from memory.")
