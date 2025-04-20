import polars as pl
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import uuid
import os
from pathlib import Path

# Configuration parameters (easily adjustable)
CONFIG = {
    # Date range
    "start_date": datetime(2022, 1, 1),
    "end_date": datetime(2024, 12, 31),
    
    # Volume parameters
    "num_patients": 2_000_000,     # Up to 2 million patients
    "num_providers": 65_000,       # Between 50k-75k providers
    "num_payers": 20,              # 20 insurance companies
    "claims_per_year": 500_000,    # 500k claims per year
    
    # Claim status parameters
    "claim_status_format": "numeric",  # Options: "numeric" (0/1), "boolean" (True/False), "string" ("APPROVED"/"DENIED")
    "overall_denial_rate": 0.19,       # Target overall denial rate (18-20%)
    
    # Denial reason distribution
    "denial_reason_distribution": {
        "other": 0.34,                       # Other reasons - 34%
        "administrative": 0.18,              # Administrative issues - 18%
        "excluded_service": 0.16,            # Excluded service - 16%
        "no_prior_auth": 0.09,               # Lack of prior auth - 9%
        "not_medically_necessary": 0.06,     # Not medically necessary - 6%
        "remaining": 0.17                    # All other reasons - 17%
    },
    
    # Output configuration
    "output_dir": "./output",
    "output_formats": ["csv", "parquet"]     # Output formats
}

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
fake = Faker()
Faker.seed(42)

# Calculate total number of claims based on date range and claims per year
years_span = (CONFIG["end_date"].year - CONFIG["start_date"].year) + (1 if CONFIG["start_date"].month <= CONFIG["end_date"].month else 0)
CONFIG["total_claims"] = int(CONFIG["claims_per_year"] * years_span)

# Ensure output directory exists
Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

# ICD-10 diagnosis codes with descriptions (expanded list of common codes)
diagnosis_codes = {
    "E11.9": "Type 2 diabetes mellitus without complications",
    "I10": "Essential (primary) hypertension",
    "J45.909": "Unspecified asthma, uncomplicated",
    "M54.5": "Low back pain",
    "F41.9": "Anxiety disorder, unspecified",
    "F32.9": "Major depressive disorder, single episode, unspecified",
    "K21.9": "Gastro-esophageal reflux disease without esophagitis",
    "M17.9": "Osteoarthritis of knee, unspecified",
    "J40": "Bronchitis, not specified as acute or chronic",
    "N39.0": "Urinary tract infection, site not specified",
    "H40.9": "Unspecified glaucoma",
    "E78.5": "Hyperlipidemia, unspecified",
    "I25.10": "Atherosclerotic heart disease of native coronary artery without angina pectoris",
    "G47.00": "Insomnia, unspecified",
    "G43.909": "Migraine, unspecified, not intractable, without status migrainosus",
    "J02.9": "Acute pharyngitis, unspecified",
    "J01.90": "Acute sinusitis, unspecified",
    "M25.511": "Pain in right shoulder",
    "M25.512": "Pain in left shoulder",
    "M79.604": "Pain in right leg",
    "M79.605": "Pain in left leg",
    "M79.621": "Pain in right upper arm",
    "M79.622": "Pain in left upper arm",
    "R10.9": "Unspecified abdominal pain",
    "R07.9": "Chest pain, unspecified",
    "R51": "Headache",
    "Z00.00": "Encounter for general adult medical examination without abnormal findings",
    "Z00.129": "Encounter for routine child health examination without abnormal findings",
    "Z23": "Encounter for immunization",
    "Z12.11": "Encounter for screening for malignant neoplasm of colon",
    "Z12.31": "Encounter for screening mammogram for malignant neoplasm of breast"
}

# CPT procedure codes with descriptions (expanded list of common codes)
procedure_codes = {
    "99213": "Office/outpatient visit est (15 min)",
    "99214": "Office/outpatient visit est (25 min)",
    "99203": "Office/outpatient visit new (30 min)",
    "99204": "Office/outpatient visit new (45 min)",
    "80053": "Comprehensive metabolic panel",
    "85025": "Complete blood count (CBC)",
    "82607": "Vitamin B-12 blood test",
    "80061": "Lipid panel",
    "71045": "X-ray examination of chest, single view",
    "71046": "X-ray examination of chest, 2 views",
    "72100": "X-ray examination of lower spine, 2-3 views",
    "70450": "CT scan of head/brain without contrast",
    "93000": "Electrocardiogram, routine",
    "94640": "Inhalation treatment for airway obstruction",
    "97110": "Therapeutic exercises",
    "99385": "Preventive visit new, age 18-39",
    "99386": "Preventive visit new, age 40-64",
    "99395": "Preventive visit est, age 18-39",
    "99396": "Preventive visit est, age 40-64",
    "90471": "Immunization administration",
    "90686": "Influenza vaccine, quadrivalent (IIV4), 0.5 mL dosage",
    "90715": "Tdap vaccine, intramuscular",
    "99243": "Office consultation, 40 min",
    "99244": "Office consultation, 60 min",
    "96372": "Therapeutic, prophylactic, or diagnostic injection",
    "99283": "Emergency dept visit, moderate severity",
    "99284": "Emergency dept visit, high severity",
    "99285": "Emergency dept visit, high severity with threat",
    "29125": "Application of short arm splint",
    "29515": "Application of short leg splint",
    "45378": "Colonoscopy, diagnostic",
    "45380": "Colonoscopy with biopsy",
    "77067": "Screening mammography, bilateral",
    "77066": "Diagnostic mammography, bilateral"
}

# Revenue codes
revenue_codes = {
    "0120": "Room & Board - Semi-Private",
    "0250": "Pharmacy - General",
    "0270": "Medical/Surgical Supplies",
    "0300": "Laboratory - General",
    "0320": "Radiology - Diagnostic",
    "0370": "Anesthesia",
    "0420": "Physical Therapy",
    "0450": "Emergency Room",
    "0510": "Clinic - General",
    "0636": "Drugs requiring detailed coding",
    "0260": "IV Therapy - General",
    "0410": "Respiratory Services - General",
    "0610": "Magnetic Resonance Technology - General",
    "0730": "EKG/ECG - General",
    "0921": "Peripheral Vascular Lab"
}

# Place of service codes
place_of_service_codes = {
    "11": "Office",
    "21": "Inpatient Hospital",
    "22": "Outpatient Hospital",
    "23": "Emergency Room - Hospital",
    "24": "Ambulatory Surgical Center",
    "31": "Skilled Nursing Facility",
    "32": "Nursing Facility",
    "33": "Custodial Care Facility",
    "41": "Ambulance - Land",
    "50": "Federally Qualified Health Center",
    "65": "End-Stage Renal Disease Treatment Facility",
    "71": "State or Local Public Health Clinic",
    "72": "Rural Health Clinic",
    "20": "Urgent Care Facility",
    "12": "Home",
    "81": "Independent Laboratory"
}

# Create expanded denial reason codes with realistic categories
denial_reasons = {
    # Administrative reasons (18%)
    "A1": "Missing or invalid subscriber/insured ID number",
    "A2": "Claim lacks required information",
    "A3": "Duplicate claim submission",
    "A4": "Claim filed after filing deadline",
    "A5": "Claim form incomplete or invalid",
    "A6": "Missing or invalid provider information",
    "A7": "Invalid place of service for procedure",
    "A8": "Incorrect provider specialty for service",
    
    # Excluded service reasons (16%)
    "B1": "Service specifically excluded from coverage",
    "B2": "Service not covered under patient's plan",
    "B3": "Cosmetic procedure not covered",
    "B4": "Routine service not covered",
    "B5": "Non-emergency service performed out of network",
    "B6": "Annual benefit maximum met",
    "B7": "Service considered experimental/investigational",
    "B8": "Frequency limitation exceeded",
    
    # Lack of prior authorization (9%)
    "C1": "Prior authorization/precertification required but not obtained",
    "C2": "Prior authorization number invalid",
    "C3": "Service differs from authorized service",
    "C4": "Authorization expired or date of service outside approval range",
    
    # Medical necessity reasons (6%)
    "D1": "Service not medically necessary based on diagnosis",
    "D2": "Upcoding detected - service level not supported by documentation",
    "D3": "Unbundling detected - services should be billed as single procedure",
    "D4": "Diagnosis does not support medical necessity for procedure",
    
    # Other reasons (34%)
    "E1": "Patient not eligible on date of service",
    "E2": "Coverage terminated prior to service date",
    "E3": "Patient has primary insurance with another carrier",
    "E4": "Preexisting condition limitations apply",
    "E5": "Service processed under different procedure code",
    "E6": "Claim pending for additional information",
    "E7": "Modifier inappropriate or missing",
    "E8": "Coordination of benefits information required",
    "E9": "Provider not in-network for this service",
    "E10": "Claim requires manual review",
    "E11": "Invalid diagnosis code",
    "E12": "Invalid procedure code",
    "E13": "Patient responsibility (deductible/copay/coinsurance)",
    "E14": "Billed amount exceeds fee schedule allowance",
    "E15": "Charges included in global procedure payment",
    
    # Remaining reasons (17%)
    "F1": "Service previously adjudicated",
    "F2": "Refund or reversal of previous claim payment",
    "F3": "Services properly billed to facility",
    "F4": "Claim awaiting medical records review",
    "F5": "Waiting for response to payer initiated correspondence",
    "F6": "Coding inconsistent with national standards",
    "F7": "Claim requires specialized handling",
    "F8": "Non-covered provider specialty"
}

# Group denial reasons by category
denial_categories = {
    "administrative": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
    "excluded_service": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"],
    "no_prior_auth": ["C1", "C2", "C3", "C4"],
    "not_medically_necessary": ["D1", "D2", "D3", "D4"],
    "other": ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12", "E13", "E14", "E15"],
    "remaining": ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
}

# Invert for lookup
category_for_denial_code = {}
for category, codes in denial_categories.items():
    for code in codes:
        category_for_denial_code[code] = category

# Create lists of medical specialties with specialty-specific denial rate modifiers
specialties = [
    {"name": "Family Medicine", "denial_modifier": 0.9},  # Lower than average denials
    {"name": "Internal Medicine", "denial_modifier": 1.0},  # Average denials
    {"name": "Cardiology", "denial_modifier": 1.2},  # Higher than average denials
    {"name": "Dermatology", "denial_modifier": 1.1},
    {"name": "Orthopedics", "denial_modifier": 1.3},  # Much higher denials
    {"name": "Neurology", "denial_modifier": 1.2},
    {"name": "Pediatrics", "denial_modifier": 0.8},  # Lower denials
    {"name": "Obstetrics", "denial_modifier": 1.2},
    {"name": "Gynecology", "denial_modifier": 1.0},
    {"name": "Psychiatry", "denial_modifier": 1.1},
    {"name": "Oncology", "denial_modifier": 1.3},  # Much higher denials
    {"name": "Radiology", "denial_modifier": 1.1},
    {"name": "Urology", "denial_modifier": 1.0},
    {"name": "Gastroenterology", "denial_modifier": 1.1},
    {"name": "Endocrinology", "denial_modifier": 1.0},
    {"name": "Nephrology", "denial_modifier": 1.2},
    {"name": "Pulmonology", "denial_modifier": 1.1},
    {"name": "Rheumatology", "denial_modifier": 1.2},
    {"name": "Allergy & Immunology", "denial_modifier": 0.9},
    {"name": "Emergency Medicine", "denial_modifier": 1.0},
    {"name": "Physical Medicine & Rehabilitation", "denial_modifier": 1.3},  # High denials
    {"name": "Infectious Disease", "denial_modifier": 1.0},
    {"name": "General Surgery", "denial_modifier": 1.2},
    {"name": "Vascular Surgery", "denial_modifier": 1.3},
    {"name": "Plastic Surgery", "denial_modifier": 1.4}  # Very high denials
]

# Payer types and names with payer-specific denial rate modifiers
payer_types = ["Commercial", "Medicare", "Medicaid", "Self-Pay", "Workers Comp"]

payer_info = [
    {"name": "Blue Cross Blue Shield", "type": "Commercial", "denial_modifier": 1.0},  # Average denial rate
    {"name": "UnitedHealthcare", "type": "Commercial", "denial_modifier": 1.2},  # Higher denial rate
    {"name": "Aetna", "type": "Commercial", "denial_modifier": 1.1},
    {"name": "Cigna", "type": "Commercial", "denial_modifier": 1.3},  # Much higher denial rate
    {"name": "Humana", "type": "Commercial", "denial_modifier": 1.0},
    {"name": "Kaiser Permanente", "type": "Commercial", "denial_modifier": 0.9},  # Lower denial rate
    {"name": "Medicare", "type": "Medicare", "denial_modifier": 0.8},  # Lower denial rate
    {"name": "Medicare Advantage", "type": "Medicare", "denial_modifier": 1.1},
    {"name": "Medicaid", "type": "Medicaid", "denial_modifier": 1.2},  # Higher denial rate
    {"name": "Centene", "type": "Commercial", "denial_modifier": 1.1},
    {"name": "Molina Healthcare", "type": "Commercial", "denial_modifier": 1.2},
    {"name": "Anthem", "type": "Commercial", "denial_modifier": 1.0},
    {"name": "Health Net", "type": "Commercial", "denial_modifier": 1.1},
    {"name": "CareFirst", "type": "Commercial", "denial_modifier": 0.9},
    {"name": "Wellcare", "type": "Commercial", "denial_modifier": 1.0},
    {"name": "Tricare", "type": "Commercial", "denial_modifier": 0.9},
    {"name": "Optum", "type": "Commercial", "denial_modifier": 1.1},
    {"name": "CVS Caremark", "type": "Commercial", "denial_modifier": 1.0},
    {"name": "Self-Pay", "type": "Self-Pay", "denial_modifier": 0.7},  # Lower denial rate
    {"name": "Workers Compensation", "type": "Workers Comp", "denial_modifier": 1.2}  # Higher denial rate
]

# Procedures with higher denial rates (based on realistic patterns)
high_denial_procedures = [
    "70450",  # CT scan
    "93000",  # EKG
    "45378",  # Colonoscopy
    "77067",  # Mammography
    "96372",  # Injections
    "97110",  # Physical therapy
    "99284",  # ER visits
    "99244",  # Consultations
    "45380",  # Colonoscopy with biopsy
    "77066"   # Diagnostic mammography
]

# Create patient data (optimized for large dataset)
def generate_patients(num_patients):
    print(f"Generating {num_patients} patients...")
    
    # Generate data in chunks for memory efficiency
    chunk_size = 100000
    all_chunks = []
    
    for i in range(0, num_patients, chunk_size):
        end = min(i + chunk_size, num_patients)
        chunk_size_actual = end - i
        
        patient_ids = [f"P{str(j+1).zfill(8)}" for j in range(i, end)]
        genders = [random.choice(['M', 'F']) for _ in range(chunk_size_actual)]
        
        first_names = []
        for g in genders:
            if g == 'M':
                first_names.append(fake.first_name_male())
            else:
                first_names.append(fake.first_name_female())
        
        dobs = [fake.date_of_birth(minimum_age=1, maximum_age=95) for _ in range(chunk_size_actual)]
        
        # Generate random insurance IDs with weighted distribution (some payers are more common)
        insurance_ids = [f"INS{str(random.randint(1, CONFIG['num_payers'])).zfill(3)}" for _ in range(chunk_size_actual)]
        
        # Generate some chronic conditions
        chronic_conditions = []
        for _ in range(chunk_size_actual):
            num_conditions = random.choices([0, 1, 2, 3, 4], weights=[0.6, 0.2, 0.1, 0.07, 0.03])[0]
            if num_conditions > 0:
                conditions = random.sample(list(diagnosis_codes.keys()), num_conditions)
                chronic_conditions.append(','.join(conditions))
            else:
                chronic_conditions.append(None)
        
        chunk_data = {
            'patient_id': patient_ids,
            'first_name': first_names,
            'last_name': [fake.last_name() for _ in range(chunk_size_actual)],
            'dob': dobs,
            'gender': genders,
            'address': [fake.street_address() for _ in range(chunk_size_actual)],
            'city': [fake.city() for _ in range(chunk_size_actual)],
            'state': [fake.state_abbr() for _ in range(chunk_size_actual)],
            'zip': [fake.zipcode() for _ in range(chunk_size_actual)],
            'phone': [fake.phone_number() for _ in range(chunk_size_actual)],
            'email': [fake.email() for _ in range(chunk_size_actual)],
            'insurance_id': insurance_ids,
            'membership_id': [f"MEM{fake.numerify('##########')}" for _ in range(chunk_size_actual)],
            'chronic_conditions': chronic_conditions
        }
        
        all_chunks.append(pl.DataFrame(chunk_data))
        
        if i % (chunk_size * 10) == 0:
            print(f"  Generated {i + chunk_size_actual} patients...")
    
    return pl.concat(all_chunks)

# Create provider data
def generate_providers(num_providers):
    print(f"Generating {num_providers} providers...")
    
    providers = []
    
    # Create provider data in a more efficient way
    provider_ids = [f"DR{str(i+1).zfill(7)}" for i in range(num_providers)]
    
    # Randomly select specialties with their denial modifiers
    selected_specialties = [random.choice(specialties) for _ in range(num_providers)]
    specialty_names = [s["name"] for s in selected_specialties]
    specialty_modifiers = [s["denial_modifier"] for s in selected_specialties]
    
    providers_data = {
        'provider_id': provider_ids,
        'provider_name': [f"Dr. {fake.last_name()}" for _ in range(num_providers)],
        'npi': [fake.numerify('##########') for _ in range(num_providers)],  # National Provider Identifier
        'specialty': specialty_names,
        'specialty_denial_modifier': specialty_modifiers,
        'facility_name': [f"ANEC Medical Center {random.randint(1, 25)}" for _ in range(num_providers)],
        'address': [fake.street_address() for _ in range(num_providers)],
        'city': [fake.city() for _ in range(num_providers)],
        'state': [fake.state_abbr() for _ in range(num_providers)],
        'zip': [fake.zipcode() for _ in range(num_providers)],
        'phone': [fake.phone_number() for _ in range(num_providers)],
        'network_status': [random.choice(['In-Network', 'Out-of-Network']) for _ in range(num_providers)],
        'years_experience': [random.randint(1, 40) for _ in range(num_providers)],
        'credentials': [random.choice(['MD', 'DO', 'NP', 'PA']) for _ in range(num_providers)],
        'average_patients_per_day': [random.randint(5, 40) for _ in range(num_providers)]
    }
    
    return pl.DataFrame(providers_data)

# Create payer data
def generate_payers(num_payers):
    print(f"Generating {num_payers} payers...")
    
    # Ensure we don't try to generate more payers than we have in our template list
    actual_num_payers = min(num_payers, len(payer_info))
    
    # Select a random subset of payers if num_payers < len(payer_info)
    selected_payers = random.sample(payer_info, actual_num_payers)
    
    payers_data = {
        'payer_id': [f"INS{str(i+1).zfill(3)}" for i in range(actual_num_payers)],
        'payer_name': [p["name"] for p in selected_payers],
        'payer_type': [p["type"] for p in selected_payers],
        'payer_denial_modifier': [p["denial_modifier"] for p in selected_payers],
        'average_processing_days': [random.randint(7, 45) for _ in range(actual_num_payers)],
        'electronic_claim_submission': [random.choice([True, False]) for _ in range(actual_num_payers)],
        'prior_auth_required_procedures': [','.join(random.sample(list(procedure_codes.keys()), random.randint(5, 15))) for _ in range(actual_num_payers)],
        'base_denial_rate': [round(CONFIG["overall_denial_rate"] * p["denial_modifier"], 3) for p in selected_payers],
        'average_reimbursement_rate': [round(random.uniform(0.50, 0.95), 2) for _ in range(actual_num_payers)],
        'timely_filing_limit_days': [random.choice([30, 60, 90, 120, 180, 365]) for _ in range(actual_num_payers)],
        'appeal_timeframe_days': [random.choice([30, 45, 60, 90]) for _ in range(actual_num_payers)],
        'contact_phone': [fake.phone_number() for _ in range(actual_num_payers)],
        'website': [f"www.{p['name'].lower().replace(' ', '')}.com" for p in selected_payers]
    }
    
    return pl.DataFrame(payers_data)

# Create claims data - optimized for large datasets
def generate_claims(num_claims, patients_df, providers_df, payers_df, start_date, end_date):
    print(f"Generating {num_claims} claims...")
    
    # Create a map of payer ID to denial rate for faster lookups
    payer_base_denial_rates = dict(zip(payers_df['payer_id'].to_list(), payers_df['base_denial_rate'].to_list()))
    payer_prior_auth_lists = dict(zip(payers_df['payer_id'].to_list(), payers_df['prior_auth_required_procedures'].to_list()))
    
    # Create a map of provider ID to denial modifier for faster lookups
    provider_denial_modifiers = dict(zip(providers_df['provider_id'].to_list(), providers_df['specialty_denial_modifier'].to_list()))
    provider_network_status = dict(zip(providers_df['provider_id'].to_list(), providers_df['network_status'].to_list()))
    
    # Time between service and claim submission (days)
    submission_lag_days = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90]
    submission_lag_weights = [0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02]
    
    # Generate claims in chunks to save memory
    chunk_size = 100000
    all_claim_chunks = []
    
    # Convert patient IDs to list for random selection
    patient_ids = patients_df['patient_id'].to_list()
    provider_ids = providers_df['provider_id'].to_list()
    
    # Calculate days in range for date generation
    days_in_range = (end_date - start_date).days
    
    # Setup denial reason selection based on configuration distribution
    denial_reason_weights = {}
    for category, codes in denial_categories.items():
        weight = CONFIG["denial_reason_distribution"].get(category, 0.0) / len(codes)
        for code in codes:
            denial_reason_weights[code] = weight
    
    for chunk_start in range(0, num_claims, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_claims)
        actual_chunk_size = chunk_end - chunk_start
        
        # Generate claim IDs
        claim_ids = [f"CLM{str(i+1).zfill(10)}" for i in range(chunk_start, chunk_end)]
        
        # Randomly select patients
        selected_patient_ids = [random.choice(patient_ids) for _ in range(actual_chunk_size)]
        
        # Randomly select providers
        selected_provider_ids = [random.choice(provider_ids) for _ in range(actual_chunk_size)]
        
        # Randomly select payer IDs
        selected_payer_ids = [f"INS{str(random.randint(1, CONFIG['num_payers'])).zfill(3)}" for _ in range(actual_chunk_size)]
        
        # Generate service dates (within the date range)
        service_dates = [start_date + timedelta(days=random.randint(0, days_in_range)) for _ in range(actual_chunk_size)]
        
        # Calculate submission lags and dates
        submission_lags = [random.choices(submission_lag_days, submission_lag_weights)[0] for _ in range(actual_chunk_size)]
        submission_dates = [min(service_date + timedelta(days=lag), end_date) for service_date, lag in zip(service_dates, submission_lags)]
        
        # Generate diagnoses
        primary_diagnoses = [random.choice(list(diagnosis_codes.keys())) for _ in range(actual_chunk_size)]
        has_secondary = [random.random() < 0.4 for _ in range(actual_chunk_size)]  # 40% chance of secondary diagnosis
        secondary_diagnoses = [random.choice(list(diagnosis_codes.keys())) if has_sec else None for has_sec in has_secondary]
        
        # Generate procedures
        procedure_codes_list = list(procedure_codes.keys())
        selected_procedures = [random.choice(procedure_codes_list) for _ in range(actual_chunk_size)]
        
        # Generate place of service
        places_of_service = [random.choice(list(place_of_service_codes.keys())) for _ in range(actual_chunk_size)]
        
        # Generate revenue codes
        has_revenue_code = [random.random() < 0.7 for _ in range(actual_chunk_size)]  # 70% chance of having revenue code
        revenue_codes_list = [random.choice(list(revenue_codes.keys())) if has_rev else None for has_rev in has_revenue_code]
        
        # Determine prior authorization status
        prior_auth_required = []
        prior_auth_obtained = []
        
        for i in range(actual_chunk_size):
            payer_id = selected_payer_ids[i]
            procedure = selected_procedures[i]
            payer_auth_list = payer_prior_auth_lists.get(payer_id, '').split(',')
            
            if procedure in payer_auth_list:
                prior_auth_required.append(True)
                prior_auth_obtained.append(random.random() < 0.85)  # 85% compliance rate
            else:
                prior_auth_required.append(False)
                prior_auth_obtained.append(False)
        
        # Calculate charge amounts
        base_charges = [random.uniform(50, 5000) for _ in range(actual_chunk_size)]
        charge_amounts = []
        
        for i in range(actual_chunk_size):
            charge = base_charges[i]
            place = places_of_service[i]
            
            # Hospital settings have higher charges
            if place in ["21", "23"]:
                charge *= random.uniform(1.5, 3.0)
                
            charge_amounts.append(round(charge, 2))
        
        # Determine claim status and denial reason
        claim_statuses = []
        denial_reason_codes = []
        denial_reason_descriptions = []
        
        for i in range(actual_chunk_size):
            payer_id = selected_payer_ids[i]
            provider_id = selected_provider_ids[i]
            procedure = selected_procedures[i]
            
            # Base denial rate from payer
            base_denial_prob = payer_base_denial_rates.get(payer_id, CONFIG["overall_denial_rate"])
            
            # Modify by provider specialty
            provider_modifier = provider_denial_modifiers.get(provider_id, 1.0)
            denial_probability = base_denial_prob * provider_modifier
            
            # Adjust denial probability based on various factors
            if prior_auth_required[i] and not prior_auth_obtained[i]:
                denial_probability += 0.60  # Significant increase if no prior auth
            
            if provider_network_status.get(provider_id) == 'Out-of-Network':
                denial_probability += 0.20  # Higher for out-of-network
            
            if procedure in high_denial_procedures:
                denial_probability += 0.15  # Higher for certain procedures
            
            timely_filing_limit = random.choice([30, 60, 90, 120, 180, 365])  # Simulating payer's limit
            if submission_lags[i] > (timely_filing_limit * 0.8):
                denial_probability += 0.30  # Higher if close to timely filing limit
                
            # Cap the denial probability at 0.95
            denial_probability = min(denial_probability, 0.95)
            
            # Determine claim status
            is_denied = random.random() < denial_probability
            
            # Format claim status according to configuration
            if is_denied:
                if CONFIG["claim_status_format"] == "numeric":
                    claim_statuses.append(1)  # 1 = DENIED
                elif CONFIG["claim_status_format"] == "boolean":
                    claim_statuses.append(True)  # True = DENIED
                else:  # string format
                    claim_statuses.append("DENIED")
                    
                # Select denial reason based on configured distribution
                if prior_auth_required[i] and not prior_auth_obtained[i]:
                    denial_code = random.choice(denial_categories["no_prior_auth"])
                elif provider_network_status.get(provider_id) == 'Out-of-Network' and random.random() < 0.5:
                    denial_code = random.choice(denial_categories["excluded_service"])
                elif submission_lags[i] > timely_filing_limit:
                    denial_code = random.choice(["A4", "H1"])  # Timely filing related codes
                else:
                    # Use weighted random selection based on distribution
                    denial_code = random.choices(
                        list(denial_reason_weights.keys()),
                        weights=list(denial_reason_weights.values())
                    )[0]
                
                denial_reason_codes.append(denial_code)
                denial_reason_descriptions.append(denial_reasons.get(denial_code, "Unknown reason"))
            else:
                if CONFIG["claim_status_format"] == "numeric":
                    claim_statuses.append(0)  # 0 = APPROVED
                elif CONFIG["claim_status_format"] == "boolean":
                    claim_statuses.append(False)  # False = APPROVED
                else:  # string format
                    claim_statuses.append("APPROVED")
                
                denial_reason_codes.append(None)
                denial_reason_descriptions.append(None)
        
        # Calculate payment amounts
        payment_amounts = []
        patient_responsibilities = []
        
        for i in range(actual_chunk_size):
            charge = charge_amounts[i]
            is_denied = (claim_statuses[i] == 1 if CONFIG["claim_status_format"] == "numeric" else 
                        claim_statuses[i] == True if CONFIG["claim_status_format"] == "boolean" else 
                        claim_statuses[i] == "DENIED")
            
            if is_denied:
                payment_amounts.append(None)
                patient_responsibilities.append(charge)  # Patient responsible for full amount if denied
            else:
                # Calculate payment based on random reimbursement rate
                reimbursement_rate = random.uniform(0.5, 0.95)
                payment = round(charge * reimbursement_rate, 2)
                payment_amounts.append(payment)
                patient_responsibilities.append(round(charge - payment, 2))
        
        # Generate processing dates
        processing_dates = []
        for i in range(actual_chunk_size):
            processing_days = random.randint(3, 30)
            proc_date = submission_dates[i] + timedelta(days=processing_days)
            if proc_date > end_date:
                proc_date = end_date
            processing_dates.append(proc_date)
        
        # Calculate days to payment
        days_to_payment = []
        for i in range(actual_chunk_size):
            is_denied = (claim_statuses[i] == 1 if CONFIG["claim_status_format"] == "numeric" else 
                        claim_statuses[i] == True if CONFIG["claim_status_format"] == "boolean" else 
                        claim_statuses[i] == "DENIED")
            
            if is_denied:
                days_to_payment.append(None)
            else:
                days_to_payment.append((processing_dates[i] - submission_dates[i]).days)
        
        # Claim frequency types
        claim_frequencies = [random.choices([1, 2, 3], weights=[0.85, 0.10, 0.05])[0] for _ in range(actual_chunk_size)]
        
        # Create diagnostic and procedure description columns
        primary_diag_descriptions = [diagnosis_codes[code] for code in primary_diagnoses]
        secondary_diag_descriptions = [diagnosis_codes.get(code) if code else None for code in secondary_diagnoses]
        procedure_descriptions = [procedure_codes[code] for code in selected_procedures]
        place_descriptions = [place_of_service_codes[code] for code in places_of_service]
        revenue_descriptions = [revenue_codes.get(code) if code else None for code in revenue_codes_list]
        
        # Combine all data into a dataframe
        claims_data = {
            'claim_id': claim_ids,
            'patient_id': selected_patient_ids,
            'provider_id': selected_provider_ids,
            'payer_id': selected_payer_ids,
            'service_date': service_dates,
            'submission_date': submission_dates,
            'processing_date': processing_dates,
            'primary_diagnosis_code': primary_diagnoses,
            'primary_diagnosis_description': primary_diag_descriptions,
            'secondary_diagnosis_code': secondary_diagnoses,
            'secondary_diagnosis_description': secondary_diag_descriptions,
            'procedure_code': selected_procedures,
            'procedure_description': procedure_descriptions,
            'revenue_code': revenue_codes_list,
            'revenue_description': revenue_descriptions,
            'place_of_service_code': places_of_service,
            'place_of_service_description': place_descriptions,
            'prior_auth_required': prior_auth_required,
            'prior_auth_obtained': prior_auth_obtained,
            'charge_amount': charge_amounts,
            'claim_status': claim_statuses,
            'denial_reason_code': denial_reason_codes,
            'denial_reason_description': denial_reason_descriptions,
            'payment_amount': payment_amounts,
            'patient_responsibility': patient_responsibilities,
            'claim_frequency': claim_frequencies,
            'days_to_payment': days_to_payment
        }
        
        all_claim_chunks.append(pl.DataFrame(claims_data))
        
        if chunk_start % (chunk_size * 5) == 0:
            print(f"  Generated {chunk_start + actual_chunk_size} claims...")
    
    return pl.concat(all_claim_chunks)

# Function to save dataframe in specified formats
def save_dataframe(df, filename, formats):
    base_path = os.path.join(CONFIG["output_dir"], filename)
    
    for fmt in formats:
        if fmt.lower() == 'csv':
            df.write_csv(f"{base_path}.csv")
            print(f"Saved {filename}.csv")
        elif fmt.lower() == 'parquet':
            df.write_parquet(f"{base_path}.parquet")
            print(f"Saved {filename}.parquet")

# Calculate overall denial rate from generated data
def calculate_denial_rate(claims_df):
    if CONFIG["claim_status_format"] == "numeric":
        denied_count = claims_df.filter(pl.col("claim_status") == 1).height
    elif CONFIG["claim_status_format"] == "boolean":
        denied_count = claims_df.filter(pl.col("claim_status") == True).height
    else:  # string format
        denied_count = claims_df.filter(pl.col("claim_status") == "DENIED").height
        
    total_count = claims_df.height
    return denied_count / total_count if total_count > 0 else 0

# Function to get denial reasons distribution
def get_denial_reason_distribution(claims_df):
    # Filter for denied claims
    if CONFIG["claim_status_format"] == "numeric":
        denied_claims = claims_df.filter(pl.col("claim_status") == 1)
    elif CONFIG["claim_status_format"] == "boolean":
        denied_claims = claims_df.filter(pl.col("claim_status") == True)
    else:  # string format
        denied_claims = claims_df.filter(pl.col("claim_status") == "DENIED")
    
    # Count by category
    category_counts = {}
    for category in denial_categories.keys():
        category_counts[category] = 0
    
    # Count denials by category
    reason_codes = denied_claims["denial_reason_code"].to_list()
    for code in reason_codes:
        if code:
            category = category_for_denial_code.get(code, "other")
            category_counts[category] = category_counts.get(category, 0) + 1
    
    # Calculate percentages
    total_denials = len(reason_codes)
    category_percentages = {}
    for category, count in category_counts.items():
        category_percentages[category] = round(count / total_denials, 4) if total_denials > 0 else 0
    
    return category_percentages

# Main execution
def main():
    print(f"Starting data generation with:")
    print(f"  - Date range: {CONFIG['start_date'].date()} to {CONFIG['end_date'].date()}")
    print(f"  - Total claims: {CONFIG['total_claims']:,}")
    print(f"  - Patients: {CONFIG['num_patients']:,}")
    print(f"  - Providers: {CONFIG['num_providers']:,}")
    print(f"  - Payers: {CONFIG['num_payers']}")
    print(f"  - Target denial rate: {CONFIG['overall_denial_rate']:.2%}")
    
    # Generate all datasets
    patients_df = generate_patients(CONFIG["num_patients"])
    
    providers_df = generate_providers(CONFIG["num_providers"])
    
    payers_df = generate_payers(CONFIG["num_payers"])
    
    claims_df = generate_claims(
        CONFIG["total_claims"], 
        patients_df, 
        providers_df, 
        payers_df, 
        CONFIG["start_date"], 
        CONFIG["end_date"]
    )
    
    # Save to files in specified formats
    save_dataframe(patients_df, "synthetic_patients", CONFIG["output_formats"])
    save_dataframe(providers_df, "synthetic_providers", CONFIG["output_formats"])
    save_dataframe(payers_df, "synthetic_payers", CONFIG["output_formats"])
    save_dataframe(claims_df, "synthetic_claims", CONFIG["output_formats"])
    
    # Calculate and display statistics
    actual_denial_rate = calculate_denial_rate(claims_df)
    
    print("\nData Generation Complete!")
    print("\nData Statistics:")
    print(f"  - Total patients: {patients_df.height:,}")
    print(f"  - Total providers: {providers_df.height:,}")
    print(f"  - Total payers: {payers_df.height:,}")
    print(f"  - Total claims: {claims_df.height:,}")
    print(f"  - Overall denial rate: {actual_denial_rate:.2%}")
    
    # Show denial reason distribution
    print("\nDenial Reason Distribution:")
    distribution = get_denial_reason_distribution(claims_df)
    for category, percentage in distribution.items():
        print(f"  - {category}: {percentage:.2%}")
    
    print("\nFiles saved to:", CONFIG["output_dir"])

if __name__ == "__main__":
    main()