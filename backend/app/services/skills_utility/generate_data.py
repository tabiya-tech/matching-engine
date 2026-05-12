import json

# Taxonomy IDs mapped from your skills.csv sample
# I have linked these logically to your job titles (e.g., Python Developer -> Haskell ID)
skill_ids = {
    "tech": "68934ca60f3f20b5b2ed9e4f",      # Haskell (Tech Proxy)
    "const": "68934ca60f3f20b5b2ed9e49",     # Supervise procedures
    "med": "68934ca60f3f20b5b2ed9e4a",       # Anti-oppressive practices
    "sales": "68934ca60f3f20b5b2ed9e52",     # Diplomatic principles
    "design": "68934ca60f3f20b5b2ed9e57",    # Work with soloists
    "logistics": "68934ca60f3f20b5b2ed9e4b", # Railway vehicle regulations
    "edu": "68934ca60f3f20b5b2ed9e5d",       # Teach housekeeping
    "fin": "68934ca60f3f20b5b2ed9e60",       # Risk management
    "hosp": "68934ca60f3f20b5b2ed9e51",      # Food waste reduction
    "admin": "68934ca60f3f20b5b2ed9e4c",     # Identify available services
    "sec": "68934ca60f3f20b5b2ed9e53",       # Lead investigations
    "agri": "68934ca60f3f20b5b2ed9e62",      # Maintain aquaculture
    "manuf": "68934ca60f3f20b5b2ed9e5c",     # Handle equipment
    "beauty": "68934ca60f3f20b5b2ed9e5d"     # Housekeeping/Service proxy
}

# Skill Groups from your hierarchy snippet
group_ids = ["68934ca40f3f20b5b2ed9bc7", "68934ca40f3f20b5b2ed9bc8", 
             "68934ca40f3f20b5b2ed9bc9", "68934ca40f3f20b5b2ed9bca"]

# --- 1. GENERATE DEMAND (20 Original Jobs + Skill Injection) ---
raw_demand = [
    {"uuid": "job_001", "originUuid": "tech_01", "opportunity_title": "Senior Python Developer", "location": "Remote", "contract_type": "full_time", "attributes": {"earnings_per_month": "earn_70k", "task_content": "task_creative", "physical_demand": "phys_light", "work_flexibility": "flex_high", "social_interaction": "soc_alone", "career_growth": "growth_high", "social_meaning": "mean_low", "expected_demand": "Very Low Expected Demand"}, "essential": [skill_ids["tech"]], "group": group_ids[0]},
    {"uuid": "job_002", "originUuid": "const_01", "opportunity_title": "Site Foreman", "location": "Nairobi", "contract_type": "contract", "attributes": {"earnings_per_month": "earn_50k", "task_content": "task_mix", "physical_demand": "phys_heavy", "work_flexibility": "flex_fixed", "social_interaction": "soc_people", "career_growth": "growth_med", "social_meaning": "mean_low", "expected_demand": "Very Low Expected Demand"}, "essential": [skill_ids["const"]], "group": group_ids[1]},
    {"uuid": "job_003", "originUuid": "med_01", "opportunity_title": "Community Health Nurse", "location": "Kisumu", "contract_type": "full_time", "attributes": {"earnings_per_month": "earn_30k", "task_content": "task_mix", "physical_demand": "phys_light", "work_flexibility": "flex_some", "social_interaction": "soc_people", "career_growth": "growth_high", "social_meaning": "mean_high", "expected_demand": "Very Low Expected Demand"}, "essential": [skill_ids["med"]], "group": group_ids[2]},
    {"uuid": "job_004", "originUuid": "sales_01", "opportunity_title": "Regional Sales Manager", "location": "Mombasa", "contract_type": "full_time", "attributes": {"earnings_per_month": "earn_70k", "task_content": "task_mix", "physical_demand": "phys_light", "work_flexibility": "flex_some", "social_interaction": "soc_people", "career_growth": "growth_high", "social_meaning": "mean_low", "expected_demand": "High Expected Demand"}, "essential": [skill_ids["sales"]], "group": group_ids[3]},
    {"uuid": "job_005", "originUuid": "design_01", "opportunity_title": "Freelance Graphic Designer", "location": "Remote", "contract_type": "freelance", "attributes": {"earnings_per_month": "earn_30k", "task_content": "task_creative", "physical_demand": "phys_light", "work_flexibility": "flex_high", "social_interaction": "soc_alone", "career_growth": "growth_med", "social_meaning": "mean_high", "expected_demand": "Very Low Expected Demand"}, "essential": [skill_ids["design"]], "group": group_ids[0]},
    {"uuid": "job_006", "originUuid": "log_01", "opportunity_title": "Delivery Driver", "location": "Nakuru", "contract_type": "contract", "attributes": {"earnings_per_month": "earn_30k", "task_content": "task_routine", "physical_demand": "phys_heavy", "work_flexibility": "flex_some", "social_interaction": "soc_alone", "career_growth": "growth_low", "social_meaning": "mean_low", "expected_demand": "Moderate Expected Demand"}, "essential": [skill_ids["logistics"]], "group": group_ids[1]},
    {"uuid": "job_007", "originUuid": "edu_01", "opportunity_title": "Primary School Teacher", "location": "Eldoret", "contract_type": "full_time", "attributes": {"earnings_per_month": "earn_30k", "task_content": "task_routine", "physical_demand": "phys_light", "work_flexibility": "flex_fixed", "social_interaction": "soc_people", "career_growth": "growth_med", "social_meaning": "mean_high", "expected_demand": "High Expected Demand"}, "essential": [skill_ids["edu"]], "group": group_ids[2]},
    {"uuid": "job_008", "originUuid": "fin_01", "opportunity_title": "Junior Accountant", "location": "Nairobi", "contract_type": "internship", "attributes": {"earnings_per_month": "earn_30k", "task_content": "task_routine", "physical_demand": "phys_light", "work_flexibility": "flex_fixed", "social_interaction": "soc_alone", "career_growth": "growth_high", "social_meaning": "mean_low", "expected_demand": "Moderate Expected Demand"}, "essential": [skill_ids["fin"]], "group": group_ids[3]},
    {"uuid": "job_009", "originUuid": "tech_02", "opportunity_title": "Electrician Assistant", "location": "Thika", "contract_type": "contract", "attributes": {"earnings_per_month": "earn_30k", "task_content": "task_mix", "physical_demand": "phys_heavy", "work_flexibility": "flex_some", "social_interaction": "soc_people", "career_growth": "growth_med", "social_meaning": "mean_low", "expected_demand": "Very Low Expected Demand"}, "essential": [skill_ids["tech"]], "group": group_ids[0]},
    {"uuid": "job_010", "originUuid": "hosp_01", "opportunity_title": "Line Cook", "location": "Nairobi", "contract_type": "full_time", "attributes": {"earnings_per_month": "earn_30k", "task_content": "task_creative", "physical_demand": "phys_heavy", "work_flexibility": "flex_fixed", "social_interaction": "soc_people", "career_growth": "growth_med", "social_meaning": "mean_low", "expected_demand": "Very Low Expected Demand"}, "essential": [skill_ids["hosp"]], "group": group_ids[1]},
    {"uuid": "job_011", "originUuid": "admin_01", "opportunity_title": "Data Entry Clerk", "location": "Remote", "contract_type": "part_time", "attributes": {"earnings_per_month": "earn_15k", "task_content": "task_routine", "physical_demand": "phys_light", "work_flexibility": "flex_high", "social_interaction": "soc_alone", "career_growth": "growth_low", "social_meaning": "mean_low", "expected_demand": "Low Expected Demand"}, "essential": [skill_ids["admin"]], "group": group_ids[2]},
    {"uuid": "job_012", "originUuid": "sec_01", "opportunity_title": "Night Guard", "location": "Nairobi", "contract_type": "full_time", "attributes": {"earnings_per_month": "earn_15k", "task_content": "task_routine", "physical_demand": "phys_light", "work_flexibility": "flex_fixed", "social_interaction": "soc_alone", "career_growth": "growth_low", "social_meaning": "mean_high", "expected_demand": "Low Expected Demand"}, "essential": [skill_ids["sec"]], "group": group_ids[3]},
    {"uuid": "job_013", "originUuid": "agri_01", "opportunity_title": "Farm Hand", "location": "Naivasha", "contract_type": "seasonal", "attributes": {"earnings_per_month": "earn_15k", "task_content": "task_routine", "physical_demand": "phys_heavy", "work_flexibility": "flex_some", "social_interaction": "soc_alone", "career_growth": "growth_low", "social_meaning": "mean_high", "expected_demand": "Moderate Expected Demand"}, "essential": [skill_ids["agri"]], "group": group_ids[0]},
    {"uuid": "job_014", "originUuid": "const_02", "opportunity_title": "Carpenter", "location": "Kisumu", "contract_type": "contract", "attributes": {"earnings_per_month": "earn_30k", "task_content": "task_creative", "physical_demand": "phys_heavy", "work_flexibility": "flex_some", "social_interaction": "soc_people", "career_growth": "growth_med", "social_meaning": "mean_low", "expected_demand": "Very Low Expected Demand"}, "essential": [skill_ids["const"]], "group": group_ids[1]},
    {"uuid": "job_015", "originUuid": "serv_01", "opportunity_title": "Call Center Agent", "location": "Nairobi", "contract_type": "full_time", "attributes": {"earnings_per_month": "earn_30k", "task_content": "task_routine", "physical_demand": "phys_light", "work_flexibility": "flex_fixed", "social_interaction": "soc_people", "career_growth": "growth_med", "social_meaning": "mean_low", "expected_demand": "Moderate Expected Demand"}, "essential": [skill_ids["admin"]], "group": group_ids[2]},
    {"uuid": "job_016", "originUuid": "tech_03", "opportunity_title": "Frontend Developer", "location": "Remote", "contract_type": "contract", "attributes": {"earnings_per_month": "earn_70k", "task_content": "task_creative", "physical_demand": "phys_light", "work_flexibility": "flex_high", "social_interaction": "soc_alone", "career_growth": "growth_high", "social_meaning": "mean_low", "expected_demand": "Very Low Expected Demand"}, "essential": [skill_ids["tech"]], "group": group_ids[3]},
    {"uuid": "job_017", "originUuid": "const_03", "opportunity_title": "Painter", "location": "Mombasa", "contract_type": "contract", "attributes": {"earnings_per_month": "earn_30k", "task_content": "task_mix", "physical_demand": "phys_light", "work_flexibility": "flex_some", "social_interaction": "soc_alone", "career_growth": "growth_low", "social_meaning": "mean_low", "expected_demand": "Moderate Expected Demand"}, "essential": [skill_ids["const"]], "group": group_ids[0]},
    {"uuid": "job_018", "originUuid": "man_01", "opportunity_title": "Factory Machine Operator", "location": "Industrial Area", "contract_type": "full_time", "attributes": {"earnings_per_month": "earn_30k", "task_content": "task_routine", "physical_demand": "phys_heavy", "work_flexibility": "flex_fixed", "social_interaction": "soc_alone", "career_growth": "growth_low", "social_meaning": "mean_low", "expected_demand": "Moderate Expected Demand"}, "essential": [skill_ids["manuf"]], "group": group_ids[1]},
    {"uuid": "job_019", "originUuid": "beauty_01", "opportunity_title": "Hair Stylist", "location": "Westlands", "contract_type": "commission", "attributes": {"earnings_per_month": "earn_30k", "task_content": "task_creative", "physical_demand": "phys_light", "work_flexibility": "flex_some", "social_interaction": "soc_people", "career_growth": "growth_med", "social_meaning": "mean_high", "expected_demand": "Low Expected Demand"}, "essential": [skill_ids["beauty"]], "group": group_ids[2]},
    {"uuid": "job_020", "originUuid": "const_04", "opportunity_title": "Welder", "location": "Thika", "contract_type": "contract", "attributes": {"earnings_per_month": "earn_30k", "task_content": "task_mix", "physical_demand": "phys_heavy", "work_flexibility": "flex_fixed", "social_interaction": "soc_alone", "career_growth": "growth_med", "social_meaning": "mean_low", "expected_demand": "Low Expected Demand"}, "essential": [skill_ids["const"]], "group": group_ids[3]}
]

# --- 2. FORMAT AND SAVE DEMAND.JSONL ---
with open('demand.jsonl', 'w') as f:
    for item in raw_demand:
        # We inject the new scoring fields while keeping original location and attributes
        record = {
            "uuid": item["uuid"],
            "originUuid": item["originUuid"],
            "opportunity_title": item["opportunity_title"],
            "location": item["location"],
            "city": item["location"], # Helper for U_loc
            "province": item["location"], # Helper for U_loc
            "contract_type": item["contract_type"],
            "essential_skills": [{"id": str(u), "label": ""} for u in item["essential"]],
            "optional_skills": [],
            "skill_groups_origin_uuids": [item["group"]],
            "attributes": item["attributes"]
        }
        f.write(json.dumps(record) + '\n')

print("Success: demand.jsonl generated with all original attributes + matching UUIDs.")