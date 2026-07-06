import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time
os.chdir('')# enter input file directory

# =========================================================================
# 1. DYNAMICALLY GENERATE THE CONFIGURATION DICTIONARY
# =========================================================================
covariates = [
    'C(cancer_YN)', 'C(dyslipidemia_YN)', 
    'C(anticoagulant_antiplatelet_YN)', 'C(antidiabetic_drugs_YN)', 
    'C(antihistamines_YN)', 'C(antipsychotics_YN)', 'C(antispasmodics_YN)',
    'C(h2ra_YN)', 'C(sedative_hypnotics_YN)'
]
covariates_str = " + ".join(covariates)

pps = ['pp0', 'pp1', 'pp2']
endpoints = {
    'aki': {'file': 'aki_df_260415.csv', 'tau': 4.0},
    'mace': {'file': 'mace_df_260415.csv', 'tau': 6.0},
    'itching': {'file': 'itching_df_260415.csv', 'tau': 6.0},
    'urticaria': {'file': 'urticaria_df_260415.csv', 'tau': 6.0}
}
cohorts = ['older', 'younger', 'overall']
model_types = ['covariatesadjusted', 'crude']

ades_config = {}

for pp in pps:
    for ep_name, ep_meta in endpoints.items():
        for cohort in cohorts:
            for m_type in model_types:
                config_key = f"{pp}_{ep_name}_{cohort}_iptwapplied_{m_type}"
                
                if m_type == 'crude':
                    if cohort == 'overall':
                        formula = "rmst_pseudo_jittered ~ C(age_65_YN) * C(group)"
                    else:
                        formula = "rmst_pseudo_jittered ~ C(group)"
                else:
                    if cohort == 'overall':
                        formula = f"rmst_pseudo_jittered ~ C(age_65_YN) * C(group) + {covariates_str}"
                    else:
                        formula = f"rmst_pseudo_jittered ~ C(group) + {covariates_str}"
                
                ades_config[config_key] = {
                    'event_col': 'adverse_event_YN',
                    'tau': ep_meta['tau'],
                    'rmst_formula': formula,
                    'ade_file': ep_meta['file'],
                    'iptw_file': f"{pp}_{cohort}_patients_with_propensity_scores.csv",
                    'result_file_name': config_key
                }

# =========================================================================
# 2. PROFILE-BINNED PSEUDO-VALUE GENERATOR (Optimized for 350k+ Patients)
# =========================================================================
def compute_pseudo_obs_large_sample(df, time_col, event_col, tau):
    clean_df = df.reset_index(drop=True)
    n = len(clean_df)
    durations = clean_df[time_col].to_numpy()
    events = clean_df[event_col].to_numpy()
    
    # Fit baseline full model once
    kmf_full = KaplanMeierFitter()
    kmf_full.fit(durations, events)
    theta_full = restricted_mean_survival_time(kmf_full, t=tau)
    
    # Compress 356,000 patients into unique survival day profiles
    unique_profiles = clean_df.groupby([time_col, event_col]).size().reset_index(name='count')
    print(f"    (Big Data Optimization: Compressed {n:,} rows into {len(unique_profiles):,} unique KM loops)")
    
    theta_minus_i_dict = {}
    
    # Loop over profiles instead of individual records
    for _, row in unique_profiles.iterrows():
        t_val = row[time_col]
        e_val = row[event_col]
        
        # Drop exactly one representative matching this survival profile
        idx_to_drop = np.where((durations == t_val) & (events == e_val))[0][0]
        dur_minus_i = np.delete(durations, idx_to_drop)
        eve_minus_i = np.delete(events, idx_to_drop)
        
        kmf = KaplanMeierFitter()
        kmf.fit(dur_minus_i, eve_minus_i)
        theta_minus_i_dict[(t_val, e_val)] = restricted_mean_survival_time(kmf, t=tau)
    
    # Re-map profile vector values to match original dataframe length
    theta_minus_i_list = np.array([theta_minus_i_dict[(t, e)] for t, e in zip(durations, events)])
    
    pseudo_values = (n * theta_full) - ((n - 1) * theta_minus_i_list)
    return pd.Series(pseudo_values, index=df.index)

# =========================================================================
# 3. AUTOMATION LOOP WITH CACHING MATRIX
# =========================================================================
all_results = []
file_cache = {} 

print(f"Starting Highly Optimized Automated RMST Pipeline ({len(ades_config)} models)...")

for ade_name, config in ades_config.items():
    event_col = config['event_col']
    tau = config['tau']
    formula = config['rmst_formula']
    
    print(f"\n>>> Processing: {ade_name} (Tau = {tau} years/days) <<<")
    
    # Avoid reading large CSVs from disk repeatedly
    if config['ade_file'] not in file_cache:
        file_cache[config['ade_file']] = pd.read_csv(config['ade_file'], header=0, sep=',')
    if config['iptw_file'] not in file_cache:
        file_cache[config['iptw_file']] = pd.read_csv(config['iptw_file'], header=0, sep=',')
        
    df = file_cache[config['ade_file']]
    iptw = file_cache[config['iptw_file']]
    
    # Slicing columns early to preserve memory
    merge_cols = ['person_id', 'age_65_YN', 'adverse_event_YN', 'search_start', 'end_date', 
                  'max_ingredient_cnt', 'pp_group1', 'pp_group2']
    subset_df = df[df.columns.intersection(merge_cols)]
    
    iptw_df = pd.merge(subset_df, iptw, how='inner', on='person_id')
    
    # Fast date calculation
    iptw_df['time'] = (pd.to_datetime(iptw_df['end_date']) - pd.to_datetime(iptw_df['search_start'])).dt.days
    iptw_df.loc[iptw_df['time'] < 0, 'time'] = tau
    
    # Step C: Compute Fast Pseudo-values using the unique profile trick
    iptw_df['rmst_pseudo'] = compute_pseudo_obs_large_sample(iptw_df, 'time', event_col, tau)
    
    # Step D: Apply noise to the target regression variable column
    np.random.seed(42)
    iptw_df['rmst_pseudo_jittered'] = iptw_df['rmst_pseudo'] + np.random.normal(loc=0.0, scale=1e-5, size=len(iptw_df))
    
    # Step E: Fit GLM
    print("  - Fitting Doubly Robust GLM...")
    try:
        model = smf.glm(formula=formula, 
                        data=iptw_df, 
                        var_weights=iptw_df['iptw'], 
                        family=sm.families.Gaussian())
        results = model.fit(cov_type='HC0')
        conf_int = results.conf_int()
        
        for term in results.params.index:
            all_results.append({
                'Endpoint': ade_name,
                'Time Horizon (Tau)': tau,
                'Model Term': term,
                'Coefficient (Days)': results.params[term],
                'Lower 95% CI': conf_int.loc[term, 0],
                'Upper 95% CI': conf_int.loc[term, 1],
                'P-Value': results.pvalues[term]
            })
    except Exception as e:
        print(f"  - ⚠️ Modeling failed for {ade_name}: {e}")


# =========================================================================
# 4. EXPORT RESULTS WITH STRICT PP0 MULTIPLE TESTING CORRECTION (FDR)
# =========================================================================
if all_results:
    final_results_df = pd.DataFrame(all_results)
    
    # 1. Parse the 'Endpoint' string back into its metadata components for filtering
    final_results_df['Polypharmacy_System'] = final_results_df['Endpoint'].apply(lambda x: x.split('_')[0])
    final_results_df['Adverse_Event'] = final_results_df['Endpoint'].apply(lambda x: x.split('_')[1])
    final_results_df['Cohort'] = final_results_df['Endpoint'].apply(lambda x: x.split('_')[2])
    final_results_df['Adjustment_Type'] = final_results_df['Endpoint'].apply(lambda x: x.split('_')[-1])

    # 2. Define the Target Slice: ONLY pp0, ONLY adjusted models, and ONLY main effects
    # This isolates exactly 12 models (3 cohorts x 4 AEs)
# Pool 1: Main Effects Only (n = 36 rows)
    primary_mask = (
        (final_results_df['Polypharmacy_System'] == 'pp0') &
        (final_results_df['Adjustment_Type'] == 'covariatesadjusted') & 
        (final_results_df['Model Term'].str.contains('C\(group\)')) &
        (~final_results_df['Model Term'].str.contains(':'))
    )
    
    # Pool 2: Interaction Effects Only (n = 36 rows)
    interaction_mask = (
        (final_results_df['Polypharmacy_System'] == 'pp0') &
        (final_results_df['Adjustment_Type'] == 'covariatesadjusted') & 
        (final_results_df['Model Term'].str.contains(':'))
    )
    # 3. Apply Benjamini-Hochberg FDR correction exclusively to the pp0 slice
    final_results_df['FDR_Adjusted_Q_Value'] = np.nan
    final_results_df['Significant_After_FDR'] = False
    
    import statsmodels.stats.multitest as smm
  # Apply FDR to Main Effects
    if primary_mask.any():
        raw_p_main = final_results_df.loc[primary_mask, 'P-Value'].to_numpy()
        rej_main, q_main, _, _ = smm.multipletests(raw_p_main, alpha=0.05, method='fdr_bh')
        final_results_df.loc[primary_mask, 'FDR_Adjusted_Q_Value'] = q_main
        final_results_df.loc[primary_mask, 'Significant_After_FDR'] = rej_main

    # Apply FDR to Interaction Effects separately
    if interaction_mask.any():
        raw_p_inter = final_results_df.loc[interaction_mask, 'P-Value'].to_numpy()
        rej_inter, q_inter, _, _ = smm.multipletests(raw_p_inter, alpha=0.05, method='fdr_bh')
        final_results_df.loc[interaction_mask, 'FDR_Adjusted_Q_Value'] = q_inter
        final_results_df.loc[interaction_mask, 'Significant_After_FDR'] = rej_inter
        
    # 4. Reorder columns for optimal readability
    col_order = [
        'Endpoint', 'Polypharmacy_System', 'Adverse_Event', 'Cohort', 'Adjustment_Type',
        'Time Horizon (Tau)', 'Model Term', 'Coefficient (Days)', 'Lower 95% CI', 'Upper 95% CI', 
        'P-Value', 'FDR_Adjusted_Q_Value', 'Significant_After_FDR'
    ]
    final_results_df = final_results_df[col_order]

    # 5. Export to disk
    output_path_csv = "RMST_Multiple_Jittering_Endpoints_Results.csv"
    output_path_xlsx = "RMST_Multiple_Jittering_Endpoints_Results.xlsx"
    
    final_results_df.to_csv(output_path_csv, index=False)
    final_results_df.to_excel(output_path_xlsx, index=False)
    
    print(f"\n✅ Done! Combined matrix compiled. FDR correction applied to pp0; pp1 and pp2 left as sensitivity checks.")
