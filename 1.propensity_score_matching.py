"""
============================================================
1. Propensity Score Matching — 9 scenarios
============================================================
Runs the full PSM pipeline 9 times across every combination of:

  Polypharmacy definition (3):
    pp0 (main)          : 1-4 / 5-9 / 10-19 / 20+        (4 groups)
    pp1 (sensitivity 1) : 1-14 / 15-29 / 30+             (3 groups)
    pp2 (sensitivity 2) : 1-19 / 20-39 / 40+             (3 groups)

  Age stratum (3):
    overall : all patients
    older   : age >= 65
    younger : age <  65

Total: 3 × 3 = 9 independent PSM analyses.

──────────────────── Paths ────────────────────
  INPUT  : <script_dir>/input/base_cohort_260416.csv
  OUTPUT : <script_dir>/output/psm_results/
             - pp{0,1,2}_{overall,older,younger}_patients_*.csv/png/txt (9 × 5 files)
             - _all_scenarios_overview.csv

All paths are resolved relative to THIS script's directory,
so it works no matter where you run it from.

Dependencies:
    pip install pandas numpy scikit-learn matplotlib scipy
============================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PATHS — resolved from this script's directory
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(SCRIPT_DIR, "input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", "psm_results")

COHORT_CSV = os.path.join(INPUT_DIR, "base_cohort_260416.csv")


# ============================================================
# SECTION 1: DATA DEFINITIONS
# ============================================================

# Demographic variables: age, sex, and CCI (Charlson Comorbidity Index) score.
DEMOGRAPHIC_VARS = [
    'age', 'sex', 'score_sum',
]

# Comorbidity variables (19): used as confounders in the propensity score model.
COMORBIDITY_VARS = [
    'atrial_fibrillation_YN', 'cancer_YN', 'cerebrovascular_disease_YN',
    'chronic_kidney_disease_YN', 'chronic_liver_disease_YN',
    'chronic_respiratory_disease_YN', 'coronary_artery_disease_YN',
    'depression_YN', 'diabetes_mellitus_YN', 'dyslipidemia_YN',
    'hearing_loss_YN', 'hemiplegia_YN', 'hypertension_YN', 'insomnia_YN',
    'obstructive_sleep_apnea_YN', 'parkinsons_disease_YN', 'peptic_ulcer_YN',
    'peripheral_artery_disease_YN', 'traumatic_head_injury_YN',
]

# Co-medication variables (13): prescription-related confounders.
COMEDICATION_VARS = [
    'anti_parkinson_agents_YN', 'anticoagulant_antiplatelet_YN',
    'antidepressants_YN', 'antidiabetic_drugs_YN', 'antihistamines_YN',
    'antipsychotics_YN', 'antispasmodics_YN', 'anxiolytics_YN',
    'bladder_antimuscarinics_YN', 'h2ra_YN', 'sedative_hypnotics_YN',
    'skeletal_muscle_relaxants_YN', 'statins_YN',
]

# Full covariate set (35) and the columns kept for analysis.
NUMERIC_VARS = DEMOGRAPHIC_VARS + COMORBIDITY_VARS + COMEDICATION_VARS
ANALYSIS_COLUMNS = ['person_id', 'group'] + NUMERIC_VARS

# Polypharmacy definitions: main analysis (pp0) + 2 sensitivity analyses (pp1, pp2).
PP_DEFINITIONS = {
    "pp0": {
        "groups": [(0, 1, 4), (1, 5, 9), (2, 10, 19), (3, 20, None)],
        "names": {
            0: "non-polypharmacy", 1: "minor polypharmacy",
            2: "moderate polypharmacy", 3: "major polypharmacy",
        },
        "colors": {0: "#378ADD", 1: "#1D9E75", 2: "#D85A30", 3: "#D4537E"},
    },
    "pp1": {
        "groups": [(0, 1, 14), (1, 15, 29), (2, 30, None)],
        "names": {
            0: "non-polypharmacy (1-14)",
            1: "moderate polypharmacy (15-29)",
            2: "major polypharmacy (30+)",
        },
        "colors": {0: "#378ADD", 1: "#D85A30", 2: "#D4537E"},
    },
    "pp2": {
        "groups": [(0, 1, 19), (1, 20, 39), (2, 40, None)],
        "names": {
            0: "non-polypharmacy (1-19)",
            1: "moderate polypharmacy (20-39)",
            2: "major polypharmacy (40+)",
        },
        "colors": {0: "#378ADD", 1: "#D85A30", 2: "#D4537E"},
    },
}

# Age-stratum filters: overall / older (>=65) / younger (<65).
AGE_STRATA = {
    "overall": lambda df: df,
    "older":   lambda df: df[df["age"] >= 65],
    "younger": lambda df: df[df["age"] <  65],
}


# ============================================================
# SECTION 2: DATA LOADING
# ============================================================

def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load the CSV, drop missing values, coerce covariates to numeric, and
    return a cleaned DataFrame."""
    # [1] Load the raw CSV.
    df = pd.read_csv(csv_path)
    n_raw = len(df)
    print(f"  [Preprocessing] Raw rows loaded                : {n_raw:>8,}")

    # [2] Drop rows missing any covariate or max_ingredient_cnt.
    needed = NUMERIC_VARS + ['max_ingredient_cnt']
    df = df.dropna(subset=[c for c in needed if c in df.columns]).copy()
    n_after_dropna1 = len(df)
    print(f"  [Preprocessing] After dropping NAs (covariates) : {n_after_dropna1:>8,} "
          f"(removed {n_raw - n_after_dropna1:,})")

    # [3] Coerce to numeric (any embedded strings become NaN).
    df['sex'] = df['sex'].astype(int)
    df['max_ingredient_cnt'] = pd.to_numeric(df['max_ingredient_cnt'], errors='coerce')
    for col in NUMERIC_VARS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # [4] Drop NaNs introduced by numeric coercion (based on max_ingredient_cnt).
    df = df.dropna(subset=['max_ingredient_cnt']).copy()
    n_final = len(df)
    print(f"  [Preprocessing] After numeric coercion cleanup  : {n_final:>8,} "
          f"(removed {n_after_dropna1 - n_final:,})")
    print(f"  [Preprocessing] Total rows removed              : {n_raw - n_final:,} "
          f"({(n_raw - n_final) / n_raw * 100:.2f}%)")
    return df


def assign_polypharm_groups(df: pd.DataFrame, pp_scheme: str) -> pd.DataFrame:
    """Assign a group label based on max_ingredient_cnt according to the
    chosen pp_scheme (pp0/pp1/pp2)."""
    if pp_scheme not in PP_DEFINITIONS:
        raise ValueError(f"Unknown pp_scheme: {pp_scheme}")
    n_before = len(df)
    df = df.copy()
    df["group"] = np.nan
    for code, lo, hi in PP_DEFINITIONS[pp_scheme]["groups"]:
        mask = df["max_ingredient_cnt"] >= lo
        if hi is not None:
            mask &= df["max_ingredient_cnt"] <= hi
        df.loc[mask, "group"] = code
    # Drop patients not falling into any group (e.g. max_ingredient_cnt == 0).
    df = df.dropna(subset=["group"]).copy()
    df["group"] = df["group"].astype(int)
    n_after = len(df)
    print(f"  [Group assignment] {pp_scheme}: {n_before:,} → {n_after:,} "
          f"(removed {n_before - n_after:,} rows not matching any group range)")
    return df


def build_analysis_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the analysis columns and sort by group to produce the final
    analytic dataset."""
    return (df[ANALYSIS_COLUMNS]
            .sort_values('group', kind='stable')
            .reset_index(drop=True))


# ============================================================
# SECTION 3: PROPENSITY SCORE MODEL
# ============================================================

class PropensityScoreModel:
    """Multinomial logistic regression propensity score model
    (fit → predict → IPTW → evaluate)."""

    def __init__(self, covariates, group_col="group", random_state=42,
                 C=1.0, max_iter=1000):
        """Store the covariate list and logistic regression hyperparameters
        (C, max_iter, random_state)."""
        self.covariates = covariates
        self.group_col = group_col
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            solver="lbfgs", max_iter=max_iter, C=C, random_state=random_state,
        )
        self.data_with_scores = None
        self.classes_ = None
        self._class_to_idx = None

    def _scale(self, df, fit=False):
        """Standardize the covariates (mean 0, variance 1). fit=True fits the
        scaler; fit=False only transforms."""
        X = df[self.covariates].values
        return self.scaler.fit_transform(X) if fit else self.scaler.transform(X)

    def _check_fitted(self):
        """Raise RuntimeError if the model has not been fitted yet."""
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

    def fit(self, df):
        """Standardize the covariates and fit a multinomial logistic regression
        to learn group-membership probabilities."""
        X_scaled = self._scale(df, fit=True)
        y = df[self.group_col].values
        self.model.fit(X_scaled, y)
        self.classes_ = self.model.classes_
        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        print(f"  Model fitted on {len(df)} patients across {len(self.classes_)} groups.")
        return self

    def predict_scores(self, df):
        """Compute each patient's per-group propensity scores (ps_group_*) and
        the score of their actually assigned group (ps_assigned)."""
        self._check_fitted()
        X_scaled = self._scale(df)
        probs = self.model.predict_proba(X_scaled)
        result = df.copy()
        for i, cls in enumerate(self.classes_):
            result[f"ps_group_{cls}"] = probs[:, i]
        assigned_idx = df[self.group_col].map(self._class_to_idx).to_numpy()
        result["ps_assigned"] = probs[np.arange(len(df)), assigned_idx]
        self.data_with_scores = result
        return result

    def compute_iptw(self, stabilized=True, truncate=(0.01, 0.99),
                     truncate_by_group=False):
        """Compute inverse-probability-of-treatment weights (IPTW), with optional
        stabilization and extreme-value truncation."""
        if self.data_with_scores is None:
            raise RuntimeError("Run predict_scores() first.")
        result = self.data_with_scores.copy()
        # Stabilized weights: w = P(group) / P(group|X) → dampens extreme values.
        if stabilized:
            marginal = result[self.group_col].map(
                result[self.group_col].value_counts(normalize=True)
            )
            result["iptw"] = marginal / result["ps_assigned"]
        else:
            result["iptw"] = 1.0 / result["ps_assigned"]
        # Clip weights at the 1%/99% quantiles to limit extreme influence.
        if truncate is not None:
            lo_q, hi_q = truncate
            if truncate_by_group:
                result["iptw_truncated"] = (
                    result.groupby(self.group_col, observed=True)["iptw"]
                          .transform(lambda s: s.clip(*s.quantile([lo_q, hi_q])))
                )
            else:
                lo, hi = result["iptw"].quantile([lo_q, hi_q])
                result["iptw_truncated"] = result["iptw"].clip(lo, hi)
        self.data_with_scores = result
        return result

    def evaluate(self, df, cv=5):
        """Evaluate model performance with 5-fold cross-validation
        (one-vs-rest AUC and accuracy)."""
        self._check_fitted()
        X_scaled = self._scale(df)
        y = df[self.group_col].values
        auc_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring="roc_auc_ovr")
        acc_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring="accuracy")
        return {
            "cv_auc_mean": auc_scores.mean(), "cv_auc_std": auc_scores.std(),
            "cv_accuracy_mean": acc_scores.mean(), "cv_accuracy_std": acc_scores.std(),
            "n_patients": len(df), "n_groups": len(self.classes_),
            "groups": [str(c) for c in self.classes_],
        }


# ============================================================
# SECTION 4: COVARIATE BALANCE (SMD)
# ============================================================

def _group_moments(df, covariates, group_col, weight_col=None):
    """Compute the per-group mean and variance of each covariate (weighted
    mean/variance if a weight column is given)."""
    rows = []
    for g, sub in df.groupby(group_col, observed=True):
        if weight_col is None:
            means = sub[covariates].mean()
            varis = sub[covariates].var(ddof=1)
        else:
            w = sub[weight_col].to_numpy(dtype=float)
            w_sum = w.sum()
            x = sub[covariates].to_numpy(dtype=float)
            mean_arr = (w[:, None] * x).sum(axis=0) / w_sum
            var_arr = (w[:, None] * (x - mean_arr) ** 2).sum(axis=0) / w_sum
            means = pd.Series(mean_arr, index=covariates)
            varis = pd.Series(var_arr, index=covariates)
        for cov in covariates:
            rows.append({group_col: g, "covariate": cov,
                         "mean": means[cov], "var": varis[cov]})
    return pd.DataFrame(rows)


def compute_smd(df, covariates, group_col="group", reference_group=0,
                weight_col=None, binary_as_proportion=True):
    """Compute the standardized mean difference (SMD) of each group vs the
    reference group. Binary variables use a proportion-based pooled SD."""
    moments = _group_moments(df, covariates, group_col, weight_col)
    ref_moments = moments[moments[group_col] == reference_group].set_index("covariate")
    # For binary variables, use the p(1-p) formula for the pooled SD (recommended).
    binary_cols = ({c for c in covariates if df[c].dropna().isin([0, 1]).all()}
                   if binary_as_proportion else set())
    rows = []
    for g, sub in moments.groupby(group_col, observed=True):
        if g == reference_group:
            continue
        sub = sub.set_index("covariate")
        for cov in covariates:
            m_ref, v_ref = ref_moments.loc[cov, ["mean", "var"]]
            m_grp, v_grp = sub.loc[cov, ["mean", "var"]]
            if cov in binary_cols:
                pooled_sd = np.sqrt((m_ref * (1 - m_ref) + m_grp * (1 - m_grp)) / 2)
            else:
                pooled_sd = np.sqrt((v_ref + v_grp) / 2)
            smd = (m_grp - m_ref) / pooled_sd if pooled_sd > 0 else 0.0
            rows.append({"covariate": cov, "group": g,
                         "mean_ref": m_ref, "mean_grp": m_grp,
                         "smd": smd, "abs_smd": abs(smd)})
    return pd.DataFrame(rows)


def balance_table(df, covariates, group_col="group", reference_group=0,
                  weight_col="iptw_truncated"):
    """Compare SMDs before and after IPTW side by side to judge whether
    balance improved."""
    unw = compute_smd(df, covariates, group_col, reference_group, weight_col=None)
    wtd = compute_smd(df, covariates, group_col, reference_group, weight_col=weight_col)
    merged = (unw[["covariate", "group", "smd", "abs_smd"]]
              .rename(columns={"smd": "smd_unweighted", "abs_smd": "abs_smd_unweighted"})
              .merge(wtd[["covariate", "group", "smd", "abs_smd"]]
                     .rename(columns={"smd": "smd_weighted", "abs_smd": "abs_smd_weighted"}),
                     on=["covariate", "group"]))
    merged["improved"] = merged["abs_smd_weighted"] < merged["abs_smd_unweighted"]
    return merged.sort_values(["group", "abs_smd_unweighted"], ascending=[True, False])


# ============================================================
# SECTION 5: VISUALIZATIONS
# ============================================================

def _finalize(fig, filename, dpi=150, show=True):
    """Shared helper that saves a plot to file and releases its memory."""
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"  Saved: {filename}")
    return filename


def _style_axis(ax):
    """Apply a clean style by removing the top and right spines."""
    ax.spines[["top", "right"]].set_visible(False)


def _iter_axes(axes):
    """Flatten single/multiple subplot axes into a consistent 1D array
    (adapts flexibly to the number of groups)."""
    return np.atleast_1d(axes).ravel()


def plot_ps_distributions(df, out_dir, group_col="group",
                          prefix="pp0_overall_patients",
                          group_names=None, group_colors=None, show=False):
    """Build and save per-group propensity score histograms (used to check
    common support)."""
    group_names = group_names or {}
    group_colors = group_colors or {}
    ps_cols = sorted(c for c in df.columns if c.startswith("ps_group_"))
    n_groups = len(ps_cols)
    fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 4))
    axes = _iter_axes(axes)
    groups = sorted(df[group_col].unique())
    for ax, ps_col in zip(axes, ps_cols):
        gid = int(ps_col.split("_")[-1])
        for g in groups:
            vals = df.loc[df[group_col] == g, ps_col]
            ax.hist(vals, bins=40, density=True, alpha=0.55,
                    color=group_colors.get(g, "gray"),
                    label=group_names.get(g, f"Group {g}"), edgecolor="none")
        ax.set_title(f"PS for {group_names.get(gid, f'Group {gid}')}", fontsize=11)
        ax.set_xlabel("Propensity score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        _style_axis(ax)
    fig.suptitle(f"Propensity Score Distributions — {prefix}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _finalize(fig, os.path.join(out_dir, f"{prefix}_ps_distributions.png"), show=show)


def plot_love_plot(balance_df, out_dir, prefix="pp0_overall_patients",
                   group_names=None, threshold=0.1, show=False):
    """Love plot: compare |SMD| before/after IPTW per covariate (with the 0.1
    threshold line)."""
    group_names = group_names or {}
    groups = sorted(balance_df["group"].unique())
    n_groups = len(groups)
    n_covs = balance_df["covariate"].nunique()
    fig, axes = plt.subplots(1, n_groups,
                             figsize=(5 * n_groups, max(4, n_covs * 0.45 + 1)),
                             sharex=True)
    axes = _iter_axes(axes)
    for ax, gid in zip(axes, groups):
        gdf = (balance_df[balance_df["group"] == gid]
               .sort_values("abs_smd_unweighted", ascending=True))
        y = np.arange(len(gdf))
        ax.scatter(gdf["abs_smd_unweighted"], y,
                   color="#378ADD", s=50, zorder=3, label="Before IPTW")
        ax.scatter(gdf["abs_smd_weighted"], y,
                   color="#1D9E75", marker="D", s=50, zorder=3, label="After IPTW")
        ax.axvline(threshold, color="#D85A30", linestyle="--", linewidth=1,
                   label=f"|SMD| = {threshold}")
        ax.set_yticks(y)
        ax.set_yticklabels(gdf["covariate"], fontsize=9)
        ax.set_xlabel("|Standardized Mean Difference|")
        ax.set_title(f"vs {group_names.get(gid, f'Group {gid}')}", fontsize=11)
        ax.set_xlim(left=0)
        ax.legend(fontsize=8)
        _style_axis(ax)
    fig.suptitle(f"Love Plot — {prefix}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _finalize(fig, os.path.join(out_dir, f"{prefix}_love_plot.png"), show=show)


def plot_weight_distribution(df, out_dir, group_col="group",
                             weight_col="iptw_truncated",
                             prefix="pp0_overall_patients",
                             group_names=None, group_colors=None, show=False):
    """Build a boxplot of the IPTW weight distribution (used to detect extreme
    or abnormal weights)."""
    group_names = group_names or {}
    group_colors = group_colors or {}
    groups = sorted(df[group_col].unique())
    data = [df.loc[df[group_col] == g, weight_col].to_numpy() for g in groups]
    labels = [group_names.get(g, f"Group {g}") for g in groups]
    colors = [group_colors.get(g, "gray") for g in groups]
    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.5})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel(weight_col)
    ax.set_title(f"IPTW Weight Distribution — {prefix}", fontsize=12)
    _style_axis(ax)
    fig.tight_layout()
    return _finalize(fig, os.path.join(out_dir, f"{prefix}_weight_distributions.png"), show=show)


# ============================================================
# SECTION 6: SUMMARY
# ============================================================

def _group_stats(df, group_col, value_col, agg):
    """Helper returning a per-group aggregate (mean/std/max, etc.) of a column
    as a dict."""
    return df.groupby(group_col, observed=True)[value_col].agg(agg).to_dict()


def _format_line(label, body, width=28):
    """Format a 'label : body' line for the summary report at an aligned width."""
    return f"  {label:<{width}s}: {body}"


def build_summary(df_scored, eval_results, balance_df,
                  group_col="group", weight_col="iptw_truncated",
                  ps_assigned_col="ps_assigned", smd_threshold=0.1,
                  scenario_label="", group_names=None):
    """Build a combined text summary for a single scenario
    (patient counts / AUC / SMD, etc.)."""
    group_names = group_names or {}
    lines = []
    sep = "=" * 70
    lines.append("\n" + sep)
    lines.append(f"  PROPENSITY SCORE ANALYSIS — {scenario_label}")
    lines.append(sep)
    lines.append(f"\n  Patients: {eval_results['n_patients']}")
    lines.append(f"  Groups:   "
                 f"{', '.join(group_names.get(int(g), g) for g in eval_results['groups'])}")
    lines.append(f"\n  Model: Multinomial Logistic Regression")
    if "cv_auc_mean" in eval_results:
        lines.append(f"  CV AUC (one-vs-rest): "
                     f"{eval_results['cv_auc_mean']:.3f} ± {eval_results['cv_auc_std']:.3f}")
    lines.append(f"  CV Accuracy:          "
                 f"{eval_results['cv_accuracy_mean']:.3f} ± {eval_results['cv_accuracy_std']:.3f}")
    lines.append(f"\n  ── Group sizes ──────────────────────────────")
    sizes = df_scored[group_col].value_counts().sort_index()
    for g, cnt in sizes.items():
        lines.append(_format_line(group_names.get(g, f"Group {g}"), f"{cnt:6d} patients"))
    lines.append(f"\n  ── Mean assigned propensity score ───────────")
    ps_mean = _group_stats(df_scored, group_col, ps_assigned_col, "mean")
    ps_std = _group_stats(df_scored, group_col, ps_assigned_col, "std")
    for g in sizes.index:
        lines.append(_format_line(group_names.get(g, f"Group {g}"),
                                  f"{ps_mean[g]:.3f} ± {ps_std[g]:.3f}"))
    if weight_col in df_scored.columns:
        lines.append(f"\n  ── IPTW weight summary ──────────────────────")
        w_mean = _group_stats(df_scored, group_col, weight_col, "mean")
        w_max = _group_stats(df_scored, group_col, weight_col, "max")
        for g in sizes.index:
            lines.append(_format_line(group_names.get(g, f"Group {g}"),
                                      f"mean={w_mean[g]:.2f}, max={w_max[g]:.2f}"))
    lines.append(f"\n  ── Covariate balance (max |SMD|) ────────────")
    for g in sorted(balance_df["group"].unique()):
        gdf = balance_df[balance_df["group"] == g]
        before = gdf["abs_smd_unweighted"].max()
        after = gdf["abs_smd_weighted"].max()
        flag = " ✓" if after < smd_threshold else f" ✗ (>{smd_threshold})"
        lines.append(f"  vs {group_names.get(g, f'Group {g}'):<26s}: "
                     f"before={before:.3f}  after={after:.3f}{flag}")
    n_still = (balance_df["abs_smd_weighted"] >= smd_threshold).sum()
    n_total = len(balance_df)
    lines.append(f"\n  Covariates still imbalanced after IPTW: {n_still} / {n_total}")
    lines.append("\n" + sep + "\n")
    return "\n".join(lines)


# ============================================================
# SECTION 7: SINGLE-SCENARIO PIPELINE
# ============================================================

def run_propensity_analysis(data, covariates, prefix, group_names, group_colors,
                            out_dir, group_col="group", reference_group=0,
                            stabilized=True, truncate=(0.01, 0.99),
                            truncate_by_group=False, save_plots=True,
                            save_csv=True, save_summary=True, show_plots=False):
    """End-to-end pipeline for a single scenario: fit model → IPTW →
    balance assessment → visualization/saving."""
    model = PropensityScoreModel(covariates=covariates, group_col=group_col)
    model.fit(data)
    model.predict_scores(data)
    scored = model.compute_iptw(stabilized=stabilized, truncate=truncate,
                                 truncate_by_group=truncate_by_group)
    balance = balance_table(scored, covariates=covariates, group_col=group_col,
                            reference_group=reference_group)
    evaluation = model.evaluate(data)
    summary_text = build_summary(scored, evaluation, balance,
                                 group_col=group_col, scenario_label=prefix,
                                 group_names=group_names)
    print(summary_text)

    # Save the 3 plots: PS distributions / Love plot / weight boxplot.
    if save_plots:
        plot_ps_distributions(scored, out_dir, group_col=group_col, prefix=prefix,
                              group_names=group_names, group_colors=group_colors,
                              show=show_plots)
        plot_love_plot(balance, out_dir, prefix=prefix,
                       group_names=group_names, show=show_plots)
        plot_weight_distribution(scored, out_dir, group_col=group_col, prefix=prefix,
                                 group_names=group_names, group_colors=group_colors,
                                 show=show_plots)

    # Save per-patient PS/IPTW results to CSV (person_id as the first column).
    if save_csv:
        csv_path = os.path.join(out_dir, f"{prefix}_with_propensity_scores.csv")
        cols = ["person_id"] + [c for c in scored.columns if c != "person_id"]
        scored[cols].to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    # Save the text summary file.
    if save_summary:
        txt_path = os.path.join(out_dir, f"{prefix}_summary.txt")
        with open(txt_path, "w") as f:
            f.write(summary_text)
        print(f"  Saved: {txt_path}")

    return {"prefix": prefix, "model": model, "scored": scored,
            "balance": balance, "evaluation": evaluation,
            "summary": summary_text}


# ============================================================
# SECTION 8: 9-SCENARIO RUNNER
# ============================================================

def build_scenarios():
    """Build the list of 9 scenario combinations (3 pp schemes × 3 age strata)."""
    return [{"pp_scheme": pp, "age_stratum": age,
             "prefix": f"{pp}_{age}_patients"}
            for pp in ["pp0", "pp1", "pp2"]
            for age in ["overall", "older", "younger"]]


def run_all_scenarios(cohort_csv=COHORT_CSV, out_dir=OUTPUT_DIR,
                      covariates=None, save_plots=True, save_csv=True,
                      save_summary=True, show_plots=False):
    """Master function that runs all 9 scenarios sequentially and produces the
    final overview CSV."""
    covariates = covariates or NUMERIC_VARS
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("  LOADING BASE COHORT")
    print("=" * 70)
    print(f"  Input : {cohort_csv}")
    print(f"  Output: {out_dir}")

    # Load the base cohort (shared across all 9 scenarios).
    df_base = load_and_prepare_data(cohort_csv)
    print(f"  Total patients in base cohort: {df_base['person_id'].nunique():,}")
    print(f"  Age range: {df_base['age'].min():.0f} – {df_base['age'].max():.0f}")
    print(f"  Max ingredient count range: "
          f"{df_base['max_ingredient_cnt'].min():.0f} – "
          f"{df_base['max_ingredient_cnt'].max():.0f}")

    scenarios = build_scenarios()
    all_results = {}

    # Run each scenario independently (one failure does not abort the rest).
    for i, sc in enumerate(scenarios, start=1):
        pp_scheme = sc["pp_scheme"]
        age_stratum = sc["age_stratum"]
        prefix = sc["prefix"]

        print("\n" + "#" * 70)
        print(f"#  SCENARIO {i}/{len(scenarios)}: {prefix}")
        print(f"#    pp scheme   : {pp_scheme}")
        print(f"#    age stratum : {age_stratum}")
        print("#" * 70)

        # Age filter → polypharmacy group assignment → build analytic dataset.
        n_base = len(df_base)
        df_age = AGE_STRATA[age_stratum](df_base)
        n_age = len(df_age)
        print(f"  [Age filter] {age_stratum}: {n_base:,} → {n_age:,} "
              f"(removed {n_base - n_age:,})")

        df_grouped = assign_polypharm_groups(df_age, pp_scheme)
        data = build_analysis_dataset(df_grouped)
        print(f"  [Final analytic dataset] rows: {len(data):,} "
              f"(overall retention: {len(data) / n_base * 100:.2f}% of base cohort)")

        group_counts = data['group'].value_counts().sort_index()
        pp_names = PP_DEFINITIONS[pp_scheme]["names"]
        print(f"  Group breakdown:")
        for g, cnt in group_counts.items():
            print(f"    {pp_names.get(g, f'Group {g}')}: {cnt:,}")

        # Safeguards: skip if fewer than 2 groups; warn if any group has < 10 patients.
        if len(group_counts) < 2:
            print(f"  ⚠ Skipping: only {len(group_counts)} group(s).")
            continue
        if (group_counts < 10).any():
            print(f"  ⚠ Warning: some groups have fewer than 10 patients.")

        try:
            result = run_propensity_analysis(
                data, covariates=covariates, prefix=prefix, out_dir=out_dir,
                group_names=PP_DEFINITIONS[pp_scheme]["names"],
                group_colors=PP_DEFINITIONS[pp_scheme]["colors"],
                save_plots=save_plots, save_csv=save_csv,
                save_summary=save_summary, show_plots=show_plots,
            )
            result["pp_scheme"] = pp_scheme
            result["age_stratum"] = age_stratum
            all_results[prefix] = result
        except Exception as e:
            print(f"  ✗ Scenario failed: {type(e).__name__}: {e}")

    # Build and save the overview table summarizing all 9 scenarios' performance.
    print("\n" + "=" * 70)
    print(f"  ALL {len(all_results)}/{len(scenarios)} SCENARIOS COMPLETED")
    print("=" * 70)
    overview = pd.DataFrame([{
        "scenario": prefix, "pp_scheme": r["pp_scheme"],
        "age_stratum": r["age_stratum"], "n_patients": r["evaluation"]["n_patients"],
        "n_groups": r["evaluation"]["n_groups"],
        "cv_auc": round(r["evaluation"].get("cv_auc_mean", np.nan), 3),
        "cv_accuracy": round(r["evaluation"]["cv_accuracy_mean"], 3),
        "max_smd_before": round(r["balance"]["abs_smd_unweighted"].max(), 3),
        "max_smd_after":  round(r["balance"]["abs_smd_weighted"].max(), 3),
        "n_imbalanced_after": int((r["balance"]["abs_smd_weighted"] >= 0.1).sum()),
    } for prefix, r in all_results.items()])
    print("\nOverview across scenarios:")
    print(overview.to_string(index=False))

    overview_path = os.path.join(out_dir, "_all_scenarios_overview.csv")
    overview.to_csv(overview_path, index=False)
    print(f"\nSaved: {overview_path}")

    return all_results


# ============================================================
# MAIN
# ============================================================

def main():
    """Script entry point: run all 9 scenarios with the default settings."""
    return run_all_scenarios(
        cohort_csv=COHORT_CSV,
        out_dir=OUTPUT_DIR,
        covariates=NUMERIC_VARS,
        save_plots=True,
        save_csv=True,
        save_summary=True,
        show_plots=False,
    )


if __name__ == "__main__":
    results = main()
