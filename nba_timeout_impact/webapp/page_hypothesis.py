"""Hypothesis Playground — structured hypothesis testing over NBA data."""

import json

import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st

from nba_timeout_impact.webapp.helpers import MODEL_OPUS, call_llm, execute_sql, get_client, load_data, parse_response

# ── Metrics ──────────────────────────────────────────────────────────────────

METRICS = {
    "Possession points": {"col": "possession_points", "base": "possession", "distribution": "normal"},
    "Shot result (made/missed)": {"col": "shot_result_binary", "base": "shot", "distribution": "bernoulli"},
    "Possession outcome": {"col": "possession_outcome", "base": "possession", "distribution": "categorical"},
    "Points scored per event": {"col": "points_scored", "base": "event", "distribution": "normal"},
    "Shot distance": {"col": "shotDistance", "base": "shot", "distribution": "normal"},
    "Custom SQL": {"col": None, "base": "custom", "distribution": None},
}

TEAMS = [
    "ATL",
    "BKN",
    "BOS",
    "CHA",
    "CHI",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GSW",
    "HOU",
    "IND",
    "LAC",
    "LAL",
    "MEM",
    "MIA",
    "MIL",
    "MIN",
    "NOP",
    "NYK",
    "OKC",
    "ORL",
    "PHI",
    "PHX",
    "POR",
    "SAC",
    "SAS",
    "TOR",
    "UTA",
    "WAS",
]

PREV_EVENTS = {
    "Any": None,
    "After timeout": "timeout",
    "After offensive rebound": "rebound",
    "After steal": "steal",
    "After turnover": "turnover",
    "After made shot": "2pt",
}

# ── Distribution & Test Info ─────────────────────────────────────────────────

DISTRIBUTIONS = {
    "bernoulli": "Bernoulli — binary outcomes (made/missed, 0/1)",
    "categorical": "Categorical — discrete outcomes with 3+ categories",
    "poisson": "Poisson — non-negative count data",
    "normal": "Normal — continuous or approximately continuous",
    "nonparametric": "Nonparametric — no distribution assumption",
}

TESTS = {
    "neyman_pearson": {
        "name": "Neyman-Pearson Test",
        "distributions": ["bernoulli", "normal", "poisson"],
        "params": ["alpha", "theta_0", "theta_1"],
        "description": "Optimal test for simple hypotheses with known H0 and H1 parameters. Maximizes detection probability for a given false alarm rate α.",
    },
    "glrt": {
        "name": "GLRT (Generalized Likelihood Ratio)",
        "distributions": ["bernoulli", "categorical", "poisson", "normal"],
        "params": ["alpha"],
        "description": "For composite hypotheses where H1 parameters are unknown. Uses MLE under both hypotheses, compares via likelihood ratio.",
    },
    "joint_detection_estimation": {
        "name": "Joint Detection & Estimation",
        "distributions": ["bernoulli", "categorical", "normal"],
        "params": ["alpha"],
        "description": "Simultaneously detect if an effect exists AND estimate its magnitude. Bridges detection theory and estimation theory.",
    },
    "sprt": {
        "name": "SPRT (Sequential)",
        "distributions": ["bernoulli", "normal"],
        "params": ["alpha", "beta", "theta_0", "theta_1"],
        "description": "Sequential test — processes observations one at a time. The 'random walk between fences.' Optimal for earliest possible detection.",
    },
    "proportions_z": {
        "name": "Two-Proportion Z-Test",
        "distributions": ["bernoulli"],
        "params": ["alpha"],
        "description": "Compare two proportions using the normal approximation to the binomial.",
    },
    "ttest": {
        "name": "Two-Sample t-Test (Welch's)",
        "distributions": ["normal"],
        "params": ["alpha"],
        "description": "Compare means of two independent samples. Does not assume equal variances.",
    },
    "chi_squared": {
        "name": "Chi-Squared Test",
        "distributions": ["categorical"],
        "params": ["alpha"],
        "description": "Test whether two categorical distributions differ using a contingency table.",
    },
    "mann_whitney": {
        "name": "Mann-Whitney U",
        "distributions": ["nonparametric"],
        "params": ["alpha"],
        "description": "Nonparametric rank-based test for stochastic dominance.",
    },
    "ks_test": {
        "name": "Kolmogorov-Smirnov",
        "distributions": ["nonparametric", "normal"],
        "params": ["alpha"],
        "description": "Compares ECDFs of two samples. Sensitive to any difference in distribution shape.",
    },
    "permutation": {
        "name": "Permutation Test",
        "distributions": ["nonparametric", "bernoulli", "normal"],
        "params": ["alpha", "n_resamples"],
        "description": "Exact nonparametric test. Shuffles group labels to build null distribution.",
    },
    "bootstrap": {
        "name": "Bootstrap CI",
        "distributions": ["bernoulli", "normal", "nonparametric"],
        "params": ["confidence", "n_resamples"],
        "description": "Estimate sampling distribution of the difference via resampling.",
    },
}

# ── LaTeX Formulas ───────────────────────────────────────────────────────────

LATEX = {
    "neyman_pearson": [
        r"\text{Log-Likelihood Ratio: } \Lambda = \sum_{i=1}^{n} \log \frac{f(x_i \mid \theta_1)}{f(x_i \mid \theta_0)}",
        r"\text{Decision: Reject } H_0 \text{ if } \Lambda > \gamma",
        r"\text{Threshold } \gamma \text{: } P(\Lambda > \gamma \mid H_0) = \alpha",
        r"\text{Power: } \beta = P(\Lambda > \gamma \mid H_1)",
    ],
    "glrt": [
        r"\Lambda_{\text{GLRT}} = -2 \log \frac{\sup_{\theta \in \Theta_0} L(\theta)}{\sup_{\theta \in \Theta} L(\theta)}",
        r"\text{Under } H_0\!: \Lambda_{\text{GLRT}} \xrightarrow{d} \chi^2(k)",
        r"\text{Reject } H_0 \text{ if } \Lambda > \chi^2_{1-\alpha}(k)",
    ],
    "sprt": [
        r"S_n = \sum_{i=1}^{n} \log \frac{f(x_i \mid \theta_1)}{f(x_i \mid \theta_0)}",
        r"A = \log \frac{1-\beta}{\alpha}, \quad B = \log \frac{\beta}{1-\alpha}",
        r"\text{Reject } H_0 \text{ if } S_n \geq A; \quad \text{Accept } H_0 \text{ if } S_n \leq B",
    ],
    "joint_detection_estimation": [
        r"\text{Detection: } \Lambda_{\text{GLRT}} > \chi^2_{1-\alpha}(k)",
        r"\text{If detected: } \hat{\theta}_{\text{MLE}} = \arg\max L(\theta \mid \mathbf{x})",
        r"\text{Effect size: } d = (\hat{\theta}_1 - \theta_0) / \text{SE}",
    ],
    "proportions_z": [
        r"\hat{p} = \frac{x_1+x_2}{n_1+n_2}, \quad Z = \frac{\hat{p}_1-\hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})(1/n_1+1/n_2)}}",
        r"\text{Reject } H_0 \text{ if } |Z| > z_{1-\alpha/2}",
    ],
    "ttest": [
        r"t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}",
    ],
    "chi_squared": [
        r"\chi^2 = \sum_{i,j} \frac{(O_{ij}-E_{ij})^2}{E_{ij}}, \quad V = \sqrt{\frac{\chi^2}{n \cdot \min(r-1,c-1)}}",
    ],
    "ks_test": [
        r"D = \sup_x |F_1(x) - F_2(x)|",
    ],
}

# ── Presets ───────────────────────────────────────────────────────────────────

PRESETS = {
    "Timeout effect on scoring": {
        "metric": "Possession points",
        "distribution": "normal",
        "test_type": "glrt",
        "a": {"prev_event": "After timeout"},
        "b": {},
    },
    "Clutch vs non-clutch FG%": {
        "metric": "Shot result (made/missed)",
        "distribution": "bernoulli",
        "test_type": "proportions_z",
        "a": {"clutch": True},
        "b": {"clutch": False},
    },
    "Home vs away 3PT%": {
        "metric": "Shot result (made/missed)",
        "distribution": "bernoulli",
        "test_type": "proportions_z",
        "a": {"home_away": "Home", "shot_type": "3pt"},
        "b": {"home_away": "Away", "shot_type": "3pt"},
    },
    "Post-OREB shot accuracy": {
        "metric": "Shot result (made/missed)",
        "distribution": "bernoulli",
        "test_type": "glrt",
        "a": {"prev_event": "After offensive rebound"},
        "b": {},
    },
    "OT vs regulation possession outcomes": {
        "metric": "Possession outcome",
        "distribution": "categorical",
        "test_type": "chi_squared",
        "a": {"period": "OT"},
        "b": {"period": "Regulation"},
    },
}

# ── Test Implementations ─────────────────────────────────────────────────────


def _to_numpy(series):
    return series.drop_nulls().to_numpy().astype(float)


def run_test(test_type, a, b, params):
    alpha = params.get("alpha", 0.05)

    if test_type == "proportions_z":
        n1, n2 = len(a), len(b)
        x1, x2 = a.sum(), b.sum()
        p1, p2 = x1 / n1, x2 / n2
        p_pool = (x1 + x2) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        z = (p1 - p2) / se if se > 0 else 0
        p_val = 2 * stats.norm.sf(abs(z))
        crit = stats.norm.ppf(1 - alpha / 2)
        return {
            "statistic": z,
            "p_value": p_val,
            "critical_value": crit,
            "decision": "Reject H0" if abs(z) > crit else "Fail to Reject H0",
            "details": {"p1": p1, "p2": p2, "pooled": p_pool, "n1": n1, "n2": n2},
        }

    elif test_type == "ttest":
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        crit = stats.t.ppf(1 - alpha / 2, df=min(len(a), len(b)) - 1)
        return {
            "statistic": t_stat,
            "p_value": p_val,
            "critical_value": crit,
            "decision": "Reject H0" if p_val < alpha else "Fail to Reject H0",
            "details": {"mean_a": a.mean(), "mean_b": b.mean(), "std_a": a.std(), "std_b": b.std()},
        }

    elif test_type == "chi_squared":
        all_cats = sorted(set(a.tolist() + b.tolist()))
        table = np.array([[np.sum(a == c) for c in all_cats], [np.sum(b == c) for c in all_cats]])
        chi2, p_val, dof, expected = stats.chi2_contingency(table)
        n = table.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1))) if n > 0 else 0
        crit = stats.chi2.ppf(1 - alpha, dof)
        return {
            "statistic": chi2,
            "p_value": p_val,
            "critical_value": crit,
            "decision": "Reject H0" if chi2 > crit else "Fail to Reject H0",
            "details": {
                "dof": dof,
                "cramers_v": cramers_v,
                "categories": all_cats,
                "observed": table.tolist(),
                "expected": expected.tolist(),
            },
        }

    elif test_type == "mann_whitney":
        u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
        return {
            "statistic": u_stat,
            "p_value": p_val,
            "critical_value": None,
            "decision": "Reject H0" if p_val < alpha else "Fail to Reject H0",
            "details": {"median_a": np.median(a), "median_b": np.median(b)},
        }

    elif test_type == "ks_test":
        d_stat, p_val = stats.ks_2samp(a, b)
        return {
            "statistic": d_stat,
            "p_value": p_val,
            "critical_value": None,
            "decision": "Reject H0" if p_val < alpha else "Fail to Reject H0",
            "details": {},
        }

    elif test_type == "permutation":
        n_resamples = params.get("n_resamples", 9999)

        def stat_fn(x, y, axis):
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)

        result = stats.permutation_test((a, b), stat_fn, n_resamples=n_resamples, alternative="two-sided")
        return {
            "statistic": result.statistic,
            "p_value": result.pvalue,
            "critical_value": None,
            "decision": "Reject H0" if result.pvalue < alpha else "Fail to Reject H0",
            "details": {"null_distribution": result.null_distribution},
        }

    elif test_type == "bootstrap":
        confidence = params.get("confidence", 0.95)
        n_resamples = params.get("n_resamples", 9999)

        def diff_means(x, y, axis):
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)

        result = stats.bootstrap(
            (a, b), diff_means, n_resamples=n_resamples, confidence_level=confidence, method="percentile"
        )
        ci = result.confidence_interval
        obs_diff = a.mean() - b.mean()
        sig = ci.low > 0 or ci.high < 0
        return {
            "statistic": obs_diff,
            "p_value": None,
            "critical_value": None,
            "decision": f"CI [{ci.low:.4f}, {ci.high:.4f}]" + (" — significant" if sig else " — not significant"),
            "details": {"ci_low": ci.low, "ci_high": ci.high, "bootstrap_distribution": result.bootstrap_distribution},
        }

    elif test_type == "glrt":
        n1, n2 = len(a), len(b)
        if len(np.unique(a)) == 2:
            x1, x2 = a.sum(), b.sum()
            p1, p2 = x1 / n1, x2 / n2
            p_pool = (x1 + x2) / (n1 + n2)
            eps = 1e-15
            ll_r = (
                x1 * np.log(p_pool + eps)
                + (n1 - x1) * np.log(1 - p_pool + eps)
                + x2 * np.log(p_pool + eps)
                + (n2 - x2) * np.log(1 - p_pool + eps)
            )
            ll_u = (
                x1 * np.log(p1 + eps)
                + (n1 - x1) * np.log(1 - p1 + eps)
                + x2 * np.log(p2 + eps)
                + (n2 - x2) * np.log(1 - p2 + eps)
            )
            lambda_stat = -2 * (ll_r - ll_u)
            dof = 1
        else:
            mu_pool = np.concatenate([a, b]).mean()
            sigma_pool = np.concatenate([a, b]).std() + 1e-15
            ll_r = stats.norm.logpdf(a, mu_pool, sigma_pool).sum() + stats.norm.logpdf(b, mu_pool, sigma_pool).sum()
            ll_u = (
                stats.norm.logpdf(a, a.mean(), a.std() + 1e-15).sum()
                + stats.norm.logpdf(b, b.mean(), b.std() + 1e-15).sum()
            )
            lambda_stat = -2 * (ll_r - ll_u)
            dof = 1
        p_val = stats.chi2.sf(lambda_stat, dof)
        crit = stats.chi2.ppf(1 - alpha, dof)
        return {
            "statistic": lambda_stat,
            "p_value": p_val,
            "critical_value": crit,
            "decision": "Reject H0" if lambda_stat > crit else "Fail to Reject H0",
            "details": {"dof": dof},
        }

    elif test_type == "neyman_pearson":
        theta_0 = params.get("theta_0", b.mean())
        theta_1 = params.get("theta_1", a.mean())
        n = len(a)
        if len(np.unique(a)) == 2:
            p0, p1 = theta_0, theta_1
            eps = 1e-15
            log_lr = a * np.log((p1 + eps) / (p0 + eps)) + (1 - a) * np.log((1 - p1 + eps) / (1 - p0 + eps))
        else:
            sigma = a.std() if a.std() > 0 else 1
            log_lr = (theta_1 - theta_0) * a / sigma**2 - (theta_1**2 - theta_0**2) / (2 * sigma**2)
        total_llr = log_lr.sum()
        mu_0 = log_lr.mean()
        var_0 = log_lr.var()
        gamma = n * mu_0 + stats.norm.ppf(1 - alpha) * np.sqrt(n * var_0) if var_0 > 0 else 0
        return {
            "statistic": total_llr,
            "p_value": stats.norm.sf((total_llr - n * mu_0) / (np.sqrt(n * var_0) + 1e-15)),
            "critical_value": gamma,
            "decision": "Reject H0" if total_llr > gamma else "Fail to Reject H0",
            "details": {"theta_0": theta_0, "theta_1": theta_1, "log_lr": log_lr},
        }

    elif test_type == "sprt":
        theta_0 = params.get("theta_0", b.mean())
        theta_1 = params.get("theta_1", a.mean())
        beta = params.get("beta", 0.1)
        eps = 1e-15
        log_A = np.log((1 - beta) / (alpha + eps))
        log_B = np.log((beta + eps) / (1 - alpha))
        if len(np.unique(a)) == 2:
            p0, p1 = theta_0, theta_1
            log_lr = a * np.log((p1 + eps) / (p0 + eps)) + (1 - a) * np.log((1 - p1 + eps) / (1 - p0 + eps))
        else:
            sigma = a.std() if a.std() > 0 else 1
            log_lr = (theta_1 - theta_0) / sigma**2 * (a - (theta_0 + theta_1) / 2)
        cum_lr = np.cumsum(log_lr)
        decision_idx = None
        decision = "Continue (inconclusive)"
        for i, s in enumerate(cum_lr):
            if s >= log_A:
                decision = "Reject H0 (signal detected)"
                decision_idx = i
                break
            elif s <= log_B:
                decision = "Accept H0 (no signal)"
                decision_idx = i
                break
        return {
            "statistic": cum_lr[-1] if len(cum_lr) else 0,
            "p_value": None,
            "critical_value": None,
            "decision": decision,
            "details": {
                "cum_lr": cum_lr,
                "log_A": log_A,
                "log_B": log_B,
                "decision_idx": decision_idx,
                "theta_0": theta_0,
                "theta_1": theta_1,
            },
        }

    elif test_type == "joint_detection_estimation":
        glrt_result = run_test("glrt", a, b, params)
        detected = glrt_result["p_value"] < alpha if glrt_result["p_value"] is not None else False
        theta_hat = a.mean()
        theta_0 = b.mean()
        se = a.std() / np.sqrt(len(a)) if len(a) > 0 else 0
        ci_low = theta_hat - stats.norm.ppf(1 - alpha / 2) * se
        ci_high = theta_hat + stats.norm.ppf(1 - alpha / 2) * se
        effect_size = (theta_hat - theta_0) / (b.std() + 1e-15)
        return {
            "statistic": glrt_result["statistic"],
            "p_value": glrt_result["p_value"],
            "critical_value": glrt_result["critical_value"],
            "decision": f"Detected, θ̂={theta_hat:.4f} [{ci_low:.4f}, {ci_high:.4f}]" if detected else "Not detected",
            "details": {
                "detected": detected,
                "theta_hat": theta_hat,
                "theta_0": theta_0,
                "effect_size": effect_size,
                "ci_low": ci_low,
                "ci_high": ci_high,
            },
        }

    return {"statistic": None, "p_value": None, "critical_value": None, "decision": "Unknown test", "details": {}}


# ── Visualization ────────────────────────────────────────────────────────────


def make_visualization(test_type, a, b, results, label_a="Sample A", label_b="Sample B"):
    fig = go.Figure()
    d = results.get("details", {})

    if test_type == "proportions_z":
        p1, p2 = d.get("p1", 0), d.get("p2", 0)
        n1, n2 = d.get("n1", 1), d.get("n2", 1)
        se1, se2 = np.sqrt(p1 * (1 - p1) / n1), np.sqrt(p2 * (1 - p2) / n2)
        fig.add_trace(
            go.Bar(
                x=[label_a, label_b],
                y=[p1, p2],
                error_y=dict(type="data", array=[1.96 * se1, 1.96 * se2]),
                marker_color=["#636EFA", "#EF553B"],
            )
        )
        fig.update_layout(title="Proportions with 95% CI", yaxis_title="Proportion", template="plotly_dark")

    elif test_type in ("ttest", "glrt", "neyman_pearson", "joint_detection_estimation"):
        fig.add_trace(go.Histogram(x=a, name=label_a, opacity=0.6, nbinsx=50))
        fig.add_trace(go.Histogram(x=b, name=label_b, opacity=0.6, nbinsx=50))
        fig.update_layout(barmode="overlay", title="Distribution Comparison", template="plotly_dark")

    elif test_type == "chi_squared":
        cats = d.get("categories", [])
        obs = np.array(d.get("observed", []))
        if len(obs) == 2:
            fig.add_trace(go.Bar(x=[str(c) for c in cats], y=obs[0], name=label_a))
            fig.add_trace(go.Bar(x=[str(c) for c in cats], y=obs[1], name=label_b))
            fig.update_layout(barmode="group", title="Category Frequencies", template="plotly_dark")

    elif test_type == "ks_test":
        a_s, b_s = np.sort(a), np.sort(b)
        fig.add_trace(go.Scatter(x=a_s, y=np.arange(1, len(a_s) + 1) / len(a_s), mode="lines", name=label_a))
        fig.add_trace(go.Scatter(x=b_s, y=np.arange(1, len(b_s) + 1) / len(b_s), mode="lines", name=label_b))
        fig.update_layout(title=f"ECDFs (D={results['statistic']:.4f})", template="plotly_dark")

    elif test_type == "sprt":
        cum_lr = d.get("cum_lr", np.array([]))
        if len(cum_lr):
            fig.add_trace(go.Scatter(y=cum_lr.tolist(), mode="lines", name="Cumulative Log-LR"))
            fig.add_hline(y=d.get("log_A", 0), line_dash="dash", line_color="red", annotation_text="Reject H0")
            fig.add_hline(y=d.get("log_B", 0), line_dash="dash", line_color="green", annotation_text="Accept H0")
            if d.get("decision_idx") is not None:
                fig.add_vline(x=d["decision_idx"], line_dash="dot", line_color="yellow")
            fig.update_layout(
                title="SPRT: Random Walk Between Fences",
                xaxis_title="Observation",
                yaxis_title="Cumulative Log-LR",
                template="plotly_dark",
            )

    elif test_type in ("permutation", "bootstrap"):
        null = d.get("null_distribution") or d.get("bootstrap_distribution")
        if null is not None:
            fig.add_trace(go.Histogram(x=np.array(null).flatten(), nbinsx=80, name="Null", opacity=0.7))
            if results["statistic"] is not None:
                fig.add_vline(x=results["statistic"], line_color="red", line_width=2, annotation_text=f"Observed")
            if test_type == "bootstrap" and d.get("ci_low") is not None:
                fig.add_vrect(x0=d["ci_low"], x1=d["ci_high"], fillcolor="green", opacity=0.15)
            fig.update_layout(
                title=f"{'Permutation' if test_type == 'permutation' else 'Bootstrap'} Distribution",
                template="plotly_dark",
            )

    elif test_type == "mann_whitney":
        fig.add_trace(go.Box(y=a, name=label_a))
        fig.add_trace(go.Box(y=b, name=label_b))
        fig.update_layout(title="Box Plots", template="plotly_dark")

    return fig


# ── LLM SQL Generation ──────────────────────────────────────────────────────

HYPOTHESIS_SQL_SUFFIX = """

## SPECIAL TASK: Hypothesis Test SQL
You are generating SQL for a hypothesis test. The user has specified a metric and filters for two samples.
Generate TWO SQL queries that each return a single column named 'value'.

Return ONLY a JSON object:
{"sample_a_sql": "SELECT ... AS value FROM ...", "sample_b_sql": "SELECT ... AS value FROM ..."}

Metric definitions:
- "Possession points": SELECT possession_points AS value, first row per possession (ROW_NUMBER OVER PARTITION BY gameId, possession_id)
- "Shot result (made/missed)": SELECT CASE WHEN shotResult='Made' THEN 1 ELSE 0 END AS value WHERE isFieldGoal=1
- "Possession outcome": SELECT possession_outcome AS value, first row per possession
- "Points scored per event": SELECT points_scored AS value
- "Shot distance": SELECT shotDistance AS value WHERE isFieldGoal=1

Only return the JSON object. No markdown fences."""


def generate_sql(client, metric, filters_a, filters_b, system_prompt):
    system = system_prompt + HYPOTHESIS_SQL_SUFFIX
    prompt = json.dumps({"metric": metric, "sample_a_filters": filters_a, "sample_b_filters": filters_b})
    raw = call_llm(client, [{"role": "user", "content": prompt}], system=system, model=MODEL_OPUS, max_tokens=2048)
    return parse_response(raw)


# ── Filter UI ────────────────────────────────────────────────────────────────


def _render_filters(label, key_prefix):
    """Render filter controls for one sample. Returns dict of filter values."""
    filters = {}
    st.markdown(f"**{label}**")
    seasons = st.multiselect("Seasons", [2019, 2020, 2021, 2022, 2023, 2024, 2025], key=f"{key_prefix}_seasons")
    if seasons:
        filters["seasons"] = seasons
    st_type = st.selectbox("Season type", ["Both", "Regular season", "Playoffs"], key=f"{key_prefix}_stype")
    if st_type == "Regular season":
        filters["season_type"] = "rg"
    elif st_type == "Playoffs":
        filters["season_type"] = "po"
    teams = st.multiselect("Teams", TEAMS, key=f"{key_prefix}_teams")
    if teams:
        filters["teams"] = teams
    player = st.text_input("Player (name)", key=f"{key_prefix}_player")
    if player:
        filters["player"] = player
    prev = st.selectbox("Previous event", list(PREV_EVENTS.keys()), key=f"{key_prefix}_prev")
    if PREV_EVENTS[prev]:
        filters["prev_event"] = prev
    period = st.selectbox(
        "Period", ["All", "Regulation (Q1-Q4)", "OT", "Q1", "Q2", "Q3", "Q4"], key=f"{key_prefix}_period"
    )
    if period != "All":
        filters["period"] = period
    clutch = st.selectbox("Clutch", ["Either", "Clutch only", "Non-clutch only"], key=f"{key_prefix}_clutch")
    if clutch == "Clutch only":
        filters["clutch"] = True
    elif clutch == "Non-clutch only":
        filters["clutch"] = False
    shot_type = st.selectbox("Shot type", ["All", "2pt", "3pt", "freethrow"], key=f"{key_prefix}_shottype")
    if shot_type != "All":
        filters["shot_type"] = shot_type
    home_away = st.selectbox("Home/Away", ["Both", "Home", "Away"], key=f"{key_prefix}_homeaway")
    if home_away != "Both":
        filters["home_away"] = home_away
    return filters


# ── Main Page ────────────────────────────────────────────────────────────────


def _init_state():
    for key, default in [("hyp_results", None), ("hyp_sql", None)]:
        if key not in st.session_state:
            st.session_state[key] = default


def hypothesis_page():
    _init_state()
    st.title("Hypothesis Playground")
    st.caption("Define two samples, pick a distribution and test, then run.")

    client = get_client()
    if not client:
        st.error("Set ANTHROPIC_API_KEY in .streamlit/secrets.toml")
        return
    conn, pbp, player_stats, player_advanced = load_data()
    from helpers import get_system_prompt

    SYSTEM_PROMPT = get_system_prompt(pbp, player_stats, player_advanced)

    # Presets
    st.markdown("#### Quick Start")
    preset_cols = st.columns(len(PRESETS))
    for i, (name, cfg) in enumerate(PRESETS.items()):
        with preset_cols[i % len(PRESETS)]:
            if st.button(name, key=f"preset_{i}"):
                for k, v in cfg.get("a", {}).items():
                    st.session_state[f"a_{k}"] = v
                for k, v in cfg.get("b", {}).items():
                    st.session_state[f"b_{k}"] = v
                st.rerun()

    st.markdown("---")

    # Section 1: Metric
    metric = st.selectbox("Metric (what to compare)", list(METRICS.keys()), key="hyp_metric")
    metric_info = METRICS[metric]

    custom_sql_a, custom_sql_b = "", ""
    if metric == "Custom SQL":
        col1, col2 = st.columns(2)
        with col1:
            custom_sql_a = st.text_area("Sample A SQL (must return 'value' column)", height=120, key="custom_a")
        with col2:
            custom_sql_b = st.text_area("Sample B SQL (must return 'value' column)", height=120, key="custom_b")

    # Section 2: Filters for each sample
    filters_a, filters_b = {}, {}
    if metric != "Custom SQL":
        st.markdown("#### Sample Definitions")
        col1, col2 = st.columns(2)
        with col1:
            freeform_a = st.text_input(
                "Sample A description (optional)",
                key="freeform_a",
                placeholder="e.g., Celtics in clutch time, after timeouts",
            )
            if freeform_a:
                filters_a["freeform"] = freeform_a
        with col2:
            freeform_b = st.text_input(
                "Sample B description (optional)",
                key="freeform_b",
                placeholder="e.g., All other possessions, league average",
            )
            if freeform_b:
                filters_b["freeform"] = freeform_b

        with st.expander("Structured Filters (optional — refine the samples)"):
            col1, col2 = st.columns(2)
            with col1:
                filters_a.update(_render_filters("Sample A", "a"))
            with col2:
                filters_b.update(_render_filters("Sample B", "b"))

    st.markdown("---")

    # Section 3: Distribution
    default_dist = metric_info.get("distribution", "normal")
    dist_options = list(DISTRIBUTIONS.keys())
    dist_idx = dist_options.index(default_dist) if default_dist and default_dist in dist_options else 0
    distribution = st.selectbox(
        "Distribution model", dist_options, index=dist_idx, format_func=lambda k: DISTRIBUTIONS[k], key="hyp_dist"
    )

    # Section 4: Test type (filtered by distribution)
    compatible = {k: v for k, v in TESTS.items() if distribution in v["distributions"]}
    test_type = st.selectbox(
        "Test type",
        list(compatible.keys()),
        format_func=lambda k: f"{compatible[k]['name']} — {compatible[k]['description'][:80]}",
        key="hyp_test",
    )
    test_info = TESTS[test_type]

    # Parameters
    st.markdown("#### Parameters")
    params = {}
    pcols = st.columns(4)
    if "alpha" in test_info["params"]:
        with pcols[0]:
            params["alpha"] = st.slider("α (significance)", 0.01, 0.20, 0.05, 0.01, key="hyp_alpha")
    if "theta_0" in test_info["params"]:
        with pcols[1]:
            params["theta_0"] = st.number_input("θ₀ (H0 parameter)", value=0.0, format="%.4f", key="hyp_t0")
        with pcols[2]:
            params["theta_1"] = st.number_input("θ₁ (H1 parameter)", value=0.0, format="%.4f", key="hyp_t1")
    if "beta" in test_info["params"]:
        with pcols[3]:
            params["beta"] = st.slider("β (miss rate)", 0.01, 0.20, 0.10, 0.01, key="hyp_beta")
    if "n_resamples" in test_info["params"]:
        with pcols[1]:
            params["n_resamples"] = st.number_input("Resamples", 999, 99999, 9999, key="hyp_resamp")
    if "confidence" in test_info["params"]:
        with pcols[0]:
            params["confidence"] = st.slider("Confidence", 0.80, 0.99, 0.95, 0.01, key="hyp_conf")

    # Run
    st.markdown("---")
    if st.button("Run Test", type="primary", key="run_test_btn"):
        with st.spinner("Generating SQL..."):
            if metric == "Custom SQL":
                sql_result = {"sample_a_sql": custom_sql_a, "sample_b_sql": custom_sql_b}
            else:
                sql_result = generate_sql(client, metric, filters_a, filters_b, SYSTEM_PROMPT)

        if not sql_result.get("sample_a_sql") or not sql_result.get("sample_b_sql"):
            st.error("Could not generate SQL queries.")
            if sql_result.get("error"):
                with st.expander("Raw response"):
                    st.text(str(sql_result.get("error", ""))[:500])
            return

        st.session_state.hyp_sql = sql_result
        with st.expander("Generated SQL"):
            st.code(sql_result["sample_a_sql"], language="sql")
            st.code(sql_result["sample_b_sql"], language="sql")

        with st.spinner("Running queries..."):
            try:
                df_a = execute_sql(conn, sql_result["sample_a_sql"])
                df_b = execute_sql(conn, sql_result["sample_b_sql"])
            except Exception as e:
                st.error(f"SQL execution failed: {e}")
                return

        if "value" not in df_a.columns or "value" not in df_b.columns:
            st.error("SQL must return a column named 'value'.")
            return

        if distribution == "categorical":
            a = df_a["value"].drop_nulls().to_numpy()
            b = df_b["value"].drop_nulls().to_numpy()
        else:
            a = _to_numpy(df_a["value"])
            b = _to_numpy(df_b["value"])

        if len(a) == 0 or len(b) == 0:
            st.error(f"Empty samples: A={len(a)}, B={len(b)}")
            return

        # Subsample for performance
        if len(a) > 100000:
            rng = np.random.default_rng(42)
            a = rng.choice(a, 100000, replace=False)
            st.caption(f"Sample A subsampled to 100K (from {df_a.height:,})")
        if len(b) > 100000:
            rng = np.random.default_rng(43)
            b = rng.choice(b, 100000, replace=False)
            st.caption(f"Sample B subsampled to 100K (from {df_b.height:,})")

        # Auto-fill theta params from data if not set
        if "theta_0" in test_info["params"] and params.get("theta_0", 0) == 0 and params.get("theta_1", 0) == 0:
            params["theta_0"] = float(b.mean())
            params["theta_1"] = float(a.mean())

        with st.spinner("Running test..."):
            results = run_test(test_type, a, b, params)

        st.session_state.hyp_results = {
            "test_results": results,
            "a": a,
            "b": b,
            "label_a": "Sample A",
            "label_b": "Sample B",
        }

    # Display results
    if st.session_state.hyp_results:
        r = st.session_state.hyp_results
        results = r["test_results"]
        a, b = r["a"], r["b"]

        st.markdown("---")
        st.subheader("Results")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Sample A** (n={len(a):,}) — mean: {a.mean():.4f}, std: {a.std():.4f}")
        with col2:
            st.markdown(f"**Sample B** (n={len(b):,}) — mean: {b.mean():.4f}, std: {b.std():.4f}")

        mcols = st.columns(4)
        with mcols[0]:
            st.metric("Test Statistic", f"{results['statistic']:.4f}" if results["statistic"] is not None else "N/A")
        with mcols[1]:
            st.metric("p-value", f"{results['p_value']:.4e}" if results["p_value"] is not None else "N/A")
        with mcols[2]:
            st.metric(
                "Critical Value", f"{results['critical_value']:.4f}" if results["critical_value"] is not None else "N/A"
            )
        with mcols[3]:
            dec = results["decision"]
            st.markdown(f"**{dec}**")

        fig = make_visualization(test_type, a, b, results, r.get("label_a", "A"), r.get("label_b", "B"))
        st.plotly_chart(fig, width="stretch", theme="streamlit")

        formulas = LATEX.get(test_type, [])
        if formulas:
            with st.expander("Mathematical Details"):
                for f in formulas:
                    st.latex(f)

        with st.expander("Full Details"):
            clean = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in results.get("details", {}).items()
                if not isinstance(v, np.ndarray) or len(v) < 50
            }
            st.json(clean)
