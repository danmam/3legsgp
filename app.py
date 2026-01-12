import streamlit as st
from dataclasses import dataclass
import numpy as np
from scipy import stats
from scipy.optimize import minimize

# -----------------------------
# Conversion Helpers
# -----------------------------

def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def prob_to_american(p):
    if p <= 0:
        return float("inf")
    if p >= 1:
        return float("inf")
    dec = 1 / p
    if dec >= 2:
        return int((dec - 1) * 100)
    else:
        return int(-100 / (dec - 1))

def remove_juice_proportional(probs):
    """Remove juice by normalizing probabilities proportionally."""
    total = sum(probs)
    if total == 0:
        return probs
    return [p / total for p in probs]

def remove_juice_power(probs, k=None):
    """Remove juice using power method (Shin's method approximation)."""
    if k is None:
        low, high = 0.5, 2.0
        for _ in range(50):
            mid = (low + high) / 2
            total = sum(p ** mid for p in probs)
            if total > 1:
                low = mid
            else:
                high = mid
        k = mid
    
    fair_probs = [p ** k for p in probs]
    total = sum(fair_probs)
    return [p / total for p in fair_probs]


# -----------------------------
# Data Classes
# -----------------------------

@dataclass
class EightWay:
    p111: float  # S=1, A=1, B=1
    p110: float  # S=1, A=1, B=0
    p101: float  # S=1, A=0, B=1
    p100: float  # S=1, A=0, B=0
    p011: float  # S=0, A=1, B=1
    p010: float  # S=0, A=1, B=0
    p001: float  # S=0, A=0, B=1
    p000: float  # S=0, A=0, B=0

@dataclass
class FairOdds:
    pS: float
    pA: float
    pB: float
    pSA: float
    pSB: float

@dataclass
class ThreeLegResult:
    pSAB_final: float
    pSAB_indep: float
    pSAB_kappa: float
    pSAB_copula_unconstrained: float
    american_final: int
    american_indep: int
    american_kappa: int
    american_copula_unconstrained: int
    pAB_from_8way: float
    pAB_given_S_final: float
    correlation_matrix: np.ndarray
    juice_total: float
    juice_pct: float
    warnings: list


# -----------------------------
# Copula Methods
# -----------------------------

def extract_correlation_from_8way(eight_fair: EightWay):
    """
    Extract the correlation matrix from the fair 8-way distribution.
    Uses the Gaussian copula approach - convert marginals to standard normal,
    then estimate correlation.
    """
    # Calculate all marginals from 8-way
    pS = eight_fair.p111 + eight_fair.p110 + eight_fair.p101 + eight_fair.p100
    pA = eight_fair.p111 + eight_fair.p110 + eight_fair.p011 + eight_fair.p010
    pB = eight_fair.p111 + eight_fair.p101 + eight_fair.p011 + eight_fair.p001
    
    pSA = eight_fair.p111 + eight_fair.p110
    pSB = eight_fair.p111 + eight_fair.p101
    pAB = eight_fair.p111 + eight_fair.p011
    
    # Calculate pairwise correlations using Pearson correlation for binary variables
    # For binary variables: cor(X,Y) = (P(XY) - P(X)P(Y)) / sqrt(P(X)(1-P(X))P(Y)(1-P(Y)))
    
    def binary_correlation(p_joint, p_x, p_y):
        """Calculate correlation between two binary variables."""
        if p_x == 0 or p_x == 1 or p_y == 0 or p_y == 1:
            return 0
        cov = p_joint - p_x * p_y
        std_x = np.sqrt(p_x * (1 - p_x))
        std_y = np.sqrt(p_y * (1 - p_y))
        if std_x == 0 or std_y == 0:
            return 0
        return cov / (std_x * std_y)
    
    rho_SA = binary_correlation(pSA, pS, pA)
    rho_SB = binary_correlation(pSB, pS, pB)
    rho_AB = binary_correlation(pAB, pA, pB)
    
    # Construct correlation matrix
    corr_matrix = np.array([
        [1.0, rho_SA, rho_SB],
        [rho_SA, 1.0, rho_AB],
        [rho_SB, rho_AB, 1.0]
    ])
    
    # Ensure positive semi-definite (adjust if needed)
    eigvals = np.linalg.eigvalsh(corr_matrix)
    if np.min(eigvals) < -1e-10:
        # Adjust to nearest positive semi-definite matrix
        corr_matrix = nearest_positive_definite(corr_matrix)
    
    return corr_matrix, pAB

def nearest_positive_definite(A):
    """Find the nearest positive definite matrix."""
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    
    if is_positive_definite(A3):
        return A3
    
    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    
    return A3

def is_positive_definite(B):
    """Check if matrix is positive definite."""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

def gaussian_copula_prob(pS, pA, pB, corr_matrix):
    """
    Calculate P(S∩A∩B) using Gaussian copula with given correlation matrix.
    This is the baseline copula estimate without 2-way constraints.
    """
    # Convert marginal probabilities to standard normal quantiles
    # Add small epsilon to avoid infinite values
    eps = 1e-10
    pS_clip = np.clip(pS, eps, 1 - eps)
    pA_clip = np.clip(pA, eps, 1 - eps)
    pB_clip = np.clip(pB, eps, 1 - eps)
    
    z_S = stats.norm.ppf(pS_clip)
    z_A = stats.norm.ppf(pA_clip)
    z_B = stats.norm.ppf(pB_clip)
    
    # Use multivariate normal CDF to calculate joint probability
    mean = [0, 0, 0]
    upper = [z_S, z_A, z_B]
    lower = [-np.inf, -np.inf, -np.inf]
    
    # Calculate P(S∩A∩B) using multivariate normal CDF
    try:
        from scipy.stats import mvn as mvn_module
        p, _ = mvn_module.mvnun(lower, upper, mean, corr_matrix)
        return p
    except:
        # Fallback: use Monte Carlo approximation
        n_samples = 10000
        samples = np.random.multivariate_normal(mean, corr_matrix, n_samples)
        count = np.sum((samples[:, 0] <= z_S) & 
                      (samples[:, 1] <= z_A) & 
                      (samples[:, 2] <= z_B))
        return count / n_samples

def constrained_copula_prob(pS, pA, pB, pSA, pSB, pAB, corr_matrix):
    """
    Calculate P(S∩A∩B) using copula structure while respecting authoritative 2-way constraints.
    
    This finds P(S∩A∩B) that:
    1. Respects P(S∩A) = pSA (authoritative)
    2. Respects P(S∩B) = pSB (authoritative)  
    3. Has dependence structure similar to the copula
    
    We use P(AB) from 8-way as a guide for the A-B correlation.
    """
    # Extract correlation between A and B given S from the copula
    # This comes from the 8-way and represents how A and B relate when S occurs
    rho_AB_given_S = corr_matrix[1, 2]  # Correlation between A and B
    
    # Calculate conditional probabilities from authoritative 2-way odds
    pA_given_S = pSA / pS if pS > 0 else 0
    pB_given_S = pSB / pS if pS > 0 else 0
    
    # Calculate P(A∩B|S) using the copula-derived correlation
    # Start with independence baseline
    pAB_given_S_indep = pA_given_S * pB_given_S
    
    # The maximum possible P(A∩B|S)
    max_pAB_given_S = min(pA_given_S, pB_given_S)
    
    # The minimum possible P(A∩B|S) (Fréchet lower bound)
    min_pAB_given_S = max(0, pA_given_S + pB_given_S - 1)
    
    # Use the correlation to interpolate between independence and perfect correlation
    # For binary variables, we can use the correlation coefficient directly
    # to adjust from independence toward the bounds
    
    # Calculate what P(A∩B|S) should be based on the binary correlation
    std_A = np.sqrt(pA_given_S * (1 - pA_given_S))
    std_B = np.sqrt(pB_given_S * (1 - pB_given_S))
    
    if std_A > 0 and std_B > 0:
        # P(AB|S) = P(A|S)*P(B|S) + rho * std(A|S) * std(B|S)
        pAB_given_S = pAB_given_S_indep + rho_AB_given_S * std_A * std_B
        
        # Ensure within valid bounds
        pAB_given_S = np.clip(pAB_given_S, min_pAB_given_S, max_pAB_given_S)
    else:
        pAB_given_S = pAB_given_S_indep
    
    # Final 3-way probability
    pSAB = pS * pAB_given_S
    
    # Validate consistency with P(SA) and P(SB)
    # P(SAB) must be <= min(P(SA), P(SB))
    max_pSAB = min(pSA, pSB)
    if pSAB > max_pSAB:
        pSAB = max_pSAB
    
    return pSAB, pAB_given_S


# -----------------------------
# Core Computation
# -----------------------------

def compute_three_leg_fair(eight: EightWay, fair: FairOdds, devig_method: str) -> ThreeLegResult:
    warnings = []
    
    # Get raw 8-way probabilities
    raw_probs = [
        eight.p111, eight.p110, eight.p101, eight.p100,
        eight.p011, eight.p010, eight.p001, eight.p000
    ]
    
    # Calculate juice
    juice_total = sum(raw_probs)
    juice_pct = (juice_total - 1.0) * 100
    
    if juice_total < 0.99:
        warnings.append(f"8-way probabilities sum to {juice_total:.4f} < 1.0, which is unusual")
    
    # Remove juice based on selected method
    if devig_method == "Proportional":
        fair_probs = remove_juice_proportional(raw_probs)
    else:  # Power/Shin
        fair_probs = remove_juice_power(raw_probs)
    
    # Create devigged 8-way
    eight_fair = EightWay(*fair_probs)
    
    # Extract correlation structure and P(AB) from 8-way
    corr_matrix, pAB_from_8way = extract_correlation_from_8way(eight_fair)
    
    # Calculate P(SAB) using constrained copula that respects authoritative SA and SB
    pSAB_copula, pAB_given_S_copula = constrained_copula_prob(
        fair.pS, fair.pA, fair.pB, fair.pSA, fair.pSB, pAB_from_8way, corr_matrix
    )
    
    # Also calculate unconstrained copula for comparison
    pSAB_copula_unconstrained = gaussian_copula_prob(fair.pS, fair.pA, fair.pB, corr_matrix)
    
    # Also calculate using traditional kappa method for comparison
    pS_8 = eight_fair.p111 + eight_fair.p110 + eight_fair.p101 + eight_fair.p100
    pSA_8 = eight_fair.p111 + eight_fair.p110
    pSB_8 = eight_fair.p111 + eight_fair.p101
    
    pA_given_S = fair.pSA / fair.pS if fair.pS > 0 else 0
    pB_given_S = fair.pSB / fair.pS if fair.pS > 0 else 0
    
    pAB_given_S_8way = eight_fair.p111 / pS_8 if pS_8 > 0 else 0
    pAB_given_S_indep = (pSA_8 / pS_8) * (pSB_8 / pS_8) if pS_8 > 0 else 0
    
    if pAB_given_S_indep > 0:
        kappa = pAB_given_S_8way / pAB_given_S_indep
    else:
        kappa = 1.0
        warnings.append("Cannot calculate kappa (division by zero)")
    
    pAB_given_S_kappa = kappa * pA_given_S * pB_given_S
    max_pAB = min(pA_given_S, pB_given_S)
    if pAB_given_S_kappa > max_pAB:
        pAB_given_S_kappa = max_pAB
        warnings.append(f"Kappa method exceeded bounds, capped at {max_pAB:.4f}")
    
    pSAB_kappa = fair.pS * pAB_given_S_kappa
    
    # True independence baseline
    pSAB_indep = fair.pS * fair.pA * fair.pB
    
    return ThreeLegResult(
        pSAB_final=pSAB_copula,
        pSAB_indep=pSAB_indep,
        pSAB_kappa=pSAB_kappa,
        pSAB_copula_unconstrained=pSAB_copula_unconstrained,
        american_final=prob_to_american(pSAB_copula),
        american_indep=prob_to_american(pSAB_indep),
        american_kappa=prob_to_american(pSAB_kappa),
        american_copula_unconstrained=prob_to_american(pSAB_copula_unconstrained),
        pAB_from_8way=pAB_from_8way,
        pAB_given_S_final=pAB_given_S_copula,
        correlation_matrix=corr_matrix,
        juice_total=juice_total,
        juice_pct=juice_pct,
        warnings=warnings
    )


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("📊 3-Leg SGP Fair Value Calculator (Copula Method)")
st.write("""
**8-way inputs are labeled as (S, A, B) in that exact order.**  
Example:  
- (1,1,1) means **S=1, A=1, B=1**  
- (1,0,1) means **S=1, A=0, B=1**

This calculator uses a **Gaussian copula** to extract correlation structure from 8-way odds 
and apply it to authoritative 2-leg parlay odds.
""")

# Devigging method selection
devig_method = st.radio(
    "Select devigging method for 8-way odds:",
    ["Proportional", "Power/Shin"],
    help="Proportional: scales all probabilities equally. Power/Shin: uses exponential adjustment, typically more accurate for correlated outcomes."
)

st.header("Step 1 — Enter 8-Way American Odds (S, A, B)")

cols = st.columns(4)
o111 = cols[0].number_input("Odds(S=1,A=1,B=1)", value=0, step=1)
o110 = cols[1].number_input("Odds(S=1,A=1,B=0)", value=0, step=1)
o101 = cols[2].number_input("Odds(S=1,A=0,B=1)", value=0, step=1)
o100 = cols[3].number_input("Odds(S=1,A=0,B=0)", value=0, step=1)

cols2 = st.columns(4)
o011 = cols2[0].number_input("Odds(S=0,A=1,B=1)", value=0, step=1)
o010 = cols2[1].number_input("Odds(S=0,A=1,B=0)", value=0, step=1)
o001 = cols2[2].number_input("Odds(S=0,A=0,B=1)", value=0, step=1)
o000 = cols2[3].number_input("Odds(S=0,A=0,B=0)", value=0, step=1)

eight = EightWay(
    american_to_prob(o111),
    american_to_prob(o110),
    american_to_prob(o101),
    american_to_prob(o100),
    american_to_prob(o011),
    american_to_prob(o010),
    american_to_prob(o001),
    american_to_prob(o000)
)

st.header("Step 2 — Enter Authoritative American Odds (S, A, B, SA, SB)")

oS = st.number_input("Odds(S)", value=0, step=1)
oA = st.number_input("Odds(A)", value=0, step=1)
oB = st.number_input("Odds(B)", value=0, step=1)
oSA = st.number_input("Odds(SA)", value=0, step=1)
oSB = st.number_input("Odds(SB)", value=0, step=1)

fair = FairOdds(
    american_to_prob(oS),
    american_to_prob(oA),
    american_to_prob(oB),
    american_to_prob(oSA),
    american_to_prob(oSB)
)

if st.button("Compute 3-Leg Fair Value"):
    result = compute_three_leg_fair(eight, fair, devig_method)

    # Show juice information
    st.info(f"📊 8-Way Juice: {result.juice_pct:.2f}% (sum = {result.juice_total:.4f})")

    if result.warnings:
        st.warning("⚠️ Warnings:")
        for w in result.warnings:
            st.write(f"- {w}")

    st.subheader("📊 Extracted Correlation Structure")
    st.write("**Correlation Matrix (from 8-way):**")
    corr_df = {
        "": ["S", "A", "B"],
        "S": [f"{result.correlation_matrix[0,0]:.4f}", f"{result.correlation_matrix[1,0]:.4f}", f"{result.correlation_matrix[2,0]:.4f}"],
        "A": [f"{result.correlation_matrix[0,1]:.4f}", f"{result.correlation_matrix[1,1]:.4f}", f"{result.correlation_matrix[2,1]:.4f}"],
        "B": [f"{result.correlation_matrix[0,2]:.4f}", f"{result.correlation_matrix[1,2]:.4f}", f"{result.correlation_matrix[2,2]:.4f}"]
    }
    st.table(corr_df)
    st.caption(f"P(A∩B) extracted from 8-way: {result.pAB_from_8way:.6f}")

    st.subheader("📉 Independence Baseline")
    st.metric("P(S∩A∩B) under full independence", f"{result.pSAB_indep:.6f}")
    st.metric("American Odds (Independence)", f"{result.american_indep}")
    st.caption("= P(S) × P(A) × P(B)")

    st.subheader("🔥 Final 3-Leg Fair Value (Copula Method)")
    st.metric("P(S∩A∩B) Final", f"{result.pSAB_final:.6f}")
    st.metric("American Odds (Final)", f"{result.american_final}")
    st.caption("Uses Gaussian copula with correlation structure from 8-way, applied to authoritative marginals")
    
    st.subheader("📊 Comparison: Kappa Method")
    st.metric("P(S∩A∩B) Kappa", f"{result.pSAB_kappa:.6f}")
    st.metric("American Odds (Kappa)", f"{result.american_kappa}")
    st.caption("Traditional correlation multiplier method (for comparison)")
    
    diff_pct = abs(result.pSAB_final - result.pSAB_kappa) / result.pSAB_kappa * 100
    st.info(f"Difference between Copula and Kappa methods: {diff_pct:.2f}%")
