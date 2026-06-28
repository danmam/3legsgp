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

def devig(probs, devig_method):
    """Remove juice from a list of raw implied probabilities using the chosen method."""
    if devig_method == "Proportional":
        return remove_juice_proportional(probs)
    else:  # Power/Shin
        return remove_juice_power(probs)

def binary_correlation(p_joint, p_x, p_y):
    """Calculate the Pearson correlation between two binary variables.

    cor(X,Y) = (P(X∩Y) - P(X)P(Y)) / sqrt(P(X)(1-P(X)) P(Y)(1-P(Y)))
    """
    if p_x <= 0 or p_x >= 1 or p_y <= 0 or p_y >= 1:
        return 0.0
    cov = p_joint - p_x * p_y
    std_x = np.sqrt(p_x * (1 - p_x))
    std_y = np.sqrt(p_y * (1 - p_y))
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)

def joint_from_marginals_and_corr(p_x, p_y, rho):
    """Compute P(X∩Y) for two binary events given their marginals and correlation.

    P(X∩Y) = P(X)P(Y) + rho * sqrt(P(X)(1-P(X))) * sqrt(P(Y)(1-P(Y)))

    The result is clipped to the Fréchet bounds so it stays a valid probability.
    """
    std_x = np.sqrt(p_x * (1 - p_x))
    std_y = np.sqrt(p_y * (1 - p_y))
    p_joint = p_x * p_y + rho * std_x * std_y
    lower = max(0.0, p_x + p_y - 1.0)
    upper = min(p_x, p_y)
    return float(np.clip(p_joint, lower, upper))


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
class SGPResult:
    ok: bool                         # False if not enough info to compute
    error: str                       # populated when ok is False
    pSAB_final: float
    american_final: int
    method_name: str                 # which method produced the final value
    estimates: dict                  # label -> probability (all methods that could run)
    marginals: dict                  # 'S'/'A'/'B' -> (value, source)
    correlations: dict               # 'SA'/'SB'/'AB' -> (value, source)
    correlation_matrix: np.ndarray
    derived_pairs: dict              # 'SA'/'SB'/'AB' -> dict of display values (entered pairs only)
    inputs_detected: list            # human-readable list of which inputs were used
    overdefined: list                # human-readable over-definition notes
    warnings: list


# -----------------------------
# Two-Way Market Helpers
# -----------------------------

@dataclass
class PairMarket:
    """What a single 2-way (4-corner) market tells us, after devigging."""
    rho: float                  # correlation between the two legs
    pX_market: float            # P(X) implied by the (devigged) 2-way market
    pY_market: float            # P(Y) implied by the (devigged) 2-way market
    pJoint_market: float        # devigged P(X∩Y) straight from the market (the OO corner)
    corners_fair: tuple         # devigged (OO, OU, UO, UU)
    juice_total: float          # raw sum of the 4 corner probabilities (pre-devig)

def odds_entered(o) -> bool:
    """A single American-odds box counts as entered when it is non-zero."""
    return o != 0

def corners_entered(four) -> bool:
    """A 4-corner market counts as entered when any of its boxes is non-zero."""
    return any(o != 0 for o in four)

def pair_from_corners(odds_oo, odds_ou, odds_uo, odds_uu, devig_method) -> PairMarket:
    """Devig the 4 corner American odds of a 2-way market and read off its structure.

    Corners (X = first leg, Y = second leg, with O = leg hits, U = leg misses):
        OO -> both legs hit          == the 2-way parlay
        OU -> X hits,  Y misses
        UO -> X misses, Y hits
        UU -> neither hits

    Returns the market-implied marginals, the OO joint, and the X-Y correlation.
    The fair joint using *authoritative* marginals is computed later, once the
    marginals have been resolved across all available inputs.
    """
    raw = [american_to_prob(o) for o in (odds_oo, odds_ou, odds_uo, odds_uu)]
    juice_total = sum(raw)

    p_oo, p_ou, p_uo, p_uu = devig(raw, devig_method)

    pX_market = p_oo + p_ou
    pY_market = p_oo + p_uo

    rho = binary_correlation(p_oo, pX_market, pY_market)

    return PairMarket(
        rho=rho,
        pX_market=pX_market,
        pY_market=pY_market,
        pJoint_market=p_oo,
        corners_fair=(p_oo, p_ou, p_uo, p_uu),
        juice_total=juice_total,
    )


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
    # (binary_correlation is defined at module level)

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
    
    z = [stats.norm.ppf(pS_clip), stats.norm.ppf(pA_clip), stats.norm.ppf(pB_clip)]
    mean = [0, 0, 0]

    # P(S∩A∩B) = P(Z_S <= z_S, Z_A <= z_A, Z_B <= z_B) under the Gaussian copula
    try:
        mvn = stats.multivariate_normal(mean=mean, cov=corr_matrix, allow_singular=True)
        return float(mvn.cdf(z))
    except Exception:
        # Fallback: Monte Carlo approximation
        n_samples = 100000
        samples = np.random.multivariate_normal(mean, corr_matrix, n_samples)
        count = np.sum((samples[:, 0] <= z[0]) &
                       (samples[:, 1] <= z[1]) &
                       (samples[:, 2] <= z[2]))
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

def eight_marginals(e: EightWay):
    pS = e.p111 + e.p110 + e.p101 + e.p100
    pA = e.p111 + e.p110 + e.p011 + e.p010
    pB = e.p111 + e.p101 + e.p011 + e.p001
    return pS, pA, pB


def compute_sgp(eight_odds, oS, oA, oB, sa_odds, sb_odds, ab_odds, devig_method) -> SGPResult:
    """Compute fair P(S∩A∩B) from whichever inputs are provided.

    Any box left at 0 is treated as "not entered". The method is selected from
    what is available:

      * marginals P(S),P(A),P(B): authoritative single  >  pair market(s)  >  8-way
      * correlations ρ_SA,ρ_SB,ρ_AB: 8-way  >  dedicated pair market  >  independence(0)
      * final estimate: if the 8-way is present we use the constrained copula
        (it carries genuine 3rd-order structure); otherwise the Gaussian copula
        over the resolved marginals + pairwise correlation matrix.
    """
    warnings = []
    overdefined = []
    inputs_detected = []

    # ---- Parse the optional pair markets -------------------------------------
    sa = pair_from_corners(*sa_odds, devig_method) if corners_entered(sa_odds) else None
    sb = pair_from_corners(*sb_odds, devig_method) if corners_entered(sb_odds) else None
    ab = pair_from_corners(*ab_odds, devig_method) if corners_entered(ab_odds) else None

    # ---- Parse the optional 8-way --------------------------------------------
    have_eight = corners_entered(eight_odds)
    eight_fair = None
    corr8 = None
    e8S = e8A = e8B = None
    if have_eight:
        raw = [american_to_prob(o) for o in eight_odds]
        juice_total = sum(raw)
        if juice_total < 0.99:
            warnings.append(f"8-way probabilities sum to {juice_total:.4f} < 1.0, which is unusual")
        eight_fair = EightWay(*devig(raw, devig_method))
        corr8, _ = extract_correlation_from_8way(eight_fair)
        e8S, e8A, e8B = eight_marginals(eight_fair)
        inputs_detected.append(f"8-way market (juice {(juice_total - 1) * 100:.2f}%)")

    # ---- Authoritative singles -----------------------------------------------
    sgl = {
        'S': american_to_prob(oS) if odds_entered(oS) else None,
        'A': american_to_prob(oA) if odds_entered(oA) else None,
        'B': american_to_prob(oB) if odds_entered(oB) else None,
    }
    for leg in ('S', 'A', 'B'):
        if sgl[leg] is not None:
            inputs_detected.append(f"Authoritative single {leg}")
    for label, pair in (('S&A', sa), ('S&B', sb), ('A&B', ab)):
        if pair is not None:
            inputs_detected.append(f"{label} 2-way market (juice {(pair.juice_total - 1) * 100:.2f}%)")

    # ---- Resolve each marginal: single > pair-market(s) > 8-way ---------------
    def pair_marginals(leg):
        """List of (source_label, value) for a leg's marginal from entered pairs."""
        out = []
        if leg == 'S':
            if sa: out.append(('S&A', sa.pX_market))
            if sb: out.append(('S&B', sb.pX_market))
        elif leg == 'A':
            if sa: out.append(('S&A', sa.pY_market))
            if ab: out.append(('A&B', ab.pX_market))
        elif leg == 'B':
            if sb: out.append(('S&B', sb.pY_market))
            if ab: out.append(('A&B', ab.pY_market))
        return out

    eight_marg = {'S': e8S, 'A': e8A, 'B': e8B}

    def resolve_marginal(leg):
        candidates = []  # (label, value) for over-definition reporting
        if sgl[leg] is not None:
            candidates.append((f"single {leg}", sgl[leg]))
        candidates.extend(pair_marginals(leg))
        if eight_marg[leg] is not None:
            candidates.append(("8-way", eight_marg[leg]))

        if not candidates:
            return None, None

        # priority: single > average of pair markets > 8-way
        if sgl[leg] is not None:
            chosen, source = sgl[leg], f"authoritative single {leg}"
        else:
            pm = pair_marginals(leg)
            if pm:
                chosen = sum(v for _, v in pm) / len(pm)
                source = " & ".join(l for l, _ in pm) + (" (avg)" if len(pm) > 1 else "")
            else:
                chosen, source = eight_marg[leg], "8-way"

        if len(candidates) > 1:
            vals = [v for _, v in candidates]
            overdefined.append(
                f"P({leg}) is over-defined: " +
                ", ".join(f"{l}={v:.4f}" for l, v in candidates) +
                f" (spread {max(vals) - min(vals):.4f}). Using {source} = {chosen:.4f}."
            )
        return chosen, source

    marginals = {}
    for leg in ('S', 'A', 'B'):
        val, src = resolve_marginal(leg)
        marginals[leg] = (val, src)

    missing = [leg for leg in ('S', 'A', 'B') if marginals[leg][0] is None]
    if missing:
        return SGPResult(
            ok=False,
            error=("Not enough information: no source for marginal(s) " +
                   ", ".join(f"P({m})" for m in missing) +
                   ". Enter the single, a 2-way market containing it, or the 8-way."),
            pSAB_final=0.0, american_final=0, method_name="",
            estimates={}, marginals=marginals, correlations={},
            correlation_matrix=np.eye(3), derived_pairs={},
            inputs_detected=inputs_detected, overdefined=overdefined, warnings=warnings,
        )

    pS = marginals['S'][0]
    pA = marginals['A'][0]
    pB = marginals['B'][0]

    # ---- Resolve each correlation: 8-way > pair market > independence ---------
    eight_corr = None
    if corr8 is not None:
        eight_corr = {'SA': corr8[0, 1], 'SB': corr8[0, 2], 'AB': corr8[1, 2]}

    def resolve_corr(name, pair):
        candidates = []
        if pair is not None:
            candidates.append((f"{name} market", pair.rho))
        if eight_corr is not None:
            candidates.append(("8-way", eight_corr[name]))

        if eight_corr is not None:
            chosen, source = eight_corr[name], "8-way"
        elif pair is not None:
            chosen, source = pair.rho, f"{name} 2-way market"
        else:
            chosen, source = 0.0, "assumed independent"
            warnings.append(f"ρ_{name} has no source; assuming independence (0).")

        if len(candidates) > 1:
            vals = [v for _, v in candidates]
            overdefined.append(
                f"ρ_{name} is over-defined: " +
                ", ".join(f"{l}={v:.4f}" for l, v in candidates) +
                f" (spread {max(vals) - min(vals):.4f}). Using {source} = {chosen:.4f}."
            )
        return chosen, source

    correlations = {}
    for name, pair in (('SA', sa), ('SB', sb), ('AB', ab)):
        val, src = resolve_corr(name, pair)
        correlations[name] = (val, src)

    rSA = correlations['SA'][0]
    rSB = correlations['SB'][0]
    rAB = correlations['AB'][0]

    corr_matrix = np.array([
        [1.0, rSA, rSB],
        [rSA, 1.0, rAB],
        [rSB, rAB, 1.0],
    ])
    if np.min(np.linalg.eigvalsh(corr_matrix)) < -1e-10:
        corr_matrix = nearest_positive_definite(corr_matrix)
        warnings.append("Correlation matrix was not PSD; adjusted to nearest valid matrix.")

    # ---- Fair 2-way joints from resolved marginals + correlations -------------
    pSA = joint_from_marginals_and_corr(pS, pA, rSA)
    pSB = joint_from_marginals_and_corr(pS, pB, rSB)
    pAB = joint_from_marginals_and_corr(pA, pB, rAB)

    derived_pairs = {}
    for name, x, y, pair, joint in (
        ('SA', 'S', 'A', sa, pSA),
        ('SB', 'S', 'B', sb, pSB),
        ('AB', 'A', 'B', ab, pAB),
    ):
        if pair is not None:
            derived_pairs[name] = {
                'rho': pair.rho,
                'market_joint': pair.pJoint_market,
                'fair_joint': joint,
                'fair_american': prob_to_american(joint),
                'juice_pct': (pair.juice_total - 1) * 100,
            }

    # ---- Estimates -----------------------------------------------------------
    estimates = {}
    estimates['Independence'] = pS * pA * pB
    estimates['Gaussian copula'] = gaussian_copula_prob(pS, pA, pB, corr_matrix)

    if have_eight:
        # Constrained copula: authoritative marginals/correlations + 8-way 3rd order
        pSAB_constrained, _ = constrained_copula_prob(pS, pA, pB, pSA, pSB, pAB, corr_matrix)
        estimates['Constrained copula (8-way)'] = pSAB_constrained

        # Kappa multiplier method (uses 8-way conditional structure)
        pS8, _, _ = eight_marginals(eight_fair)
        pSA8 = eight_fair.p111 + eight_fair.p110
        pSB8 = eight_fair.p111 + eight_fair.p101
        pA_g_S = pSA / pS if pS > 0 else 0
        pB_g_S = pSB / pS if pS > 0 else 0
        pAB_g_S_8 = eight_fair.p111 / pS8 if pS8 > 0 else 0
        pAB_g_S_indep8 = (pSA8 / pS8) * (pSB8 / pS8) if pS8 > 0 else 0
        kappa = pAB_g_S_8 / pAB_g_S_indep8 if pAB_g_S_indep8 > 0 else 1.0
        pAB_g_S_k = min(kappa * pA_g_S * pB_g_S, min(pA_g_S, pB_g_S))
        estimates['Kappa (8-way)'] = pS * pAB_g_S_k

        # Raw devigged 8-way joint
        estimates['8-way direct'] = eight_fair.p111

    # ---- Pick the final estimate ---------------------------------------------
    if have_eight:
        final = estimates['Constrained copula (8-way)']
        method_name = "Constrained copula (authoritative marginals/correlations + 8-way 3rd-order)"
    else:
        final = estimates['Gaussian copula']
        method_name = "Gaussian copula (resolved marginals + pairwise correlation matrix)"

    return SGPResult(
        ok=True,
        error="",
        pSAB_final=final,
        american_final=prob_to_american(final),
        method_name=method_name,
        estimates=estimates,
        marginals=marginals,
        correlations=correlations,
        correlation_matrix=corr_matrix,
        derived_pairs=derived_pairs,
        inputs_detected=inputs_detected,
        overdefined=overdefined,
        warnings=warnings,
    )


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("📊 3-Leg SGP Fair Value Calculator (Copula Method)")
st.write("""
Compute the fair value of a 3-leg same-game parlay **S ∩ A ∩ B** from whatever odds you
have. **Every box is optional — leave anything you don't have at 0.** The app figures out
which calculation method the available inputs support.

What each input contributes:
- **8-way market** — the complete joint distribution (marginals, all correlations, *and*
  the 3-way interaction). On its own it fully determines the answer.
- **Single-leg odds (S, A, B)** — the marginals P(S), P(A), P(B).
- **2-way markets (S&A, S&B, A&B)** — each gives its two marginals **plus** the
  correlation between those two legs.

Resolution priority when a quantity is supplied by more than one input:
**marginals:** single → 2-way market → 8-way.&nbsp;&nbsp;
**correlations:** 8-way → dedicated 2-way market → assume independent.
If the 8-way is provided it drives the final value (it carries real 3rd-order structure);
otherwise the Gaussian copula over the resolved marginals + correlations is used.
""")

# Devigging method selection
devig_method = st.radio(
    "Select devigging method:",
    ["Proportional", "Power/Shin"],
    help="Applied to every market (8-way and each 2-way). Proportional: scales all probabilities equally. Power/Shin: exponential adjustment, often better for correlated outcomes."
)

st.header("Step 1 — 8-Way Market (optional)")
st.caption("The 8 corners of S∩A∩B, labeled (S, A, B). Leave all at 0 to skip.")

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

eight_odds = (o111, o110, o101, o100, o011, o010, o001, o000)

st.header("Step 2 — Authoritative Single-Leg Odds (optional)")
st.caption("Sharpest source of each marginal. Leave at 0 to skip.")

oS = st.number_input("Odds(S)", value=0, step=1)
oA = st.number_input("Odds(A)", value=0, step=1)
oB = st.number_input("Odds(B)", value=0, step=1)

st.header("Step 3 — 2-Way Markets (optional)")
st.write("""
Enter the **4 corners** of any 2-way market you have. Corner labels use **O = leg hits**,
**U = leg misses** (e.g. for S&A, *OO* = both hit, *OU* = S hits / A misses). Each market
contributes the correlation between its two legs (and, if needed, their marginals).
Leave a market's 4 boxes at 0 to skip it.
""")

st.subheader("S & A market")
sa_cols = st.columns(4)
oSA_OO = sa_cols[0].number_input("Odds(S hit, A hit)",   value=0, step=1, key="sa_oo")
oSA_OU = sa_cols[1].number_input("Odds(S hit, A miss)",  value=0, step=1, key="sa_ou")
oSA_UO = sa_cols[2].number_input("Odds(S miss, A hit)",  value=0, step=1, key="sa_uo")
oSA_UU = sa_cols[3].number_input("Odds(S miss, A miss)", value=0, step=1, key="sa_uu")

st.subheader("S & B market")
sb_cols = st.columns(4)
oSB_OO = sb_cols[0].number_input("Odds(S hit, B hit)",   value=0, step=1, key="sb_oo")
oSB_OU = sb_cols[1].number_input("Odds(S hit, B miss)",  value=0, step=1, key="sb_ou")
oSB_UO = sb_cols[2].number_input("Odds(S miss, B hit)",  value=0, step=1, key="sb_uo")
oSB_UU = sb_cols[3].number_input("Odds(S miss, B miss)", value=0, step=1, key="sb_uu")

st.subheader("A & B market")
ab_cols = st.columns(4)
oAB_OO = ab_cols[0].number_input("Odds(A hit, B hit)",   value=0, step=1, key="ab_oo")
oAB_OU = ab_cols[1].number_input("Odds(A hit, B miss)",  value=0, step=1, key="ab_ou")
oAB_UO = ab_cols[2].number_input("Odds(A miss, B hit)",  value=0, step=1, key="ab_uo")
oAB_UU = ab_cols[3].number_input("Odds(A miss, B miss)", value=0, step=1, key="ab_uu")

sa_odds = (oSA_OO, oSA_OU, oSA_UO, oSA_UU)
sb_odds = (oSB_OO, oSB_OU, oSB_UO, oSB_UU)
ab_odds = (oAB_OO, oAB_OU, oAB_UO, oAB_UU)

if st.button("Compute 3-Leg Fair Value"):
    result = compute_sgp(eight_odds, oS, oA, oB, sa_odds, sb_odds, ab_odds, devig_method)

    if not result.ok:
        st.error(result.error)
        st.stop()

    # ---- Which inputs were used ------------------------------------------------
    st.subheader("🧾 Inputs Detected")
    if result.inputs_detected:
        for item in result.inputs_detected:
            st.write(f"- {item}")
    else:
        st.write("- (none)")

    # ---- Over-definition notes -------------------------------------------------
    if result.overdefined:
        st.subheader("♻️ Over-Defined Inputs")
        st.caption("These quantities were supplied by more than one input. The priority "
                   "rules above decide which value is used; large spreads mean your sources disagree.")
        for note in result.overdefined:
            st.write(f"- {note}")

    if result.warnings:
        st.warning("⚠️ Warnings:")
        for w in result.warnings:
            st.write(f"- {w}")

    # ---- Resolved marginals ----------------------------------------------------
    st.subheader("📐 Resolved Marginals")
    marg_df = {
        "Leg": ["S", "A", "B"],
        "P(leg)": [f"{result.marginals[k][0]:.6f}" for k in ("S", "A", "B")],
        "American": [f"{prob_to_american(result.marginals[k][0])}" for k in ("S", "A", "B")],
        "Source": [result.marginals[k][1] for k in ("S", "A", "B")],
    }
    st.table(marg_df)

    # ---- Resolved correlations + matrix ---------------------------------------
    st.subheader("📊 Resolved Correlation Structure")
    corr_src_df = {
        "Pair": ["S↔A", "S↔B", "A↔B"],
        "ρ": [f"{result.correlations[k][0]:.4f}" for k in ("SA", "SB", "AB")],
        "Source": [result.correlations[k][1] for k in ("SA", "SB", "AB")],
    }
    st.table(corr_src_df)
    cm = result.correlation_matrix
    st.write("**Correlation matrix:**")
    st.table({
        "": ["S", "A", "B"],
        "S": [f"{cm[0,0]:.4f}", f"{cm[1,0]:.4f}", f"{cm[2,0]:.4f}"],
        "A": [f"{cm[0,1]:.4f}", f"{cm[1,1]:.4f}", f"{cm[2,1]:.4f}"],
        "B": [f"{cm[0,2]:.4f}", f"{cm[1,2]:.4f}", f"{cm[2,2]:.4f}"],
    })

    # ---- Derived 2-way fair values (entered pairs only) -----------------------
    if result.derived_pairs:
        st.subheader("🧩 Derived 2-Way Fair Values")
        names = list(result.derived_pairs.keys())
        st.table({
            "Pair": names,
            "ρ": [f"{result.derived_pairs[n]['rho']:.4f}" for n in names],
            "Market P(both)": [f"{result.derived_pairs[n]['market_joint']:.6f}" for n in names],
            "Fair P(both)": [f"{result.derived_pairs[n]['fair_joint']:.6f}" for n in names],
            "Fair American": [f"{result.derived_pairs[n]['fair_american']}" for n in names],
            "Market juice": [f"{result.derived_pairs[n]['juice_pct']:.2f}%" for n in names],
        })
        st.caption(
            "Fair P(both) = P(x)·P(y) + ρ·√(P(x)(1−P(x)))·√(P(y)(1−P(y))), using the resolved "
            "marginals and each market's correlation."
        )

    # ---- Final answer ----------------------------------------------------------
    st.subheader("🔥 Final 3-Leg Fair Value")
    st.metric("P(S∩A∩B) Final", f"{result.pSAB_final:.6f}")
    st.metric("American Odds (Final)", f"{result.american_final}")
    st.caption(f"Method: {result.method_name}")

    # ---- All estimates for comparison -----------------------------------------
    st.subheader("📊 All Estimates")
    est_names = list(result.estimates.keys())
    st.table({
        "Method": est_names,
        "P(S∩A∩B)": [f"{result.estimates[n]:.6f}" for n in est_names],
        "American": [f"{prob_to_american(result.estimates[n])}" for n in est_names],
    })
