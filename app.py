import streamlit as st
from dataclasses import dataclass

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
    american_final: int
    american_indep: int
    pA_given_S: float
    pB_given_S: float
    pAB_given_S: float
    correlation_metric: float
    warnings: list


# -----------------------------
# Core Computation
# -----------------------------

def compute_three_leg_fair(eight: EightWay, fair: FairOdds) -> ThreeLegResult:
    warnings = []
    
    # Validate 8-way probabilities sum to 1
    total = (eight.p111 + eight.p110 + eight.p101 + eight.p100 + 
             eight.p011 + eight.p010 + eight.p001 + eight.p000)
    if abs(total - 1.0) > 0.01:
        warnings.append(f"8-way probabilities sum to {total:.4f}, not 1.0")
    
    # Extract P(S∩A∩B) directly from 8-way distribution
    pSAB_from_8way = eight.p111
    
    # Calculate marginals from 8-way for comparison
    pS_8 = eight.p111 + eight.p110 + eight.p101 + eight.p100
    pA_8 = eight.p111 + eight.p110 + eight.p011 + eight.p010
    pB_8 = eight.p111 + eight.p101 + eight.p011 + eight.p001
    pSA_8 = eight.p111 + eight.p110
    pSB_8 = eight.p111 + eight.p101
    
    # Check consistency between fair odds and 8-way
    if abs(pS_8 - fair.pS) > 0.05:
        warnings.append(f"P(S) mismatch: 8-way={pS_8:.4f}, fair={fair.pS:.4f}")
    if abs(pSA_8 - fair.pSA) > 0.05:
        warnings.append(f"P(S∩A) mismatch: 8-way={pSA_8:.4f}, fair={fair.pSA:.4f}")
    if abs(pSB_8 - fair.pSB) > 0.05:
        warnings.append(f"P(S∩B) mismatch: 8-way={pSB_8:.4f}, fair={fair.pSB:.4f}")
    
    # Use authoritative fair odds for calculation
    pA_given_S = fair.pSA / fair.pS if fair.pS > 0 else 0
    pB_given_S = fair.pSB / fair.pS if fair.pS > 0 else 0
    
    # Calculate conditional correlation from 8-way
    pAB_given_S_8way = eight.p111 / pS_8 if pS_8 > 0 else 0
    pAB_given_S_indep = (pSA_8 / pS_8) * (pSB_8 / pS_8) if pS_8 > 0 else 0
    
    # Correlation metric: how much does P(A∩B|S) deviate from independence?
    if pAB_given_S_indep > 0:
        correlation_metric = pAB_given_S_8way / pAB_given_S_indep
    else:
        correlation_metric = 1.0
        warnings.append("Cannot calculate correlation metric (division by zero)")
    
    # Apply correlation structure to authoritative odds
    # P(A∩B|S) = correlation_metric × P(A|S) × P(B|S)
    pAB_given_S_final = correlation_metric * pA_given_S * pB_given_S
    
    # Validate bounds
    max_pAB_given_S = min(pA_given_S, pB_given_S)
    if pAB_given_S_final > max_pAB_given_S:
        warnings.append(f"P(A∩B|S)={pAB_given_S_final:.4f} exceeds max={max_pAB_given_S:.4f}, capping")
        pAB_given_S_final = max_pAB_given_S
    
    # Final 3-leg probability
    pSAB_final = fair.pS * pAB_given_S_final
    
    # True independence baseline (using marginal probabilities)
    pSAB_indep = fair.pS * fair.pA * fair.pB
    
    return ThreeLegResult(
        pSAB_final=pSAB_final,
        pSAB_indep=pSAB_indep,
        american_final=prob_to_american(pSAB_final),
        american_indep=prob_to_american(pSAB_indep),
        pA_given_S=pA_given_S,
        pB_given_S=pB_given_S,
        pAB_given_S=pAB_given_S_final,
        correlation_metric=correlation_metric,
        warnings=warnings
    )


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("📊 3-Leg SGP Fair Value Calculator (Corrected)")
st.write("""
**8-way inputs are labeled as (S, A, B) in that exact order.**  
Example:  
- (1,1,1) means **S=1, A=1, B=1**  
- (1,0,1) means **S=1, A=0, B=1**

This calculator extracts the correlation structure from 8-way odds and applies it to authoritative 2-leg parlay odds.
""")

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
    result = compute_three_leg_fair(eight, fair)

    if result.warnings:
        st.warning("⚠️ Warnings:")
        for w in result.warnings:
            st.write(f"- {w}")

    st.subheader("📊 Correlation Analysis")
    st.metric("Correlation Multiplier", f"{result.correlation_metric:.6f}")
    st.caption("Ratio of actual P(A∩B|S) to independent P(A∩B|S) from 8-way data")

    st.subheader("📈 Conditional Probabilities")
    st.write(f"P(A | S) = {result.pA_given_S:.6f}")
    st.write(f"P(B | S) = {result.pB_given_S:.6f}")
    st.write(f"P(A ∩ B | S) = {result.pAB_given_S:.6f}")

    st.subheader("📉 True Independence Baseline")
    st.metric("P(S∩A∩B) under full independence", f"{result.pSAB_indep:.6f}")
    st.metric("American Odds (Independence)", f"{result.american_indep}")
    st.caption("= P(S) × P(A) × P(B)")

    st.subheader("🔥 Final 3-Leg Fair Value")
    st.metric("P(S∩A∩B) Final", f"{result.pSAB_final:.6f}")
    st.metric("American Odds (Final)", f"{result.american_final}")
    st.caption("= P(S) × P(A∩B|S), where P(A∩B|S) is adjusted by correlation from 8-way data")
