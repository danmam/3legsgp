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
class SA_SB_Fair:
    pS: float
    pA: float
    pB: float
    pSA: float
    pSB: float

@dataclass
class ThreeLegResult:
    kappa: float
    pSAB_final: float
    pSAB_indep: float
    pA_given_S: float
    pB_given_S: float
    pAB_given_S_final: float
    american_final: float
    american_indep: float


# -----------------------------
# Core Computation
# -----------------------------

def compute_three_leg_fair(eight: EightWay, fair: SA_SB_Fair) -> ThreeLegResult:
    p111 = eight.p111
    p110 = eight.p110
    p101 = eight.p101
    p100 = eight.p100

    # 8-way marginals
    pS_8 = p111 + p110 + p101 + p100
    pSA_8 = p111 + p110
    pSB_8 = p111 + p101

    # Conditional probabilities from 8-way
    pA_given_S_8 = pSA_8 / pS_8
    pB_given_S_8 = pSB_8 / pS_8
    pAB_given_S_8 = p111 / pS_8

    # Incremental correlation multiplier
    kappa = pAB_given_S_8 / (pA_given_S_8 * pB_given_S_8)

    # Apply to authoritative SA/SB
    pA_given_S = fair.pSA / fair.pS
    pB_given_S = fair.pSB / fair.pS

    # Independence baseline
    pSAB_indep = fair.pS * pA_given_S * pB_given_S

    # Apply correlation multiplier
    pAB_given_S_final = kappa * pA_given_S * pB_given_S
    pSAB_final = fair.pS * pAB_given_S_final

    return ThreeLegResult(
        kappa=kappa,
        pSAB_final=pSAB_final,
        pSAB_indep=pSAB_indep,
        pA_given_S=pA_given_S,
        pB_given_S=pB_given_S,
        pAB_given_S_final=pAB_given_S_final,
        american_final=prob_to_american(pSAB_final),
        american_indep=prob_to_american(pSAB_indep)
    )


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("📊 3‑Leg SGP Fair Value Calculator (American Odds Input)")
st.write("""
**8‑way inputs are labeled as (S, A, B) in that exact order.**  
Example:  
- (1,1,1) means **S=1, A=1, B=1**  
- (1,0,1) means **S=1, A=0, B=1**
""")

st.header("Step 1 — Enter 8‑Way American Odds (S, A, B)")

cols = st.columns(4)
o111 = cols[0].number_input("Odds(S=1,A=1,B=1)", step=1)
o110 = cols[1].number_input("Odds(S=1,A=1,B=0)", step=1)
o101 = cols[2].number_input("Odds(S=1,A=0,B=1)", step=1)
o100 = cols[3].number_input("Odds(S=1,A=0,B=0)", step=1)

cols2 = st.columns(4)
o011 = cols2[0].number_input("Odds(S=0,A=1,B=1)", step=1)
o010 = cols2[1].number_input("Odds(S=0,A=1,B=0)", step=1)
o001 = cols2[2].number_input("Odds(S=0,A=0,B=1)", step=1)
o000 = cols2[3].number_input("Odds(S=0,A=0,B=0)", step=1)

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

oS = st.number_input("Odds(S)", step=1)
oA = st.number_input("Odds(A)", step=1)
oB = st.number_input("Odds(B)", step=1)
oSA = st.number_input("Odds(SA)", step=1)
oSB = st.number_input("Odds(SB)", step=1)

fair = SA_SB_Fair(
    american_to_prob(oS),
    american_to_prob(oA),
    american_to_prob(oB),
    american_to_prob(oSA),
    american_to_prob(oSB)
)

if st.button("Compute 3‑Leg Fair Value"):
    result = compute_three_leg_fair(eight, fair)

    st.subheader("🔍 Incremental Correlation Extracted from 8‑Way")
    st.metric("Correlation Multiplier κ", f"{result.kappa:.6f}")

    st.subheader("📈 Conditional Probabilities (Using Your SA/SB)")
    st.write(f"P(A | S) = {result.pA_given_S:.6f}")
    st.write(f"P(B | S) = {result.pB_given_S:.6f}")

    st.subheader("📉 Independence Baseline")
    st.metric("P(SAB) under independence", f"{result.pSAB_indep:.6f}")
    st.metric("American Odds (Independence)", f"{result.american_indep}")

    st.subheader("🔥 Final 3‑Leg Fair Value (Adjusted by κ)")
    st.metric("P(SAB) Final", f"{result.pSAB_final:.6f}")
    st.metric("American Odds (Final)", f"{result.american_final}")

    st.subheader("🧮 Final Joint Conditional")
    st.write(f"P(A ∩ B | S) Final = {result.pAB_given_S_final:.6f}")
