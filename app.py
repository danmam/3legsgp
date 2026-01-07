import streamlit as st
from dataclasses import dataclass

# -----------------------------
# Data Classes
# -----------------------------

@dataclass
class EightWay:
    p111: float
    p110: float
    p101: float
    p100: float
    p011: float
    p010: float
    p001: float
    p000: float

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


# -----------------------------
# Core Computation
# -----------------------------

def compute_three_leg_fair(eight: EightWay, fair: SA_SB_Fair) -> ThreeLegResult:
    # Extract from 8-way
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
        pAB_given_S_final=pAB_given_S_final
    )


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("📊 3‑Leg SGP Fair Value Calculator")
st.write("""
This tool extracts the **incremental 3‑way correlation** from your 8‑way devig,
then applies that correlation to your authoritative **SA** and **SB** fair values
to compute the final **3‑leg fair probability**.
""")

st.header("Step 1 — Enter 8‑Way Vig‑Free Probabilities")

cols = st.columns(4)
p111 = cols[0].number_input("P(1,1,1)", min_value=0.0, max_value=1.0, step=0.0001)
p110 = cols[1].number_input("P(1,1,0)", min_value=0.0, max_value=1.0, step=0.0001)
p101 = cols[2].number_input("P(1,0,1)", min_value=0.0, max_value=1.0, step=0.0001)
p100 = cols[3].number_input("P(1,0,0)", min_value=0.0, max_value=1.0, step=0.0001)

cols2 = st.columns(4)
p011 = cols2[0].number_input("P(0,1,1)", min_value=0.0, max_value=1.0, step=0.0001)
p010 = cols2[1].number_input("P(0,1,0)", min_value=0.0, max_value=1.0, step=0.0001)
p001 = cols2[2].number_input("P(0,0,1)", min_value=0.0, max_value=1.0, step=0.0001)
p000 = cols2[3].number_input("P(0,0,0)", min_value=0.0, max_value=1.0, step=0.0001)

eight = EightWay(p111, p110, p101, p100, p011, p010, p001, p000)

st.header("Step 2 — Enter Authoritative Fair Values (S, A, B, SA, SB)")

pS = st.number_input("P(S)", min_value=0.0, max_value=1.0, step=0.0001)
pA = st.number_input("P(A)", min_value=0.0, max_value=1.0, step=0.0001)
pB = st.number_input("P(B)", min_value=0.0, max_value=1.0, step=0.0001)
pSA = st.number_input("P(SA)", min_value=0.0, max_value=1.0, step=0.0001)
pSB = st.number_input("P(SB)", min_value=0.0, max_value=1.0, step=0.0001)

fair = SA_SB_Fair(pS, pA, pB, pSA, pSB)

if st.button("Compute 3‑Leg Fair Value"):
    result = compute_three_leg_fair(eight, fair)

    st.subheader("🔍 Incremental Correlation Extracted from 8‑Way")
    st.metric("Correlation Multiplier κ", f"{result.kappa:.6f}")

    st.subheader("📈 Conditional Probabilities (Using Your SA/SB)")
    st.write(f"P(A | S) = {result.pA_given_S:.6f}")
    st.write(f"P(B | S) = {result.pB_given_S:.6f}")

    st.subheader("📉 Independence Baseline")
    st.metric("P(SAB) under independence", f"{result.pSAB_indep:.6f}")

    st.subheader("🔥 Final 3‑Leg Fair Value (Adjusted by κ)")
    st.metric("P(SAB) Final", f"{result.pSAB_final:.6f}")

    st.subheader("🧮 Final Joint Conditional")
    st.write(f"P(A ∩ B | S) Final = {result.pAB_given_S_final:.6f}")
