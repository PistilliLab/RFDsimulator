import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Nonlinear RFD prediction model
# -----------------------------
def predict_rfd(RFD_peak, duration, k_RFD=0.8, a=0.25, d0=2):
    """Predict RFD after aerobic exercise using exponential decay model."""
    scaling = k_RFD * (1 - np.exp(-a * (duration - d0))) if duration > d0 else 0
    RFD_pred = RFD_peak * (1 - scaling)
    loss_percent = 100 * (1 - RFD_pred / RFD_peak)
    return RFD_pred, loss_percent


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="RFD Simulator", page_icon="🏋️", layout="centered")

st.title("Nonlinear RFD Prediction Model")

st.markdown("""
Use this tool to estimate **rate of force development (RFD)** decline after aerobic exercise.  
The model assumes an exponential decay in RFD with increasing exercise duration.
""")

# --- User Input Section ---
st.header("Model Input Parameters")

with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        RFD_peak = st.number_input("Peak RFD (N/s)", min_value=0.0, value=10000.0, step=100.0)
        a = st.number_input("Rate constant (a)", min_value=0.0, value=0.25, step=0.05)
    with col2:
        k_RFD = st.slider("Scaling factor (k_RFD)", 0.0, 1.0, 0.8, 0.05)
        d0 = st.number_input("Onset time (d₀, min)", min_value=0.0, value=2.0, step=0.5)
    
    duration = st.slider("Aerobic Exercise Duration (minutes)", 0.0, 60.0, 20.0, 1.0)

# --- Compute Prediction ---
RFD_pred, loss_percent = predict_rfd(RFD_peak, duration, k_RFD, a, d0)

# --- Results Display ---
st.header("Predicted Results")
st.write(f"**Baseline Peak RFD:** {RFD_peak:.1f} N/s")
st.write(f"**Exercise Duration:** {duration:.1f} min")
st.write(f"**Predicted RFD After Exercise:** {RFD_pred:.1f} N/s")
st.write(f"**Predicted % Loss:** {loss_percent:.1f}%")

# --- Plot ---
durations = np.linspace(0, 60, 200)
rfd_curve = [predict_rfd(RFD_peak, d, k_RFD, a, d0)[0] for d in durations]

fig, ax = plt.subplots()
ax.plot(durations, rfd_curve, label="Predicted RFD", color="royalblue")
ax.axvline(duration, color="gray", linestyle="--", label="Selected Duration")
ax.set_xlabel("Exercise Duration (min)")
ax.set_ylabel("RFD (N/s)")
ax.set_title("Predicted Force Development Decline")
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.caption(f"Model parameters: k_RFD = {k_RFD:.2f}, a = {a:.2f}, d₀ = {d0:.1f}")
