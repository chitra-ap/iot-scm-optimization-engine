import streamlit as st
import pandas as pd
import numpy as np

# --- 1. The TOPSIS Engine ---
class DynamicTOPSISModel:
    def __init__(self, excel_path='TOPSIS_Analysis.xlsx'):
        df_matrix = pd.read_excel(excel_path, sheet_name='1. Decision Matrix', skiprows=2)
        df_matrix.columns = df_matrix.iloc[0]
        self.decision_matrix = df_matrix.iloc[1:26].reset_index(drop=True)
        
        self.criteria = ['Cost', 'Reliability', 'Ease of implementation', 'Compatibility', 
                         'Real time information', 'Efficiency', 'Risk Reduction', 'Sustainability']
        
        self.is_benefit = [False, True, True, True, True, True, True, True]

    def predict_best_combination(self, user_weights):
        X = self.decision_matrix[self.criteria].astype(float).values
        norm_X = np.divide(X, np.sqrt((X**2).sum(axis=0)), out=np.zeros_like(X), where=np.sqrt((X**2).sum(axis=0))!=0)
        weighted_X = norm_X * user_weights
        
        ideal_best = np.zeros(len(self.criteria))
        ideal_worst = np.zeros(len(self.criteria))
        
        for i in range(len(self.criteria)):
            if self.is_benefit[i]:
                ideal_best[i] = np.max(weighted_X[:, i])
                ideal_worst[i] = np.min(weighted_X[:, i])
            else: 
                ideal_best[i] = np.min(weighted_X[:, i])
                ideal_worst[i] = np.max(weighted_X[:, i])
                
        S_plus = np.sqrt(((weighted_X - ideal_best)**2).sum(axis=1))
        S_minus = np.sqrt(((weighted_X - ideal_worst)**2).sum(axis=1))
        
        denominator = S_plus + S_minus
        C_star = np.divide(S_minus, denominator, out=np.zeros_like(S_minus), where=denominator!=0)
        
        results = self.decision_matrix[['IoT Tool', 'SCM Technique']].copy()
        results['Priority Score'] = C_star
        results['Integration'] = results['IoT Tool'] + " + " + results['SCM Technique']
        
        return results.sort_values(by='Priority Score', ascending=False)

# --- 2. The Enhanced Graphical Interface ---
st.set_page_config(page_title="IoT & SCM Optimizer", page_icon="✨", layout="wide")

# Custom CSS for stacked ranking boxes
st.markdown("""
    <style>
    .big-font { font-size:20px !important; font-weight: 500; color: #4F4F4F;}
    .rank-1-box { background-color: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50; margin-bottom: 15px;}
    .rank-2-box { background-color: #E3F2FD; padding: 15px; border-radius: 10px; border-left: 5px solid #2196F3; margin-bottom: 15px;}
    .rank-3-box { background-color: #FFF3E0; padding: 15px; border-radius: 10px; border-left: 5px solid #FF9800; margin-bottom: 25px;}
    </style>
""", unsafe_allow_html=True)

st.title("✨ Intelligent Decision Support System")
st.markdown("<p class='big-font'>Dynamic Prioritization of Integrated IoT and SCM Tools</p>", unsafe_allow_html=True)
st.markdown("---")

@st.cache_data 
def load_model():
    return DynamicTOPSISModel()

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading Excel file. Details: {e}")
    st.stop()

# --- Sidebar: Grouped and clean controls ---
st.sidebar.title("🎛️ Priority Controls")
st.sidebar.markdown("Define the importance of each metric for your current scenario.")

st.sidebar.markdown("### 💰 Financial & Risk")
w_cost = st.sidebar.slider("Cost", 0.0, 10.0, 5.0, 0.5)
w_risk = st.sidebar.slider("Risk Reduction", 0.0, 10.0, 5.0, 0.5)

st.sidebar.markdown("### 🚀 Performance")
w_eff = st.sidebar.slider("Efficiency", 0.0, 10.0, 5.0, 0.5)
w_rel = st.sidebar.slider("Reliability", 0.0, 10.0, 5.0, 0.5)
w_real = st.sidebar.slider("Real-time Information", 0.0, 10.0, 5.0, 0.5)

st.sidebar.markdown("### 🌍 Integration")
w_ease = st.sidebar.slider("Ease of Implementation", 0.0, 10.0, 5.0, 0.5)
w_comp = st.sidebar.slider("Compatibility", 0.0, 10.0, 5.0, 0.5)
w_sust = st.sidebar.slider("Sustainability", 0.0, 10.0, 5.0, 0.5)

weights_dict = {
    'Cost': w_cost, 'Reliability': w_rel, 'Ease of implementation': w_ease,
    'Compatibility': w_comp, 'Real time information': w_real, 'Efficiency': w_eff,
    'Risk Reduction': w_risk, 'Sustainability': w_sust
}

raw_weights = [weights_dict[c] for c in model.criteria]
total_weight = sum(raw_weights)
normalized_weights = [w / total_weight for w in raw_weights] if total_weight > 0 else [0.125] * 8

# --- Main Results Area ---
results_df = model.predict_best_combination(normalized_weights)
top_3 = results_df.head(3).reset_index(drop=True)

st.markdown("### 🏆 Top 3 Optimal Recommendations")

# Rank 1
st.markdown(f"""
<div class="rank-1-box">
    <h2 style="margin-top:0px; color:#2E7D32;">🥇 1. {top_3.loc[0, 'Integration']}</h2>
    <p style="margin-bottom:0px; font-size:18px;"><b>Match Score:</b> {top_3.loc[0, 'Priority Score']:.4f}</p>
</div>
""", unsafe_allow_html=True)

# Rank 2
st.markdown(f"""
<div class="rank-2-box">
    <h3 style="margin-top:0px; color:#1565C0;">🥈 2. {top_3.loc[1, 'Integration']}</h3>
    <p style="margin-bottom:0px; font-size:16px;"><b>Match Score:</b> {top_3.loc[1, 'Priority Score']:.4f}</p>
</div>
""", unsafe_allow_html=True)

# Rank 3
st.markdown(f"""
<div class="rank-3-box">
    <h3 style="margin-top:0px; color:#E65100;">🥉 3. {top_3.loc[2, 'Integration']}</h3>
    <p style="margin-bottom:0px; font-size:16px;"><b>Match Score:</b> {top_3.loc[2, 'Priority Score']:.4f}</p>
</div>
""", unsafe_allow_html=True)

# --- Clean Data Table ---
with st.expander("📄 View Complete 25-Item Ranking Data"):
    display_df = results_df[['IoT Tool', 'SCM Technique', 'Priority Score']].copy()
    display_df.index = np.arange(1, len(display_df) + 1)
    st.dataframe(
        display_df.style.background_gradient(cmap='Blues', subset=['Priority Score']).format({'Priority Score': '{:.4f}'}), 
        use_container_width=True
    )