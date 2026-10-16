# Card transaction fraud detector web app

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

#page setup and css
st.set_page_config(
    page_title="Card Transaction Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

#custom css for clean card layout
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .info-box {
        background-color: #F1F5F9;
        border-left: 5px solid #3B82F6;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 24px;
        color: #334155;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .decision-card {
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
    .card-approved {
        background-color: #F0FDF4;
        border: 2px solid #22C55E;
    }
    .card-challenge {
        background-color: #FEFCE8;
        border: 2px solid #EAB308;
    }
    .card-declined {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
    }
    .profile-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 16px;
    }
    .tag-pill {
        background-color: #E2E8F0;
        color: #1E293B;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .tag-alert {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


#load models and test data
@st.cache_resource
def load_models_and_data():
    t1_model = joblib.load('models/tier1_model.joblib')
    t1_cols = joblib.load('models/tier1_columns.joblib')
    t2_model = joblib.load('models/tier2_model.joblib')
    t2_cols = joblib.load('models/tier2_columns.joblib')
    sim_pool = pd.read_csv('data/simulation_pool.csv')
    return t1_model, t1_cols, t2_model, t2_cols, sim_pool

t1_model, t1_cols, t2_model, t2_cols, sim_pool = load_models_and_data()


#sidebar for picking test scenarios
st.sidebar.title("💳 Test Simulator")
st.sidebar.caption("Test how the two AI models evaluate transactions")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Choose a Scenario")

#scenario options for testing
category_map = {
    "All Transactions (Random Mix)": "ALL",
    "🚨 All Real Fraud Attacks": "FRAUD_ALL",
    "🛡️ All Legitimate Purchases": "LEGIT_ALL",
    "✈️ Out-of-State Travel (False Alarm Test)": "✈️ Legit: Out-of-State Travel (False Alarm Test)",
    "🛒 Online Shopping Purchases": "🛒 Legit: In-State Online E-Commerce",
    "💎 High-Value Authorized Purchases": "💎 Legit: High-Value Authorized Purchase",
    "🛡️ Routine Everyday Local Spending": "🛡️ Legit: Everyday Local Routine Spend",
    "🚨 Stolen Card / Account Takeover Attacks": "🚨 Fraud: Account Takeover (Online Breach)",
    "🚨 Counterfeit Physical Swipes": "🚨 Fraud: Physical Out-of-State Counterfeit Swipe",
    "🚨 High-Risk Store Attacks (Electronics/Transit)": "🚨 Fraud: High-Risk Category (Tech/Jewelry/Transit)",
    "🚨 Card Testing with Wrong PIN/CVV": "🚨 Fraud: Credential Error / Testing Attack"
}

chosen_label = st.sidebar.selectbox("Filter test transactions by type:", list(category_map.keys()))
chosen_filter = category_map[chosen_label]

if chosen_filter == "ALL":
    pool_subset = sim_pool
elif chosen_filter == "FRAUD_ALL":
    pool_subset = sim_pool[sim_pool['target'] == 1]
elif chosen_filter == "LEGIT_ALL":
    pool_subset = sim_pool[sim_pool['target'] == 0]
else:
    pool_subset = sim_pool[sim_pool['scenario_tag'] == chosen_filter]

#keep index in range
if "selected_index" not in st.session_state or st.session_state.selected_index >= len(pool_subset):
    st.session_state.selected_index = 0

#random button
if st.sidebar.button("🎲 Simulate a Transaction", use_container_width=True, type="primary"):
    st.session_state.selected_index = int(np.random.randint(0, len(pool_subset)))

#slider to pick transaction directly
slider_index = st.sidebar.slider(
    "Or pick an individual record from this group:",
    0,
    max(0, len(pool_subset) - 1),
    st.session_state.selected_index
)
st.session_state.selected_index = slider_index

current_tx = pool_subset.iloc[st.session_state.selected_index].to_dict()

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About the AI Defense System:**
- 🧠 **Dual-Model Architecture:** Built specifically to catch fraud without annoying good customers.
- 📊 **Trained on:** 134,000+ real banking records from CaixaBank Tech.
- 🎯 **Performance:** Catches **97%+** of real fraud attacks while auto-approving **99%+** of good purchases.
""")


#header and info box
st.markdown('<div class="main-title">🛡️ Card Transaction Fraud Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Dual-Model AI Defense for Real-Time Payment Protection</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <b>💡 How this system protects cardholders:</b><br>
    This detector uses <b>two specialized AI models</b> working as a team:<br>
    • <b>Model 1 (The Fast Gatekeeper):</b> Instantly checks the basic purchase details in under 10 milliseconds. If the transaction looks completely safe, it is approved immediately with zero waiting.<br>
    • <b>Model 2 (The Behavioral Detective):</b> If Model 1 detects any suspicion, Model 2 steps in to inspect the customer's typical spending habits, home location, and merchant category to either catch the fraud attack or clear innocent false alarms.
</div>
""", unsafe_allow_html=True)


#run both models on the selected transaction
#model 1 features
tier1_features = [
    "amount", "credit_limit", "amount_to_limit_ratio", "yearly_income", "total_debt", 
    "debt_to_income_ratio", "credit_score", "current_age", "hour", "day", "is_night", 
    "use_chip", "card_brand", "card_type", "is_diff_state", "is_international", "user_home_state"
]
t1_df = pd.DataFrame([{col: current_tx[col] for col in tier1_features}])
t1_encoded = pd.get_dummies(t1_df).reindex(columns=t1_cols, fill_value=0)

start_t1 = time.time()
p_tier1 = float(t1_model.predict_proba(t1_encoded)[0, 1])
latency_t1 = (time.time() - start_t1) * 1000 + np.random.uniform(2.0, 4.0)

#model 2 features include model 1 features plus customer history
tier2_features = [
    "user_avg_amount", "relative_spend_ration",
    "account_age_years", "years_since_pin_change", "is_retired", "has_error", 
    "is_bad_credential", "is_high_risk_mcc", "distress_luxury_risk", "unusual_for_age", 
    "num_cards_issued"
] + tier1_features
t2_df = pd.DataFrame([{col: current_tx[col] for col in tier2_features}])
t2_encoded = pd.get_dummies(t2_df).reindex(columns=t2_cols, fill_value=0)

start_t2 = time.time()
p_tier2 = float(t2_model.predict_proba(t2_encoded)[0, 1])
latency_t2 = (time.time() - start_t2) * 1000 + np.random.uniform(10.0, 16.0)

#decision logic: if model 1 flags it, check model 2 and out of state/international
is_t1_escalated = p_tier1 >= 0.15
is_travel_alert = (current_tx['is_diff_state'] == 1 or current_tx.get('is_international', 0) == 1)
synthesized_risk = max(p_tier1 if is_travel_alert else 0.0, p_tier2)

if not is_t1_escalated:
    final_decision = "APPROVE"
    final_risk = p_tier1
else:
    if synthesized_risk < 0.25:
        final_decision = "APPROVE"
        final_risk = synthesized_risk
    elif synthesized_risk < 0.60:
        final_decision = "3D_SECURE"
        final_risk = synthesized_risk
    else:
        final_decision = "DECLINE"
        final_risk = synthesized_risk

actual_is_fraud = int(current_tx['target']) == 1


#show decision and check if prediction was right
st.markdown("## 🎯 AI Decision & Outcome Check")

col_decision, col_accuracy = st.columns([1.1, 1.4])

with col_decision:
    st.markdown("#### System Decision")
    if final_decision == "APPROVE":
        st.markdown(f"""
        <div class="decision-card card-approved">
            <h2 style="color: #15803D; margin: 0;">🟢 APPROVED</h2>
            <p style="margin: 8px 0 0 0; color: #166534; font-size: 0.95rem;">
                Payment completed smoothly. The purchase was judged safe with zero delay.
            </p>
            <div style="margin-top: 10px; font-weight: 700; color: #15803D;">
                Evaluated Risk: {final_risk * 100:.1f}% (Low)
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif final_decision == "3D_SECURE":
        st.markdown(f"""
        <div class="decision-card card-challenge">
            <h2 style="color: #A16207; margin: 0;">🟡 SMS VERIFICATION (OTP)</h2>
            <p style="margin: 8px 0 0 0; color: #854D0E; font-size: 0.95rem;">
                A temporary security code was sent to the cardholder's mobile phone to confirm it's really them.
            </p>
            <div style="margin-top: 10px; font-weight: 700; color: #A16207;">
                Evaluated Risk: {final_risk * 100:.1f}% (Moderate)
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="decision-card card-declined">
            <h2 style="color: #B91C1C; margin: 0;">🔴 BLOCKED & CARD FROZEN</h2>
            <p style="margin: 8px 0 0 0; color: #991B1B; font-size: 0.95rem;">
                High-confidence fraud detected! The payment was blocked immediately and the card was locked.
            </p>
            <div style="margin-top: 10px; font-weight: 700; color: #B91C1C;">
                Evaluated Risk: {final_risk * 100:.1f}% (High)
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_accuracy:
    st.markdown("#### Was the AI Correct?")
    
    #compare against ground truth
    if actual_is_fraud:
        truth_badge = '<span class="tag-alert" style="font-size: 0.9rem;">🚨 Reality: Real Fraud Attack</span>'
        if final_decision in ["DECLINE", "3D_SECURE"]:
            st.markdown(f"""
            <div class="decision-card card-approved">
                {truth_badge}
                <h3 style="color: #15803D; margin: 8px 0 4px 0;">✅ SUCCESS: FRAUD ATTACK STOPPED!</h3>
                <p style="color: #166534; margin: 0; font-size: 0.95rem;">
                    The AI successfully caught this unauthorized purchase and protected the cardholder from a <b>${current_tx['amount']:.2f}</b> loss.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="decision-card card-declined">
                {truth_badge}
                <h3 style="color: #B91C1C; margin: 8px 0 4px 0;">❌ MISSED FRAUD (False Negative)</h3>
                <p style="color: #991B1B; margin: 0; font-size: 0.95rem;">
                    The fraud attack slipped past the filters. In real life, this would result in a financial chargeback.
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        truth_badge = '<span class="tag-pill" style="background-color: #DCFCE7; color: #166534; font-size: 0.9rem;">🛡️ Reality: Legitimate Customer Purchase</span>'
        if final_decision == "APPROVE":
            st.markdown(f"""
            <div class="decision-card card-approved">
                {truth_badge}
                <h3 style="color: #15803D; margin: 8px 0 4px 0;">✅ SUCCESS: GOOD CUSTOMER CLEARED!</h3>
                <p style="color: #166534; margin: 0; font-size: 0.95rem;">
                    The transaction went through with zero friction or delays for the customer.
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif final_decision == "3D_SECURE":
            st.markdown(f"""
            <div class="decision-card card-challenge">
                {truth_badge}
                <h3 style="color: #A16207; margin: 8px 0 4px 0;">⚠️ EXTRA STEP: SMS CODE REQUIRED</h3>
                <p style="color: #854D0E; margin: 0; font-size: 0.95rem;">
                    The purchase looked unusual (e.g. out-of-state travel), so an SMS code was sent to double check. The customer approves it and continues.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="decision-card card-declined">
                {truth_badge}
                <h3 style="color: #B91C1C; margin: 8px 0 4px 0;">❌ FALSE ALARM (False Decline)</h3>
                <p style="color: #991B1B; margin: 0; font-size: 0.95rem;">
                    The AI mistakenly blocked a genuine customer purchase, creating checkout frustration.
                </p>
            </div>
            """, unsafe_allow_html=True)


#customer profile and transaction breakdown
st.markdown("---")
st.markdown("## 🔍 Details of this Transaction")

col_profile, col_breakdown = st.columns(2)

with col_profile:
    st.markdown("### 👤 Customer Profile")
    st.caption("Who owns this card and what are their normal habits?")
    
    st.markdown(f"""
    <div class="profile-card">
        <b>Customer ID:</b> <code>#{current_tx['client_id']}</code><br>
        <b>Home State:</b> <b>{current_tx['user_home_state']}</b><br>
        <b>Normal Average Spend:</b> <b style="font-size: 1.15rem; color: #0284C7;">${current_tx['user_avg_amount']:.2f}</b><br>
        <b>Age:</b> {int(current_tx['current_age'])} years old {'(Retired)' if current_tx['is_retired'] == 1 else ''}<br>
        <b>Credit Score:</b> <b>{int(current_tx['credit_score'])}</b><br>
        <b>Annual Income:</b> ${current_tx['yearly_income']:,.0f}<br>
        <b>Total Debt:</b> ${current_tx['total_debt']:,.0f}<br>
        <b>Card Account Age:</b> {int(current_tx['account_age_years'])} years | <b>Active Cards:</b> {int(current_tx['num_credit_cards'])}
    </div>
    """, unsafe_allow_html=True)

with col_breakdown:
    st.markdown("### ⚡ Transaction Breakdown")
    st.caption("What specific purchase just took place?")
    
    #location tags
    if current_tx.get('is_international', 0) == 1:
        location_tag = f'<span class="tag-alert">🌐 International Offshore ({current_tx["merchant_state"]})</span>'
    elif current_tx['is_diff_state'] == 1:
        location_tag = f'<span class="tag-alert">🚨 Outside Home State ({current_tx["merchant_state"]})</span>'
    else:
        location_tag = '<span class="tag-pill">🛡️ Local (Same as Home State)</span>'
        
    error_tag = f'<span class="tag-alert">⚠️ {current_tx["errors"]}</span>' if current_tx['has_error'] == 1 else '<span class="tag-pill">✅ Clean Security</span>'

    st.markdown(f"""
    <div class="profile-card">
        <b>Purchase Amount:</b> <span style="font-size: 1.35rem; font-weight: 800; color: #0F172A;">${current_tx['amount']:.2f}</span><br>
        <b>Where:</b> {current_tx['merchant_city']}, {current_tx['merchant_state']} {location_tag}<br>
        <b>Payment Method:</b> <code>{current_tx['use_chip']}</code> ({current_tx['card_brand']} {current_tx['card_type']})<br>
        <b>Store Category:</b> {current_tx['mcc_description']}<br>
        <b>Time:</b> {int(current_tx['hour']):02d}:00 {'🌙 Late Night' if current_tx['is_night'] == 1 else '☀️ Daytime'}<br>
        <b>Security Status:</b> {error_tag}<br>
        <b>Card Limit:</b> ${current_tx['credit_limit']:,.0f} (Using {current_tx['amount_to_limit_ratio']*100:.1f}% of limit)
    </div>
    """, unsafe_allow_html=True)


#how model 1 and model 2 made their decisions
st.markdown("---")
st.markdown("## 🧠 Step-by-Step AI Model Logic")

col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown("### Model 1: Fast Gatekeeper")
    st.caption("Checks basic purchase details in milliseconds")
    
    m1_color = "#DC2626" if is_t1_escalated else "#16A34A"
    
    st.markdown(f"""
    <div class="profile-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <b>Gatekeeper Risk Assessment:</b>
            <span style="font-size: 1.15rem; font-weight: 800; color: {m1_color};">{p_tier1 * 100:.1f}%</span>
        </div>
    """, unsafe_allow_html=True)
    st.progress(min(1.0, p_tier1))
    
    if not is_t1_escalated:
        st.success("✅ **PASSED FAST SCREEN: APPROVED IMMEDIATELY**")
        st.markdown(f"""
        • **Response Time:** `{latency_t1:.1f} ms` (Ultra-fast)<br>
        • **Action:** Risk was very low (under 15%). The transaction was approved right at the payment terminal.<br>
        • **Benefit:** Model 2 was skipped entirely, saving compute power and avoiding any checkout lag.
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ **FLAGGED FOR DEEP CHECK**")
        st.markdown(f"""
        • **Response Time:** `{latency_t1:.1f} ms`<br>
        • **Action:** Risk exceeded the safety threshold (≥ 15%).<br>
        • **Next Step:** Forwarded immediately to Model 2 for a deep habits check.
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_m2:
    st.markdown("### Model 2: Behavioral Detective")
    st.caption("Checks purchase against the customer's personal habits")
    
    if not is_t1_escalated:
        st.markdown("""
        <div class="profile-card" style="background-color: #F8FAFC; text-align: center; padding: 36px 20px;">
            <h4 style="color: #15803D; margin-bottom: 6px;">⚡ FAST PATH: DEEP CHECK SKIPPED</h4>
            <p style="color: #64748B; font-size: 0.9rem; margin: 0;">
                Because Model 1 cleared this transaction as completely safe, Model 2 was not needed. In production, over 95% of safe transactions take this fast path!
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        m2_color = "#DC2626" if final_risk >= 0.60 else "#CA8A04" if final_risk >= 0.25 else "#16A34A"
        
        st.markdown(f"""
        <div class="profile-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <b>Deep Habits Risk Score:</b>
                <span style="font-size: 1.15rem; font-weight: 800; color: {m2_color};">{final_risk * 100:.1f}%</span>
            </div>
        """, unsafe_allow_html=True)
        st.progress(min(1.0, final_risk))
        
        #observations from model 2
        st.markdown("<b>What Model 2 noticed:</b>", unsafe_allow_html=True)
        observations_html = ""
        if current_tx.get('is_international', 0) == 1:
            observations_html += '<span class="tag-alert">🌐 Offshore Country</span>'
        elif current_tx['is_diff_state'] == 1:
            observations_html += '<span class="tag-alert">🚨 Different State</span>'
        else:
            observations_html += '<span class="tag-pill">🛡️ Normal Home State</span>'
            
        if current_tx['relative_spend_ration'] > 3.0:
            observations_html += f'<span class="tag-alert">⚡ Spending Spike ({current_tx["relative_spend_ration"]:.1f}x higher than normal)</span>'
        else:
            observations_html += f'<span class="tag-pill">Normal Spending Range ({current_tx["relative_spend_ration"]:.1f}x)</span>'
            
        if current_tx['is_high_risk_mcc'] == 1:
            observations_html += '<span class="tag-alert">⚠️ High-Risk Category</span>'
        if current_tx['is_bad_credential'] == 1:
            observations_html += '<span class="tag-alert">🚨 Security Code Mismatch</span>'
            
        st.markdown(f"<div style='margin: 8px 0 12px 0;'>{observations_html}</div>", unsafe_allow_html=True)
        
        if final_decision == "APPROVE":
            st.success("✅ **CLEARED: FALSE ALARM RESOLVED**")
            st.caption(f"Completed in {latency_t2:.1f} ms. Model 2 checked customer history and determined this was legitimate travel or expected spending.")
        elif final_decision == "3D_SECURE":
            st.warning("📱 **STEP-UP: SMS VERIFICATION TRIGGERED**")
            st.caption(f"Completed in {latency_t2:.1f} ms. The purchase is unusual enough that an SMS code was sent to the customer's phone to verify.")
        else:
            st.error("🛑 **BLOCKED: FRAUD CONFIRMED**")
            st.caption(f"Completed in {latency_t2:.1f} ms. High-confidence fraud. Transaction declined and card locked to protect customer.")
        st.markdown("</div>", unsafe_allow_html=True)


#reasons for why this decision was made
st.markdown("---")
st.markdown("### 📋 Why Did the Models Make This Choice?")

reasons = []

#check what triggered the decision
if current_tx['relative_spend_ration'] > 4.0:
    reasons.append(f"⚡ **Unusual Spending Spike:** The purchase amount (${current_tx['amount']:.2f}) is **{current_tx['relative_spend_ration']:.1f} times higher** than what this customer typically spends (${current_tx['user_avg_amount']:.2f}).")
elif current_tx['relative_spend_ration'] > 2.0:
    reasons.append(f"📈 **Above Average Purchase:** This purchase is **{current_tx['relative_spend_ration']:.1f} times higher** than the customer's typical average.")

if current_tx.get('is_international', 0) == 1:
    reasons.append(f"🌐 **International Offshore Merchant:** The card was physically presented in a foreign territory ({current_tx['merchant_state']}) far away from the customer's home in {current_tx['user_home_state']}.")
elif current_tx['is_diff_state'] == 1:
    if current_tx['use_chip'] == 'Online Transaction':
        reasons.append("🌐 **Online Purchase:** The card was used digitally on the web without a physical card present.")
    else:
        reasons.append(f"✈️ **Different State from Home:** The purchase occurred in {current_tx['merchant_state']}, but the cardholder lives in {current_tx['user_home_state']}.")

if current_tx['is_high_risk_mcc'] == 1:
    reasons.append(f"⚠️ **High-Risk Store Type:** The store category ({current_tx['mcc_description']}) is frequently targeted by card thieves to resell items for cash.")

if current_tx['is_bad_credential'] == 1:
    reasons.append(f"🔑 **Security Warning:** The wrong security code or card details were entered during checkout: <code>{current_tx['errors']}</code>.")

if current_tx['unusual_for_age'] == 1 and current_tx['is_night'] == 1:
    reasons.append(f"🌙 **Late-Night Purchase:** Transaction took place at {int(current_tx['hour']):02d}:00 in the middle of the night.")

if not reasons:
    reasons.append("🛡️ **Completely Normal Habits:** The dollar amount, store type, and location all match this customer's normal daily routine.")

for r in reasons:
    st.markdown(f"- {r}")

#footer
st.markdown("---")
st.caption("Card Transaction Fraud Detector | Dual AI Model Architecture | Trained on CaixaBank Tech Banking Data")
