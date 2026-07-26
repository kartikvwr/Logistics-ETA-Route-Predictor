import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Logistics ETA Predictor",
    page_icon="🚚",
    layout="wide",
)

# Hide Streamlit menu/footer
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}
</style>
""", unsafe_allow_html=True)

model = joblib.load("models/eta_model.pkl")

st.title("🚚 Logistics ETA Prediction System")

st.markdown(
"""
Predict delivery time using an **XGBoost Machine Learning model**
trained on over **43,000 delivery records**.
"""
)

st.divider()

left, right = st.columns([2,1])

with left:

    c1,c2 = st.columns(2)

    with c1:
        age = st.slider("Agent Age",18,60,28)
        rating = st.slider("Agent Rating",1.0,5.0,4.8)
        distance = st.slider("Distance (km)",1.0,30.0,7.5)
        pickup_delay = st.slider("Pickup Delay (min)",0,30,10)

    with c2:
        weather = st.selectbox(
            "Weather",
            ["Sunny","Cloudy","Fog","Stormy","Sandstorms","Windy"]
        )

        traffic = st.selectbox(
            "Traffic",
            ["Low","Medium","High","Jam"]
        )

        vehicle = st.selectbox(
            "Vehicle",
            ["motorcycle","scooter","electric_scooter"]
        )

        area = st.selectbox(
            "Area",
            ["Urban","Semi-Urban","Metropolitian"]
        )

    category = st.selectbox(
        "Category",
        [
            "Food",
            "Clothing",
            "Electronics",
            "Sports",
            "Cosmetics",
            "Toys"
        ]
    )

    predict = st.button(
        "🚀 Predict ETA",
        use_container_width=True
    )

with right:

    st.subheader("Project Information")

    st.info(
"""
**Model:** XGBoost Regressor

**Dataset:** Amazon Delivery Dataset

**Training Samples:** 43,594

**R² Score:** 0.8146

**MAE:** 17.11 minutes
"""
    )

if predict:

    sample = pd.DataFrame([{

        "Agent_Age": age,
        "Agent_Rating": rating,
        "Weather": weather,
        "Traffic": traffic,
        "Vehicle": vehicle,
        "Area": area,
        "Category": category,
        "Distance_km": distance,
        "Day": 26,
        "Month": 7,
        "Weekday": 5,
        "Order_Hour": 18,
        "Pickup_Delay_Min": pickup_delay,

    }])

    eta = model.predict(sample)[0]

    st.divider()

    a,b,c = st.columns(3)

    a.metric(
        "Estimated ETA",
        f"{eta:.1f} min"
    )

    b.metric(
        "Distance",
        f"{distance:.1f} km"
    )

    c.metric(
        "Weather",
        weather)