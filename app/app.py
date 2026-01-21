import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Vehicle Acceptability Predictor",
    layout="centered"
)

# -------------------- LOAD MODEL --------------------
model = joblib.load("models/random_forest_model.pkl")

# -------------------- HEADER --------------------
st.title(" Vehicle Acceptability Prediction")
st.markdown(
    """
    This application predicts **vehicle acceptability** based on  
    **pricing, safety, capacity, and utility features** using a  
    **Random Forest Machine Learning model**.
    """
)

st.divider()

# -------------------- INPUT SECTION --------------------
st.subheader("Enter Vehicle Details")

col1, col2 = st.columns(2)

with col1:
    buying_price_inr = st.number_input(
        "Expected Vehicle Price (INR)",
        min_value=100000,
        max_value=10000000,
        step=50000,
        value=500000,
        help="Approximate on-road price"
    )

    if buying_price_inr < 400000:
        buying = "low"
    elif buying_price_inr < 800000:
        buying = "med"
    elif buying_price_inr < 1500000:
        buying = "high"
    else:
        buying = "vhigh"

    doors = st.selectbox(
        "Number of Doors",
        ["2", "3", "4", "5more"]
    )

    lug_boot = st.selectbox(
        "Luggage Boot Size",
        ["small", "med", "big"]
    )


with col2:
    maintenance_cost_inr = st.number_input(
        "Estimated Annual Maintenance Cost (INR)",
        min_value=5000,
        max_value=300000,
        step=5000,
        value=20000,
        help="Approximate yearly maintenance cost"
    )

    if maintenance_cost_inr < 15000:
        maint = "low"
    elif maintenance_cost_inr < 40000:
        maint = "med"
    elif maintenance_cost_inr < 80000:
        maint = "high"
    else:
        maint = "vhigh"

    persons = st.selectbox(
        "Passenger Capacity",
        ["2", "4", "more"]
    )

    safety_rating = st.slider(
        "Safety Rating (1–5)",
        min_value=1,
        max_value=5,
        value=3,
        help="Higher rating indicates better safety"
    )

    if safety_rating <= 2:
        safety = "low"
    elif safety_rating == 3:
        safety = "med"
    else:
        safety = "high"


# -------------------- DATA PREP --------------------
input_df = pd.DataFrame([{
    "buying": buying,
    "maint": maint,
    "doors": doors,
    "persons": persons,
    "lug_boot": lug_boot,
    "safety": safety
}])

input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(
    columns=model.feature_names_in_,
    fill_value=0
)

# -------------------- PREDICTION --------------------
st.divider()

if st.button(" Predict Acceptability", use_container_width=True):

    prediction = model.predict(input_encoded)[0]
    probabilities = model.predict_proba(input_encoded)[0]
    classes = model.classes_
st.divider()
st.subheader("Pricing Insight")

if buying_price_inr < 400000:
    st.write(
        "This vehicle is positioned in the **budget segment**, making it attractive for cost-sensitive buyers. "
        "Competitive pricing can significantly improve acceptability, even with basic features."
    )

elif buying_price_inr < 800000:
    st.write(
        "This vehicle falls in the **mid-range segment**. Buyers typically expect a balance between price, safety, "
        "and comfort features. Strong safety ratings can positively influence acceptance."
    )

elif buying_price_inr < 1500000:
    st.write(
        "This vehicle is priced in the **upper mid-range segment**. Customers in this segment are more selective and "
        "expect higher safety standards and better overall utility."
    )

else:
    st.write(
        "This vehicle is positioned in the **premium segment**. High pricing may limit mass-market adoption unless "
        "supported by excellent safety ratings, premium features, and strong brand value."
    )

    # -------------------- MAIN RESULT --------------------
    st.subheader(" Prediction Result")

    if prediction == "unacc":
        st.error(" **Unacceptable Vehicle**")
    elif prediction == "acc":
        st.warning(" **Acceptable Vehicle**")
    elif prediction == "good":
        st.success(" **Good Vehicle**")
    else:
        st.success(" **Very Good Vehicle**")

    # -------------------- PROBABILITY TABLE --------------------
    prob_df = pd.DataFrame({
        "Acceptability": classes,
        "Probability": probabilities
    }).sort_values(by="Probability", ascending=False)

    st.subheader("Prediction Confidence")
    st.dataframe(
        prob_df.style.format({"Probability": "{:.2%}"}),
        use_container_width=True
    )

    # -------------------- BAR CHART --------------------
    st.subheader("Probability Distribution")

    fig, ax = plt.subplots()
    ax.bar(prob_df["Acceptability"], prob_df["Probability"])
    ax.set_ylabel("Probability")
    ax.set_xlabel("Acceptability Class")
    ax.set_ylim(0, 1)

    st.pyplot(fig)

    # -------------------- BUSINESS INSIGHTS --------------------
    st.subheader("Business Insights")

    insights = []

    if safety == "high":
        insights.append("High safety significantly improves vehicle acceptability.")
    else:
        insights.append("Low or medium safety negatively impacts vehicle acceptance.")

    if buying in ["high", "vhigh"] and safety == "high":
        insights.append("Customers may still accept higher prices due to strong safety features.")

    if persons == "more":
        insights.append("Higher passenger capacity increases suitability for family use.")

    if lug_boot == "big":
        insights.append("Larger luggage space adds value for long-distance and family travel.")

    if prediction in ["good", "vgood"]:
        insights.append("This configuration is suitable for premium or urban markets.")
    else:
        insights.append("This configuration may require price or feature optimization.")

    for insight in insights:
        st.write("•", insight)

    # -------------------- CONFIDENCE SUMMARY --------------------
    top_class = prob_df.iloc[0]
    st.info(
        f"The model is **{top_class['Probability']:.1%} confident** "
        f"that the vehicle falls under **'{top_class['Acceptability']}'** category."
    )

# -------------------- FOOTER --------------------
st.divider()
st.caption(
    "End-to-End ML Project • Random Forest • Streamlit • Business-Oriented Insights"
)
