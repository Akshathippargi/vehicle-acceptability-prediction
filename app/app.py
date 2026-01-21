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
    buying = st.selectbox("Buying Price", ["low", "med", "high", "vhigh"])
    doors = st.selectbox("Number of Doors", ["2", "3", "4", "5more"])
    lug_boot = st.selectbox("Luggage Boot Size", ["small", "med", "big"])

with col2:
    maint = st.selectbox("Maintenance Cost", ["low", "med", "high", "vhigh"])
    persons = st.selectbox("Passenger Capacity", ["2", "4", "more"])
    safety = st.selectbox("Safety Level", ["low", "med", "high"])

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
