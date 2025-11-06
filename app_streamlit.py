import streamlit as st
from predict import predict_ml, predict_cnn
import pandas as pd
import datetime

st.set_page_config(page_title="Hemoglobin Estimation", layout="centered")

st.title("🩸 Non-Invasive Hemoglobin Estimation")
st.write("Use your **Android phone camera** to capture fingernail or conjunctiva image for instant hemoglobin estimation.")

# Capture image via Android camera
img_data = st.camera_input("📷 Capture Image Using Phone Camera")

# Function to save results
def log_result(name, hb_value):
    df = pd.DataFrame([[name, hb_value, datetime.datetime.now()]],
                      columns=["Name", "Hb (g/dL)", "Timestamp"])
    df.to_csv("patient_records.csv", mode='a', header=False, index=False)

if img_data:
    with open("captured_phone.jpg", "wb") as f:
        f.write(img_data.getbuffer())
    st.image("captured_phone.jpg", caption="Captured Image", use_column_width=True)
    
    st.subheader("🔍 Prediction Results")
    hb_ml = predict_ml("captured_phone.jpg")
    hb_cnn = predict_cnn("captured_phone.jpg")
    hb_final = (hb_ml + hb_cnn) / 2

    st.success(f"Predicted Hemoglobin: **{hb_final:.2f} g/dL**")

    if hb_final < 11:
        st.error("⚠️ Low Hemoglobin (Possible Anemia)")
    elif hb_final > 17:
        st.warning("⚠️ High Hemoglobin (Check Hydration / Polycythemia)")
    else:
        st.info("✅ Normal Hemoglobin Level")

    # Save patient name
    patient_name = st.text_input("Enter Patient Name:")
    if st.button("Save Result"):
        log_result(patient_name, hb_final)
        st.success("Record saved successfully!")
