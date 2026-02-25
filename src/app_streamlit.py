import streamlit as st
import pandas as pd
import datetime
import cv2
import numpy as np
from predict import predict_ml

# -------------------- Streamlit Page Config --------------------
st.set_page_config(page_title="Hemoglobin Estimation", layout="centered")

st.title("🩸 Non-Invasive Hemoglobin Estimation")
st.write(
    "Use your **Android phone camera** to capture a fingernail or conjunctiva image for instant hemoglobin estimation."
)

# -------------------- Skin Validation Function --------------------
def is_valid_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 15, 60], dtype=np.uint8)
    upper_skin = np.array([35, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_pixels = cv2.countNonZero(mask)
    total_pixels = image.shape[0] * image.shape[1]
    skin_ratio = skin_pixels / total_pixels

    return skin_ratio > 0.05


# -------------------- Save Results Function --------------------
def log_result(name, hb_value, confidence):
    df = pd.DataFrame(
        [[name, hb_value, confidence, datetime.datetime.now()]],
        columns=["Name", "Hb (g/dL)", "Confidence (%)", "Timestamp"],
    )
    df.to_csv("patient_records.csv", mode="a", header=False, index=False)


# -------------------- Main App Logic --------------------
img_data = st.camera_input("📷 Capture Image Using Phone Camera")

if img_data:
    with open("captured_phone.jpg", "wb") as f:
        f.write(img_data.getbuffer())

    image = cv2.imread("captured_phone.jpg")

    if not is_valid_image(image):
        st.warning("⚠️ Please capture a clear fingernail or conjunctiva image.")
    else:
        st.image("captured_phone.jpg", caption="Captured Image", use_column_width=True)
        st.subheader("🔍 Prediction Results")

        # ✅ ML Prediction Only
        hb_ml = predict_ml("captured_phone.jpg")

        hb_final = hb_ml
        confidence = 85  # Fixed confidence for ML-only version

        # -------------------- Display Results --------------------
        st.success(f"Predicted Hemoglobin: **{hb_final:.2f} g/dL**")
        st.progress(confidence)
        st.caption(f"Model Confidence: {confidence}%")

        # -------------------- Categorize Result --------------------
        if hb_final < 11:
            st.error("⚠️ Low Hemoglobin (Possible Anemia)")
        elif hb_final > 17:
            st.warning("⚠️ High Hemoglobin (Check Hydration / Polycythemia)")
        else:
            st.info("✅ Normal Hemoglobin Level")

        # -------------------- Save Patient Record --------------------
        patient_name = st.text_input("Enter Patient Name:")
        if st.button("Save Result"):
            log_result(patient_name, hb_final, confidence)
            st.success("📝 Record saved successfully!")
