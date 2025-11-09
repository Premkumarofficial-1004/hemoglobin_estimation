import streamlit as st
import pandas as pd
import datetime
import cv2
import numpy as np
from predict import predict_ml, predict_cnn

# -------------------- Streamlit Page Config --------------------
st.set_page_config(page_title="Hemoglobin Estimation", layout="centered")

st.title("🩸 Non-Invasive Hemoglobin Estimation")
st.write("Use your **Android phone camera** to capture a fingernail or conjunctiva image for instant hemoglobin estimation.")

# -------------------- Skin Validation Function --------------------
def is_valid_image(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a wide skin-color range (for all skin tones)
    lower_skin = np.array([0, 15, 60], dtype=np.uint8)
    upper_skin = np.array([35, 255, 255], dtype=np.uint8)

    # Create mask and count skin-colored pixels
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_pixels = cv2.countNonZero(mask)
    total_pixels = image.shape[0] * image.shape[1]
    skin_ratio = skin_pixels / total_pixels

    # Accept only if ≥ 5% of the image looks like skin
    return skin_ratio > 0.05

# -------------------- Helper Function to Save Results --------------------
def log_result(name, hb_value):
    df = pd.DataFrame([[name, hb_value, datetime.datetime.now()]],
                      columns=["Name", "Hb (g/dL)", "Timestamp"])
    df.to_csv("patient_records.csv", mode='a', header=False, index=False)

# -------------------- Main App Logic --------------------
img_data = st.camera_input("📷 Capture Image Using Phone Camera")

if img_data:
    # Save image
    with open("captured_phone.jpg", "wb") as f:
        f.write(img_data.getbuffer())

    # Read saved image using OpenCV
    image = cv2.imread("captured_phone.jpg")

    # ✅ Validate before prediction
    if not is_valid_image(image):
        st.warning("⚠️ Please capture a clear fingernail or conjunctiva image.")
    else:
        st.image("captured_phone.jpg", caption="Captured Image", use_column_width=True)
        st.subheader("🔍 Prediction Results")

        # Predict using both ML and CNN models
        hb_ml = predict_ml("captured_phone.jpg")
        hb_cnn = predict_cnn("captured_phone.jpg")
        hb_final = (hb_ml + hb_cnn) / 2

        st.success(f"Predicted Hemoglobin: **{hb_final:.2f} g/dL**")

        # Categorize result
        if hb_final < 11:
            st.error("⚠️ Low Hemoglobin (Possible Anemia)")
        elif hb_final > 17:
            st.warning("⚠️ High Hemoglobin (Check Hydration / Polycythemia)")
        else:
            st.info("✅ Normal Hemoglobin Level")

        # Save patient name and record
        patient_name = st.text_input("Enter Patient Name:")
        if st.button("Save Result"):
            log_result(patient_name, hb_final)
            st.success("📝 Record saved successfully!")
