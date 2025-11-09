import streamlit as st
import pandas as pd
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from predict import predict_ml, predict_cnn

# -------------------- Streamlit Page Config --------------------
st.set_page_config(page_title="Hemoglobin Estimation", layout="centered")

st.title("🩸 Non-Invasive Hemoglobin Estimation")
st.write("Use your **Android phone camera** to capture a fingernail or conjunctiva image for instant hemoglobin estimation.")

# -------------------- Smart Image Validation --------------------
def is_valid_image(image):
    """Check if the captured image likely contains a fingernail or conjunctiva region."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Broad skin color range to handle all skin tones
    lower_skin = np.array([0, 15, 60], dtype=np.uint8)
    upper_skin = np.array([35, 255, 255], dtype=np.uint8)

    # Create mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = image.shape[:2]
    total_area = h * w
    valid_regions = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < (0.01 * total_area) or area > (0.25 * total_area):
            continue  # Ignore very small or very large regions

        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / ch if ch != 0 else 0

        # Fingernails/conjunctiva are roughly rectangular and near center
        if 0.5 < aspect < 2.5 and (w * 0.2 < x < w * 0.8) and (h * 0.2 < y < h * 0.8):
            valid_regions += 1

    # Accept only if a valid region exists
    return valid_regions > 0

# -------------------- Save Results Helper --------------------
def log_result(name, hb_value, confidence):
    df = pd.DataFrame([[name, hb_value, confidence, datetime.datetime.now()]],
                      columns=["Name", "Hb (g/dL)", "Confidence (%)", "Timestamp"])
    df.to_csv("patient_records.csv", mode='a', header=False, index=False)

# -------------------- Main App Logic --------------------
img_data = st.camera_input("📷 Capture Image Using Phone Camera")

if img_data:
    # Save the captured image
    with open("captured_phone.jpg", "wb") as f:
        f.write(img_data.getbuffer())

    # Read image using OpenCV
    image = cv2.imread("captured_phone.jpg")

    # Validate before prediction
    if not is_valid_image(image):
        st.warning("⚠️ Please capture a clear fingernail or conjunctiva image.")
    else:
        st.image("captured_phone.jpg", caption="Captured Image", use_column_width=True)
        st.subheader("🔍 Prediction Results")

        # Predict with both ML and CNN models
        hb_ml = predict_ml("captured_phone.jpg")
        hb_cnn = predict_cnn("captured_phone.jpg")
        hb_final = (hb_ml + hb_cnn) / 2

        # -------------------- Confidence Level Calculation --------------------
        hb_diff = abs(hb_ml - hb_cnn)
        confidence = max(0, 100 - (hb_diff * 8))  # Smaller difference = higher confidence
        confidence = min(confidence, 100)

        # Display predicted values
        st.success(f"Predicted Hemoglobin: **{hb_final:.2f} g/dL**")
        st.progress(int(confidence))
        st.caption(f"Model Confidence: {confidence:.1f}%")

        # -------------------- Bar Chart for Model Comparison --------------------
        fig, ax = plt.subplots()
        ax.bar(["ML Model", "CNN Model"], [hb_ml, hb_cnn], color=["#2196F3", "#E91E63"])
        ax.axhline(hb_final, color="green", linestyle="--", label="Average")
        ax.set_ylabel("Hemoglobin (g/dL)")
        ax.set_title("Model Prediction Comparison")
        ax.legend()
        st.pyplot(fig)

        # -------------------- Result Interpretation --------------------
        if hb_final < 11:
            st.error("⚠️ Low Hemoglobin (Possible Anemia)")
        elif hb_final > 17:
            st.warning("⚠️ High Hemoglobin (Check Hydration / Polycythemia)")
        else:
            st.info("✅ Normal Hemoglobin Level")

        # -------------------- Save Record --------------------
        patient_name = st.text_input("Enter Patient Name:")
        if st.button("Save Result"):
            log_result(patient_name, hb_final, confidence)
            st.success("📝 Record saved successfully!")
