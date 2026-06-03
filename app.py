import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter
import cv2
from skimage.filters import threshold_otsu
from fpdf import FPDF
import os
from datetime import datetime
import re


# Load the trained model
model = tf.keras.models.load_model('model/vgg16_model.h5')

# Define class labels
class_names = ['NORMAL LUNGS', 'NON-SMALL CELL LUNG CANCER']

# Streamlit app title and instructions
st.title("🩺 DSU Team-127 | Non-Small Cell Lung Cancer Detection using VGG-16 from CT Scan Images")
st.write("Upload a CT scan image to classify it using deep learning and image preprocessing.")

# Upload section
uploaded_file = st.file_uploader("📂 Choose a CT scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display original image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='🖼️ Original CT Scan Image', use_container_width=True)

    # --- Preprocessing Section ---
    st.subheader("🧪 Preprocessing Results")

    # 1. Noise Reduction
    img_denoised = img.filter(ImageFilter.MedianFilter(size=3))
    st.image(img_denoised, caption='🧹 After Noise Reduction (Median Filter)', use_container_width=True)

    # Convert to OpenCV image
    img_cv = np.array(img_denoised)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # 2. Edge Detection
    edges = cv2.Canny(gray, 100, 200)
    st.image(edges, caption='🧩 Edge Detection (Canny)', use_container_width=True, channels="GRAY")

    # 3. Watershed Preparation
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(thresh, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    unknown = cv2.subtract(np.uint8(sure_bg), np.uint8(sure_fg))
    st.image(unknown, caption='🧬 Watershed Unknown Region Map', use_container_width=True, channels="GRAY")

    # 4. Region-Based Segmentation using Otsu
    thresh_val = threshold_otsu(gray)
    binary_mask = gray > thresh_val
    st.image(binary_mask.astype(np.uint8) * 255, caption='🗺️ Region-Based Segmentation (Otsu)', use_container_width=True, channels="GRAY")


    # --- Prediction Section ---
    st.subheader("🔍 Prediction Result")
    img_resized = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)

    if predicted_index < len(class_names):
        prediction_result = class_names[predicted_index]
        st.success(f"Prediction: **{prediction_result}**")
    else:
        st.error("Prediction index out of range – check model output and class names.")


# --- Patient Details Form ---
st.subheader("👤 Patient Details")

with st.form("patient_form"):
    name = st.text_input("Full Name")
    age = st.number_input("Age", min_value=0)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    contact = st.text_input("Contact Number")
    email = st.text_input("Email Address")
    symptoms = st.text_area("Symptoms Description")
    submit_button = st.form_submit_button("Submit")

if submit_button:
    if name and age and gender and contact and email and symptoms:
        # Process and save patient details
        patient_info = (
            f"Name: {name}\n"
            f"Age: {age}\n"
            f"Gender: {gender}\n"
            f"Contact: {contact}\n"
            f"Email: {email}\n"
            f"Symptoms: {symptoms}\n"
            f"Prediction: {prediction_result}\n"
            + "-"*50 + "\n"
        )
        with open("patient_details.txt", "a", encoding="utf-8") as f:
            f.write(patient_info)

        # Save uploaded image locally
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if not os.path.exists("uploaded_images"):
            os.makedirs("uploaded_images")
        image_path = f"uploaded_images/ct_scan_{timestamp}.png"
        img.save(image_path)

        # Prepare PDF report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Print patient info on left half (width ~100mm)
        pdf.multi_cell(100, 10, patient_info)

        # Place the CT scan image on the right half of the page
        # Set x to 110 mm (right half), y aligned with top of text
        pdf.set_xy(110, pdf.get_y() - (len(patient_info.split('\n')) * 10))
        pdf.image(image_path, w=80)  # Width 80mm, height auto

        
        # Output PDF as bytes
        pdf_output = pdf.output(dest='S').encode('latin1')

        # Provide download buttons outside the form
        st.download_button(
            label="Download PDF Report",
            data=pdf_output,
            file_name="patient_report.pdf",
            mime='application/octet-stream'
        )

        with open("patient_details.txt", "r", encoding="utf-8") as f:
            patient_details_content = f.read()
        st.download_button(
            label="Download Patient Details",
            data=patient_details_content,
            file_name="patient_details.txt",
            mime='text/plain'
        )
    else:
        st.warning("⚠️ Please fill in all fields before submitting.")


# --- Contact Form ---
st.subheader("📬 Contact Us")

def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email)

with st.form("contact_form"):
    contact_name = st.text_input("Your Name")
    contact_email = st.text_input("Your Email")
    contact_message = st.text_area("Your Message")
    contact_submit = st.form_submit_button("Send")

    if contact_submit:
        if not contact_name or not contact_email or not contact_message:
            st.warning("⚠️ Please fill in all fields before submitting.")
        elif not validate_email(contact_email):
            st.error("⚠️ Please enter a valid email address.")
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            contact_info = (
                f"Timestamp: {timestamp}\n"
                f"Name: {contact_name}\n"
                f"Email: {contact_email}\n"
                f"Message: {contact_message}\n"
                + "-"*50 + "\n"
            )
            with open("message.txt", "a", encoding="utf-8") as f:
                f.write(contact_info)
            st.success("✅ Your message has been sent successfully!")

            # --- Display Overall Accuracy ---
st.markdown("## 📊 Model Evaluation Summary")

accuracy_displayed = False
try:
    with open("metrics/evaluation_metrics.txt", "r", encoding="utf-8") as f:
        for line in f:
            if "Validation Accuracy" in line:
                st.info(f"**Model Accuracy:** {line.strip().split(':')[-1].strip()}")
                accuracy_displayed = True
                break
except FileNotFoundError:
    st.warning("Model accuracy file not found. Please evaluate and generate metrics first.")

if not accuracy_displayed:
    st.info("Overall accuracy will appear here after model evaluation.")
