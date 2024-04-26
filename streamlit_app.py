import streamlit as st
import pandas as pd
import base64
import tensorflow as tf 
from tensorflow.keras.models import load_model 
import numpy as np
from PIL import Image, ImageOps
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load and preprocess the image
def load_and_preprocess_image(image_data, target_size=(224, 224)):
    image = ImageOps.fit(image_data, target_size, Image.LANCZOS)
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize
    return img_array

# Function to predict defects
def predict_defect(image, model):
    prediction = model.predict(image[np.newaxis, ...])
    return prediction

# Function to display results
def display_results(result, confidence):
    labels = ["Crack", "No Crack"]
    label = labels[result]
    if result == 0:  # Crack
        st.error("Crack Detected")
        st.error("The Crack present can be harmful and needs to be corrected as soon as possible.")
    else:  # No Crack
        st.success("No Crack Detected")
        st.success("There is no harmful crack present, and the structure is strong enough physically.")
    if confidence is not None:
        st.write(f"Confidence: {confidence}")
        # Plotting confidence ratio with Seaborn
        df = pd.DataFrame({
            'Class': labels,
            'Confidence': confidence.squeeze()
        })
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(6, 4))
        sns.barplot(x='Class', y='Confidence', data=df, palette='viridis', hue='Class', legend=False)
        plt.xlabel('Class')
        plt.ylabel('Confidence')
        plt.title('Confidence Ratio')
        st.pyplot(plt.gcf())  # Pass the current figure explicitly

# Function to generate report
def generate_report(images, results, user_data):
    report_data = {"Image": [], "Result": [], "User Data": []}
    for image, result, data in zip(images, results, user_data):
        report_data["Image"].append(image)
        report_data["Result"].append("Crack" if result == 0 else "No Crack")
        report_data["User Data"].append(data)
    report_df = pd.DataFrame(report_data)
    return report_df

# Function to send email with the report attached
def send_email(report_df):
    sender_email = "dwellgroup13@gmail.com"  # Enter your email address
    receiver_email = "deshmaneuttu311@gmail.com"  # Enter receiver email address
    password = "sefe viyz vqrm abgy"  # Enter your email password

    # Create message container
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Dwell-well Report"

    # Attach report CSV file to email
    csv_data = report_df.to_csv(index=False)
    part = MIMEBase('text', 'csv')
    part.set_payload(csv_data.encode())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename=report.csv")
    msg.attach(part)

    # Start SMTP session
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    # Login to SMTP server
    server.login(sender_email, password)
    # Send email
    server.sendmail(sender_email, receiver_email, msg.as_string())
    # Close SMTP session
    server.quit()

# Set page title and favicon
st.set_page_config(page_title="Dwell well", page_icon=":camera:", layout="wide")

# Background image
#background_image = Image.open("img.jpg")  
#st.image(background_image, use_column_width=True)

# Title and header
st.title("Welcome to the Dwell Well!")

# Logo
logo_image = Image.open("logo.png")
st.sidebar.image(logo_image, width=200)

# Navigation menu
menu_selection = st.sidebar.radio("Menu", ["Home", "About Us"])

# Handle menu selection
if menu_selection == "Home":
    # Collect user data
    st.sidebar.header("User Data")
    name = st.sidebar.text_input("Name", "")
    age = st.sidebar.number_input("Age", min_value=0, max_value=150, value=30, step=1)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    location = st.sidebar.text_input("Location", "")
    phone_number = st.sidebar.text_input("Phone Number", "", max_chars=10)
    aadhar_card_no = st.sidebar.text_input("Aadhar Card Number", "", max_chars=12)
    pan_card_no = st.sidebar.text_input("PAN Card Number", "", max_chars=10)

    # File uploader for multiple images
    files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

    # If files are uploaded
    if files:
        if name and age and gender and location and phone_number.isdigit() and len(phone_number) == 10 and len(aadhar_card_no) == 12 and len(pan_card_no) == 10:
            try:
                # Load the trained model
                model = load_model("my_model.h5")  # Replace "my_model.h5" with your model file path
                # Compile the model
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                
                # Lists to store processed images, results, and user data
                processed_images = []
                results = []
                confidences = []
                user_data = [(name, age, gender, location, phone_number, aadhar_card_no, pan_card_no)] * len(files)
                
                # Process each uploaded image
                for file in files:
                    # Open the image
                    image = Image.open(file)
                    # Preprocess the image
                    image_array = load_and_preprocess_image(image)
                    # Make prediction
                    prediction = predict_defect(image_array, model)
                    # Get result
                    result = np.argmax(prediction)
                    confidence = np.max(prediction)
                    # Append processed image, result, and confidence to lists
                    processed_images.append(image)
                    results.append(result)
                    confidences.append(confidence)
                
                # Display results for each image
                for image, result, confidence in zip(processed_images, results, confidences):
                    # Display results
                    display_results(result, confidence)
                    # Display uploaded image
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Generate report
                report_df = generate_report(files, results, user_data)
                
                # Save report to CSV file
                with st.spinner("Sending report via email..."):
                    # Send report via email
                    send_email(report_df)
                
                # Provide download link for the report
                st.success("Report sent successfully!")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please fill in all the required user data fields correctly.")

elif menu_selection == "About Us":
    st.subheader("About Us")
    st.write("Welcome to Dwell Well!")
    st.write("We are a team of passionate engineers dedicated to ensuring structural safety and integrity.")
    st.write("Our mission is to provide innovative solutions for detecting structural defects and promoting safety in construction.")
    st.write("Feel free to reach out to us for any inquiries or collaborations!")

# Footer
footer_html = """
<div class="stFooter">
    <p>Copyright © 2024 Dwell Well. All rights reserved.</p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
