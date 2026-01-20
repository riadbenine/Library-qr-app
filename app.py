import streamlit as st
import pandas as pd
import qrcode
from io import BytesIO
from datetime import datetime
import os
from PIL import Image
import numpy as np
import cv2
import re

# File paths
USERS_FILE = 'library_users.csv'
LOGS_FILE = 'library_logs.csv'

# Initialize CSV files
def init_files():
    if not os.path.exists(USERS_FILE):
        df = pd.DataFrame(columns=['qr_code', 'first_name', 'last_name', 'category', 'field', 'email', 'phone', 'created_at'])
        df.to_csv(USERS_FILE, index=False)
    
    if not os.path.exists(LOGS_FILE):
        df = pd.DataFrame(columns=['qr_code', 'first_name', 'last_name', 'category', 'action', 'timestamp'])
        df.to_csv(LOGS_FILE, index=False)

# Load data
def load_users():
    if os.path.exists(USERS_FILE):
        return pd.read_csv(USERS_FILE)
    return pd.DataFrame(columns=['qr_code', 'first_name', 'last_name', 'category', 'field', 'email', 'phone', 'created_at'])

def load_logs():
    if os.path.exists(LOGS_FILE):
        return pd.read_csv(LOGS_FILE)
    return pd.DataFrame(columns=['qr_code', 'first_name', 'last_name', 'category', 'action', 'timestamp'])

# Check if user already exists
def check_user_exists(email, phone):
    users_df = load_users()
    if email:
        if not users_df.empty and 'email' in users_df.columns:
            email_exists = not users_df[users_df['email'].str.lower() == email.lower()].empty
            if email_exists:
                return True, "Email already registered"
    if phone:
        if not users_df.empty and 'phone' in users_df.columns:
            phone_exists = not users_df['phone'] == phone
            if not users_df[users_df['phone'] == phone].empty:
                return True, "Phone number already registered"
    return False, ""

# Validate phone
def validate_phone(phone):
    if not phone:
        return False, "Phone number cannot be empty"
    if not re.match(r'^0\d{9}$', phone):
        return False, "Phone must be 10 digits starting with 0"
    return True, ""

# Validate email
def validate_email(email):
    if not email:
        return True, ""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False, "Invalid email format"
    return True, ""

# Generate QR code
def generate_qr(data):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

# Add user
def add_user(qr_code, first_name, last_name, category, field, email, phone):
    users_df = load_users()
    if qr_code in users_df['qr_code'].values:
        return False, "QR code already exists"
    new_user = pd.DataFrame([{
        'qr_code': qr_code,
        'first_name': first_name,
        'last_name': last_name,
        'category': category,
        'field': field,
        'email': email,
        'phone': phone,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }])
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    users_df.to_csv(USERS_FILE, index=False)
    return True, ""

# Log entry/exit
def log_action(qr_code, first_name, last_name, category, action):
    logs_df = load_logs()
    new_log = pd.DataFrame([{
        'qr_code': qr_code,
        'first_name': first_name,
        'last_name': last_name,
        'category': category,
        'action': action,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }])
    logs_df = pd.concat([logs_df, new_log], ignore_index=True)
    logs_df.to_csv(LOGS_FILE, index=False)

# Get user by QR
def get_user_by_qr(qr_code):
    users_df = load_users()
    user = users_df[users_df['qr_code'] == qr_code]
    if not user.empty:
        return user.iloc[0]
    return None

# Search users
def search_users(search_term):
    users_df = load_users()
    mask = (users_df['first_name'].str.contains(search_term, case=False, na=False) | 
            users_df['last_name'].str.contains(search_term, case=False, na=False) |
            users_df['qr_code'].str.contains(search_term, case=False, na=False) |
            users_df['phone'].str.contains(search_term, case=False, na=False) |
            users_df['email'].str.contains(search_term, case=False, na=False))
    return users_df[mask]

# Preprocess image
def preprocess_image(image):
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    processed_images = [gray]
    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    processed_images.append(adaptive)
    # Otsu threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(otsu)
    # CLAHE contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    processed_images.append(clahe.apply(gray))
    # Bilateral filter
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    processed_images.append(bilateral)
    # Morphology
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    processed_images.append(morph)
    return processed_images

# Decode QR using OpenCV
def decode_qr_opencv(image):
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(gray)
    if data:
        return data
    return None

# Enhanced decode
def decode_qr_enhanced(image):
    # Try original
    result = decode_qr_opencv(image)
    if result:
        return result
    # Try preprocessed
    for processed in preprocess_image(image):
        result = decode_qr_opencv(processed)
        if result:
            return result
    # Try rotations
    if isinstance(image, Image.Image):
        img = image
    else:
        img = Image.fromarray(image)
    for angle in [90,180,270]:
        rotated = img.rotate(angle, expand=True)
        rotated_array = np.array(rotated)
        result = decode_qr_opencv(rotated_array)
        if result:
            return result
    return None

# Main app
def main():
    st.set_page_config(page_title="Library Recognition System", layout="wide")
    init_files()
    st.title("📚 Library Recognition System")
    st.sidebar.title("📋 Menu")
    menu = st.sidebar.radio("", 
                            ["🎥 Scan QR Code", 
                             "➕ Register User", 
                             "👥 View Users", 
                             "📊 View Logs",
                             "🔍 Search User"],
                            label_visibility="collapsed")
    # Scan QR Code
    if menu == "🎥 Scan QR Code":
        st.header("Scan QR Code")
        tab1, tab2 = st.tabs(["📷 Use Camera", "📁 Upload Image"])
        with tab1:
            camera_image = st.camera_input("Point at QR code", label_visibility="collapsed")
            if camera_image:
                image = Image.open(camera_image)
                with st.spinner("🔍 Detecting QR code..."):
                    qr_data = decode_qr_enhanced(image)
                if qr_data:
                    user = get_user_by_qr(qr_data)
                    if user is not None:
                        st.success("✅ User Found!")
                        st.write(f"**Name:** {user['first_name']} {user['last_name']}")
                        st.write(f"**Category:** {user['category']}")
                        st.write(f"**Field:** {user['field']}")
                        st.write(f"**Email:** {user['email']}")
                        st.write(f"**Phone:** {user['phone']}")
                        st.markdown("---")
                        action = st.radio("Select Action:", ["Entry","Exit"], horizontal=True, key="cam_action")
                        if st.button("✅ Log Action", type="primary", key="cam_log"):
                            log_action(user['qr_code'], user['first_name'], user['last_name'], user['category'], action)
                            st.success(f"✅ {action} logged at {datetime.now().strftime('%H:%M:%S')}!")
                            st.balloons()
                    else:
                        st.error("❌ User not found")
                        st.info(f"QR Code detected: {qr_data}")
                else:
                    st.warning("⚠️ No QR code detected")
        with tab2:
            uploaded_file = st.file_uploader("Choose an image", type=['png','jpg','jpeg','webp','bmp'])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                if st.button("🔍 Scan Image", type="primary"):
                    with st.spinner("🔍 Analyzing image..."):
                        qr_data = decode_qr_enhanced(image)
                    if qr_data:
                        user = get_user_by_qr(qr_data)
                        if user is not None:
                            st.success("✅ User Found!")
                            st.write(f"**Name:** {user['first_name']} {user['last_name']}")
                            st.write(f"**Category:** {user['category']}")
                            st.write(f"**Field:** {user['field']}")
                            st.write(f"**Email:** {user['email']}")
                            st.write(f"**Phone:** {user['phone']}")
                            action = st.radio("Select Action:", ["Entry","Exit"], horizontal=True, key="upload_action")
                            if st.button("✅ Log Action", type="primary", key="upload_log"):
                                log_action(user['qr_code'], user['first_name'], user['last_name'], user['category'], action)
                                st.success(f"✅ {action} logged at {datetime.now().strftime('%H:%M:%S')}!")
                                st.balloons()
                        else:
                            st.error("❌ User not found")
                            st.info(f"QR Code detected: {qr_data}")
                    else:
                        st.error("❌ No QR code detected")

    # Register User
    elif menu == "➕ Register User":
        st.header("Register New User")
        with st.form("register_form"):
            col1, col2 = st.columns(2)
            with col1:
                first_name = st.text_input("First Name *")
                last_name = st.text_input("Last Name *")
                category = st.selectbox("Category *", ["Student","Teacher","Staff","Visitor","Other"])
                field = st.text_input("Field/Department")
            with col2:
                email = st.text_input("Email")
                phone = st.text_input("Phone *")
            submitted = st.form_submit_button("✅ Register User")
            if submitted:
                if not first_name or not last_name or not phone:
                    st.error("❌ First Name, Last Name, and Phone required")
                else:
                    phone_valid, phone_error = validate_phone(phone)
                    email_valid, email_error = validate_email(email)
                    if not phone_valid:
                        st.error(phone_error)
                    elif not email_valid:
                        st.error(email_error)
                    else:
                        exists, exists_error = check_user_exists(email, phone)
                        if exists:
                            st.error(exists_error)
                        else:
                            qr_code = f"LIB-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                            success, err_msg = add_user(qr_code, first_name, last_name, category, field or "", email or "", phone)
                            if success:
                                st.success("✅ User registered successfully!")
                                st.image(generate_qr(qr_code), width=200)
                            else:
                                st.error(err_msg)

    # View Users
    elif menu == "👥 View Users":
        st.header("Registered Users")
        df = load_users()
        if not df.empty:
            st.dataframe(df)
        else:
            st.info("No users registered yet.")

    # View Logs
    elif menu == "📊 View Logs":
        st.header("Entry/Exit Logs")
        df = load_logs()
        if not df.empty:
            st.dataframe(df)
        else:
            st.info("No logs available.")

    # Search User
    elif menu == "🔍 Search User":
        st.header("Search User")
        term = st.text_input("Enter name or QR code")
        if term:
            results = search_users(term)
            if not results.empty:
                st.dataframe(results)
            else:
                st.warning("No users found.")

if __name__ == "__main__":
    main()
