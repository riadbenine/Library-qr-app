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
import traceback

# File paths
USERS_FILE = 'library_users.csv'
LOGS_FILE = 'library_logs.csv'

# Initialize CSV files
def init_files():
    """Initialize CSV files if they don't exist"""
    try:
        if not os.path.exists(USERS_FILE):
            df = pd.DataFrame(columns=['qr_code', 'first_name', 'last_name', 'category', 'field', 'email', 'phone', 'created_at'])
            df.to_csv(USERS_FILE, index=False)
        
        if not os.path.exists(LOGS_FILE):
            df = pd.DataFrame(columns=['qr_code', 'first_name', 'last_name', 'category', 'action', 'timestamp'])
            df.to_csv(LOGS_FILE, index=False)
        return True, ""
    except PermissionError:
        return False, "Permission denied. Cannot create/access files. Check file permissions."
    except Exception as e:
        return False, f"Error initializing files: {str(e)}"

# Load data with error handling
def load_users():
    """Load users from CSV with error handling"""
    try:
        if os.path.exists(USERS_FILE):
            df = pd.read_csv(USERS_FILE)
            # Ensure all required columns exist
            required_cols = ['qr_code', 'first_name', 'last_name', 'category', 'field', 'email', 'phone', 'created_at']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = ''
            return df
        return pd.DataFrame(columns=['qr_code', 'first_name', 'last_name', 'category', 'field', 'email', 'phone', 'created_at'])
    except pd.errors.EmptyDataError:
        st.warning("Users file is empty. Creating new structure.")
        return pd.DataFrame(columns=['qr_code', 'first_name', 'last_name', 'category', 'field', 'email', 'phone', 'created_at'])
    except Exception as e:
        st.error(f"Error loading users: {str(e)}")
        return pd.DataFrame(columns=['qr_code', 'first_name', 'last_name', 'category', 'field', 'email', 'phone', 'created_at'])

def load_logs():
    """Load logs from CSV with error handling"""
    try:
        if os.path.exists(LOGS_FILE):
            df = pd.read_csv(LOGS_FILE)
            # Ensure all required columns exist
            required_cols = ['qr_code', 'first_name', 'last_name', 'category', 'action', 'timestamp']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = ''
            return df
        return pd.DataFrame(columns=['qr_code', 'first_name', 'last_name', 'category', 'action', 'timestamp'])
    except pd.errors.EmptyDataError:
        st.warning("Logs file is empty. Creating new structure.")
        return pd.DataFrame(columns=['qr_code', 'first_name', 'last_name', 'category', 'action', 'timestamp'])
    except Exception as e:
        st.error(f"Error loading logs: {str(e)}")
        return pd.DataFrame(columns=['qr_code', 'first_name', 'last_name', 'category', 'action', 'timestamp'])

# Check if user already exists
def check_user_exists(email, phone):
    """Check if user exists by email or phone"""
    try:
        users_df = load_users()
        
        if users_df.empty:
            return False, ""
        
        # Check email
        if email and email.strip():
            if 'email' in users_df.columns:
                # Handle NaN values
                email_matches = users_df['email'].fillna('').str.lower() == email.lower()
                if email_matches.any():
                    return True, "Email already registered"
        
        # Check phone
        if phone and phone.strip():
            if 'phone' in users_df.columns:
                # Handle NaN values and convert to string
                phone_matches = users_df['phone'].fillna('').astype(str) == str(phone)
                if phone_matches.any():
                    return True, "Phone number already registered"
        
        return False, ""
    except Exception as e:
        st.error(f"Error checking user existence: {str(e)}")
        return False, ""

# Validate phone
def validate_phone(phone):
    """Validate phone number format"""
    try:
        if not phone or not phone.strip():
            return False, "Phone number cannot be empty"
        
        phone = phone.strip()
        
        if not re.match(r'^0\d{9}$', phone):
            return False, "Phone must be 10 digits starting with 0"
        
        return True, ""
    except Exception as e:
        return False, f"Error validating phone: {str(e)}"

# Validate email
def validate_email(email):
    """Validate email format"""
    try:
        if not email or not email.strip():
            return True, ""  # Email is optional
        
        email = email.strip()
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(pattern, email):
            return False, "Invalid email format"
        
        return True, ""
    except Exception as e:
        return False, f"Error validating email: {str(e)}"

# Generate QR code
def generate_qr(data):
    """Generate QR code image"""
    try:
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Error generating QR code: {str(e)}")
        return None

# Add user
def add_user(qr_code, first_name, last_name, category, field, email, phone):
    """Add new user to database"""
    try:
        users_df = load_users()
        
        # Check if QR code already exists
        if not users_df.empty and qr_code in users_df['qr_code'].values:
            return False, "QR code already exists"
        
        new_user = pd.DataFrame([{
            'qr_code': qr_code,
            'first_name': first_name.strip(),
            'last_name': last_name.strip(),
            'category': category,
            'field': field.strip() if field else "",
            'email': email.strip() if email else "",
            'phone': phone.strip(),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])
        
        users_df = pd.concat([users_df, new_user], ignore_index=True)
        users_df.to_csv(USERS_FILE, index=False)
        
        return True, ""
    except PermissionError:
        return False, "Permission denied. Cannot write to users file."
    except Exception as e:
        return False, f"Error adding user: {str(e)}"

# Log entry/exit
def log_action(qr_code, first_name, last_name, category, action):
    """Log user entry/exit action"""
    try:
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
        
        return True, ""
    except PermissionError:
        return False, "Permission denied. Cannot write to logs file."
    except Exception as e:
        return False, f"Error logging action: {str(e)}"

# Get user by QR
def get_user_by_qr(qr_code):
    """Retrieve user by QR code"""
    try:
        users_df = load_users()
        
        if users_df.empty:
            return None
        
        user = users_df[users_df['qr_code'] == qr_code]
        
        if not user.empty:
            return user.iloc[0]
        
        return None
    except Exception as e:
        st.error(f"Error retrieving user: {str(e)}")
        return None

# Search users
def search_users(search_term):
    """Search users by name, QR code, phone, or email"""
    try:
        users_df = load_users()
        
        if users_df.empty:
            return pd.DataFrame()
        
        search_term = search_term.strip()
        
        # Create mask for searching across multiple columns
        mask = (users_df['first_name'].fillna('').str.contains(search_term, case=False, na=False) | 
                users_df['last_name'].fillna('').str.contains(search_term, case=False, na=False) |
                users_df['qr_code'].fillna('').str.contains(search_term, case=False, na=False) |
                users_df['phone'].fillna('').astype(str).str.contains(search_term, case=False, na=False) |
                users_df['email'].fillna('').str.contains(search_term, case=False, na=False))
        
        return users_df[mask]
    except Exception as e:
        st.error(f"Error searching users: {str(e)}")
        return pd.DataFrame()

# Preprocess image
def preprocess_image(image):
    """Preprocess image for better QR detection"""
    try:
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
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return []

# Decode QR using OpenCV
def decode_qr_opencv(image):
    """Decode QR code using OpenCV"""
    try:
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
    except Exception as e:
        return None

# Enhanced decode
def decode_qr_enhanced(image):
    """Enhanced QR decoding with multiple strategies"""
    try:
        # Try original
        result = decode_qr_opencv(image)
        if result:
            return result
        
        # Try preprocessed versions
        preprocessed = preprocess_image(image)
        for processed in preprocessed:
            result = decode_qr_opencv(processed)
            if result:
                return result
        
        # Try rotations
        if isinstance(image, Image.Image):
            img = image
        else:
            img = Image.fromarray(image)
        
        for angle in [90, 180, 270]:
            rotated = img.rotate(angle, expand=True)
            rotated_array = np.array(rotated)
            result = decode_qr_opencv(rotated_array)
            if result:
                return result
        
        return None
    except Exception as e:
        st.error(f"Error decoding QR code: {str(e)}")
        return None

# Main app
def main():
    """Main application function"""
    try:
        st.set_page_config(page_title="Library Recognition System", layout="wide")
        
        # Initialize files
        init_success, init_error = init_files()
        if not init_success:
            st.error(f"❌ Initialization Error: {init_error}")
            st.stop()
        
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
                    try:
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
                                    log_success, log_error = log_action(user['qr_code'], user['first_name'], user['last_name'], user['category'], action)
                                    
                                    if log_success:
                                        st.success(f"✅ {action} logged at {datetime.now().strftime('%H:%M:%S')}!")
                                        st.balloons()
                                    else:
                                        st.error(f"❌ {log_error}")
                            else:
                                st.error("❌ User not found in database")
                                st.info(f"QR Code detected: {qr_data}")
                        else:
                            st.warning("⚠️ No QR code detected. Try adjusting lighting or camera angle.")
                    except Exception as e:
                        st.error(f"❌ Error processing camera image: {str(e)}")
            
            with tab2:
                uploaded_file = st.file_uploader("Choose an image", type=['png','jpg','jpeg','webp','bmp'])
                
                if uploaded_file:
                    try:
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
                                        log_success, log_error = log_action(user['qr_code'], user['first_name'], user['last_name'], user['category'], action)
                                        
                                        if log_success:
                                            st.success(f"✅ {action} logged at {datetime.now().strftime('%H:%M:%S')}!")
                                            st.balloons()
                                        else:
                                            st.error(f"❌ {log_error}")
                                else:
                                    st.error("❌ User not found in database")
                                    st.info(f"QR Code detected: {qr_data}")
                            else:
                                st.error("❌ No QR code detected in image. Ensure the QR code is clear and well-lit.")
                    except Exception as e:
                        st.error(f"❌ Error processing uploaded image: {str(e)}")

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
                    phone = st.text_input("Phone *", help="10 digits starting with 0")
                
                submitted = st.form_submit_button("✅ Register User", type="primary")
                
                if submitted:
                    try:
                        # Validate required fields
                        if not first_name or not first_name.strip():
                            st.error("❌ First Name is required")
                        elif not last_name or not last_name.strip():
                            st.error("❌ Last Name is required")
                        elif not phone or not phone.strip():
                            st.error("❌ Phone is required")
                        else:
                            # Validate phone
                            phone_valid, phone_error = validate_phone(phone)
                            
                            if not phone_valid:
                                st.error(f"❌ {phone_error}")
                            else:
                                # Validate email
                                email_valid, email_error = validate_email(email)
                                
                                if not email_valid:
                                    st.error(f"❌ {email_error}")
                                else:
                                    # Check if user exists
                                    exists, exists_error = check_user_exists(email, phone)
                                    
                                    if exists:
                                        st.error(f"❌ {exists_error}")
                                    else:
                                        # Generate QR code
                                        qr_code = f"LIB-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                                        
                                        # Add user
                                        success, err_msg = add_user(
                                            qr_code, 
                                            first_name, 
                                            last_name, 
                                            category, 
                                            field or "", 
                                            email or "", 
                                            phone
                                        )
                                        
                                        if success:
                                            st.success("✅ User registered successfully!")
                                            
                                            # Generate and display QR code
                                            qr_img = generate_qr(qr_code)
                                            if qr_img:
                                                st.image(qr_img, width=200, caption=f"QR Code: {qr_code}")
                                                st.download_button(
                                                    label="📥 Download QR Code",
                                                    data=qr_img,
                                                    file_name=f"{qr_code}.png",
                                                    mime="image/png"
                                                )
                                            else:
                                                st.warning("⚠️ User registered but QR code generation failed")
                                        else:
                                            st.error(f"❌ {err_msg}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error during registration: {str(e)}")

        # View Users
        elif menu == "👥 View Users":
            st.header("Registered Users")
            
            try:
                df = load_users()
                
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                    st.info(f"Total users: {len(df)}")
                else:
                    st.info("No users registered yet.")
            except Exception as e:
                st.error(f"❌ Error displaying users: {str(e)}")

        # View Logs
        elif menu == "📊 View Logs":
            st.header("Entry/Exit Logs")
            
            try:
                df = load_logs()
                
                if not df.empty:
                    # Add filters
                    col1, col2 = st.columns(2)
                    with col1:
                        action_filter = st.multiselect("Filter by Action", options=df['action'].unique())
                    with col2:
                        category_filter = st.multiselect("Filter by Category", options=df['category'].unique())
                    
                    # Apply filters
                    filtered_df = df.copy()
                    if action_filter:
                        filtered_df = filtered_df[filtered_df['action'].isin(action_filter)]
                    if category_filter:
                        filtered_df = filtered_df[filtered_df['category'].isin(category_filter)]
                    
                    st.dataframe(filtered_df, use_container_width=True)
                    st.info(f"Total logs: {len(filtered_df)} (of {len(df)})")
                else:
                    st.info("No logs available.")
            except Exception as e:
                st.error(f"❌ Error displaying logs: {str(e)}")

        # Search User
        elif menu == "🔍 Search User":
            st.header("Search User")
            
            try:
                term = st.text_input("Enter name, QR code, phone, or email")
                
                if term and term.strip():
                    results = search_users(term)
                    
                    if not results.empty:
                        st.success(f"✅ Found {len(results)} user(s)")
                        st.dataframe(results, use_container_width=True)
                    else:
                        st.warning("No users found matching your search.")
                elif term is not None and term.strip() == "":
                    st.info("Enter a search term to find users.")
            except Exception as e:
                st.error(f"❌ Error searching users: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ Critical Application Error: {str(e)}")
        st.error("Stack trace:")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
