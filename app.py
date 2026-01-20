import streamlit as st
import pandas as pd
import qrcode
from io import BytesIO
from datetime import datetime
import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from pyzbar import pyzbar
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
    # Check by email if provided
    if email:
        if not users_df.empty and 'email' in users_df.columns:
            email_exists = not users_df[users_df['email'].str.lower() == email.lower()].empty
            if email_exists:
                return True, "Email already registered"
    
    # Check by phone if provided
    if phone:
        if not users_df.empty and 'phone' in users_df.columns:
            phone_exists = not users_df[users_df['phone'] == phone].empty
            if phone_exists:
                return True, "Phone number already registered"
    
    return False, ""

# Validate phone number format
def validate_phone(phone):
    if not phone:
        return False, "Phone number cannot be empty"
    
    # Check if phone starts with 0 and has 10 digits
    if not re.match(r'^0\d{9}$', phone):
        return False, "Phone must be 10 digits starting with 0 (e.g., 0123456789)"
    
    return True, ""

# Validate email format
def validate_email(email):
    if not email:
        return True, ""  # Email is optional
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
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

# Get user by QR code
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

# Enhanced QR code detection with multiple preprocessing techniques
def preprocess_image(image):
    """Apply various preprocessing techniques to improve QR detection"""
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Convert to grayscale if color
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    processed_images = []
    
    # Original grayscale
    processed_images.append(gray)
    
    # Adaptive thresholding
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    processed_images.append(adaptive)
    
    # Otsu's thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(otsu)
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    processed_images.append(enhanced)
    
    # Bilateral filter (reduces noise while keeping edges sharp)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    processed_images.append(bilateral)
    
    # Morphological operations to enhance QR structure
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    processed_images.append(morph)
    
    return processed_images

# Enhanced decode function using OpenCV's QRCodeDetector
def decode_qr_opencv(image):
    """Try to decode QR code using OpenCV's detector"""
    try:
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Use OpenCV's QRCodeDetector
        detector = cv2.QRCodeDetector()
        data, bbox, straight_qrcode = detector.detectAndDecode(gray)
        
        if data:
            return data
    except Exception as e:
        pass
    
    return None

# Enhanced decode function with multiple attempts
def decode_qr_enhanced(image):
    """Try multiple methods to decode QR code"""
    
    # Method 1: Try pyzbar on original image
    try:
        decoded = pyzbar.decode(image)
        if decoded:
            return decoded[0].data.decode('utf-8')
    except:
        pass
    
    # Method 2: Try OpenCV detector on original
    result = decode_qr_opencv(image)
    if result:
        return result
    
    # Method 3: Try with preprocessed images
    processed_images = preprocess_image(image)
    
    for processed in processed_images:
        # Try pyzbar
        try:
            decoded = pyzbar.decode(processed)
            if decoded:
                return decoded[0].data.decode('utf-8')
        except:
            pass
        
        # Try OpenCV detector
        result = decode_qr_opencv(processed)
        if result:
            return result
    
    # Method 4: Try with different rotations
    if isinstance(image, Image.Image):
        img = image
    else:
        img = Image.fromarray(image)
    
    for angle in [90, 180, 270]:
        rotated = img.rotate(angle, expand=True)
        rotated_array = np.array(rotated)
        
        # Try pyzbar
        try:
            decoded = pyzbar.decode(rotated_array)
            if decoded:
                return decoded[0].data.decode('utf-8')
        except:
            pass
        
        # Try OpenCV
        result = decode_qr_opencv(rotated_array)
        if result:
            return result
    
    return None

# Main app
def main():
    st.set_page_config(page_title="Library Recognition System", layout="wide")
    
    # Initialize files
    init_files()
    
    st.title("📚 Library Recognition System")
    
    # Sidebar navigation
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
        
        # Tabs for different scanning methods
        tab1, tab2 = st.tabs(["📷 Use Camera", "📁 Upload Image"])
        
        with tab1:
            st.subheader("📷 Scan with Camera")
            
            col_cam, col_info = st.columns([1, 2])
            
            with col_cam:
                camera_image = st.camera_input("Point at QR code", label_visibility="collapsed")
            
            with col_info:
                if camera_image:
                    image = Image.open(camera_image)
                    
                    with st.spinner("🔍 Detecting QR code..."):
                        # Enhanced QR detection
                        qr_data = decode_qr_enhanced(image)
                    
                    if qr_data:
                        user = get_user_by_qr(qr_data)
                        if user is not None:
                            st.success(f"✅ User Found!")
                            
                            # Display user info
                            st.write(f"**Name:** {user['first_name']} {user['last_name']}")
                            st.write(f"**Category:** {user['category']}")
                            st.write(f"**Field:** {user['field']}")
                            st.write(f"**Email:** {user['email']}")
                            st.write(f"**Phone:** {user['phone']}")
                            
                            st.markdown("---")
                            
                            # Action selection
                            action = st.radio("Select Action:", ["Entry", "Exit"], horizontal=True, key="camera_action")
                            
                            if st.button("✅ Log Action", type="primary", key="camera_log"):
                                log_action(user['qr_code'], user['first_name'], 
                                         user['last_name'], user['category'], action)
                                st.success(f"✅ {action} logged at {datetime.now().strftime('%H:%M:%S')}!")
                                st.balloons()
                        else:
                            st.error("❌ User not found")
                            st.info(f"QR Code detected: {qr_data}")
                    else:
                        st.warning("⚠️ No QR code detected. Try again with:")
                        st.markdown("""
                        - Better lighting
                        - Holding the QR code steady
                        - Getting closer to the QR code
                        - Making sure the QR code is clearly visible
                        """)
                else:
                    st.info("👈 Point your camera at the QR code and capture the image.")
        
        with tab2:
            st.subheader("📁 Upload QR Code Image")
            uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg', 'webp', 'bmp'])
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                
                col_img, col_result = st.columns([1, 2])
                
                with col_img:
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                
                with col_result:
                    if st.button("🔍 Scan Image", type="primary"):
                        with st.spinner("🔍 Analyzing image with advanced detection..."):
                            qr_data = decode_qr_enhanced(image)
                        
                        if qr_data:
                            user = get_user_by_qr(qr_data)
                            if user is not None:
                                st.success(f"✅ User Found!")
                                
                                # Display user info
                                st.markdown("---")
                                st.write(f"**Name:** {user['first_name']} {user['last_name']}")
                                st.write(f"**Category:** {user['category']}")
                                st.write(f"**Field:** {user['field']}")
                                st.write(f"**Email:** {user['email']}")
                                st.write(f"**Phone:** {user['phone']}")
                                st.markdown("---")
                                
                                # Action selection
                                action = st.radio("Select Action:", ["Entry", "Exit"], horizontal=True, key="upload_action")
                                
                                if st.button("✅ Log Action", type="primary", key="upload_log"):
                                    log_action(user['qr_code'], user['first_name'], 
                                             user['last_name'], user['category'], action)
                                    st.success(f"✅ {action} logged successfully at {datetime.now().strftime('%H:%M:%S')}!")
                                    st.balloons()
                            else:
                                st.error("❌ User not found in database")
                                st.info(f"QR Code detected: {qr_data}")
                        else:
                            st.error("❌ No QR code detected in image")
                            st.info("💡 Tips: Make sure the image is clear, well-lit, and the QR code is fully visible.")
        
    
    # Register User
    elif menu == "➕ Register User":
        st.header("Register New User")
        
        with st.form("register_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                first_name = st.text_input("First Name *", placeholder="Benine")
                last_name = st.text_input("Last Name *", placeholder="Riad")
                category = st.selectbox("Category *", 
                                       ["Student", "Teacher", "Staff", "Visitor", "Other"])
                field = st.text_input("Field/Department", placeholder="Computer Science, Mathematics, etc.")
            
            with col2:
                email = st.text_input("Email", placeholder="exemple@gmail.com")
                phone = st.text_input("Phone *", placeholder="0123456789")
                st.caption("Phone must be 10 digits starting with 0")
            
            submitted = st.form_submit_button("✅ Register User", type="primary")
            
            if submitted:
                # Validate required fields
                if not first_name or not last_name:
                    st.error("❌ First Name and Last Name are required")
                elif not phone:
                    st.error("❌ Phone number is required")
                else:
                    # Validate phone format
                    phone_valid, phone_error = validate_phone(phone)
                    if not phone_valid:
                        st.error(f"❌ {phone_error}")
                    else:
                        # Validate email format if provided
                        email_valid = True
                        if email:
                            email_valid, email_error = validate_email(email)
                            if not email_valid:
                                st.error(f"❌ {email_error}")
                        
                        if email_valid:
                            # Check if user already exists
                            exists, exists_error = check_user_exists(email, phone)
                            if exists:
                                st.error(f"❌ {exists_error}")
                            else:
                                # Generate unique QR code
                                qr_code = f"LIB-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                                
                                success, error_msg = add_user(qr_code, first_name, last_name, category, field or "", email or "", phone)
                                if success:
                                    st.success("✅ User registered successfully!")
                                    
                                    # Display user info
                                    st.markdown("---")
                                    st.subheader("User Details")
                                    col_a, col_b = st.columns([2, 1])
                                    
                                    with col_a:
                                        st.write(f"**Name:** {first_name} {last_name}")
                                        st.write(f"**Category:** {category}")
                                        st.write(f"**Field:** {field}")
                                        st.write(f"**QR Code:** {qr_code}")
                                        st.write(f"**Email:** {email if email else 'Not provided'}")
                                        st.write(f"**Phone:** {phone}")
                                    
                                    with col_b:
                                        # Generate and display QR code
                                        qr_img = generate_qr(qr_code)
                                        st.image(qr_img, caption="QR Code", width=200)
                                        
                                        # Download button
                                        st.download_button(
                                            label="📥 Download QR",
                                            data=qr_img,
                                            file_name=f"{qr_code}_{first_name}_{last_name}.png",
                                            mime="image/png"
                                        )
                                else:
                                    st.error(f"❌ Error: {error_msg}")
    
    # View Users
    elif menu == "👥 View Users":
        st.header("Registered Users")
        
        users_df = load_users()
        
        if not users_df.empty:
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Users", len(users_df))
            with col2:
                st.metric("Students", len(users_df[users_df['category'] == 'Student']))
            with col3:
                st.metric("Teachers", len(users_df[users_df['category'] == 'Teacher']))
            with col4:
                st.metric("Staff", len(users_df[users_df['category'] == 'Staff']))
            
            st.markdown("---")
            
            # Display table
            st.dataframe(users_df, use_container_width=True, hide_index=True)
            
            # Export button
            st.download_button(
                label="📥 Export Users to CSV",
                data=users_df.to_csv(index=False),
                file_name=f"library_users_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Show QR code for specific user
            st.markdown("---")
            st.subheader("Generate QR Code for User")
            selected_qr = st.selectbox("Select User", users_df['qr_code'].tolist(), 
                                       format_func=lambda x: f"{x} - {users_df[users_df['qr_code']==x]['first_name'].iloc[0]} {users_df[users_df['qr_code']==x]['last_name'].iloc[0]}")
            
            if st.button("Show QR Code"):
                user = users_df[users_df['qr_code'] == selected_qr].iloc[0]
                qr_img = generate_qr(selected_qr)
                
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.image(qr_img, width=250)
                with col_b:
                    st.write(f"**Name:** {user['first_name']} {user['last_name']}")
                    st.write(f"**Category:** {user['category']}")
                    st.write(f"**Field:** {user['field']}")
                    st.write(f"**QR Code:** {user['qr_code']}")
                    
                    st.download_button(
                        label="📥 Download QR Code",
                        data=qr_img,
                        file_name=f"{user['qr_code']}_{user['first_name']}_{user['last_name']}.png",
                        mime="image/png"
                    )
        else:
            st.info("📝 No users registered yet. Go to 'Register User' to add users.")
    
    # View Logs
    elif menu == "📊 View Logs":
        st.header("Entry/Exit Logs")
        
        logs_df = load_logs()
        
        if not logs_df.empty:
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(logs_df))
            with col2:
                entries = len(logs_df[logs_df['action'] == 'Entry'])
                st.metric("Total Entries", entries)
            with col3:
                exits = len(logs_df[logs_df['action'] == 'Exit'])
                st.metric("Total Exits", exits)
            
            st.markdown("---")
            
            # Filter options
            col_a, col_b = st.columns(2)
            with col_a:
                filter_action = st.selectbox("Filter by Action", ["All", "Entry", "Exit"])
            with col_b:
                filter_category = st.selectbox("Filter by Category", 
                                               ["All"] + list(logs_df['category'].unique()))
            
            # Apply filters
            filtered_df = logs_df.copy()
            if filter_action != "All":
                filtered_df = filtered_df[filtered_df['action'] == filter_action]
            if filter_category != "All":
                filtered_df = filtered_df[filtered_df['category'] == filter_category]
            
            # Sort by timestamp (most recent first)
            filtered_df = filtered_df.sort_values('timestamp', ascending=False)
            
            # Display table
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
            
            # Export buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Export All Logs",
                    data=logs_df.to_csv(index=False),
                    file_name=f"library_logs_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    label="📥 Export Filtered Logs",
                    data=filtered_df.to_csv(index=False),
                    file_name=f"library_logs_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("📝 No logs available yet. Start scanning QR codes to log entries/exits.")
    
    # Search User
    elif menu == "🔍 Search User":
        st.header("Search User")
        
        search_term = st.text_input("🔍 Enter name or QR code", placeholder="Search...")
        
        if search_term:
            results_df = search_users(search_term)
            
            if not results_df.empty:
                st.success(f"✅ Found {len(results_df)} user(s)")
                st.markdown("---")
                
                for idx, user in results_df.iterrows():
                    with st.expander(f"👤 {user['first_name']} {user['last_name']} - {user['category']}"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**QR Code:** {user['qr_code']}")
                            st.write(f"**Category:** {user['category']}")
                            st.write(f"**Field:** {user['field']}")
                            st.write(f"**Email:** {user['email']}")
                            st.write(f"**Phone:** {user['phone']}")
                            st.write(f"**Registered:** {user['created_at']}")
                        
                        with col2:
                            qr_img = generate_qr(user['qr_code'])
                            st.image(qr_img, width=150)
                            st.download_button(
                                label="📥 Download QR",
                                data=qr_img,
                                file_name=f"{user['qr_code']}.png",
                                mime="image/png",
                                key=f"download_{idx}"
                            )
            else:
                st.warning("❌ No users found matching your search")
        else:
            st.info("👆 Enter a search term above to find users")

if __name__ == "__main__":
    main()