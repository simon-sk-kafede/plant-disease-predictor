import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import bcrypt
import os
import csv
import datetime
import pandas as pd
import base64
import re
import secrets

# --- Page config ---
st.set_page_config(page_title="Plant Health Diagnosis", page_icon="🍃", layout="wide")

USERS_FILE = "users.yaml"
LOG_FILE = "activity_log.csv"

# --- Validation Functions ---
def validate_email(email):
    """Validate email format using regex"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, email):
        return True, ""
    return False, "Please enter a valid email address (e.g., user@example.com)"

def validate_contact(contact):
    """Validate contact number format"""
    cleaned_contact = re.sub(r'[\s\-\(\)]', '', contact)
    pattern = r'^(\+?256|0)?[7]\d{8}$'
    if re.match(pattern, cleaned_contact):
        return True, ""
    return False, "Please enter a valid Uganda phone number (e.g., +256700000000 or 0700000000)"

def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    return True, ""

# --- Load User Data ---
def load_users():
    if not os.path.exists(USERS_FILE):
        return {"credentials": {"usernames": {}},
                "cookie": {"name": "authenticator_cookie", "key": "random_key", "expiry_days": 30}}
    with open(USERS_FILE) as file:
        return yaml.load(file, Loader=SafeLoader)

def save_users(config):
    with open(USERS_FILE, "w") as file:
        yaml.dump(config, file)

config = load_users()

# --- Activity Logger ---
def log_activity(username, action, details=""):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.datetime.now(), username, action, details])

# --- Helper for embedding images in HTML ---
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return ""

# --- Page Decorations ---
def add_page_decorations(enable_background=True):
    try:
        logo_img = get_base64_image("assets/logo.jpg")
        background_img = get_base64_image("assets/background.jpg")
        if enable_background:
            if not background_img:
                background_fallback = "https://images.unsplash.com/photo-1416879595882-3373a0480b5b?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80"
                background_style = f"background: url('{background_fallback}') no-repeat center center fixed; background-size: cover;"
            else:
                background_style = f"background: url('data:image/jpeg;base64,{background_img}') no-repeat center center fixed; background-size: cover;"
        else:
            background_style = "background: none !important;"
        st.markdown(
            f"""
            <style>
            .centered-column {{
                max-width: 380px;
                margin: 0 auto;
                padding: 2rem;
                background: rgba(255, 255, 255, 0.98);
                border-radius: 16px;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.3);
            }}
            .title {{
                text-align: center;
                color: #ffffff;
                font-size: 24px;
                font-weight: 700;
                margin-bottom: 1.2rem;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid #ffffff;
            }}
            .stTextInput>div>div>input {{
                border-radius: 10px;
                border: 1.5px solid #e0e0e0;
                padding: 10px 14px;
                font-size: 18px;
                transition: all 0.3s;
            }}
            .stTextInput>div>div>input:focus {{
                border-color: #2E8B57;
                box-shadow: 0 0 0 2px rgba(46, 139, 87, 0.2);
            }}
            .stButton>button {{
                width: 100%;
                border-radius: 10px;
                background: linear-gradient(135deg, #2E8B57, #3CB371);
                color: white;
                font-weight: 600;
                padding: 10px 20px;
                border: none;
                transition: all 0.3s;
                margin-top: 0.8rem;
                font-size: 14px;
            }}
            .stButton>button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(46, 139, 87, 0.4);
            }}
            .link-button {{
                background: transparent !important;
                color: #2E8B57 !important;
                border: 1.5px solid #2E8B57 !important;
                font-size: 14px;
            }}
            .link-button:hover {{
                background: rgba(46, 139, 87, 0.1) !important;
                transform: none !important;
                box-shadow: none !important;
            }}
            .logo-container {{
                text-align: center;
                margin-bottom: 1.2rem;
            }}
            .logo {{
                width: 80px;
                height: 80px;
                border-radius: 50%;
                object-fit: cover;
                border: 3px solid #e6f4ea;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }}
            .stApp {{
                {background_style}
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        return logo_img
    except Exception as e:
        st.warning(f"⚠️ Some decoration images are missing: {e}")
        return ""

# --- Forgot Password Page ---
def forgot_password_page():
    logo_img = add_page_decorations(enable_background=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if logo_img:
            st.markdown(f"<div class='logo-container'><img class='logo' src='data:image/jpeg;base64,{logo_img}'></div>", unsafe_allow_html=True)
        st.markdown("<h1 class='title'>Reset Password</h1>", unsafe_allow_html=True)

        # Initialize session state for reset process
        if "reset_code" not in st.session_state:
            st.session_state.reset_code = None
            st.session_state.reset_email = None
            st.session_state.reset_username = None

        # Step 1: Request email
        if not st.session_state.reset_code:
            email = st.text_input("Enter your email", placeholder="your.email@example.com")
            if st.button("Send Reset Code", use_container_width=True):
                email_valid, email_msg = validate_email(email)
                if not email_valid:
                    st.error(email_msg)
                else:
                    # Check if email exists in users.yaml
                    username = None
                    for u, data in config["credentials"]["usernames"].items():
                        if data["email"] == email:
                            username = u
                            break
                    if username:
                        # Generate a 6-digit reset code
                        st.session_state.reset_code = secrets.token_hex(3).upper()
                        st.session_state.reset_email = email
                        st.session_state.reset_username = username
                        st.success(f"Reset code generated: **{st.session_state.reset_code}** (For demo purposes, code is shown here. In production, it would be emailed.)")
                        log_activity(username, "Password Reset Requested", f"Email: {email}")
                    else:
                        st.error("No account found with this email.")
        # Step 2: Enter reset code and new password
        else:
            st.write(f"Code sent to {st.session_state.reset_email}. Enter the code below.")
            reset_code = st.text_input("Reset Code", placeholder="Enter the 6-character code")
            new_password = st.text_input("New Password", type="password", placeholder="Create a strong password")
            confirm_password = st.text_input("Confirm New Password", type="password", placeholder="Re-enter your password")
            
            # Validate password in real-time
            password_valid = False
            passwords_match = False
            if new_password:
                password_valid, password_msg = validate_password(new_password)
                if password_valid:
                    st.markdown(f'<div class="validation-success">✓ Password strength: Strong</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="validation-error">✗ {password_msg}</div>', unsafe_allow_html=True)
            if new_password and confirm_password:
                if new_password == confirm_password:
                    passwords_match = True
                    st.markdown(f'<div class="validation-success">✓ Passwords match</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="validation-error">✗ Passwords do not match</div>', unsafe_allow_html=True)
            
            if st.button("Reset Password", use_container_width=True):
                if reset_code != st.session_state.reset_code:
                    st.error("Invalid reset code.")
                elif not password_valid:
                    st.error("New password does not meet strength requirements.")
                elif not passwords_match:
                    st.error("Passwords do not match.")
                else:
                    # Update password in users.yaml
                    hashed_pw = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                    config["credentials"]["usernames"][st.session_state.reset_username]["password"] = hashed_pw
                    save_users(config)
                    st.success("🎉 Password reset successful! Please log in with your new password.")
                    log_activity(st.session_state.reset_username, "Password Reset", f"Email: {st.session_state.reset_email}")
                    # Clear reset session state
                    st.session_state.reset_code = None
                    st.session_state.reset_email = None
                    st.session_state.reset_username = None
                    st.session_state.page = "login"
                    st.rerun()  # Immediate redirect to login page

        if st.button("Back to Login", use_container_width=True, type="secondary"):
            st.session_state.page = "login"
            st.session_state.reset_code = None
            st.session_state.reset_email = None
            st.session_state.reset_username = None
            st.rerun()

# --- Registration Page ---
def registration_page():
    logo_img = add_page_decorations()
    
    if not st.session_state.get("authentication_status"):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if logo_img:
                st.markdown(f"<div class='logo-container'><img class='logo' src='data:image/jpeg;base64,{logo_img}'></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='logo-container'><div class='logo'>🌿</div></div>", unsafe_allow_html=True)
                
            st.markdown("<h1 class='title'>Create Account</h1>", unsafe_allow_html=True)
            
            reg_username = st.text_input("Username", placeholder="Enter your username")
            reg_name = st.text_input("Full Name", placeholder="Your full name")
            reg_email = st.text_input("Email", placeholder="your.email@example.com")
            reg_contact = st.text_input("Contact", placeholder="e.g. +256700000000 or 0700000000")
            reg_password = st.text_input("Password", type="password", placeholder="Create a strong password (min. 8 chars)")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
            
            email_valid = False
            contact_valid = False
            password_valid = False
            passwords_match = False
            
            if reg_email:
                email_valid, email_msg = validate_email(reg_email)
                if email_valid:
                    st.markdown(f'<div class="validation-success">✓ {email_msg}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="validation-error">✗ {email_msg}</div>', unsafe_allow_html=True)
            
            if reg_contact:
                contact_valid, contact_msg = validate_contact(reg_contact)
                if contact_valid:
                    st.markdown(f'<div class="validation-success">✓ {contact_msg}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="validation-error">✗ {contact_msg}</div>', unsafe_allow_html=True)
            
            if reg_password:
                password_valid, password_msg = validate_password(reg_password)
                if password_valid:
                    st.markdown(f'<div class="validation-success">✓ Password strength: Strong</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="validation-error">✗ {password_msg}</div>', unsafe_allow_html=True)
            
            if reg_password and confirm_password:
                if reg_password == confirm_password:
                    passwords_match = True
                    st.markdown(f'<div class="validation-success">✓ Passwords match</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="validation-error">✗ Passwords do not match</div>', unsafe_allow_html=True)
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                reg_btn = st.button("Register", use_container_width=True)
            with col_btn2:
                if st.button("Back to Login", use_container_width=True, type="secondary"):
                    st.session_state.page = "login"
                    st.rerun()

            if reg_btn:
                validation_errors = []
                if not reg_username:
                    validation_errors.append("Username is required")
                if not reg_name:
                    validation_errors.append("Full name is required")
                if not reg_email:
                    validation_errors.append("Email is required")
                elif not email_valid:
                    validation_errors.append("Please enter a valid email address")
                if not reg_contact:
                    validation_errors.append("Contact number is required")
                elif not contact_valid:
                    validation_errors.append("Please enter a valid contact number")
                if not reg_password:
                    validation_errors.append("Password is required")
                elif not password_valid:
                    validation_errors.append("Password does not meet strength requirements")
                if not confirm_password:
                    validation_errors.append("Please confirm your password")
                elif not passwords_match:
                    validation_errors.append("Passwords do not match")
                if reg_username in config["credentials"]["usernames"]:
                    validation_errors.append("Username already exists. Choose another.")
                
                if validation_errors:
                    for error in validation_errors:
                        st.error(error)
                else:
                    hashed_pw = bcrypt.hashpw(reg_password.encode(), bcrypt.gensalt()).decode()
                    config["credentials"]["usernames"][reg_username] = {
                        "name": reg_name,
                        "email": reg_email,
                        "contact": reg_contact,
                        "password": hashed_pw
                    }
                    save_users(config)
                    st.session_state.registration_success = True  # Flag for success message
                    log_activity(reg_username, "Registration", f"New user: {reg_email}")
                    st.session_state.page = "login"
                    st.rerun()  # Immediate redirect to login page

    else:
        main_app(st.session_state.get("username"), authenticator=None)

# --- Login Page ---
def login_page():
    logo_img = add_page_decorations(enable_background=True)
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
    if not st.session_state.get("authentication_status"):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if logo_img:
                st.markdown(f"<div class='logo-container'><img class='logo' src='data:image/jpeg;base64,{logo_img}'></div>", unsafe_allow_html=True)
            st.markdown("<h1 class='title'>Welcome Back</h1>", unsafe_allow_html=True)

            # Show registration success message if set
            if st.session_state.get("registration_success"):
                st.success("🎉 Registration successful! Please log in.")
                st.session_state.registration_success = False  # Clear flag

            authenticator.login(location='main')

            if st.session_state.get("authentication_status") is False:
                st.error('Username/password is incorrect')
            elif st.session_state.get("authentication_status") is None:
                st.warning('Please enter your username and password')

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("📝 Don't have an account? Register here", use_container_width=True, type="secondary"):
                    st.session_state.page = "register"
                    st.rerun()
            with col_btn2:
                if st.button("🔑 Forgot Password?", use_container_width=True, type="secondary"):
                    st.session_state.page = "forgot_password"
                    st.rerun()
    elif st.session_state.get("authentication_status"):
        main_app(st.session_state.get("username"), authenticator)

# --- Main App ---
def main_app(username, authenticator):
    add_page_decorations(enable_background=False)
    if authenticator:
        authenticator.logout('Logout', location='sidebar')
    st.sidebar.title(f'Welcome {st.session_state.get("name", "")}')

    if username == "admin":
        st.subheader("👥 Registered Users")
        users_data = config["credentials"]["usernames"]
        st.table([{"username": u, "name": d.get("name"), "email": d.get("email"), "contact": d.get("contact", "")} for u, d in users_data.items()])

        st.subheader("📊 User Activity Log")
        try:
            df = pd.read_csv(LOG_FILE, names=["Time", "User", "Action", "Details"])
            st.dataframe(df)
        except FileNotFoundError:
            st.info("No activity logged yet.")

    st.title("🌿 Crop Disease Diagnosis System")
    st.markdown("Upload a leaf image (BANANA, CASSAVA, MAIZE & COFFEE) and let the model predict its health status.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    threshold = st.slider("Confidence Threshold:", 0, 100, 50)

    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model('model/simon_model.keras')

    model = load_model()

    class_names = [
        'Banana_Bacterial_Wilt', 'Banana_Black_Sigatoka', 'Banana_Health',
        'Cassava_Health', 'Cassava_Mosaic', 'Cassava_brown_streak',
        'Coffe_Berry', 'Coffee_Health', 'Coffee_Leaf_Rust',
        'Maize_Lethal_Necrosis', 'Maize_Health', 'Maize_Streak'
    ]

    recommendations = {
        'Banana_Bacterial_Wilt': "Use clean planting materials. Remove and destroy infected plants.",
        'Banana_Black_Sigatoka': "Use resistant varieties. Apply fungicides and ensure proper spacing.",
        'Banana_Health': "Your plant is healthy. Keep monitoring.",
        'Cassava_Mosaic': "Use virus-free cuttings. Remove infected plants immediately.",
        'Cassava_brown_streak': "Avoid planting in infected areas. Use resistant varieties.",
        'Cassava_Health': "Your plant is healthy. Continue using clean planting materials.",
        'Coffe_Berry': "Harvest ripe berries promptly. Avoid overripe berries.",
        'Coffee_Leaf_Rust': "Use resistant coffee varieties. Apply recommended fungicides.",
        'Coffee_Health': "Your plant is healthy. Keep monitoring for disease.",
        'Maize_Lethal_Necrosis': "Avoid mixed cropping of maize and susceptible crops.",
        'Maize_Streak': "Plant resistant varieties. Control insect vectors.",
        'Maize_Health': "Your plant is healthy. Practice crop rotation."
    }

    def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
        try:
            grad_model = tf.keras.models.Model(
                [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
            )
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                if isinstance(predictions, list):
                    predictions = predictions[0]
                if isinstance(conv_outputs, list):
                    conv_outputs = conv_outputs[0]
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0]).numpy()
                class_channel = predictions[0, pred_index]
            grads = tape.gradient(class_channel, conv_outputs)
            if grads is None:
                raise ValueError("Gradients are None, check model and layer compatibility")
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            pooled_grads = tf.reshape(pooled_grads, (1, 1, -1))
            heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
            heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
            return heatmap.numpy(), pred_index
        except Exception as e:
            st.error(f"Grad-CAM computation error: {e}")
            raise

    def get_gradcam_image(image_pil, heatmap, alpha=0.4):
        import cv2
        img = np.array(image_pil)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        superimposed_img = heatmap * alpha + img
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        return img, heatmap, superimposed_img

    if uploaded_file is not None:
        try:
            image_original = Image.open(uploaded_file).convert("RGB")
            st.image(image_original, caption="Selected Image Preview", width=250)
        except Exception as e:
            st.error(f"Failed to open image: {e}")
        else:
            image_for_model = image_original.resize((224, 224))
            img_array = tf.keras.utils.img_to_array(image_for_model) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            if confidence >= threshold / 100:
                st.success(f"**Prediction: {predicted_class} ({confidence * 100:.2f}%)**")
                recommendation = recommendations.get(predicted_class, "No recommendation available.")
                st.subheader("✅ Prevention Recommendation")
                st.info(recommendation)
                log_activity(username, "Prediction", predicted_class)
            else:
                st.warning("Model is unsure. Try another image or lower the threshold.")

            st.subheader("Grad-CAM Visualization")
            try:
                heatmap, pred_index = get_gradcam_heatmap(model, img_array, last_conv_layer_name='Conv_1')
                original_img, heatmap_img, superimposed_img = get_gradcam_image(image_original, heatmap)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(original_img, caption="Original Image", use_container_width=True)
                with col2:
                    st.image(heatmap_img, caption="Grad-CAM Heatmap", use_container_width=True)
                with col3:
                    st.image(superimposed_img, caption="Superimposed", use_container_width=True)
                
                st.markdown("The heatmap highlights regions the model focused on for the prediction.")
                log_activity(username, "Grad-CAM Viewed", f"Class: {predicted_class}")
            except Exception as e:
                st.error(f"Failed to generate Grad-CAM: {e}")

            st.subheader("Confidence Scores for All Classes")
            chart_data = pd.DataFrame({k: [float(v)] for k, v in zip(class_names, prediction)})
            st.bar_chart(chart_data.T)

    st.markdown("---")
    if st.button("Show Model Performance"):
        try:
            with open('model/simon_history.pkl', 'rb') as file:
                history = pickle.load(file)
            st.subheader("Training & Validation Metrics")
            st.line_chart(pd.DataFrame({
                "Training Accuracy": history.get('accuracy', []),
                "Validation Accuracy": history.get('val_accuracy', [])
            }))
            st.line_chart(pd.DataFrame({
                "Training Loss": history.get('loss', []),
                "Validation Loss": history.get('val_loss', [])
            }))
            log_activity(username, "Viewed Model Performance")
        except FileNotFoundError:
            st.error("Model performance history file not found.")
        except Exception as e:
            st.error(f"Error loading performance history: {e}")

# --- Page Routing ---
if "page" not in st.session_state:
    st.session_state.page = "login"

if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "register":
    registration_page()
elif st.session_state.page == "forgot_password":
    forgot_password_page()