import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle


import matplotlib.pyplot as plt
import pickle

# --- Load Authentication Configuration ---
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

authenticator.login(location='main')

if st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')

elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

elif st.session_state["authentication_status"]:
    authenticator.logout('Logout', location='sidebar')
    st.sidebar.title(f'Welcome {st.session_state["name"]}')
    st.title("🌿 PLANT HEALTH DIAGNOSIS")

    st.markdown("Upload a leaf image and let the model predict its health status.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    threshold = st.slider("Confidence Threshold:", 0, 100, 50)

    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model('model/simo_argumented_model.keras')

    model = load_model()

    class_names = [
        'Banana_Bacterial_Wilt',
        'Banana_Black_Sigatoka',
        'Banana_Health',
        'Cassava_Mosaic',
        'Cassava_brown_streak',
        'Cassava_Health',
        'Coffe_Berry',
        'Coffee_Leaf_Rust',
        'Coffee_Health',
        'Maize Lethal Necrosis',
        'Maize_Streak',
        'Maize_Health',

    ]
    recommendations = {
    'Banana_Bacterial_Wilt': "Use clean planting materials. Remove and destroy infected plants.",
    'Banana_Black_Sigatoka': "Use resistant varieties. Apply fungicides and ensure proper spacing.",
    'Banana_Health': "Your plant is healthy. Keep monitoring and maintain proper farming practices.",
    'Cassava_Mosaic': "Use virus-free cuttings. Remove infected plants immediately.",
    'Cassava_brown_streak': "Avoid planting in infected areas. Use resistant varieties.",
    'Cassava_Health': "Your plant is healthy. Continue using clean planting materials and good agronomic practices.",
    'Coffe_Berry': "Harvest ripe berries promptly. Avoid overripe berries on the tree.",
    'Coffee_Leaf_Rust': "Use resistant coffee varieties. Apply recommended fungicides regularly.",
    'Coffee_Health': "Your plant is healthy. Keep monitoring for any signs of disease and ensure proper care.",
    'Maize Lethal Necrosis': "Avoid mixed cropping of maize and susceptible crops. Use certified seeds.",
    'Maize_Streak': "Plant resistant varieties. Control insect vectors like leafhoppers.",
    'Maize_Health': "Your plant is healthy. Practice crop rotation and monitor regularly to prevent infections."
}

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # 🔥 Normalize and expand dimensions
        img_array = tf.keras.utils.img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        if confidence >= threshold / 100:
            st.success(f"**Prediction: {predicted_class} ({confidence * 100:.2f}%)**")
            # Show recommendation
            recommendation = recommendations.get(predicted_class, "No specific recommendation available.")
            st.subheader("✅ Prevention Recommendation")
            st.info(recommendation)
        else:
            st.warning("Model is unsure. Try another image or lower the threshold.")
        # Show bar chart of all class probabilities
        st.subheader("📊 Confidence Scores for All Classes")
        chart_data = {class_names[i]: float(prediction[i]) for i in range(len(class_names))}
        st.bar_chart(chart_data)

    st.markdown("---")
    if st.button("📈 Show Model Performance"):
        try:
            with open('model/history.pkl', 'rb') as file:
                history = pickle.load(file)

            acc = history['accuracy']
            val_acc = history['val_accuracy']
            loss = history['loss']
            val_loss = history['val_loss']
            epochs = range(1, len(acc) + 1)

            # Plot training accuracy & loss
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))

            ax[0].plot(epochs, acc, 'b', label='Training Accuracy')
            ax[0].plot(epochs, val_acc, 'r', label='Validation Accuracy')
            ax[0].set_title('Training vs Validation Accuracy')
            ax[0].legend()

            ax[1].plot(epochs, loss, 'b', label='Training Loss')
            ax[1].plot(epochs, val_loss, 'r', label='Validation Loss')
            ax[1].set_title('Training vs Validation Loss')
            ax[1].legend()

            st.pyplot(fig)
        except FileNotFoundError:
            st.error("Model performance history file (`model/history.pkl`) not found.")