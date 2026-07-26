import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image

from waste_info import waste_info

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="EcoVision AI",
    page_icon="♻️",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main{
    background-color:#f5fff7;
}

.title{
    text-align:center;
    color:#2E8B57;
    font-size:45px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.result{
    background:#E8F5E9;
    padding:20px;
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load AI Model
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("waste_classifier.keras")
    return model

model = load_model()

# -----------------------------
# Class Names
# -----------------------------
class_names = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass"
]

# -----------------------------
# Header
# -----------------------------
st.markdown("<h1 class='title'>♻️ EcoVision AI</h1>", unsafe_allow_html=True)

st.markdown(
"<p class='subtitle'>Smart Waste Classification & Recycling Assistant</p>",
unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("About")

st.sidebar.info(
"""
EcoVision AI identifies waste using
Artificial Intelligence and recommends
the correct recycling method.

Model:
EfficientNetB0

Accuracy:
95.84%

Classes:
12
"""
)

st.sidebar.markdown("---")

st.sidebar.write("Supported Waste Categories")

for item in class_names:
    st.sidebar.write("•", item.title())

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📷 Upload a Waste Image",
    type=["jpg","jpeg","png"]
)
# -----------------------------
# Prediction Section
# -----------------------------
if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image")

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)

    # EfficientNet preprocessing
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)[0]

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = float(prediction[predicted_index] * 100)

    # Top 3 predictions
    top3 = np.argsort(prediction)[::-1][:3]

    with col2:

        st.subheader("🤖 AI Prediction")

        st.success(f"**{predicted_class.upper()}**")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(min(confidence / 100, 1.0))

        st.write("### Top 3 Predictions")

        for idx in top3:

            st.write(
                f"**{class_names[idx].title()}** : {prediction[idx]*100:.2f}%"
            )

        st.divider()
# -----------------------------
# Waste Information
# -----------------------------
        if predicted_class in waste_info:

            info = waste_info[predicted_class]

            st.subheader("♻️ Recycling Information")

            st.info(f"📖 **Description:** {info['description']}")

            colA, colB = st.columns(2)

            with colA:
                st.success(f"♻️ **Recyclable:** {info['recycle']}")

                st.write(f"🗑️ **Recommended Bin:** {info['bin']}")

            with colB:
                st.warning(f"⏳ **Decomposition Time:** {info['time']}")

            st.divider()

            st.subheader("🌍 Environmental Tip")

            tips = {
                "battery":"Never throw batteries into household waste. Use authorized battery collection centers.",
                "biological":"Convert food waste into compost whenever possible.",
                "brown-glass":"Separate glass by color before recycling.",
                "cardboard":"Keep cardboard clean and dry before recycling.",
                "clothes":"Donate wearable clothes and recycle damaged textiles.",
                "green-glass":"Glass can be recycled endlessly without losing quality.",
                "metal":"Rinse metal cans before placing them in recycling bins.",
                "paper":"Avoid mixing paper with wet waste.",
                "plastic":"Reduce single-use plastics and recycle clean plastic items.",
                "shoes":"Donate usable shoes or recycle through textile collection drives.",
                "trash":"Try to minimize landfill waste by separating recyclables.",
                "white-glass":"Recycle glass bottles and jars instead of discarding them."
            }

            st.success(tips[predicted_class])

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.markdown(
"""
<center>

### 🌱 EcoVision AI

Helping build a cleaner and greener future ♻️

</center>
""",
unsafe_allow_html=True
)
