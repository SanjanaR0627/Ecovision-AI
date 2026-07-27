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

.stApp{
    background: linear-gradient(135deg,#0B1120,#111827,#1F2937);
    color:white;
}

[data-testid="stHeader"]{
    background: rgba(0,0,0,0);
}

[data-testid="stSidebar"]{
    background:#111827;
}

.block-container{
    padding-top:2rem;
}

.title{
    text-align:center;
    font-size:55px;
    font-weight:800;
    color:#00E676;
}

.subtitle{
    text-align:center;
    font-size:20px;
    color:#B0BEC5;
}

div[data-testid="stMetric"]{
    background:#1E293B;
    border-radius:15px;
    padding:15px;
    border:1px solid #334155;
}

.stSuccess{
    background:#00C85320;
    color:white;
    border-radius:12px;
}

.stInfo{
    background:#0288D120;
    color:white;
    border-radius:12px;
}

.stWarning{
    background:#FFA00020;
    color:white;
    border-radius:12px;
}

.stButton>button{
    width:100%;
    background:#00C853;
    color:white;
    border:none;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
}

.stFileUploader{
    background:#1E293B;
    padding:20px;
    border-radius:15px;
}

img{
    border-radius:20px;
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
        st.markdown(f"""
<div style="
background:#00C853;
padding:18px;
border-radius:15px;
text-align:center;
font-size:30px;
font-weight:bold;
color:white;">
♻️ {predicted_class.upper()}
</div>
""", unsafe_allow_html=True)
        

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.markdown(f"""
<div style="background:#374151;
height:20px;
border-radius:20px;">

<div style="
background:#00E676;
width:{confidence}%;
height:20px;
border-radius:20px;">
</div>

</div>

<p style="text-align:center;
color:white;
font-size:18px;">
Confidence : {confidence:.2f}%
</p>
""", unsafe_allow_html=True)

        st.write("### Top 3 Predictions")

        for idx in top3:
            st.markdown(
f"""
<div style="
background:#1E293B;
padding:12px;
margin:8px;
border-radius:10px;
color:white;">
<b>{class_names[idx].title()}</b>
<br>
{prediction[idx]*100:.2f}%
</div>
""",
unsafe_allow_html=True
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
st.markdown("""
<hr>

<div style="text-align:center">

<h2 style="color:#00E676;">
♻️ EcoVision AI
</h2>

<p style="color:#B0BEC5;font-size:18px;">
AI Powered Smart Waste Classification System
</p>

<p style="color:white;">
Helping Build a Cleaner & Greener Future 🌍
</p>

</div>
""", unsafe_allow_html=True)
