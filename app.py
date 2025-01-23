import streamlit as st
import os
from PIL import Image
from src.chest_cancer.pipeline.final_prediction import PredictionPipeline

# Page configuration
st.set_page_config(
    page_title="Chest Cancer Classification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stProgress .st-bo {
            background-color: #ff4b4b;
        }
        .success-text {
            color: #28a745;
        }
        .danger-text {
            color: #dc3545;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    return PredictionPipeline()

def main():
    # Header
    st.title("🏥 Chest Cancer Detection System")
    st.markdown("""
    Upload a chest CT scan image to detect the presence of cancer.
    """)

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Information")
        st.info("""
        This system uses deep learning to analyze chest CT scans and detect potential cancer.

        **Supported formats:**
        - JPG/JPEG
        - PNG

        **Image requirements:**
        - Clear chest CT scan
        - Good resolution
        - Proper orientation
        """)

        # Model status
        st.header("🔧 Model Status")
        predictor = load_predictor()
        if predictor.load_model():
            st.success("Model loaded successfully")
        else:
            st.error("Model not loaded. Please check the model file.")
            st.stop()

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a chest CT scan image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear chest CT scan image"
    )

    if uploaded_file:
        try:
            # Display columns
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📤 Uploaded Image")
                # Load and preprocess image
                image = Image.open(uploaded_file)

                # Convert to RGB
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Resize image to 224x224
                image = image.resize((224, 224), Image.Resampling.LANCZOS)

                # Display original image
                st.image(image, use_container_width=True)

                # Add debug info
                st.caption(f"""
                Original Image Details:
                Size: {image.size}
                Format: {image.format}
                Mode: {image.mode}
                """)

            with col2:
                st.subheader("🔍 Analysis Results")

                if st.button("Run Analysis", use_container_width=True):
                    with st.spinner("Preprocessing and analyzing image..."):
                        try:
                            # Run prediction with error handling
                            result = predictor.predict(image)

                            if "error" not in result:
                                # Display prediction results
                                st.markdown("### Prediction Results")

                                if result["prediction"] == "Cancer":
                                    st.markdown(f"""
                                    <h4 style='color: #dc3545;'>⚠️ Cancer Detected</h4>
                                    <p>Confidence: {result['confidence_percent']}%</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.progress(result["confidence"])
                                else:
                                    st.markdown(f"""
                                    <h4 style='color: #28a745;'>✅ No Cancer Detected</h4>
                                    <p>Confidence: {result['confidence_percent']}%</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.progress(1 - result["confidence"])

                        except Exception as e:
                            st.error(f"Analysis error: {str(e)}")
                            st.error("Please ensure the image is a valid chest CT scan")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.error("Please upload a valid image file")

if __name__ == "__main__":
    main()