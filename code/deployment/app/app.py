import streamlit as st
import requests
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import time

# Configure the page
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

# Title and description
st.title("🔢 MNIST Digit Classification")
st.markdown("""
Upload an image of a handwritten digit (0-9) and the CNN model will predict which digit it is.
The model is trained on the MNIST dataset and deployed using FastAPI and Docker.
""")

# Sidebar for API information
st.sidebar.title("API Information")
st.sidebar.info("""
**API Endpoint:** `/predict`

**Method:** POST

**Input:** Image file (PNG, JPG, JPEG)

**Output:** JSON with prediction results
""")

# Initialize session state
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://api:8000"

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    
    upload_method = st.radio(
        "Choose upload method:",
        ["Upload image file", "Draw digit"]
    )
    
    if upload_method == "Upload image file":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of a handwritten digit"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert to bytes for API
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
            
    else:  # Draw digit
        st.markdown("Draw a digit in the canvas below:")
        
        # Create a drawing canvas
        canvas_result = st.canvas(
            stroke_width=10,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas"
        )
        
        if canvas_result.image_data is not None:
            # Convert canvas result to image
            drawn_image = Image.fromarray((canvas_result.image_data * 255).astype('uint8'))
            drawn_image = drawn_image.resize((28, 28))
            
            # Display the drawn image
            st.image(drawn_image, caption="Drawn Digit", use_column_width=True)
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            drawn_image.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()

with col2:
    st.header("📊 Prediction Results")
    
    if ('uploaded_file' in locals() and uploaded_file is not None) or \
       ('canvas_result' in locals() and canvas_result.image_data is not None):
        
        if st.button("🚀 Predict Digit", type="primary"):
            with st.spinner("Making prediction..."):
                try:
                    # Prepare files for API request
                    files = {"file": ("image.png", img_bytes, "image/png")}
                    
                    # Make API request
                    response = requests.post(
                        f"{st.session_state.api_url}/predict",
                        files=files
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        st.success(f"✅ Prediction successful!")
                        
                        # Show predicted digit prominently
                        st.metric(
                            label="**Predicted Digit**",
                            value=result['predicted_digit'],
                            delta=f"{result['confidence']:.2%} confidence"
                        )
                        
                        # Confidence score
                        st.progress(result['confidence'])
                        st.write(f"Confidence: {result['confidence']:.2%}")
                        
                        # Probability distribution
                        st.subheader("Probability Distribution")
                        digits = list(range(10))
                        probabilities = result['probabilities']
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        bars = ax.bar(digits, probabilities, color='skyblue')
                        
                        # Highlight the predicted digit
                        bars[result['predicted_digit']].set_color('red')
                        
                        ax.set_xlabel('Digit')
                        ax.set_ylabel('Probability')
                        ax.set_title('Probability Distribution for Each Digit')
                        ax.set_xticks(digits)
                        
                        # Add probability values on bars
                        for i, v in enumerate(probabilities):
                            ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
                        
                        st.pyplot(fig)
                        
                    else:
                        st.error(f"❌ API Error: {response.status_code} - {response.text}")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Connection error: Please make sure the API is running")
                    st.info("API URL: " + st.session_state.api_url)
                    
    else:
        st.info("👆 Please upload an image or draw a digit to get a prediction")

# Footer
st.markdown("---")
st.markdown("""
### 🛠️ Technical Details
- **Model:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow/Keras
- **API:** FastAPI
- **Frontend:** Streamlit
- **Containerization:** Docker
- **Dataset:** MNIST (70,000 handwritten digits)
""")

# Health check
if st.sidebar.button("Check API Health"):
    try:
        response = requests.get(f"{st.session_state.api_url}/health")
        if response.status_code == 200:
            st.sidebar.success("✅ API is healthy")
        else:
            st.sidebar.error("❌ API health check failed")
    except:
        st.sidebar.error("❌ Cannot connect to API")