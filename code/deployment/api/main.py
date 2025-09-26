from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import json

app = FastAPI(title="MNIST Digit Classification API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = tf.keras.models.load_model('/app/models/mnist_cnn_model.h5')

def preprocess_image(image_data):
    """Preprocess image for model prediction"""
    try:
        # Open image and convert to grayscale
        image = Image.open(io.BytesIO(image_data)).convert('L')
        
        # Resize to 28x28 pixels
        image = image.resize((28, 28))
        
        # Convert to numpy array and normalize
        image_array = np.array(image).astype('float32') / 255.0
        
        # Reshape for model input (add batch and channel dimensions)
        image_array = image_array.reshape(1, 28, 28, 1)
        
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "MNIST Digit Classification API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    """Predict digit from uploaded image"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        image_data = await file.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        # Get probabilities for all digits
        probabilities = predictions[0].tolist()
        
        return {
            "predicted_digit": predicted_digit,
            "confidence": confidence,
            "probabilities": probabilities,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-array")
async def predict_from_array(data: dict):
    """Predict digit from array data (for testing)"""
    try:
        array_data = data.get("array")
        if array_data is None or len(array_data) != 784:
            raise HTTPException(status_code=400, detail="Array must have 784 elements (28x28)")
        
        # Convert to numpy array and reshape
        image_array = np.array(array_data).reshape(1, 28, 28, 1).astype('float32') / 255.0
        
        # Make prediction
        predictions = model.predict(image_array)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        return {
            "predicted_digit": predicted_digit,
            "confidence": confidence,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)