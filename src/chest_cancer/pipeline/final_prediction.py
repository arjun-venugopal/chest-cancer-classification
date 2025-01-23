import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import subprocess
import logging
from typing import Dict, Union
from PIL import Image as PILImage

class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model_training", "model.keras")
        self.model = None
        
    def run_training_pipeline(self) -> bool:
        """Run the complete training pipeline using DVC"""
        try:
            logging.info("Starting training pipeline...")
            
            # First try DVC
            try:
                subprocess.run(['dvc', 'repro'], check=True)
                logging.info("DVC pipeline completed successfully")
                return True
            except subprocess.CalledProcessError:
                logging.warning("DVC pipeline failed, trying main.py...")
                
                # If DVC fails, try running main.py
                process = subprocess.run(['python', 'main.py'], 
                                         capture_output=True, 
                                         text=True)
                
                if process.returncode == 0:
                    logging.info("Training pipeline completed successfully")
                    return True
                else:
                    logging.error(f"Training failed: {process.stderr}")
                    return False
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            return False

    def load_model(self) -> bool:
        """Load the model, run training if model not found"""
        try:
            if not os.path.exists(self.model_path):
                logging.warning("Model not found. Starting training pipeline...")
                if not self.run_training_pipeline():
                    raise Exception("Failed to train model")
                
            self.model = load_model(self.model_path)
            return True
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return False

    def preprocess_image(self, input_image: Union[str, PILImage.Image, np.ndarray]) -> np.ndarray:
        """Preprocess the image for prediction"""
        try:
            # Convert input to PIL Image
            if isinstance(input_image, str):
                img = PILImage.open(input_image)
            elif isinstance(input_image, PILImage.Image):
                img = input_image
            elif isinstance(input_image, np.ndarray):
                img = PILImage.fromarray(input_image.astype('uint8'))
            else:
                raise ValueError("Unsupported image type")

            # Debug log
            logging.info(f"Original image size: {img.size}")
            
            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to 224x224 
            img = img.resize((224, 224), PILImage.Resampling.LANCZOS) 
            
            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32)
            
            # Verify shape before normalization
            if img_array.shape != (224, 224, 3):
                raise ValueError(f"Incorrect shape after resize: {img_array.shape}")
            
            # Normalize
            img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Final shape verification
            if img_array.shape != (1, 224, 224, 3):
                raise ValueError(f"Final shape incorrect: {img_array.shape}")
            
            logging.info(f"Preprocessed shape: {img_array.shape}")
            return img_array

        except Exception as e:
            logging.error(f"Preprocessing error: {str(e)}")
            raise

    def predict(self, image_input) -> Dict:
        try:
            if not self.load_model():
                return {"error": "Failed to load or train model"}
        
            # Add debug logging
            logging.info("Starting image preprocessing...")
            processed_image = self.preprocess_image(image_input)
        
            # Ensure processed image has expected shape
            if processed_image.shape != (1, 224, 224, 3):
                logging.error(f"Processed image has unexpected shape: {processed_image.shape}")
                return {"error": "Processed image has unexpected shape."}
        
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            confidence = float(predictions[0][0])
            
            return {
                "prediction": "Cancer" if confidence > 0.5 else "Normal",
                "confidence": confidence,
                "confidence_percent": f"{confidence:.2%}",
                "status": "success"
            }
        
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return {
                "error": str(e),
                "status": "failed"
            }

    def __check_artifacts(self) -> bool:
        """Check if all required artifacts exist"""
        required_paths = [
            os.path.join("artifacts", "data_ingestion", "chest-cancer-ct"),
            os.path.join("artifacts", "model_selection"),
            self.model_path,
            "scores.json"
        ]
        return all(os.path.exists(path) for path in required_paths)