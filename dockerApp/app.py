from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Klasyfikator Jedzenia 20 Klas")

classes = [
    'apple_pie', 'beef_tartare', 'caesar_salad', 'cheesecake', 'chicken_wings',
    'chocolate_cake', 'club_sandwich', 'dumplings', 'french_fries', 'garlic_bread',
    'hamburger', 'ice_cream', 'lasagna', 'macaroni_and_cheese', 'omelette',
    'pancakes', 'pizza', 'spaghetti_bolognese', 'sushi', 'waffles'
]

# Załadowanie modelu ONNX przy starcie aplikacji
session = ort.InferenceSession("food_classifier_20.onnx")
input_name = session.get_inputs()[0].name

def preprocess_image(image_bytes):
    # Otwarcie obrazu i transformacja do 224x224 RGB
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    
    # Zamiana na tablicę numpy i normalizacja (ImageNet)
    img_data = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_data = (img_data - mean) / std
    
    # Zmiana układu osi z [H, W, C] na [C, H, W] oraz dodanie wymiaru batcha [1, C, H, W]
    img_data = img_data.transpose(2, 0, 1)
    img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
    return img_data

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    tensor_input = preprocess_image(contents)
    
    # Inferencja w ONNX Runtime
    outputs = session.run(None, {input_name: tensor_input})
    probabilities = outputs[0][0]
    
    # Wyznaczenie klasy o najwyższym prawdopodobieństwie
    best_class_idx = int(np.argmax(probabilities))
    detected_dish = classes[best_class_idx]
    
    return {
        "status": "success",
        "predicted_class_index": best_class_idx,
        "dish_name": detected_dish
    }

@app.get("/")
def health_check():
    return {"status": "healthy", "model": "MobileNetV2 ONNX 20 Classes"}