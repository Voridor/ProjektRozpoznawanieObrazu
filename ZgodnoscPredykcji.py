import time
import torch
import torchvision.models as models
import onnxruntime as ort
import numpy as np

# 1. Definiowanie klas
classes = [
    'apple_pie', 'beef_tartare', 'caesar_salad', 'cheesecake', 'chicken_wings',
    'chocolate_cake', 'club_sandwich', 'dumplings', 'french_fries', 'garlic_bread',
    'hamburger', 'ice_cream', 'lasagna', 'macaroni_and_cheese', 'omelette',
    'pancakes', 'pizza', 'spaghetti_bolognese', 'sushi', 'waffles'
]

# 2. Przygotowanie wejścia (losowy obraz o wymiarach 224x224)
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
dummy_input_tensor = torch.tensor(dummy_input)

print("Test Zgodności Predykcji")

# Ładowanie struktury PyTorch na CPU i podmiana klasyfikatora
# (Musimy to zrobić, aby porównać "surowy" model z ONNX)
pytorch_model = models.mobilenet_v2()
pytorch_model.classifier[1] = torch.nn.Linear(pytorch_model.classifier[1].in_features, len(classes))
pytorch_model.eval()

# Ładowanie ONNX Runtime
onnx_session = ort.InferenceSession("food_classifier_20.onnx")

# Pobranie predykcji
with torch.no_grad():
    pytorch_output = pytorch_model(dummy_input_tensor).numpy()

onnx_inputs = {onnx_session.get_inputs()[0].name: dummy_input}
onnx_output = onnx_session.run(None, onnx_inputs)[0]

print("\nBenchmark Czasu Inferencji (CPU)")
num_runs = 100

# Rozgrzewka (Warm-up)
for _ in range(10):
    _ = pytorch_model(dummy_input_tensor)
    _ = onnx_session.run(None, onnx_inputs)

# Test PyTorch CPU
start_time = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        _ = pytorch_model(dummy_input_tensor)
pytorch_time = (time.time() - start_time) / num_runs
print(f"Średni czas PyTorch CPU: {pytorch_time * 1000:.2f} ms per obraz")

# Test ONNX Runtime CPU
start_time = time.time()
for _ in range(num_runs):
    _ = onnx_session.run(None, onnx_inputs)
onnx_time = (time.time() - start_time) / num_runs
print(f"Średni czas ONNX Runtime CPU: {onnx_time * 1000:.2f} ms per obraz")

speedup = pytorch_time / onnx_time
print(f"\nWynik: ONNX Runtime jest {speedup:.2f}x szybszy niż PyTorch na CPU")