import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import CLIPProcessor
import torchvision.transforms as transforms
import torch

def softmax(x):
    """Hàm tính xác suất"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

print("="*50)
print("🚀 TESTING QUANTIZED MODELS")
print("="*50)

# 1. SETUP ẢNH TEST
# Tạo một ảnh rỗng hoặc load ảnh thật
image = Image.new('RGB', (224, 224), color='red')
print("📸 Created dummy image for testing...")

# ==========================================
# TEST 1: MOBILENET V2 (INT8)
# ==========================================
print("\n📱 Testing MobileNet (INT8)...")
try:
    # Preprocessing giống lúc train (Chuẩn hóa ImageNet)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image).unsqueeze(0).numpy() # Chuyển sang numpy
    
    # Load ONNX Session
    session = ort.InferenceSession("mobilenet_quant.onnx", providers=['CPUExecutionProvider'])
    
    # Run Inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_tensor})
    
    probs = softmax(outputs[0])
    print(f"✅ MobileNet Output Shape: {outputs[0].shape}")
    print(f"✅ Probabilities: {probs}")
    print("👉 MobileNet hoạt động tốt!")

except Exception as e:
    print(f"❌ MobileNet Error: {e}")

# ==========================================
# TEST 2: CLIP VISION (INT8)
# ==========================================
print("\n🖼️  Testing CLIP Vision (INT8)...")
try:
    # Dùng Processor gốc để xử lý ảnh cho chuẩn
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(images=image, return_tensors="np") # Trả về numpy
    
    pixel_values = inputs['pixel_values']
    
    # Load ONNX Session
    session = ort.InferenceSession("clip_vision_quant.onnx", providers=['CPUExecutionProvider'])
    
    # Run Inference
    # CLIP Vision ONNX có input là 'pixel_values'
    outputs = session.run(None, {'pixel_values': pixel_values})
    
    # Output của CLIP thường là: [last_hidden_state, pooler_output]
    embedding = outputs[1] # Lấy pooler_output làm embedding
    
    print(f"✅ CLIP Output Shape (Embedding): {embedding.shape}")
    print(f"✅ Embedding Vector (First 5): {embedding[0][:5]}...")
    print("👉 CLIP hoạt động tốt!")

except Exception as e:
    print(f"❌ CLIP Error: {e}")

print("\n🎉 DONE.")