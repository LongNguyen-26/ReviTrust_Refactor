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
print("🚀 TESTING QUANTIZED MODELS (FIXED)")
print("="*50)

# 1. SETUP ẢNH TEST
# Tạo ảnh mẫu
image = Image.new('RGB', (224, 224), color='green') 
print("📸 Created dummy image for testing...")

# ==========================================
# TEST 1: MOBILENET V2 (INT8)
# ==========================================
print("\n📱 Testing MobileNet (INT8)...")
try:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image).unsqueeze(0).numpy()
    
    session = ort.InferenceSession("mobilenet_quant.onnx", providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    
    probs = softmax(outputs[0])
    print(f"✅ Output Shape: {outputs[0].shape}")
    print(f"✅ Probabilities: {probs}")

except Exception as e:
    print(f"❌ MobileNet Error: {e}")

# ==========================================
# TEST 2: CLIP VISION (INT8) - ĐÃ SỬA
# ==========================================
print("\n🖼️  Testing CLIP Vision (INT8)...")
try:
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # --- PHẦN SỬA LỖI ---
    # 1. Yêu cầu trả về PyTorch Tensor ("pt") thay vì Numpy ("np")
    inputs = processor(images=image, return_tensors="pt") 
    
    # 2. Tự chuyển sang Numpy thủ công để đưa vào ONNX
    pixel_values = inputs['pixel_values'].detach().cpu().numpy()
    # --------------------
    
    session = ort.InferenceSession("clip_vision_quant.onnx", providers=['CPUExecutionProvider'])
    
    # Run Inference
    outputs = session.run(None, {'pixel_values': pixel_values})
    
    # Lấy Embedding (thường là output thứ 2: pooler_output)
    # Output 0: last_hidden_state (sequence), Output 1: pooler_output (embedding)
    if len(outputs) > 1:
        embedding = outputs[1] 
    else:
        embedding = outputs[0]

    print(f"✅ Output Shape (Embedding): {embedding.shape}")
    print(f"✅ Embedding vector sample: {embedding[0][:5]}...")
    print("👉 CLIP hoạt động tốt!")

except Exception as e:
    print(f"❌ CLIP Error: {e}")
    # In chi tiết lỗi nếu vẫn còn
    import traceback
    traceback.print_exc()

print("\n🎉 DONE.")