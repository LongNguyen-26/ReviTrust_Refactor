import torch
import torch.nn as nn
from torchvision import models
from transformers import CLIPModel, CLIPProcessor
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

# --- CONFIG ---
DEVICE = "cpu" # Export nên dùng CPU để tránh lỗi device conflict khi convert
MOBILE_NET_PATH = "best.pt" # Đường dẫn file weight của bạn
OUTPUT_MOBI = "mobilenet.onnx"
OUTPUT_MOBI_QUANT = "mobilenet_quant.onnx"
OUTPUT_CLIP = "clip_vision.onnx"
OUTPUT_CLIP_QUANT = "clip_vision_quant.onnx"

print("🚀 Bắt đầu quá trình Export và Quantization...")

# ==========================================
# 1. Xử lý MobileNetV2
# ==========================================
print(f"1️⃣ Đang xử lý MobileNetV2 từ {MOBILE_NET_PATH}...")

def load_mobilenet():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    # Load weights
    state_dict = torch.load(MOBILE_NET_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

try:
    mobilenet = load_mobilenet()
    
    # Tạo dummy input (Batch_size=1, Channels=3, Height=224, Width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export sang ONNX
    torch.onnx.export(
        mobilenet, 
        dummy_input, 
        OUTPUT_MOBI, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=14
    )
    print(f"   ✅ Export MobileNet xong: {OUTPUT_MOBI}")

    # Quantization (Dynamic - uint8)
    quantize_dynamic(
        model_input=OUTPUT_MOBI,
        model_output=OUTPUT_MOBI_QUANT,
        weight_type=QuantType.QUInt8
    )
    print(f"   ✅ Quantize MobileNet xong: {OUTPUT_MOBI_QUANT}")
    
except Exception as e:
    print(f"   ❌ Lỗi xử lý MobileNet: {e}")

# ==========================================
# 2. Xử lý CLIP (Chỉ lấy Vision Model)
# ==========================================
print("2️⃣ Đang xử lý CLIP Vision Model...")

class CLIPVisionWrapper(nn.Module):
    """
    Wrapper để chỉ lấy phần Vision Encoder và trả về pooler_output (embedding)
    """
    def __init__(self):
        super().__init__()
        self.base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision = self.base.vision_model
        self.visual_projection = self.base.visual_projection

    def forward(self, pixel_values):
        # Lấy output từ vision transformer
        vision_outputs = self.vision(pixel_values=pixel_values)
        pooler_output = vision_outputs.pooler_output  # (Batch, 768)
        # Project sang không gian CLIP chung (Batch, 512)
        image_features = self.visual_projection(pooler_output)
        return image_features

try:
    clip_wrapper = CLIPVisionWrapper().eval()
    
    # CLIP dummy input (Batch=1, C=3, H=224, W=224)
    # Lưu ý: CLIP processor resize ảnh tùy config, nhưng vit-base-patch32 thường là 224
    dummy_clip_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        clip_wrapper,
        dummy_clip_input,
        OUTPUT_CLIP,
        input_names=['pixel_values'],
        output_names=['image_features'],
        dynamic_axes={'pixel_values': {0: 'batch_size'}, 'image_features': {0: 'batch_size'}},
        opset_version=14
    )
    print(f"   ✅ Export CLIP Vision xong: {OUTPUT_CLIP}")

    # Quantization
    quantize_dynamic(
        model_input=OUTPUT_CLIP,
        model_output=OUTPUT_CLIP_QUANT,
        weight_type=QuantType.QUInt8
    )
    print(f"   ✅ Quantize CLIP Vision xong: {OUTPUT_CLIP_QUANT}")

except Exception as e:
    print(f"   ❌ Lỗi xử lý CLIP: {e}")

# Cleanup file trung gian (không lượng tử hóa) để tiết kiệm chỗ
if os.path.exists(OUTPUT_MOBI): os.remove(OUTPUT_MOBI)
if os.path.exists(OUTPUT_CLIP): os.remove(OUTPUT_CLIP)

print("🎉 Hoàn tất! Hãy upload các file *_quant.onnx lên HF Space.")