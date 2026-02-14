import torch
import torch.nn as nn
from torchvision import models
from transformers import CLIPModel
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process # <--- THÊM CÁI NÀY
import os

# --- CONFIG ---
DEVICE = "cpu"
MOBILE_NET_PATH = "best.pt"

# Tên file trung gian
OUTPUT_MOBI = "mobilenet.onnx"
OUTPUT_MOBI_PRE = "mobilenet_pre.onnx" # File đã pre-process
OUTPUT_MOBI_QUANT = "mobilenet_quant.onnx"

OUTPUT_CLIP = "clip_vision.onnx"
OUTPUT_CLIP_PRE = "clip_vision_pre.onnx" # File đã pre-process
OUTPUT_CLIP_QUANT = "clip_vision_quant.onnx"

print("🚀 Bắt đầu Export và Quantization (Phiên bản Fix Lỗi)...")

# Hàm hỗ trợ Pre-process và Quantize
def process_quantization(input_path, pre_path, output_path):
    print(f"   ... Pre-processing {input_path} -> {pre_path}")
    # Bước này sửa lỗi ShapeInferenceError (1280 vs 2, 768 vs 512)
    try:
        quant_pre_process(input_path, pre_path, skip_symbolic_shape=False)
    except Exception as e:
        print(f"   ⚠️ Warning pre-process: {e}. Thử quantize trực tiếp...")
        import shutil
        shutil.copyfile(input_path, pre_path)

    print(f"   ... Quantizing {pre_path} -> {output_path}")
    quantize_dynamic(
        model_input=pre_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8 # Dùng QUInt8 an toàn hơn cho CPU
    )
    print(f"   ✅ Done: {output_path}")

# ==========================================
# 1. Xử lý MobileNetV2
# ==========================================
print(f"1️⃣ Đang xử lý MobileNetV2...")

def load_mobilenet():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    state_dict = torch.load(MOBILE_NET_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

try:
    mobilenet = load_mobilenet()
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export (Dùng opset 17 để tương thích tốt hơn, tránh lỗi version converter)
    torch.onnx.export(
        mobilenet, 
        dummy_input, 
        OUTPUT_MOBI, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=17, # <--- CẬP NHẬT VERSION
        do_constant_folding=True
    )
    
    # Thực hiện Pre-process và Quantize
    process_quantization(OUTPUT_MOBI, OUTPUT_MOBI_PRE, OUTPUT_MOBI_QUANT)
    
except Exception as e:
    print(f"   ❌ Lỗi MobileNet: {e}")

# ==========================================
# 2. Xử lý CLIP Vision
# ==========================================
print("2️⃣ Đang xử lý CLIP Vision Model...")

class CLIPVisionWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision = self.base.vision_model
        self.visual_projection = self.base.visual_projection

    def forward(self, pixel_values):
        vision_outputs = self.vision(pixel_values=pixel_values)
        pooler_output = vision_outputs.pooler_output
        image_features = self.visual_projection(pooler_output)
        return image_features

try:
    clip_wrapper = CLIPVisionWrapper().eval()
    dummy_clip_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        clip_wrapper,
        dummy_clip_input,
        OUTPUT_CLIP,
        input_names=['pixel_values'],
        output_names=['image_features'],
        dynamic_axes={'pixel_values': {0: 'batch_size'}, 'image_features': {0: 'batch_size'}},
        opset_version=17, # <--- CẬP NHẬT VERSION
        do_constant_folding=True
    )

    # Thực hiện Pre-process và Quantize
    process_quantization(OUTPUT_CLIP, OUTPUT_CLIP_PRE, OUTPUT_CLIP_QUANT)

except Exception as e:
    print(f"   ❌ Lỗi CLIP: {e}")

# Cleanup file rác
for f in [OUTPUT_MOBI, OUTPUT_MOBI_PRE, OUTPUT_CLIP, OUTPUT_CLIP_PRE]:
    if os.path.exists(f): os.remove(f)

print("🎉 Hoàn tất! Kiểm tra xem file *_quant.onnx có được tạo ra không.")