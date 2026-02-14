import torch
import torch.nn as nn
from torchvision import models
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
import os

# --- CONFIG ---
DEVICE = "cpu"
MOBILE_NET_PATH = "best.pt"

OUTPUT_MOBI = "mobilenet.onnx"
OUTPUT_MOBI_PRE = "mobilenet_pre.onnx"
OUTPUT_MOBI_QUANT = "mobilenet_quant.onnx"

OUTPUT_CLIP = "clip_vision.onnx"
OUTPUT_CLIP_PRE = "clip_vision_pre.onnx"
OUTPUT_CLIP_QUANT = "clip_vision_quant.onnx"

print("🚀 Bắt đầu Export V3 (Sử dụng CLIPVisionModelWithProjection)...")

# Hàm xử lý lượng tử hóa chuẩn
def process_quantization(input_path, pre_path, output_path):
    print(f"   ... Pre-processing {input_path} -> {pre_path}")
    try:
        # Pre-process giúp cố định shape, tránh lỗi (768 vs 512)
        quant_pre_process(input_path, pre_path, skip_symbolic_shape=False)
    except Exception as e:
        print(f"   ⚠️ Pre-process warning (có thể bỏ qua nếu quantize thành công): {e}")
        import shutil
        shutil.copyfile(input_path, pre_path)

    print(f"   ... Quantizing {pre_path} -> {output_path}")
    try:
        quantize_dynamic(
            model_input=pre_path,
            model_output=output_path,
            weight_type=QuantType.QUInt8 # QUInt8 nhanh và nhẹ cho CPU
        )
        print(f"   ✅ Done: {output_path}")
    except Exception as e:
        print(f"   ❌ Quantize Failed: {e}")

# ==========================================
# 1. MobileNetV2 (Giữ nguyên vì đã chạy được)
# ==========================================
print(f"1️⃣ Đang xử lý MobileNetV2...")
try:
    # Load Model
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    state_dict = torch.load(MOBILE_NET_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Dùng opset 14 để ổn định nhất, tránh lỗi version converter
    torch.onnx.export(
        model, 
        dummy_input, 
        OUTPUT_MOBI, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=14 
    )
    process_quantization(OUTPUT_MOBI, OUTPUT_MOBI_PRE, OUTPUT_MOBI_QUANT)
    
except Exception as e:
    print(f"   ❌ Lỗi MobileNet: {e}")

# ==========================================
# 2. CLIP Vision (SỬ DỤNG CLASS CHUẨN HF)
# ==========================================
print("2️⃣ Đang xử lý CLIP Vision Model...")

try:
    # [THAY ĐỔI QUAN TRỌNG] Dùng CLIPVisionModelWithProjection
    # Model này trả về output.image_embeds (batch, 512) chuẩn xác
    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    print("   ... Exporting CLIP to ONNX...")
    # Khi export HF model, input_names phải đúng là 'pixel_values'
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_CLIP,
        input_names=['pixel_values'],
        output_names=['image_embeds'], # Đặt tên output rõ ràng
        dynamic_axes={'pixel_values': {0: 'batch_size'}, 'image_embeds': {0: 'batch_size'}},
        opset_version=14 # Opset 14 hỗ trợ tốt CLIP
    )

    process_quantization(OUTPUT_CLIP, OUTPUT_CLIP_PRE, OUTPUT_CLIP_QUANT)

except Exception as e:
    print(f"   ❌ Lỗi CLIP: {e}")

# Cleanup
for f in [OUTPUT_MOBI, OUTPUT_MOBI_PRE, OUTPUT_CLIP, OUTPUT_CLIP_PRE]:
    if os.path.exists(f): os.remove(f)

print("\n🎉 Kiểm tra kết quả:")
if os.path.exists(OUTPUT_MOBI_QUANT): print(f"  - MobileNet: OK ({os.path.getsize(OUTPUT_MOBI_QUANT)/1024/1024:.2f} MB)")
else: print(f"  - MobileNet: FAILED")

if os.path.exists(OUTPUT_CLIP_QUANT): print(f"  - CLIP: OK ({os.path.getsize(OUTPUT_CLIP_QUANT)/1024/1024:.2f} MB)")
else: print(f"  - CLIP: FAILED")