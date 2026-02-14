import torch
import torch.onnx
from transformers import CLIPModel
from torchvision import models
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process
import os

def get_file_size_mb(path):
    """Tính dung lượng file (bao gồm cả external data nếu có)"""
    if not os.path.exists(path):
        return 0
    total_size = os.path.getsize(path)
    # Kiểm tra xem có file .data đi kèm không (thường sinh ra khi export model lớn)
    data_path = path + ".data"
    if os.path.exists(data_path):
        total_size += os.path.getsize(data_path)
    return total_size / (1024 * 1024)

print("=" * 60)
print("🚀 BẮT ĐẦU QUANTIZE MODELS (FINAL FIX)")
print("=" * 60)

# ==========================================
# 1. QUANTIZE MOBILENET (Giữ nguyên vì đã chạy tốt)
# ==========================================
print("\n📱 [1/2] Quantizing MobileNet v2...")

try:
    # 1. Load Model
    mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    mobilenet.eval()
    
    # Modify classifier
    num_classes = 2
    mobilenet.classifier[1] = torch.nn.Linear(mobilenet.classifier[1].in_features, num_classes)
    
    # Load checkpoint
    checkpoint_path = "best.pt"
    if os.path.exists(checkpoint_path):
        mobilenet.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"✅ Loaded trained weights from {checkpoint_path}")
    
    # 2. Export FP32
    mobilenet_fp32_path = "mobilenet_fp32.onnx"
    dummy_input = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        mobilenet,
        dummy_input,
        mobilenet_fp32_path,
        export_params=True,
        opset_version=17, # Nâng lên 17 cho đồng bộ
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # 3. PRE-PROCESSING (Cần thiết cho MobileNet để fix shape)
    mobilenet_pre_path = "mobilenet_pre.onnx"
    quant_pre_process(
        input_model_path=mobilenet_fp32_path,
        output_model_path=mobilenet_pre_path,
        skip_symbolic_shape=False
    )

    # 4. Quantize
    mobilenet_quant_path = "mobilenet_quant.onnx"
    quantize_dynamic(
        model_input=mobilenet_pre_path,
        model_output=mobilenet_quant_path,
        weight_type=QuantType.QUInt8
    )
    
    # Report
    fp32_size = get_file_size_mb(mobilenet_fp32_path)
    quant_size = get_file_size_mb(mobilenet_quant_path)
    print(f"✅ Result MobileNet:")
    print(f"   - FP32 Size: {fp32_size:.2f} MB")
    print(f"   - INT8 Size: {quant_size:.2f} MB")
    print(f"   - Compression: {(1 - quant_size/fp32_size)*100:.1f}%")
    
    # Cleanup
    if os.path.exists(mobilenet_fp32_path): os.remove(mobilenet_fp32_path)
    if os.path.exists(mobilenet_pre_path): os.remove(mobilenet_pre_path)

except Exception as e:
    print(f"❌ Error quantizing MobileNet: {e}")


# ==========================================
# 2. QUANTIZE CLIP (Đã sửa lỗi Version & Pre-process)
# ==========================================
print("\n🖼️  [2/2] Quantizing CLIP Vision Encoder...")

try:
    # 1. Load Model
    model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_name)
    vision_model = clip_model.vision_model
    vision_model.eval()
    
    # 2. Export FP32 (Sử dụng Opset 17 để tránh lỗi LayerNorm)
    clip_fp32_path = "clip_vision_fp32.onnx"
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print("   Exporting to ONNX (Opset 17)...")
    torch.onnx.export(
        vision_model,
        dummy_input,
        clip_fp32_path,
        export_params=True,
        opset_version=17,  # QUAN TRỌNG: Opset 17 hỗ trợ Transformer tốt hơn
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['last_hidden_state', 'pooler_output'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'last_hidden_state': {0: 'batch_size'},
            'pooler_output': {0: 'batch_size'}
        }
    )
    print(f"✅ Exported FP32 ONNX")

    # 3. Quantize
    # LƯU Ý: Đã BỎ qua bước quant_pre_process() cho CLIP 
    # vì nó gây lỗi "list index out of range" với cấu trúc Transformer
    
    clip_quant_path = "clip_vision_quant.onnx"
    print("   Quantizing directly (Skipping pre-process)...")
    
    quantize_dynamic(
        model_input=clip_fp32_path, # Dùng thẳng file FP32 export ra
        model_output=clip_quant_path,
        weight_type=QuantType.QUInt8
    )
    
    # Report
    fp32_size = get_file_size_mb(clip_fp32_path)
    quant_size = get_file_size_mb(clip_quant_path)
    
    print(f"✅ Result CLIP:")
    print(f"   - FP32 Size: {fp32_size:.2f} MB")
    print(f"   - INT8 Size: {quant_size:.2f} MB")
    if fp32_size > 0:
        print(f"   - Compression: {(1 - quant_size/fp32_size)*100:.1f}%")
    
    # Cleanup
    if os.path.exists(clip_fp32_path): os.remove(clip_fp32_path)
    if os.path.exists(clip_fp32_path + ".data"): os.remove(clip_fp32_path + ".data")

except Exception as e:
    print(f"❌ Error quantizing CLIP: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ HOÀN TẤT QUANTIZATION!")