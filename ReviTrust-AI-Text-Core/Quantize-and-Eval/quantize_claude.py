"""
Script để convert PyTorch models sang ONNX và áp dụng dynamic quantization
Chạy script này để chuẩn bị models cho deployment trên HF Space CPU
"""

import torch
import os
from model_defs import SpamModelVi, SentimentModelVi, SpamModelEn, SentimentModelEn
from transformers import AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process
import onnx  # <--- Thêm dòng này

DEVICE = "cpu"  # Luôn dùng CPU để export

# Cấu hình models
MODEL_CONFIGS = [
    {
        "name": "spam_vi",
        "model_class": SpamModelVi,
        "weight_file": "/content/drive/MyDrive/CodeCraze/best_spam_transform-F1-0.89.pth",
        "tokenizer": "vinai/phobert-base",
        "max_length": 156,
        "sample_text": "Sản phẩm rất tốt, giao hàng nhanh"
    },
    {
        "name": "sent_vi", 
        "model_class": SentimentModelVi,
        "weight_file": "/content/drive/MyDrive/CodeCraze/best_review_transform-F108435.pth",
        "tokenizer": "vinai/phobert-base",
        "max_length": 128,
        "sample_text": "Chất lượng tuyệt vời, giá cả hợp lý"
    },
    {
        "name": "spam_en",
        "model_class": SpamModelEn,
        "weight_file": "/content/drive/MyDrive/CodeCraze/ReviewSpamEnglish.pth",
        "tokenizer": "roberta-base",
        "max_length": 156,
        "sample_text": "Great product, fast delivery"
    },
    {
        "name": "sent_en",
        "model_class": SentimentModelEn,
        "weight_file": "/content/drive/MyDrive/CodeCraze/ReviewEnglishEmotion.pth",
        "tokenizer": "roberta-base",
        "max_length": 128,
        "sample_text": "Excellent quality, reasonable price"
    }
]

def export_to_onnx(model, tokenizer, sample_text, max_length, output_path):
    """Export PyTorch model to ONNX format"""
    model.eval()
    
    # Tạo dummy input
    encoding = tokenizer(
        sample_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    # Export sang ONNX
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        # SỬA LỖI 1: Tăng opset lên 17 để hỗ trợ LayerNorm tốt hơn
        opset_version=17, 
        do_constant_folding=True
    )
    print(f"✅ Exported to {output_path}")

def quantize_onnx_model(onnx_path, quantized_path):
    """Áp dụng dynamic quantization (Fix lỗi Shape Inference bằng cách reset shape info)"""
    print(f"🔧 Fixing ONNX shapes manually...")
    
    try:
        # 1. Load model gốc
        model = onnx.load(onnx_path)
        
        # 2. QUAN TRỌNG: Xóa thông tin shape cũ bị sai lệch
        # Lỗi (768) vs (256) do shape tĩnh bị lưu cứng sai vị trí
        del model.graph.value_info[:]
        
        # 3. Tính toán lại shape mới chuẩn xác hơn
        model = onnx.shape_inference.infer_shapes(model)
        
        # 4. Lưu ra file tạm
        temp_path = onnx_path.replace(".onnx", "_temp.onnx")
        onnx.save(model, temp_path)
        
        print(f"⚡ Applying dynamic quantization...")
        
        # 5. Quantize trên file tạm đã fix lỗi
        quantize_dynamic(
            model_input=temp_path,
            model_output=quantized_path,
            weight_type=QuantType.QUInt8,
            per_channel=False,
            reduce_range=False
        )
        print(f"✅ Quantized to {quantized_path}")
        
        # Dọn dẹp
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    except Exception as e:
        print(f"❌ Critical Error in quantization: {e}")
        # In ra chi tiết để debug nếu vẫn lỗi
        import traceback
        traceback.print_exc()
            
def main():
    os.makedirs("onnx_models", exist_ok=True)
    
    for config in MODEL_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Processing: {config['name']}")
        print(f"{'='*60}")
        
        # Kiểm tra file weights
        if not os.path.exists(config["weight_file"]):
            print(f"⚠️  Weight file {config['weight_file']} not found! Skipping...")
            continue
        
        # Load tokenizer
        print(f"📦 Loading tokenizer: {config['tokenizer']}")
        use_fast = "roberta" in config["tokenizer"]
        tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer"], 
            use_fast=use_fast
        )
        
        # Khởi tạo model
        print(f"🏗️  Initializing model: {config['model_class'].__name__}")
        model = config["model_class"]().to(DEVICE)
        
        # Load weights
        print(f"📥 Loading weights from {config['weight_file']}")
        try:
            state_dict = torch.load(config["weight_file"], map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"✅ Weights loaded successfully")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            continue
        
        # Export sang ONNX
        onnx_path = f"onnx_models/{config['name']}.onnx"
        print(f"🔄 Exporting to ONNX...")
        try:
            export_to_onnx(
                model=model,
                tokenizer=tokenizer,
                sample_text=config["sample_text"],
                max_length=config["max_length"],
                output_path=onnx_path
            )
        except Exception as e:
            print(f"❌ Error exporting to ONNX: {e}")
            continue
        
        # Quantize ONNX model
        quantized_path = f"onnx_models/{config['name']}_quantized.onnx"
        print(f"⚡ Applying dynamic quantization...")
        try:
            quantize_onnx_model(onnx_path, quantized_path)
            
            # So sánh kích thước file
            original_size = os.path.getsize(onnx_path) / (1024 * 1024)
            quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
            reduction = ((original_size - quantized_size) / original_size) * 100
            
            print(f"📊 Size comparison:")
            print(f"   Original:  {original_size:.2f} MB")
            print(f"   Quantized: {quantized_size:.2f} MB")
            print(f"   Reduction: {reduction:.1f}%")
            
        except Exception as e:
            print(f"❌ Error quantizing: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("✨ Conversion completed!")
    print(f"{'='*60}")
    print(f"\nQuantized models saved in: onnx_models/")
    print(f"Files to deploy:")
    for config in MODEL_CONFIGS:
        quantized_file = f"{config['name']}_quantized.onnx"
        print(f"  - {quantized_file}")

if __name__ == "__main__":
    main()