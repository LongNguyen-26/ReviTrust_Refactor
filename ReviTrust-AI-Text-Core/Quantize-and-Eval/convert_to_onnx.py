import torch
import os
from transformers import AutoTokenizer
from model_defs import SpamModelVi, SentimentModelVi, SpamModelEn, SentimentModelEn
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# --- CONFIG ---
DEVICE = "cpu" # Export ONNX nên dùng CPU để đảm bảo tính tương thích
dummy_input_ids = torch.randint(0, 1000, (1, 156)).to(DEVICE)
dummy_mask = torch.ones((1, 156)).to(DEVICE)

# Danh sách config model cần convert
# Format: (ClassModel, weight_path, onnx_name, output_names)
# Lưu ý: output_names của sentiment phải khớp thứ tự dict keys trong model_defs
models_to_convert = [
    {
        "model_class": SpamModelVi,
        "weight": "best_spam_transform-F1-0.89.pth",
        "output": "spam_vi.onnx",
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"],
        "tokenizer": "vinai/phobert-base"
    },
    {
        "model_class": SentimentModelVi,
        "weight": "best_review_transform-F108435.pth",
        "output": "sent_vi.onnx",
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["giao_hang", "chat_luong", "gia_ca", "dong_goi"],
        "tokenizer": "vinai/phobert-base"
    },
    {
        "model_class": SpamModelEn,
        "weight": "ReviewSpamEnglish.pth",
        "output": "spam_en.onnx",
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"],
        "tokenizer": "roberta-base"
    },
    {
        "model_class": SentimentModelEn,
        "weight": "ReviewEnglishEmotion.pth",
        "output": "sent_en.onnx",
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["giao_hang", "chat_luong", "gia_ca", "dong_goi"],
        "tokenizer": "roberta-base"
    }
]

def export_and_quantize():
    print("🚀 Bắt đầu quá trình convert sang ONNX Quantized...")
    
    for conf in models_to_convert:
        print(f"\n--- Xử lý: {conf['output']} ---")
        
        # 1. Load PyTorch Model
        try:
            model = conf["model_class"]().to(DEVICE)
            model.load_state_dict(torch.load(conf["weight"], map_location=DEVICE))
            model.eval()
        except Exception as e:
            print(f"❌ Không tìm thấy hoặc lỗi load weight {conf['weight']}: {e}")
            continue

        # 2. Export to ONNX (Float32)
        temp_onnx = f"temp_{conf['output']}"
        
        # Điều chỉnh dummy input size tùy tokenizer nếu cần (ở đây để chung 156 cũng ok)
        
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_mask),
            temp_onnx,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=conf["input_names"],
            output_names=conf["output_names"],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        print(f"✅ Đã export raw ONNX: {temp_onnx}")

        # 3. Dynamic Quantization (Int8)
        quantized_model_path = f"quantized_{conf['output']}"
        quantize_dynamic(
            temp_onnx,
            quantized_model_path,
            weight_type=QuantType.QUInt8 # Quantize weights về Int8
        )
        print(f"✅ Đã lượng tử hóa: {quantized_model_path}")
        
        # Cleanup file temp
        os.remove(temp_onnx)

    print("\n🎉 Hoàn tất! Hãy upload các file 'quantized_*.onnx' lên Space.")

if __name__ == "__main__":
    export_and_quantize()