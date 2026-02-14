import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# Cấu hình
model_path = "onnx_models/spam_vi_quantized.onnx"
tokenizer_path = "vinai/phobert-base"
text = "Sản phẩm rất tốt, giao hàng nhanh"

print(f"🧪 Testing model: {model_path}")

# 1. Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# 2. Prepare Input
inputs = tokenizer(text, return_tensors="np", padding="max_length", max_length=156, truncation=True)
input_feed = {
    "input_ids": inputs["input_ids"].astype(np.int64),
    "attention_mask": inputs["attention_mask"].astype(np.int64)
}

# 3. Run Inference
output = session.run(None, input_feed)
print(f"🎉 Output raw: {output[0]}")
print(f"✅ Prediction: {'Spam' if output[0][0][1] > output[0][0][0] else 'Not Spam'}")