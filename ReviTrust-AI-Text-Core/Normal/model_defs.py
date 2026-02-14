import torch
import torch.nn as nn
from transformers import AutoModel

class AttnPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x, mask):
        score = self.attn(x).squeeze(-1)
        score = score.masked_fill(mask == 0, -1e9)
        weight = torch.softmax(score, dim=-1).unsqueeze(-1)
        return torch.sum(weight * x, dim=1)

# Vietnamese Models
class SpamModelVi(nn.Module):
    def __init__(self, model_name="vinai/phobert-base", num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.transformer_head = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_labels))

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        tf_out = self.transformer_head(outputs.last_hidden_state)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = torch.sum(tf_out * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return self.classifier(pooled)

class SentimentModelVi(nn.Module):
    def __init__(self, model_name="vinai/phobert-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=1024, dropout=0.3, batch_first=True)
        self.transformer_head = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.pooling = AttnPool(hidden_size)
        def make_fc(): return nn.Sequential(nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 3))
        self.fc_giao_hang = make_fc(); self.fc_chat_luong = make_fc(); self.fc_gia_ca = make_fc(); self.fc_dong_goi = make_fc()

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        tf_out = self.transformer_head(outputs.last_hidden_state)
        pooled = self.pooling(tf_out, attention_mask)
        return {"giao_hang": self.fc_giao_hang(pooled), "chat_luong": self.fc_chat_luong(pooled), 
                "gia_ca": self.fc_gia_ca(pooled), "dong_goi": self.fc_dong_goi(pooled)}

# English Models (Tương tự nhưng dùng Roberta)
class SpamModelEn(nn.Module):
    def __init__(self, model_name="roberta-base", num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=256, num_layers=1, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_labels))

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = torch.sum(lstm_out * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return self.classifier(pooled)

class SentimentModelEn(nn.Module):
    def __init__(self, model_name="roberta-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=1024, dropout=0.3, batch_first=True)
        self.transformer_head = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.pooling = AttnPool(hidden_size)
        def make_fc(): return nn.Sequential(nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 3))
        self.fc_giao_hang = make_fc(); self.fc_chat_luong = make_fc(); self.fc_gia_ca = make_fc(); self.fc_dong_goi = make_fc()

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        tf_out = self.transformer_head(outputs.last_hidden_state)
        pooled = self.pooling(tf_out, attention_mask)
        return {"giao_hang": self.fc_giao_hang(pooled), "chat_luong": self.fc_chat_luong(pooled),
                "gia_ca": self.fc_gia_ca(pooled), "dong_goi": self.fc_dong_goi(pooled)}