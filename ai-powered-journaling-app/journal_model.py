import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL2ID = {"NEU": 0, "HUMOR": 1, "MH": 2, "SI": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, tokenizer_name=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        use_safetensors=True
    )
    model.to(device)
    model.eval()
    return tokenizer, model

def predict_entry(text, tokenizer, model, max_length=256):
    encodings = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    pred_id = int(probs.argmax())
    pred_label = ID2LABEL[pred_id]
    prob_dict = {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))}
    return pred_label, prob_dict
