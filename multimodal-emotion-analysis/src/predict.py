import torch
import torch.nn.functional as F
from .preprocess import process_text, process_image

def predict_emotion(text, image_path, tokenizer, bert_model, resnet_model, model, device):
    text_features = process_text(text, tokenizer, bert_model, device)
    image_features = process_image(image_path, resnet_model, device)
    model.eval()
    with torch.no_grad():
        logits = model(text_features, image_features)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return ["positive", "negative", "neutral"][pred], probs[0].cpu().numpy()