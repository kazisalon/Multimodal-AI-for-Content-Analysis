
import torch
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from PIL import Image

def load_pretrained_models():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    resnet_model = models.resnet50(pretrained=True)
    resnet_model.eval()
    resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
    return tokenizer, bert_model, resnet_model

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_text(text, tokenizer, bert_model, device):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # CLS token embedding

def process_image(image_path, resnet_model, device):
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet_model(image)
    return features.flatten().unsqueeze(0)