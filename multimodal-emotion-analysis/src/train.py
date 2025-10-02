import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from .model import MultimodalEmotionClassifier
from .preprocess import load_pretrained_models, process_text, process_image
import yaml
import logging

# Custom Dataset
class EmotionDataset(Dataset):
    def __init__(self, csv_file, tokenizer, bert_model, resnet_model, device):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.resnet_model = resnet_model
        self.device = device
        self.label_map = {"positive": 0, "negative": 1, "neutral": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["text"]
        image_path = self.data.iloc[idx]["image_path"]
        label = self.label_map[self.data.iloc[idx]["label"]]
        text_features = process_text(text, self.tokenizer, self.bert_model, self.device)
        image_features = process_image(image_path, self.resnet_model, self.device)
        return text_features.squeeze(0), image_features.squeeze(0), torch.tensor(label, dtype=torch.long)

# Training function
def train_model(config):
    logging.basicConfig(filename="logs/training.log", level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    tokenizer, bert_model, resnet_model = load_pretrained_models()
    bert_model.to(device)
    resnet_model.to(device)
    model = MultimodalEmotionClassifier(
        text_feature_dim=config["model"]["text_feature_dim"],
        image_feature_dim=config["model"]["image_feature_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"]
    ).to(device)

    # Load data
    train_dataset = EmotionDataset(config["data"]["train_path"], tokenizer, bert_model, resnet_model, device)
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_loss = 0
        for text_features, image_features, labels in train_loader:
            text_features, image_features, labels = text_features.to(device), image_features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(text_features, image_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logging.info(f"Epoch {epoch+1}/{config['training']['epochs']}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Save model
    torch.save(model.state_dict(), "models/multimodal_model.pth")
    logging.info("Model saved to models/multimodal_model.pth")

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train_model(config)