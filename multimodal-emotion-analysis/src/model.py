import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalEmotionClassifier(nn.Module):
    def __init__(self, text_feature_dim=768, image_feature_dim=2048, hidden_dim=512, num_classes=3):
        super(MultimodalEmotionClassifier, self).__init__()
        self.text_fc = nn.Linear(text_feature_dim, hidden_dim)
        self.image_fc = nn.Linear(image_feature_dim, hidden_dim)
        self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, text_features, image_features):
        text_out = F.relu(self.text_fc(text_features))
        text_out = self.dropout(text_out)
        image_out = F.relu(self.image_fc(image_features))
        image_out = self.dropout(image_out)
        combined = torch.cat((text_out, image_out), dim=1)
        fused = F.relu(self.fusion_fc(combined))
        fused = self.dropout(fused)
        output = self.output_fc(fused)
        return output