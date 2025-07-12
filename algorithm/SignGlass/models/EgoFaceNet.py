import torch
import torch.nn as nn
from transformers import TimesformerModel, TimesformerConfig

# ------------------------------
# Symmetric Fusion
# ------------------------------
class SymmetricFusion(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in * 2, d_out)

    def forward(self, feat_L, feat_R):
        fused = torch.cat([feat_L, feat_R], dim=-1)
        return self.fc(fused)

# ------------------------------
# FiLM Conditioning
# ------------------------------
class FiLMConditioning(nn.Module):
    def __init__(self, d_visual, d_gloss):
        super().__init__()
        self.gamma_fc = nn.Linear(d_gloss, d_visual)
        self.beta_fc = nn.Linear(d_gloss, d_visual)

    def forward(self, visual_feat, gloss_embed):
        gamma = self.gamma_fc(gloss_embed)
        beta = self.beta_fc(gloss_embed)
        return gamma * visual_feat + beta

# ------------------------------
# Syntax Classifier
# ------------------------------
class SyntaxClassifier(nn.Module):
    def __init__(self, d_in, num_labels):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Linear(d_in, num_labels)
        )

    def forward(self, x):
        return self.classifier(x)

# ------------------------------
# EgoFaceNet using HuggingFace TimeSFormer
# ------------------------------
class EgoFaceNet(nn.Module):
    def __init__(self, gloss_dim=256, num_labels=4, timesformer_name="facebook/timesformer-base-finetuned-k400"):
        super().__init__()
        self.timesformer_L = TimesformerModel.from_pretrained(timesformer_name)
        self.timesformer_R = TimesformerModel.from_pretrained(timesformer_name)

        self.visual_dim = self.timesformer_L.config.hidden_size  # e.g., 768
        self.fusion = SymmetricFusion(self.visual_dim, self.visual_dim)
        self.film = FiLMConditioning(self.visual_dim, gloss_dim)
        self.classifier = SyntaxClassifier(self.visual_dim, num_labels)

    def extract_feat(self, x, encoder):
        # x: [B, T, C, H, W] → HuggingFace expects [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        out = encoder(pixel_values=x)
        cls_token = out.last_hidden_state[:, 0]  # [B, D]
        return cls_token

    def forward(self, x_L, x_R, gloss_embed):
        feat_L = self.extract_feat(x_L, self.timesformer_L)
        feat_R = self.extract_feat(x_R, self.timesformer_R)
        fused = self.fusion(feat_L, feat_R)
        modulated = self.film(fused, gloss_embed)
        logits = self.classifier(modulated)
        return logits