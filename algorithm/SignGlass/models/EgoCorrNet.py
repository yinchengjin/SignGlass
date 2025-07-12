import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Early Fusion Module
# ----------------------
class EarlyFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, flow_tensor, landmark_tensor):
        """
        Concatenate flow and landmark heatmaps along channel dimension.
        flow_tensor: [B, C1, H, W]
        landmark_tensor: [B, C2, H, W]
        Output: [B, C1+C2, H, W]
        """
        return torch.cat([flow_tensor, landmark_tensor], dim=1)

# ----------------------
# Cross-Modal Attention Module
# ----------------------
class CrossModalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, flow_feat, landmark_feat):
        """
        Inputs: [B, T, D]
        Returns: cross-attended [B, T, D], [B, T, D]
        """
        Q_f = self.q_proj(flow_feat)
        K_l = self.k_proj(landmark_feat)
        V_l = self.v_proj(landmark_feat)
        attn_weights_fl = torch.matmul(Q_f, K_l.transpose(-2, -1)) / (flow_feat.shape[-1] ** 0.5)
        Z_f = torch.matmul(F.softmax(attn_weights_fl, dim=-1), V_l)

        Q_l = self.q_proj(landmark_feat)
        K_f = self.k_proj(flow_feat)
        V_f = self.v_proj(flow_feat)
        attn_weights_lf = torch.matmul(Q_l, K_f.transpose(-2, -1)) / (landmark_feat.shape[-1] ** 0.5)
        Z_l = torch.matmul(F.softmax(attn_weights_lf, dim=-1), V_f)

        return Z_f, Z_l

# ----------------------
# Residual Gated Fusion Module
# ----------------------
class ResidualGatedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, Z_f, Z_l):
        """
        Z_f, Z_l: [B, T, D]
        Returns fused: [B, T, D]
        """
        concat = torch.cat([Z_f, Z_l], dim=-1)  # [B, T, 2D]
        alpha = self.gate(concat)               # [B, T, D]
        fused = alpha * Z_f + (1 - alpha) * Z_l
        return fused

# ----------------------
# EgoCorrNet: Combined Module
# ----------------------
class EgoCorrNet(nn.Module):
    def __init__(self, backbone_cnn, temporal_decoder, input_channels=5, d_model=256):
        """
        backbone_cnn: CNN encoder like MobileNet or ResNet to extract visual features
        temporal_decoder: downstream temporal model (e.g., BiLSTM + FC for CTC)
        input_channels: total input channels (e.g., 2 for flow + 21 for landmark heatmap)
        d_model: feature dimension after backbone
        """
        super().__init__()
        self.early_fusion = EarlyFusion()
        self.backbone = backbone_cnn  # input: [B, C, H, W] → output: [B, T, D]
        self.cross_attention = CrossModalAttention(d_model)
        self.residual_fusion = ResidualGatedFusion(d_model)
        self.temporal_decoder = temporal_decoder  # output: [B, T, vocab_size]

    def forward(self, flow_tensor, landmark_tensor):
        # Step 1: Early fusion
        x = self.early_fusion(flow_tensor, landmark_tensor)  # [B, C, H, W]

        # Step 2: Backbone visual feature extraction
        feat = self.backbone(x)  # Expect [B, T, D]

        # Step 3: Split into flow and landmark branches
        D_half = feat.shape[-1] // 2
        flow_feat = feat[..., :D_half]         # [B, T, D/2]
        landmark_feat = feat[..., D_half:]     # [B, T, D/2]

        # Step 4: Cross-modal attention
        Z_f, Z_l = self.cross_attention(flow_feat, landmark_feat)

        # Step 5: Residual gated fusion
        fused_feat = self.residual_fusion(Z_f, Z_l)  # [B, T, D/2]

        # Step 6: Temporal decoder (e.g., BiLSTM + FC)
        out = self.temporal_decoder(fused_feat)      # [B, T, vocab_size]

        return out
