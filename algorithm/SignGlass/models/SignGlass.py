import torch
import torch.nn as nn
from models.EgoCorrNet import EgoCorrNet
from models.ASLTranslator import ASLTranslator
from utils.predict_gloss import predict_gloss
from models.CorrNetPlus import CorrNetPlusBackbone, BiLSTMDecoder
from models.EgoFaceNet import EgoFaceNet

class SignGlass(nn.Module):
    def __init__(self, gloss_tokenizer, translator_model_name="t5-small"):
        super().__init__()
        self.gloss_tokenizer = gloss_tokenizer

        # Manual marker recognition
        self.egocorrnet = EgoCorrNet(
            backbone_cnn=CorrNetPlusBackbone(), 
            temporal_decoder=BiLSTMDecoder(vocab_size=gloss_tokenizer.vocab_size)
        )

        # Non-manual marker classification (facial syntax)
        self.face_encoder = EgoFaceNet()  # TimeSFormer + FiLM + Syntax Classifier

        # Gloss-to-English translation
        self.translator = ASLTranslator(model_name=translator_model_name)

        # Loss functions
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
        self.syntax_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        video_tensor,                # [B, C, T, H, W]
        face_tensor=None,           # Tuple: (face_L, face_R): each [B, T, C, H, W]
        syntax_label=None,          # Tensor: [B]
        target_sentences=None,      # list[str]
        target_gloss_ids=None,      # list[list[int]]
        return_all=False
    ):
        # 1. Predict gloss logits
        gloss_logits = self.egocorrnet(video_tensor)  # [B, T, vocab_size]

        # 2. Decode gloss to text
        gloss_texts = predict_gloss(gloss_logits, self.gloss_tokenizer, method="greedy")

        # 3. Translation
        translator_output = self.translator(gloss_texts, target_sentences)

        # 4. CTC loss
        loss_ctc = None
        if target_gloss_ids is not None:
            input_lengths = torch.full(
                size=(gloss_logits.size(0),),
                fill_value=gloss_logits.size(1),
                dtype=torch.long,
                device=gloss_logits.device
            )
            target_lengths = torch.tensor(
                [len(g) for g in target_gloss_ids],
                dtype=torch.long,
                device=gloss_logits.device
            )
            target_gloss_flat = torch.cat(
                [torch.tensor(g, dtype=torch.long, device=gloss_logits.device) for g in target_gloss_ids]
            )
            loss_ctc = self.ctc_loss(
                gloss_logits.log_softmax(2).transpose(0, 1),
                target_gloss_flat,
                input_lengths,
                target_lengths
            )

        # 5. Syntax classification loss (if face & label provided)
        loss_syntax = None
        syntax_logits = None
        if face_tensor is not None and syntax_label is not None:
            face_L, face_R = face_tensor  # each is [B, T, C, H, W]
            gloss_embed = self.translator.encode_gloss(gloss_texts)  # [B, D]
            syntax_logits = self.face_encoder(face_L, face_R, gloss_embed)  # [B, 4]
            loss_syntax = self.syntax_loss_fn(syntax_logits, syntax_label)

        # 6. Translation Loss
        loss_seq2seq = translator_output.loss if hasattr(translator_output, "loss") else None

        # 7. Combine losses
        total_loss = 0
        if loss_ctc is not None:
            total_loss += loss_ctc
        if loss_seq2seq is not None:
            total_loss += loss_seq2seq
        if loss_syntax is not None:
            total_loss += loss_syntax

        # 8. Return
        if return_all:
            return {
                "loss": total_loss,
                "gloss_logits": gloss_logits,
                "gloss_str": gloss_texts,
                "translator_output": translator_output,
                "syntax_logits": syntax_logits
            }

        return total_loss
