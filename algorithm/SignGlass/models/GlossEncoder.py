import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

class ASLTranslator(nn.Module):
    def __init__(self, model_name="t5-small", max_length=64, num_beams=4):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.max_length = max_length
        self.num_beams = num_beams

    def forward(self, gloss_texts, target_texts=None, return_all=False):
        """
        gloss_texts: list of gloss strings, e.g., ["WHAT YOUR NAME"]
        target_texts: list of English strings (optional)
        return_all: whether to return decoded outputs, loss, and logits
        """
        device = self.model.device
        input_encoding = self.tokenizer(
            gloss_texts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        if target_texts is not None:
            target_encoding = self.tokenizer(
                target_texts, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            labels = target_encoding.input_ids
            labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore pad in loss

            outputs = self.model(
                input_ids=input_encoding.input_ids,
                attention_mask=input_encoding.attention_mask,
                labels=labels,
                return_dict=True,
            )

            if return_all:
                decoded = self.tokenizer.batch_decode(
                    torch.argmax(outputs.logits, dim=-1), skip_special_tokens=True
                )
                return {
                    "loss": outputs.loss,
                    "logits": outputs.logits,
                    "decoded": decoded
                }
            else:
                return outputs.loss
        else:
            outputs = self.model.generate(
                input_ids=input_encoding.input_ids,
                attention_mask=input_encoding.attention_mask,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True
            )
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
