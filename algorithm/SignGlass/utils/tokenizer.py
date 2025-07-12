import os
import json
import torch
import torch.nn.functional as F


def predict_gloss(logits, tokenizer, method="greedy", beam_width=5, blank_id=0):
    """
    Args:
        logits: [B, T, C] raw output from CorrNetPlus (before softmax)
        tokenizer: GlossTokenizer with decode() method
        method: "greedy" or "beam"
        beam_width: used if method == "beam"
        blank_id: usually 0 for CTC blank
    Returns:
        List[str]: decoded gloss strings per sample
    """
    B, T, C = logits.shape
    gloss_strings = []

    probs = F.log_softmax(logits, dim=-1)  # for CTC decoding

    if method == "greedy":
        pred_ids = torch.argmax(probs, dim=-1)  # [B, T]
        for ids in pred_ids:
            collapsed = []
            prev = -1
            for i in ids.cpu().tolist():
                if i != prev and i != blank_id:
                    collapsed.append(i)
                prev = i
            gloss = tokenizer.decode(collapsed)
            gloss_strings.append(gloss)

    elif method == "beam":
        # naive beam search per sample
        from ctcdecode import CTCBeamDecoder  # requires install
        decoder = CTCBeamDecoder(
            labels=tokenizer.id2token,
            blank_id=blank_id,
            beam_width=beam_width,
            log_probs_input=True
        )
        beam_out, _, _, out_lens = decoder.decode(probs)
        for i in range(B):
            ids = beam_out[i][0][:out_lens[i][0]].tolist()
            gloss_strings.append(tokenizer.decode(ids))

    else:
        raise ValueError("Unknown decode method")

    return gloss_strings

class GlossTokenizer:
    def __init__(self, vocab_file=None, special_tokens=None):
        """
        vocab_file: path to a json file mapping gloss tokens to IDs
        special_tokens: optional list of special tokens like ["<pad>", "<unk>", "<blank>"]
        """
        self.vocab = {}
        self.id2token = {}
        self.special_tokens = special_tokens or ["<pad>", "<unk>", "<blank>"]

        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
        else:
            # Initialize empty vocab with special tokens
            for token in self.special_tokens:
                self.add_token(token)

    def add_token(self, token):
        if token not in self.vocab:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.id2token[idx] = token

    def encode(self, gloss_seq):
        """
        gloss_seq: string of glosses, e.g., "WHAT YOUR NAME"
        returns: list of token IDs
        """
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in gloss_seq.split()]

    def decode(self, id_seq, skip_special=True):
        """
        id_seq: list of token IDs
        returns: gloss string, e.g., "WHAT YOUR NAME"
        """
        tokens = [self.id2token.get(idx, "<unk>") for idx in id_seq]
        if skip_special:
            tokens = [t for t in tokens if t not in self.special_tokens]
        return " ".join(tokens)

    def build_vocab_from_data(self, list_of_gloss_strs):
        """Takes list of gloss strings and builds a vocab."""
        for seq in list_of_gloss_strs:
            for tok in seq.split():
                self.add_token(tok)

    def save_vocab(self, path):
        with open(path, "w") as f:
            json.dump(self.vocab, f, indent=2)

    def load_vocab(self, path):
        with open(path, "r") as f:
            self.vocab = json.load(f)
        self.id2token = {v: k for k, v in self.vocab.items()}

    def __len__(self):
        return len(self.vocab)

    def pad_id(self):
        return self.vocab.get("<pad>", 0)

    def blank_id(self):
        return self.vocab.get("<blank>", self.vocab.get("<pad>", 0))