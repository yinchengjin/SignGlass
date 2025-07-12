import os
import yaml
import torch
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.SignGlass import SignGlass
from dataset.SignGlassDataset import SignGlassDataset
from tokenizer.gloss_tokenizer import GlossTokenizer
from evaluator.bleu import compute_bleu
from utils.logger import Recorder

class Processor():
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.work_dir = self.config["save_dir"]
        os.makedirs(self.work_dir, exist_ok=True)
        shutil.copy2(config_path, os.path.join(self.work_dir, "config.yaml"))

        self.recoder = Recorder(self.work_dir, print_log=True, log_interval=10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_data()
        self._load_model()

    def _load_data(self):
        gloss_tokenizer = GlossTokenizer(self.config["gloss_vocab"])
        self.tokenizer = gloss_tokenizer

        self.train_set = SignGlassDataset(self.config["train_list"], gloss_tokenizer)
        self.val_set = SignGlassDataset(self.config["val_list"], gloss_tokenizer)

        self.train_loader = DataLoader(self.train_set, batch_size=self.config["batch_size"], shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=self.config["batch_size"])

    def _load_model(self):
        self.model = SignGlass(self.tokenizer).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Train Epoch {epoch}")):
            video, sentence = batch
            video = video.to(self.device)

            self.optimizer.zero_grad()
            loss = self.model(video, target_sentences=sentence)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if batch_idx % self.recoder.log_interval == 0:
                self.recoder.print_log(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(self.train_loader)
        self.recoder.print_log(f"Epoch {epoch} | Average Loss: {avg_loss:.4f}")
        return avg_loss

    def _evaluate(self, epoch):
        self.model.eval()
        predictions = []
        references = []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                video, sentence = batch
                video = video.to(self.device)
                output = self.model(video, sentence, return_all=True)
                predictions.extend(output["translator_output"]["decoded"])
                references.extend(sentence)
        bleu = compute_bleu(predictions, references)
        self.recoder.print_log(f"Epoch {epoch} | BLEU Score: {bleu:.2f}")
        return bleu

    def start(self):
        best_bleu = 0.0
        for epoch in range(1, self.config["epochs"] + 1):
            self.recoder.print_log(f"========== Epoch {epoch} ==========")
            train_loss = self._train_one_epoch(epoch)
            bleu_score = self._evaluate(epoch)

            is_best = bleu_score > best_bleu
            if is_best:
                best_bleu = bleu_score
                best_path = os.path.join(self.work_dir, "best_model.pth")
                torch.save(self.model.state_dict(), best_path)
                self.recoder.print_log("Saved best model.")

            model_path = os.path.join(self.work_dir, f"model_epoch{epoch}.pth")
            torch.save(self.model.state_dict(), model_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="SignGlass Training Entry")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    processor = Processor(args.config)
    processor.start()
