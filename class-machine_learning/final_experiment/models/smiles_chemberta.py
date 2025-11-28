import torch
from transformers import AutoTokenizer, AutoModel


class ChemBERTaEmbedder:
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    @torch.no_grad()
    def encode(self, smiles: str):
        """
        生成 SMILES 的 768 维向量
        """
        inputs = self.tokenizer(smiles, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
