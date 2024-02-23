import torch
import torch.nn.functional as F
from namegen_MLP_model import block_size

chk_data = torch.load("MLP.pt")

itos = chk_data["itos"]
lookup_table = chk_data["lookup_table"]
W1 = chk_data["W1"]
W2 = chk_data["W2"]
b1 = chk_data["b1"]
b2 = chk_data["b2"]

def sample(num: int):
    g = torch.Generator()
    for _ in range(num):
        out = []
        context = [0] * block_size
        while True:
            embedding = lookup_table[torch.tensor([context])]
            h1_output = torch.tanh(embedding.view(1, -1) @ W1 + b1)
            logits = h1_output @ W2 + b2
            probs = F.softmax(logits, dim=1)
            #sample from distribution
            index = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [index]
            out.append(index)
            if index == 0:
                break
        print("".join(itos[i] for i in out))
if __name__ == "__main__":
    sample(15)