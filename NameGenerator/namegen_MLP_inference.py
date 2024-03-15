import torch
import torch.nn.functional as F
from namegen_MLP_torch_train import block_size, embedding_space_dimension, MLP

model = torch.load('MLP.pt')
table_data = torch.load("table.pt")
lookup_table = table_data["lookup_table"]
itos = table_data["itos"]
# itos = chk_data["itos"]
# lookup_table = chk_data["lookup_table"]
# W1 = chk_data["W1"]
# W2 = chk_data["W2"]
# b1 = chk_data["b1"]
# b2 = chk_data["b2"]

def sample(num: int):
    model.eval()
    for _ in range(num):
        out = []
        context = [0] * block_size
        while True:
            #torch inference
            embedding = lookup_table[torch.tensor([context]).to("cuda")].to("cuda")
            output = model(embedding.view(-1, block_size * embedding_space_dimension)).to("cuda")
            probs = F.softmax(output, dim=1).to("cuda")
            #--------
            # embedding = lookup_table[torch.tensor([context])]
            # h1_output = torch.tanh(embedding.view(1, -1) @ W1 + b1)
            # logits = h1_output @ W2 + b2
            # probs = F.softmax(logits, dim=1)
            #sample from distribution
            index = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [index]
            out.append(index)
            if index == 0:
                break
        print("".join(itos[i] for i in out)[:-1])
if __name__ == "__main__":
    sample(30)