import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

#hyperparameters

block_size = 8
batch_size = 32
learning_rate = 1e-3
max_train_iteration = 20000
device = "cuda" if torch.cuda.is_available() else "cpu"

#------

with open('../dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
print(chars)
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]

def get_batch(data_type):
    data = train_data if data_type == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i + 1:i+block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

xb, yb = get_batch("train")

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        logits = self.token_embedding_table(idx) #(batch, block, channel(64))
        if targets == None:
            loss = None
        else:
            batch, block, channel = logits.shape
            logits = logits.view(batch*block, channel)
            targets = targets.view(batch*block)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (Batch, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (Batch, Block + 1)

        return idx
    
model = BigramLanguageModel(vocab_size=vocab_size)
model = model.to(device)

logits, loss = model(xb, yb)
print("Before any training: ")
generated_text = model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)[0]
print(decode(generated_text.tolist()))

print("Now training")
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
for step in tqdm(range(max_train_iteration)):
    #get training data in batch
    xb, yb = get_batch("train")

    #evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("After some training: ")
print(loss.item())
generated_text = model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)[0]
print(decode(generated_text.tolist()))