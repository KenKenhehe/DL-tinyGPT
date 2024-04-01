import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

#hyperparameters

block_size = 256
batch_size = 64
learning_rate = 3e-4
max_train_iteration = 6000
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embed = 128

generate_text = True

n_head = 6
n_layer = 8
dropout = 0.2
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
        
class Head(nn.Module):
    """
    Single head self-attention
    """
    def __init__(self, head_size:int):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor):
        B,T,C = x.shape #Batch, Block, Channel
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_head:int, head_size:int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(num_head * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embed:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """Transformer block, communication followed by computation"""

    def __init__(self, n_embed:int, n_head:int):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, n_embedding, block_size = 8):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding_table = nn.Embedding(block_size, n_embedding)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embedding, vocab_size)
        
    def forward(self, idx, targets = None):
        B, T = idx.shape #Batch, Block
        token_emb = self.token_embedding_table(idx) #(batch, block, channel(64))
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(Block, Channel)
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
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
            idx_cropped = idx[:, -block_size:]
            logits, loss = self(idx_cropped)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (Batch, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (Batch, Block + 1)

        return idx
    
model = Transformer(vocab_size=vocab_size, n_embedding=n_embed, block_size=block_size)
model = model.to(device)

logits, loss = model(xb, yb)

if generate_text:
    print("Before any training: ")
    generated_text = model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)[0]
    print(decode(generated_text.tolist()))

print("Now training")
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
current_best_loss = float("inf")
for step in range(max_train_iteration):
    model.train()
    #get training data in batch
    xb, yb = get_batch("train")

    #evaluate loss
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    model.eval()
    xb, yb = get_batch("val")
    logits, val_loss = model(xb, yb)
    
    if(step % 500 == 0):
        print(f"step {step}   train loss: {loss}, val loss: {val_loss}")
    if(val_loss.item() < current_best_loss):
        current_best_loss = val_loss
        print(f"new best current loss:{val_loss}, update saved model...")
        torch.save(model, "transformer.pt")

if generate_text:
    print(f"trained model best loss: {current_best_loss.item()}")
    print("Generating text from trained model: ")
    
    model = torch.load("transformer.pt")
    model.eval()
    generated_text = model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=2000)[0]
    print(decode(generated_text.tolist()))