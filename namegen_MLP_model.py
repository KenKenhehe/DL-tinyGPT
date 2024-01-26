import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open("dataset/names.txt", "r").read().splitlines()

# create character to integer mapping
chars = sorted(list(set("".join(words))))
stoi = { ch:i+1 for i,ch in enumerate(chars) }
stoi["."] = 0
itos = { i:ch for ch, i in stoi.items() }

vocab_size = len(itos)
block_size = 3
embedding_space_dimention = 14

def build_dataset(words):
    X, Y = [], []
    for word in words:
        context = [0] * block_size
        for char in word + ".":
            char_encoded = stoi[char]
            X.append(context)
            Y.append(char_encoded)
            # print(f"X: {''.join(itos[i] for i in context)} ---> Y: {itos[char_encoded]}")
            context = context[1:] + [char_encoded]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    
    return X, Y

# train, val, test split(80, 10, 10)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
x_train, y_train = build_dataset(words[:n1])
x_dev, y_dev = build_dataset(words[n1:n2])
x_test, y_test = build_dataset(words[n2:])

lookup_table = torch.randn((vocab_size, embedding_space_dimention))
W1 = torch.randn((block_size * embedding_space_dimention, 300)) * 0.2
b1 = torch.randn(300) * 0.01
W2 = torch.randn((300, vocab_size)) * 0.01
b2 = torch.randn(vocab_size) * 0
parameters = [lookup_table, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

search_lr = False
#Search for optimal learning rate
lr_exp_search_space = torch.linspace(-3, 0, 1000)
lr_search_space = 10 **lr_exp_search_space
lri = []
lossi = []

step = 200000
batch_size = 64

for i in range(step):
    #make mini batch
    mini_batch_idx = torch.randint(0, x_train.shape[0], (batch_size,))
    
    #Forward pass
    embedding = lookup_table[x_train[mini_batch_idx]]
    # equvalent to torch.cat(torch.unbind(embedding, 1), 1)
    h1_output = torch.tanh(embedding.view(-1, block_size * embedding_space_dimention) @ W1 + b1)
    logits = h1_output @ W2 + b2
    counts = logits.exp()
    prob = counts / counts.sum(1, keepdims=True)
    loss = F.cross_entropy(logits, y_train[mini_batch_idx]) # equvalent to -prob[torch.arange(X.shape[0]), Y].log().mean()
    #Backword pass
    for p in parameters:
        p.grad = None
    loss.backward()   

    #lr = lr_search_space[i]
    if(i > 100000):
        lr = 0.01
    else:
        lr = 0.1
    for p in parameters:
        p.data -= lr * p.grad 
        
    #track learning rate stat
    if search_lr: 
        lri.append(lr_exp_search_space[i])
        lossi.append(loss.item())

if search_lr:
    plt.plot(lri, lossi)
    plt.show()
#train loss    
print(f"train loss: {loss.item()}")

#evaluate loss(Do a forward pass once)
embedding = lookup_table[x_dev]
h1_output = torch.tanh(embedding.view(-1, block_size * embedding_space_dimention) @ W1 + b1)
logits = h1_output @ W2 + b2
loss = F.cross_entropy(logits, y_dev)

print(f"test loss: {loss.item()}")
