import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import random
from tqdm import tqdm

block_size = 8
batch_size = 64
embedding_space_dimension = 20
class MLP(nn.Module):
    def __init__(self, block_size, embedding_dimension, layer_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(block_size * embedding_dimension, layer_size)
        self.fc2 = nn.Linear(layer_size, output_size)
        self.bn1d = nn.BatchNorm1d(layer_size)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.bn1d(x)
        x = self.fc2(x)
        return x
    
def build_dataset(words):
    X, Y = [], []
    for word in words:
        context = [0] * block_size
        for char in word + "_":
            char_encoded = stoi[char]
            X.append(context)
            Y.append(char_encoded)
            # print(f"X: {''.join(itos[i] for i in context)} ---> Y: {itos[char_encoded]}")
            context = context[1:] + [char_encoded]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    
    return X, Y

if __name__ == "__main__":
    #load and preprocess data
    words = open("../dataset/names.txt", "r", encoding="utf-8").read().splitlines()

    # create character to integer mapping
    chars = sorted(list(set("".join(words))))
    print(chars)
    stoi = { ch:i+1 for i,ch in enumerate(chars) }
    stoi["_"] = 0
    itos = { i:ch for ch, i in stoi.items() }
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    x_train, y_train = build_dataset(words[:n1])
    x_dev, y_dev = build_dataset(words[n1:n2])
    x_test, y_test = build_dataset(words[n2:])

    vocab_size = len(itos)

    lookup_table = torch.randn((vocab_size, embedding_space_dimension)).to("cuda:0")

    model = MLP(block_size=block_size, embedding_dimension=embedding_space_dimension, 
                layer_size=150, output_size=vocab_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    epoch = 100000

    current_best_loss = float("inf")
    model.to("cuda:0")
    x_train = x_train.to("cuda:0")
    y_train = y_train.to("cuda:0")
    
    x_dev = x_dev.to("cuda:0")
    y_dev = y_dev.to("cuda:0")
    
    x_test = x_test.to("cuda:0")
    y_test = y_test.to("cuda:0")
    for i in tqdm(range(epoch)):
        model.train()
        mini_batch_idx = torch.randint(0, x_train.shape[0], (batch_size,))
        
        batched_train_sample = x_train[mini_batch_idx]
        embedding = lookup_table[batched_train_sample].to("cuda:0")

        batched_label_sample = y_train[mini_batch_idx].to("cuda:0")

        output = model(embedding.view(-1, block_size * embedding_space_dimension))
        
        #calculate loss
        loss = F.cross_entropy(output, batched_label_sample)

        #backward pass
        optimizer.zero_grad()
        loss.backward()

        #update weights
        optimizer.step()

        if i > 50000:
             for g in optimizer.param_groups:
                g['lr'] = 0.01
        # elif i > 50000:
        #     for g in optimizer.param_groups:
        #         g['lr'] = 0.01

        # print(f"output: {output}")
        # print(f"pred: {y_train[mini_batch_idx]}")
       
        model.eval()
        #calculate validation loss, save model if this beats the current best
        embedding = lookup_table[x_dev]
        output = model(embedding.view(-1, block_size * embedding_space_dimension))
        val_loss = F.cross_entropy(output, y_dev)
        # if(i % 100 == 0):
        #     print(f"{i}: train loss: {loss.item()}, val loss: {val_loss.item()}")
        #     #print(f"{i}: train loss: {loss.item()}")

        if(val_loss.item() < current_best_loss):
            current_best_loss = val_loss
            print(f"new best current loss:{val_loss}, update saved model...")
            torch.save(model, "MLP.pt")
    
    lookup_table_to_save = {
        "lookup_table":lookup_table,
        "itos": itos
    }
    
    torch.save(lookup_table_to_save, "table.pt")

    #Evaluate test loss
    model = torch.load("MLP.pt")
    model.eval()
    embedding = lookup_table[x_test]
    output = model(embedding.view(-1, block_size * embedding_space_dimension))
    test_loss = F.cross_entropy(output, y_test)
    print(f"Final test loss: {test_loss.item()}")