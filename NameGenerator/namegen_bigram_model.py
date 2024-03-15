import torch
import matplotlib.pyplot as plt

words = open("../dataset/names.txt", "r").read().splitlines()

char_count_matrix = torch.zeros((27, 27), dtype=torch.int32)

# create character to integer mapping
chars = sorted(list(set("".join(words))))
stoi = { ch:i+1 for i,ch in enumerate(chars) }
stoi["."] = 0
itos = { i:ch for ch, i in stoi.items() }

#training bigram(counting the probability matrix)
for w in words:
    full_chars = ["."] + list(w) + ["."]
    for char1, char2 in zip(full_chars, full_chars[1:]):
        idx1 = stoi[char1]
        idx2 = stoi[char2]
        char_count_matrix[idx1, idx2] += 1

# visualization
visualize = False
if visualize:
    plt.figure(figsize=(20, 20))
    plt.imshow(char_count_matrix, cmap="Blues")
    for i in range(27):
        for j in range(27):
            char_str = itos[i] + itos[j]
            plt.text(j, i, char_str, ha="center", va="bottom", color="gray")
            plt.text(j, i, char_count_matrix[i, j].item(), ha="center", va="top", color="gray")
    plt.axis("off")
    plt.show()

#Generate name with bigram
generator = torch.Generator()
prob_matrix = char_count_matrix.float()
prob_matrix = prob_matrix / prob_matrix.sum(1, keepdim=True)

for i in range(20):
    idx = 0
    output_name = []
    while True:
        current_prob = prob_matrix[idx]
        idx = torch.multinomial(current_prob, num_samples=1, replacement=True, generator=generator).item()
        output_name.append(itos[idx])
        if idx == 0:
            break

    print("".join(output_name))