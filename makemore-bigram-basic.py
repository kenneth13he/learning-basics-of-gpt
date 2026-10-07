import torch
import matplotlib.pyplot as plt


words = open('names.txt', 'r').read().splitlines()

# creating a 27 by 27 array to store all the values for each digram
N = torch.zeros((27,27), dtype=torch.int32)

#creating a list of all alphabet characters
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
    # this is the starting character + the character + end character
    # using this we can know how like a letter is to start/end a word
    chs = ["."] + list(w) + ["."]

    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] += 1

# for efficiency, create a matrix p 
# normalize every row of N at once so each row is a probability distribution
# keepdim=True keeps the sums as a (27,1) column so it broadcasts across each row
P = (N+1).float()  # +1 smoothing: no pair gets probability 0, so log(0) = -inf never happens
P /= P.sum(1, keepdim=True)

g = torch.Generator().manual_seed(2147483647)
ix = 0
out = []
for i in range(50):
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
    out = []

# evaluate the model with the negative log likelihood (lower is better)
# likelihood = product of the probabilities the model gave every bigram in the data
# log turns that product into a sum: log(a*b*c) = log(a) + log(b) + log(c)
log_likelihood = 0.0
n = 0
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        log_likelihood += torch.log(prob)
        n += 1

nll = -log_likelihood
print(f'{log_likelihood=}')
print(f'{nll=}')
print(f'{nll/n}')  # average negative log likelihood = the loss
