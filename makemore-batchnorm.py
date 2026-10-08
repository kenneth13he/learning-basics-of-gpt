import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

block_size = 3

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647)

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out = self.out + self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []

n_embd = 10
n_hidden = 100

C = torch.randn((vocab_size, n_embd), generator=g)
layers = [
    Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size, bias=False), BatchNorm1d(vocab_size),
]

with torch.no_grad():
    layers[-1].gamma *= 0.1
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print('parameters:', sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

max_steps = 200000
batch_size = 32
lossi = []
ud = []

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    emb = C[Xb]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)

    for layer in layers:
        layer.out.retain_grad()
    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 150000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
    with torch.no_grad():
        ud.append([((lr * p.grad).std() / p.data.std()).log10().item() for p in parameters])

print('\nactivations')
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out
        print(f'layer {i:2d}: mean {t.mean():+.2f}, std {t.std():.2f}, saturated {(t.abs() > 0.97).float().mean()*100:.2f}%')

print('\ngradients')
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out.grad
        print(f'layer {i:2d}: mean {t.mean():+e}, std {t.std():e}')

print('\nweight gradients')
for p in parameters:
    if p.ndim == 2:
        print(f'{str(tuple(p.shape)):10s} | mean {p.grad.mean():+e} | std {p.grad.std():e} | grad:data ratio {p.grad.std() / p.std():e}')

fig, axes = plt.subplots(2, 3, figsize=(18, 9))

axes[0, 0].plot(torch.tensor(lossi).view(-1, 1000).mean(1))
axes[0, 0].set_title('log10 loss (mean per 1000 steps)')

legends = []
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        hy, hx = torch.histogram(layer.out.detach(), density=True)
        axes[0, 1].plot(hx[:-1], hy)
        legends.append(f'layer {i}')
axes[0, 1].legend(legends)
axes[0, 1].set_title('activation distribution')

legends = []
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        hy, hx = torch.histogram(layer.out.grad, density=True)
        axes[0, 2].plot(hx[:-1], hy)
        legends.append(f'layer {i}')
axes[0, 2].legend(legends)
axes[0, 2].set_title('gradient distribution')

legends = []
for i, p in enumerate(parameters):
    if p.ndim == 2:
        hy, hx = torch.histogram(p.grad, density=True)
        axes[1, 0].plot(hx[:-1], hy)
        legends.append(f'{i} {tuple(p.shape)}')
axes[1, 0].legend(legends)
axes[1, 0].set_title('weight gradient distribution')

legends = []
for i, p in enumerate(parameters):
    if p.ndim == 2:
        axes[1, 1].plot([ud[j][i] for j in range(len(ud))])
        legends.append(f'param {i}')
axes[1, 1].plot([0, len(ud)], [-3, -3], 'k')
axes[1, 1].legend(legends)
axes[1, 1].set_title('log10 update:data ratio')

axes[1, 2].axis('off')
plt.tight_layout()

for layer in layers:
    if isinstance(layer, BatchNorm1d):
        layer.training = False

@torch.no_grad()
def split_loss(X, Y):
    emb = C[X]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    return F.cross_entropy(x, Y).item()

print('\ntrain loss', split_loss(Xtr, Ytr))
print('dev loss', split_loss(Xdev, Ydev))

g = torch.Generator().manual_seed(2147483647 + 10)
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        with torch.no_grad():
            emb = C[torch.tensor([context])]
            x = emb.view(emb.shape[0], -1)
            for layer in layers:
                x = layer(x)
            probs = F.softmax(x, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

plt.show()
