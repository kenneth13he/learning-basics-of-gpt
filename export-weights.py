import os
os.environ['MPLBACKEND'] = 'Agg'

import json
import runpy
import torch

def flat(t, digits=4):
    return [round(v, digits) for v in t.detach().flatten().tolist()]

def pca2(C):
    X = C.detach() / C.detach().norm(dim=1, keepdim=True)
    X = X - X.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    return [[round(a, 3), round(b, 3)] for a, b in (X @ Vh[:2].T).tolist()]

bigram = runpy.run_path('makemore-bigram-basic.py')
mlp = runpy.run_path('makemore-mlp.py')
wavenet = runpy.run_path('makemore-wavenet.py')

itos = bigram['itos']
chars = [itos[i] for i in range(len(itos))]

layers = wavenet['model'].layers
blocks = []
for k in range(3):
    linear, bn = layers[2 + 4 * k], layers[3 + 4 * k]
    blocks.append({
        'fanIn': linear.weight.shape[0],
        'W': flat(linear.weight),
        'gamma': flat(bn.gamma),
        'beta': flat(bn.beta),
        'mean': flat(bn.running_mean),
        'var': flat(bn.running_var),
    })

with torch.no_grad():
    mlp_check = torch.softmax(torch.tanh(mlp['C'][torch.tensor([[0, 5, 13]])].view(1, -1) @ mlp['W1'] + mlp['b1']) @ mlp['W2'] + mlp['b2'], 1)
    wavenet_check = torch.softmax(wavenet['model'](torch.tensor([[0, 0, 0, 0, 0, 5, 13, 13]])), 1)

data = {
    'chars': chars,
    'bigram': {
        'counts': bigram['N'].tolist(),
        'loss': round((-bigram['log_likelihood'] / bigram['n']).item(), 4),
    },
    'mlp': {
        'blockSize': mlp['block_size'],
        'nEmbd': mlp['C'].shape[1],
        'nHidden': mlp['W1'].shape[1],
        'C': flat(mlp['C']),
        'W1': flat(mlp['W1']),
        'b1': flat(mlp['b1']),
        'W2': flat(mlp['W2']),
        'b2': flat(mlp['b2']),
        'emb2d': pca2(mlp['C']),
        'params': sum(p.nelement() for p in mlp['parameters']),
        'trainLoss': round(mlp['split_loss'](mlp['Xtr'], mlp['Ytr']), 4),
        'devLoss': round(mlp['split_loss'](mlp['Xdev'], mlp['Ydev']), 4),
        'check': flat(mlp_check[0], 6),
    },
    'wavenet': {
        'blockSize': wavenet['block_size'],
        'nEmbd': wavenet['n_embd'],
        'nHidden': wavenet['n_hidden'],
        'C': flat(layers[0].weight),
        'blocks': blocks,
        'Wout': flat(layers[13].weight),
        'bout': flat(layers[13].bias),
        'emb2d': pca2(layers[0].weight),
        'params': sum(p.nelement() for p in wavenet['parameters']),
        'trainLoss': round(wavenet['split_loss'](wavenet['Xtr'], wavenet['Ytr']), 4),
        'devLoss': round(wavenet['split_loss'](wavenet['Xdev'], wavenet['Ydev']), 4),
        'check': flat(wavenet_check[0], 6),
    },
}

os.makedirs('docs', exist_ok=True)
with open('docs/models.json', 'w') as f:
    json.dump(data, f, separators=(',', ':'))
print('wrote docs/models.json', os.path.getsize('docs/models.json') // 1024, 'KB')
