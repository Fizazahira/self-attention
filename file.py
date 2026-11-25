import numpy as np  # for the matrix stuff

def softmax(x):  # turns scores into probabilities
    x = x - np.max(x, axis=-1, keepdims=True)  # subtract max so exp does not blow up
    e = np.exp(x)  # exp of everything
    return e / np.sum(e, axis=-1, keepdims=True)  # divide by sum so rows add up to 1

def self_attention(X, Wq, Wk, Wv):  # forward pass with X shaped seq_len by d_model
    Q = X @ Wq  # query is what each token is looking for
    K = X @ Wk  # key is what each token has to offer
    V = X @ Wv  # value is the actual info we pass along

    d_k = K.shape[-1]  # key dimension needed for scaling
    scores = Q @ K.T / np.sqrt(d_k)  # every query dot every key then scale it down

    weights = softmax(scores)  # attention weights
    output = weights @ V  # mix the values using the weights

    return output, weights  # returning weights too so i can check them

np.random.seed(0)  # same random numbers every time
seq_len, d_model, d_k = 4, 8, 8  # 4 tokens and dims of 8

X = np.random.randn(seq_len, d_model)  # random input
Wq = np.random.randn(d_model, d_k)  # random Wq
Wk = np.random.randn(d_model, d_k)  # random Wk
Wv = np.random.randn(d_model, d_k)  # random Wv

out, attn = self_attention(X, Wq, Wk, Wv)  # run it

print("output shape:", out.shape)  # should be 4 by 8
print("attention weights:\n", attn)  # each row should add up to 1