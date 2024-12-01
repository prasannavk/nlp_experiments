#!/usr/bin/env python
import torch
import torch.nn as nn


inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your    (x^1)
     [0.55, 0.87, 0.66], # journey (x^2)
     [0.57, 0.85, 0.64], # starts  (x^3)
     [0.22, 0.58, 0.33], # with    (x^4)
     [0.77, 0.25, 0.10], # one     (x^5)
     [0.05, 0.80, 0.55]] # step    (x^6)
)



query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)


attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)


print("Sum: ", attn_weights_2_tmp.sum())



def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights: ", attn_weights_2)
print("sum: ", attn_weights_2.sum())



query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2)



attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)


x1 = torch.tensor([0.43, 0.15, 0.89])
x2 = torch.tensor([0.55, 0.87, 0.66])
out = x1.dot(x2)


x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2
print(f"x_2 = {x_2}")
print(f"d_in = {d_in}")

torch.manual_seed(123)
# Why is the matrix dimensions = d_in x d_out?
# If you look at how it is applied, the embedding matrix is applied as a right multiplication
# The question is why the embedding matrix applied as a right multiplication.
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)


query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)


keys = inputs @ W_key
values = inputs @ W_value
print(f"keys.shape {keys.shape}")
print(f"values.shape {values.shape}")


keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)


# Generalizing to a matrix multiplication to compute the attention scores
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)


# Attention scores -> Attention weights.
# Scale the attention scores by sqrt(embedding dimension of the keys),
# and then apply softmax to scale the attention scores

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
print(attn_weights_2)




class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        # print(f"keys {keys}")
        # print(f"values {values}")
        # print(f"queries {queries}")
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores/ keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec




class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)

        queries = self.W_query(x)
        values = self.W_value(x)
        # print(f"keys {keys}")
        # print(f"values {values}")
        # print(f"queries {queries}")
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

def main():
    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))
    print(f"shape of v1 W_query {sa_v1.W_query.data.shape}")

    torch.manual_seed(123)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(f"shape of W_query {sa_v2.W_query.weight.shape}")
    print(f"shape of v1 W_query {sa_v1.W_query.data.shape}")

    sa_v2.W_query.weight = torch.nn.Parameter(sa_v1.W_query.data.T)
    sa_v2.W_key.weight = torch.nn.Parameter(sa_v1.W_key.data.T)
    sa_v2.W_value.weight = torch.nn.Parameter(sa_v1.W_value.data.T)
    print(sa_v2(inputs))
    print(sa_v2.W_query.weight)
    print(sa_v1.W_query.data)


if __name__ == '__main__':
    main()

