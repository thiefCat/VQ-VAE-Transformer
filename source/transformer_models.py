import torch
import torch.nn as nn
import source.utils as ut
import einops
import math
from torch.nn import functional as F

future_mask = torch.triu(torch.zeros([1024, 1024]).fill_(float("-inf")), 1)  # maximum length 1024

def scaled_dot_product_attention(q, k, v, causal=False):
    '''
    To enable both cross attention and self attention, we separate source_len and target_len
    so that in self attention, q=k=v; in cross attention, k=v, q is the vectors to do query, 
    k=v is the vectors to be queried
    Input:
    q: [n, target_len, d]
    k: [n, source_len, d]
    v: [n, source_len, d]
    d is the dimension for each head
    n represents the total number of attention operations calculated in parallel. 
    In multi-head attention, it's the product of the batch size and the number of attention heads. 
    For each of the `n` operations, compute attention scores

    Return:
    attention output: [n, target_len, d]
    '''
    d_head = q.size(-1)
    # computes dot product from query to keys
    s = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(d_head)  # [n, target_len, source_len]
    # mask future dot products: s_ij = -inf if j > i
    if causal:
        attn_mask = future_mask[:s.size(-2), :s.size(-1)].to(s)  # [target_len, source_len]
        s += attn_mask.unsqueeze(0)
    softmax = F.softmax(s, dim=-1, dtype=torch.float32).type_as(s) # [n, target_len, source_len]
    a = torch.bmm(softmax, v) # [n, target_len, d]
    return a


class MultiheadAttention(nn.Module):
    '''
    Multihead attention: in each attention step, we have multiple keys, queries, and values, i.e., 
    multiple heads. Each head can attend to different input vectors so that the attention output
    can comprise more input informations.

    This model receives h [batch, target_len, d], producing q, k, v, and then do multihead attention
    to produce the result. The final out_proj layer integrates the outputs from all the heads into a 
    final output by using nn.Linear().
    
    Since in each attention layer, there are n_heads heads for attention operation, each head should have 
    a separate set of Wq, Wk, Wv to produce (query, key, value), we make a larger projection layer so that 
    instead of creating n_heads (Wq, Wk, Wv) of shape [d, d], we create [d, n_heads * d], and then split 
    it into n_heads [d, d] matrices.
    '''
    def __init__(self, d, n_heads):
        '''
        d: dimension of head vectors
        n_heads: number of heads
        '''
        super().__init__()
        # D is d x n_heads
        D = n_heads * d
        self.q_proj = nn.Linear(d, D)
        self.k_proj = nn.Linear(d, D)
        self.v_proj = nn.Linear(d, D)
        self.out_proj = nn.Linear(D, d)
        self.n_heads = n_heads
    
    def forward(self, q, k, v, causal=True):
        '''
        Input:
        q: [batch, target_len, d]
        k: [batch, source_len, d]
        v: [batch, source_len, d]

        Return:
        a: [bach, target_len, d]
        '''
        q = self.q_proj(q) # [batch, target_len, D]
        k = self.k_proj(k) # [batch, source_len, D]
        v = self.v_proj(v) # [batch, source_len, D]
        q = einops.rearrange(q, "b tl (nh d) -> (b nh) tl d", nh=self.n_heads)  # [n, target_len, d]
        k = einops.rearrange(k, "b sl (nh d) -> (b nh) sl d", nh=self.n_heads)
        v = einops.rearrange(v, "b sl (nh d) -> (b nh) sl d", nh=self.n_heads)
        o = scaled_dot_product_attention(q, k, v, causal) # [n, target_len, d]
        o = einops.rearrange(o, "(b nh) tl d -> b tl (nh d)", nh=self.n_heads) # [b, target_len, D]
        a = self.out_proj(o) # [b, target_len, d]
        return a


class TransformerLayer(nn.Module):
    '''
    A transformer layer is composed by an attention layer and a feed forward network.
    '''
    def __init__(self, d, n_heads, d_ffn, p_drop, causal=True):
        super().__init__()
        self.attention = MultiheadAttention(d, n_heads)
        self.dropout = nn.Dropout(p_drop)
        # Layer norm: [batch, seq, d] --> mean: [batch, seq]
        self.attention_ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d_ffn)
        self.fc2 = nn.Linear(d_ffn, d)
        self.ffn_ln = nn.LayerNorm(d)
        self.causal = causal

    def attention_layer(self, x):
        residual = x
        a = self.attention(x, x, x, self.causal)
        a = self.dropout(a)
        a_bar = self.attention_ln(a + residual)
        return a_bar
    
    def feed_forward(self, a_bar):
        residual = a_bar
        h = self.fc2(F.relu(self.fc1(a_bar)))
        h = self.dropout(h)
        h_bar = self.ffn_ln(h + residual)
        return h_bar

    def forward(self, x):
        '''
        Input:
        x: [b, seq_len, d]
        Return:
        h_bar: [b, seq_len, d]
        '''
        a_bar = self.attention_layer(x)
        h_bar = self.feed_forward(a_bar)
        return h_bar
        

def positional_encoding(length, d_model):

    def get_angles(pos, i, d_model):
        angles = 1 / torch.pow(10000, (2 * (i // 2)) / torch.tensor(d_model, dtype=torch.float32))
        return pos * angles

    angle_rads = get_angles(
        torch.arange(length)[:, None],
        torch.arange(d_model)[None, :],
        d_model
    )

    # Apply sin to even indices in the array; 2i
    sines = torch.sin(angle_rads[:, 0::2])
    # Apply cos to odd indices in the array; 2i+1
    cosines = torch.cos(angle_rads[:, 1::2])

    pos_encoding = torch.cat([sines, cosines], dim=-1)
    # pos_encoding = pos_encoding[None, ...]

    return pos_encoding


class AutoregressiveTransformer(nn.Module):
    def __init__(self, num_layers, d, K, max_seq_len, num_heads, d_ffn, p_drop, vqvae, conditional=False):
        super().__init__()
        # the (K+1)th token is the sos token
        self.K = K
        self.max_seq_len = max_seq_len
        if conditional:
            self.token_embedding = nn.Embedding(K+1, d)
        else:
            self.token_embedding = nn.Embedding(K+10, d)
        # codebook_weights = vqvae.codebook.embedding.weight.data
        # self.token_embedding.weight[:-1].data = codebook_weights.clone()
        self.positional_embedding = nn.Embedding(max_seq_len, d)
        # self.positional_embedding.weight.data = positional_encoding(max_seq_len, d)
        # for param in self.positional_embedding.parameters():
        #     param.requires_grad = False
        self.input_ln = nn.LayerNorm(d)
        # Transformer block
        self.transformer = nn.ModuleList([
            TransformerLayer(d, num_heads, d_ffn, p_drop)
            for _ in range(num_layers)
        ])
        # Final linear layer to get logits for next token prediction
        self.get_logit = nn.Linear(d, K)

    def forward(self, x):
        '''
        Inputs:
        x: [batch, seq_len]
        Return:
        logits: [b, seq_len, K], which is the values before softmax operation
        '''
        embeddings = self.token_embedding(x) # [b, seq_len+1, d]
        # add positional embeddings
        positions = torch.arange(0, embeddings.size(1)).to(x.device) # [seq_len+1,] [0, 1, 2, ..., seq_len]
        positions = positions.expand((embeddings.size(0), embeddings.size(1))) # [b, seq_len+1]
        position_embeddings = self.positional_embedding(positions) # [b, seq_len+1, d]
        embeddings += position_embeddings
        embeddings = self.input_ln(embeddings)
        for layer in self.transformer:
            embeddings = layer(embeddings)
        logits = self.get_logit(embeddings) # [b, seq_len+1, K]
        return logits
    
    def sample_from_prior(self, num_samples, temperature, k, device):
        sequences = [[self.K] for _ in range(num_samples)]
        current = torch.Tensor(sequences).long().to(device) # [b, 1]
        # print(current.shape)
        for _ in range(self.max_seq_len - 1):
            logits = self.forward(current)[:, -1, :] # [b, K]
            # the bigger the temperature, the more random samples will be
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
            sampled_indices = torch.multinomial(top_k_probs, num_samples=1)
            next_tokens = top_k_indices.gather(dim=-1, index=sampled_indices) # [b, 1]
            # print(next_tokens.shape)
            current = torch.cat((current, next_tokens), dim=1)
            # print(current.shape)
        current = current[:,1:] # [b, seq_len]
        # print(current.shape)
        return current
    
    def sample_from_class(self, num_samples, y, temperature, k, device):
        y = y + self.K
        sequences = [[y] for _ in range(num_samples)]
        current = torch.Tensor(sequences).long().to(device) # [b, 1]
        # print(current.shape)
        for _ in range(self.max_seq_len - 1):
            logits = self.forward(current)[:, -1, :] # [b, K]
            # the bigger the temperature, the more random samples will be
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
            sampled_indices = torch.multinomial(top_k_probs, num_samples=1)
            next_tokens = top_k_indices.gather(dim=-1, index=sampled_indices) # [b, 1]
            # print(next_tokens.shape)
            current = torch.cat((current, next_tokens), dim=1)
            # print(current.shape)
        current = current[:,1:] # [b, seq_len]
        # print(current.shape)
        return current

# if __name__ == '__main__':
#     x = positional_encoding(10, 5)
#     print(x)