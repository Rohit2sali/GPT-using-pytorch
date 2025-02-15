import torch
import torch.nn as nn
from torch.nn import functional as F

class Transformer(nn.Module):
    def __init__(self, n_heads : int, emd_dim : int, max_seq_len : int):
        super(Transformer, self).__init__()
        d_v = emd_dim // n_heads
        self.mhsa = MaskedMultiHeadSelfAttention(n_heads, d_v, emd_dim, max_seq_len)
        self.feed_forward = FeedForward(emd_dim)
        self.norm_1 = nn.LayerNorm(emd_dim)
        self.norm_2 = nn.LayerNorm(emd_dim)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = x + self.mhsa(self.norm_1(x))
        x = x + self.feed_forward(self.norm_2(x))
        # shape of x : (batch_size, max_seq_len, d_model)
        return x
    

class FeedForward(nn.Module):
    def __init__(self, n_emb : int, extend_width : int=4, dropout : float=0.2):
        super(FeedForward, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(n_emb, extend_width * n_emb),
            nn.ReLU(),
            nn.Linear(extend_width * n_emb, n_emb),
            nn.Dropout(dropout)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.layer(x)
    

class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads : int, d_v : int, emd_dim : int, max_seq_len : int, dropout : float=0.2):
        super(MaskedMultiHeadSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([SingleHead(d_v, emd_dim, max_seq_len) for _ in range(self.n_heads)])
        self.project = nn.Linear(emd_dim, emd_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        out = torch.cat([sh(x) for sh in self.heads], dim=-1) # here we are concatinating all the outputs we have got from all the heads
        out = self.project(out) 
        # we want the output of mmha same as input to it , which is (batch_size, max_seq_len, d_model),
        # in case if there is some miss match in the output shape of mmha then this will help and we also have additional weight matrix to learn which will help us
        # to learn , it can improve the model capacity to learn
        out = self.drop(out) # shape : out (batch_size, max_seq_len, d_model)
        return out
    

class SingleHead(nn.Module):
    def __init__(self, d_v : int, emd_dim : int, max_seq_len : int, dropout : float=0.2):
        super(SingleHead, self).__init__()
        self.w_key = nn.Linear(emd_dim, d_v, bias = False)
        self.w_query = nn.Linear(emd_dim, d_v, bias=False)
        self.w_value = nn.Linear(emd_dim, d_v, bias=False)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        # here we are creating a lower-triangular matrix of 1s below the diagonal , 
        # torch.tril converts the matrix filled with ones to lower-triangular matrix

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape # B : batch_size, T : max_seq_len, C : d_model
        k = self.w_key(x) # shape of k : (B, T, C) , k.transpose : (B, C, T)
        q = self.w_query(x) # shape of q : (B, T, C)
        weights = q @ k.transpose(-2, -1) * C**(-0.5) # shape of weights : (B, T, T)
        # .transpose(-2, -1) means taking transpose along with last two dimension
        # C**(-0.5) this is equal to 1/√d_model
        masked_weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # here we are creating a lower-triangular matrix of shape (T, T), where value with 0 will be replaced by -inf
        masked_probs = F.softmax(masked_weights, dim=-1) # masked_probs : shape (B, T, T)
        masked_probs = self.drop(masked_probs)
        v = self.w_value(x) # shape : (B, T, C)
        out = masked_probs @ v # out : shape (B, T, C)
        return out
    
class GPT(nn.Module):
    def __init__(self, vocab_size : int, max_seq_len : int, emd_dim : int, n_heads: int, n_layers : int):
        super(GPT, self).__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, emd_dim)
        self.positional_embedding_table = nn.Embedding(max_seq_len, emd_dim)
        self.blocks = nn.Sequential(*[Transformer(n_heads, emd_dim, max_seq_len) for _ in range(n_layers)],)
        self.norm = nn.LayerNorm(emd_dim)
        self.fc = nn.Linear(emd_dim, vocab_size)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        B, T = x.shape # B, T : batch_size, max_seq_len
        token_embeddings = self.embedding(x) 
        # here in the matrix x which has a shape (B, T), whatever token value is there at i,j we select that number of row vertor from the embedding 
        # matrix and put it on the third dimension of the matrix x then it gets the shape (B, T, emd_dim)
        positional_embeddings = self.positional_embedding_table(torch.arange(T, device = x.device)) 
        # this will create an positional embedding for inputs, it will create a tensor of size 0 to T-1 and will give output with shape (T, emd_dim)
        token_embeddings = token_embeddings + positional_embeddings # shape : (B, T, emd_dim)
        blocks_out = self.blocks(token_embeddings)
        blocks_out = self.norm(blocks_out)
        logits = self.fc(blocks_out) # it changes shape from (B, T, end_dim) to (B, T, vocab_size)
        logits = logits.reshape(B*T, self.vocab_size)
        # we need to calculate the cross entropy loss we need the shape to be (B*T, vocab_size), from the third dimension we will select the 
        # probability of the token which we want our model to output and only keep it
        return logits
    
    def generate(self, idx : torch.Tensor, max_tokens : int) -> torch.Tensor:
        # idx is a matrix of shape (batch_size, seq_len)
        t = idx.shape[1]
        for _ in range(max_tokens):
            idx_cond = idx[:, -self.max_seq_len:] # we dont want the sequence longer than the model was trained on 
            logits = self.forward(idx_cond)
            logits = logits.reshape(1, t, -1) # shape of logits (1, t, vocab_size)
            logits = logits[:, -1, :] # getting the logits for the last token prediction in the sequence
            
            probs = F.softmax(logits, dim=1)

            idx_next = torch.multinomial(probs, num_samples=1) # even if one token has higher probability that other we can choose other, because the model should give different answers.
            idx = torch.cat((idx, idx_next), dim=1)
            if t < self.max_seq_len:
                t += 1

        return idx
    
