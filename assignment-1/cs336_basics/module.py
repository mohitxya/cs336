import torch 
import torch.nn as nn
import math

class Linear(nn.Module): 
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None): 
        super().__init__()
        meta = {'device': device, 'dtype': dtype}

        weight_tensor = torch.empty((in_features, out_features), **meta)
        nn.init.trunc_normal_(weight_tensor)

        self.W = nn.Parameter(weight_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return torch.matmul(x, self.W)

class Embedding(nn.Module): 
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None): 
        super().__init__()
        self.num_emb = num_embeddings
        self.emb_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.empty((self.num_emb, self.emb_dim), device=self.device, dtype=self.dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor: 
        token_ids.to(dtype=torch.int64)
        return self.weight[token_ids]

class RMSNorm(nn.Module): 
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None): 
        super().__init__()
        self.d_model = d_model
        self.dtype = dtype
        self.device = device
        self.eps = eps

        self.scale = nn.Parameter(torch.ones((self.d_model,), device=self.device, dtype=self.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        in_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normalized_x = x * torch.rsqrt(variance + self.eps)
        result = normalized_x * self.scale
        return result.to(in_dtype)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class SwiGLUFeedForward(nn.Module): 
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None): 
        super().__init__()
        self.d_model = d_model
        if d_ff is not None:
            self.d_ff = d_ff
        else:
            hidden_dim = int((8.0 / 3.0) * d_model)
            self.d_ff = 64 * ((hidden_dim + 63) // 64)
        self.device = device
        self.dtype = dtype

        self.w1 = Linear(self.d_model, self.d_ff, device=self.device, dtype=self.dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=self.device, dtype=self.dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=self.device, dtype=self.dtype)
    
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        silu_gate = self.silu(gate)
        up = self.w3(x)
        intermediate = silu_gate * up
        return self.w2(intermediate)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input to match the 2x2 matrix math."""
    x1 = x[..., 0::2] # first coordinate of every pair (x, y) -> x
    x2 = x[..., 1::2] # second coordinate of every pair (x, y) -> y
    return torch.stack([-x2, x1], dim=-1).flatten(-2) # [(-y, x), (-y, x), ...]

class rope(nn.Module): 
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None): 
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device
        freq = 1.0 / (self.theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=self.device)/self.d_k))
        # element wise multiplication with seq position for each. 
        t = torch.arange(max_seq_len, dtype=torch.float32, device=self.device)
        angles = t.unsqueeze(-1) * freq  # Shape: [max_seq_len, d_k / 2]
        
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor: 
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)

        
        return (x*cos) + (rotate_half(x) * sin) # allows us to use faster vector multiply
        
class Softmax(nn.Module): 
    def __init__(self, dim: int = -1, device = None, dtype = None): 
        super().__init__()
        self.dim = dim
        self.device = device
        self.dtype = dtype
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x_max = torch.max(x, dim=self.dim, keepdim=True).values
        shifted_x = x - x_max
        exp_x = torch.exp(shifted_x)
        sum_exp_x = torch.sum(exp_x, dim=self.dim, keepdim=True)
        return exp_x / sum_exp_x

class sdpa(nn.Module): 
    def __init__(self, device=None): 
        super().__init__()
        self.device = device
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor: 
        d_k = Q.shape[-1]

        # raw attention scores: Q * K^T
        # Q shape: (..., queries, d_k)
        # K transposed shape: (..., d_k, keys)
        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(d_k)
        
        if mask is not None: 
            scores = scores.masked_fill(mask==False, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)       
        return attn_weights @ V 

class multihead_self_attention(nn.Module): 
    def __init__(self, d_model: int, num_heads: int, q_proj_weight: torch.Tensor, k_proj_weight: torch.Tensor, v_proj_weight: torch.Tensor, o_proj_weight: torch.Tensor): 
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj_weight = q_proj_weight
        self.k_proj_weight = k_proj_weight
        self.v_proj_weight = v_proj_weight
        self.o_proj_weight = o_proj_weight
        self.d_k = d_model // num_heads
        self.sdpa = sdpa()
    def forward(self, in_features: torch.Tensor): 
        q_proj = in_features @ self.q_proj_weight
        k_proj = in_features @ self.k_proj_weight
        v_proj = in_features @ self.v_proj_weight
        
        q_split = q_proj.view(*q_proj.shape[:-1], self.num_heads, self.d_k).transpose(-2,-3)
        k_split = k_proj.view(*k_proj.shape[:-1], self.num_heads, self.d_k).transpose(-2,-3)
        v_split = v_proj.view(*v_proj.shape[:-1], self.num_heads, self.d_k).transpose(-2,-3)

        seq_len = in_features.shape[-2]

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        attn_output = self.sdpa(Q=q_split, K=k_split, V=v_split, mask=mask)

        attn_output = attn_output.transpose(-2, -3)

        concat_output = attn_output.contiguous().view(*attn_output.shape[:-2], self.d_model)

        final_output = concat_output @ self.o_proj_weight
        return final_output

class multihead_self_attention_rope(nn.Module): 
    def __init__(self, d_model: int, num_heads: int, q_proj_weight: torch.Tensor, k_proj_weight: torch.Tensor, v_proj_weight: torch.Tensor, o_proj_weight: torch.Tensor, max_seq_len: int | None = None, theta: float | None = None, device=None): 
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj_weight = q_proj_weight
        self.k_proj_weight = k_proj_weight
        self.v_proj_weight = v_proj_weight
        self.o_proj_weight = o_proj_weight
        self.d_k = d_model // num_heads
        self.sdpa = sdpa()

        self.rope = None
        if max_seq_len is not None and theta is not None:
            self.rope = rope(theta, self.d_k, max_seq_len, device=device)

    def forward(self, in_features: torch.Tensor, token_positions: torch.Tensor | None = None): 
        q_proj = in_features @ self.q_proj_weight
        k_proj = in_features @ self.k_proj_weight
        v_proj = in_features @ self.v_proj_weight
        
        q_split = q_proj.view(*q_proj.shape[:-1], self.num_heads, self.d_k).transpose(-2,-3)
        k_split = k_proj.view(*k_proj.shape[:-1], self.num_heads, self.d_k).transpose(-2,-3)
        v_split = v_proj.view(*v_proj.shape[:-1], self.num_heads, self.d_k).transpose(-2,-3)

        seq_len = in_features.shape[-2]

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=in_features.device)
            q_split = self.rope(q_split, token_positions)
            k_split = self.rope(k_split, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device))

        attn_output = self.sdpa(Q=q_split, K=k_split, V=v_split, mask=mask)

        attn_output = attn_output.transpose(-2, -3)

        concat_output = attn_output.contiguous().view(*attn_output.shape[:-2], self.d_model)

        final_output = concat_output @ self.o_proj_weight
        return final_output

class transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float,
                 q_proj_weight: torch.Tensor, k_proj_weight: torch.Tensor,
                 v_proj_weight: torch.Tensor, o_proj_weight: torch.Tensor,
                 w1_weight: torch.Tensor, w2_weight: torch.Tensor, w3_weight: torch.Tensor,
                 ln1_weight: torch.Tensor, ln2_weight: torch.Tensor,
                 device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.attn = multihead_self_attention_rope(
            d_model, num_heads,
            q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
            max_seq_len=max_seq_len, theta=theta, device=device,
        )
        self.ffn = SwiGLUFeedForward(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        with torch.no_grad():
            # Your Linear.W is (in_features, out_features); reference is (out, in)
            self.ffn.w1.W.copy_(w1_weight.T)
            self.ffn.w2.W.copy_(w2_weight.T)
            self.ffn.w3.W.copy_(w3_weight.T)
            self.ln1.scale.copy_(ln1_weight)
            self.ln2.scale.copy_(ln2_weight)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x