from cs336_basics.module import *

class TransformerLM(nn.Module): 
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, theta): 
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model) # no matmul; lookup
        self.layers = nn.ModuleList([
            transformer_block(d_model, num_heads, d_ff, context_length, theta) 
            for _ in range(num_layers)
            ])
        self.ln_final = RMSNorm(d_model)

        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor: 
        token_ids.to(dtype=torch.int64)
        
        x = self.token_embeddings(token_ids)

        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device)
        token_positions = positions.expand(batch_size, seq_len)
        
        for layer in self.layers: 
            x = layer(x, token_positions)

        x = self.ln_final(x)
        return self.lm_head(x)