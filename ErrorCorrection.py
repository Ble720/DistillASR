import torch
import torch.nn as nn
import torch.nn.functional as F

class ErrorCorrectionModel(nn.Module):
    def __init__(self, vocab_size=32, hidden_size=128, num_layers=1, num_heads=2, dropout=0.1, zero_init=False):
        super().__init__()
        self.input_proj = nn.Linear(vocab_size, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_size, vocab_size)

        if zero_init:
            # --- Conservative zero init ---
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

            nn.init.zeros_(self.input_proj.bias)  # remove spurious offset at input
            # (input_proj.weight stays default Xavier for learning capacity)

    def forward(self, x, attention_mask=None):
        """
        x: [batch, seq, vocab_size] logits from student
        Returns: [batch, seq, vocab_size] corrected logits (residual added)
        """
        _, T_feat, _ = x.shape
        if attention_mask is not None:
            padding_mask = ~F.interpolate(
                attention_mask.float().unsqueeze(1), 
                size=T_feat, 
                mode='nearest'
            ).squeeze(1).bool()
        else:
            padding_mask = None

        delta = self.input_proj(x)                  
        delta = self.encoder(delta, src_key_padding_mask=padding_mask)
        delta = self.output_proj(delta)              
        return x + delta