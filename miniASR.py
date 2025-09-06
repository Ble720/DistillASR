import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from SpecAugment import SpecAugment

class ConvPositionalEncoding(nn.Module):
    def __init__(self, d_model, kernel_size=129, groups=16):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=(kernel_size-1)// 2,
            groups=groups,
            bias=False
        )
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.conv.weight, mean=0, std=d_model ** -0.5)

    def forward(self, x):
        residual = x                         # x: (B, T, D)
        x = x.transpose(1, 2)                # (B, D, T)
        conv_out = self.conv(x)              # (B, D, T)
        conv_out = conv_out.transpose(1, 2)  # (B, T, D)
        return self.layer_norm(residual + self.activation(conv_out))
    
class ConvFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, out_channels=512):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=10, stride=5),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.conv_layers(x)  # (B, C, T)
        x = x.transpose(1, 2)    # (B, T, C)
        return x
    
class FeatureProjection(nn.Module):
    def __init__(self, input_dim=512, output_dim=256, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, T, C) from conv feature extractor
        x = self.layer_norm(x)
        x = self.projection(x)
        return self.dropout(x)
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=12, dim_feedforward=768):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            activation='gelu')
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers


    def forward(self, x, attention_mask=None, return_all_hidden_states=False):
        x = x.transpose(0, 1)
        hidden_states = []

        if return_all_hidden_states:
            hidden_states.append(x)  # input embeddings before any layer

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=attention_mask)
            if return_all_hidden_states:
                hidden_states.append(x)
        x = x.transpose(0, 1)
        if return_all_hidden_states:
            return x, [h.transpose(0,1) for h in hidden_states]
        return x

class MiniS2T(nn.Module):
    def __init__(
        self, 
        vocab_size=32, 
        d_model=256,
        n_head=8,
        num_layers=12,
        dim_feedforward=768, 
        hidden_dim=768
        ):
        super().__init__()
        self.feature_extractor = ConvFeatureExtractor(in_channels=1, out_channels=512)

        self.project_encoder = FeatureProjection(input_dim=512, output_dim=d_model, dropout=0.1)

        self.encoder = TransformerEncoder(d_model=d_model, nhead=n_head, num_layers=num_layers, dim_feedforward=dim_feedforward)

        self.project_out = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),               
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)    
        )
        self.spec_augment = SpecAugment(
            freq_mask_param=15,
            time_mask_param=50,
            num_freq_masks=2,
            num_time_masks=2,
            time_warp=True,
            max_time_warp=5
        )

        self.pos_encoder = ConvPositionalEncoding(d_model=d_model)

    def forward(self, x, attention_mask=None, output_hidden_states=False, epoch=None, specaug_start_epoch=15):
        x = x.unsqueeze(1)  # (B, 1, T)

        x = self.feature_extractor(x)
        x = self.project_encoder(x)

        # Apply SpecAugment here during training only
        if self.training and epoch is not None and epoch >= specaug_start_epoch:
            x = x.transpose(1, 2)
            x = self.spec_augment(x)
            x = x.transpose(1, 2)

        x = self.pos_encoder(x)

        if attention_mask is not None:
            B, T_feat, _ = x.shape
            # Downsample attention mask roughly to T_feat
            attention_mask = F.interpolate(attention_mask.unsqueeze(1).float(), size=T_feat, mode='nearest').squeeze(1)
            attention_mask = attention_mask.bool() 
            attention_mask = ~attention_mask

        if output_hidden_states:
            x, states = self.encoder(x, attention_mask, return_all_hidden_states=True)
            x = self.project_out(x)
            return self.output_layer(x), states
        else:
            x = self.encoder(x, attention_mask, return_all_hidden_states=False)
            x = self.project_out(x)

        return self.output_layer(x)  # (B, T', output_dim)