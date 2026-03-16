import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel, SDPBackend
    
class GenTabularData(nn.Module):
    def __init__(
        self,  
        cat_card, 
        bin_card,
        d_model=256, 
        nhead=8, 
        num_layers=6, 
        dim_feedforward=1024,
        dropout=0.1
    ):
        super().__init__()
        self.cat_card = cat_card
        self.bin_card = bin_card
        
        self.cat_embs = nn.ModuleList([nn.Embedding(c, d_model) for c in cat_card])
        self.bin_embs = nn.ModuleList([nn.Embedding(b, d_model) for b in bin_card])
        
        self.total_cols = len(cat_card) + len(bin_card)
        self.column_embed = nn.Embedding(self.total_cols, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cat_heads = nn.ModuleList([nn.Linear(d_model, count) for count in cat_card])
        self.bin_heads = nn.ModuleList([nn.Linear(d_model, count) for count in bin_card])
        
        self.offset_heads = nn.ModuleList([nn.Linear(d_model, count) for count in bin_card])

    def forward(self, x_cat, x_bin):
        embeddings = []
        
        for i in range(len(self.cat_embs)):
            embeddings.append(self.cat_embs[i](x_cat[:, i]).unsqueeze(1))
            
        for i in range(len(self.bin_embs)):
            embeddings.append(self.bin_embs[i](x_bin[:, i]).unsqueeze(1))
            
        x = torch.cat(embeddings, dim=1)
        
        col_indices = torch.arange(self.total_cols, device=x.device).unsqueeze(0).repeat(x.size(0), 1)
        x = x + self.column_embed(col_indices)
        
        with sdpa_kernel([SDPBackend.MATH]):
            z = self.transformer(x)
    
        pred_cat = []
        for i in range(len(self.cat_heads)):
            pred_cat.append(self.cat_heads[i](z[:, i]))

        bin_start = len(self.cat_card)
        pred_bin = []
        pred_off = []

        for i in range(len(self.bin_heads)):
            col_z = z[:, bin_start + i]
            pred_bin.append(self.bin_heads[i](col_z))
            raw_off = self.offset_heads[i](col_z)
            pred_off.append(torch.sigmoid(raw_off))
        
        return pred_cat, pred_bin, pred_off