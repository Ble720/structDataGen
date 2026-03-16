import torch

def apply_fixed_count_mask(x, cardinalities, mask_pct):
    if x.shape[1] == 0 or len(cardinalities) == 0:
        return x, torch.zeros_like(x).float()
    
    batch_size, num_cols = x.shape
    num_to_mask = int(num_cols * mask_pct)
    
    num_to_mask = max(1, min(num_to_mask, num_cols - 1))
    
    noise = torch.rand(batch_size, num_cols, device=x.device)
    mask_indices = torch.topk(noise, num_to_mask, dim=1).indices
    
    mask = torch.zeros_like(noise).scatter_(1, mask_indices, 1).bool()
    
    x_masked = x.clone()
    for i, count in enumerate(cardinalities):
        col_mask = mask[:, i]
        x_masked[col_mask, i] = 0 # Use the [MASK] token
    
    return x_masked, mask.float()