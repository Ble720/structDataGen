import argparse, csv, random, json, os

from dataloader import TabularDataHandler
from model import GenTabularData
from scheduler import get_cosine_warmup_scheduler
from utils.mask import apply_fixed_count_mask
from log_var import UncertaintyWeights

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

import numpy as np

def train_tabular_model(model, uncertainty_net, dataloader, optimizer, accountant,log_var_optimizer, scheduler, mask_pct=[0.3,0.5], epochs=50, save_interval=1000, save_path="./train", device="cuda"):
    model = model.to(device)
    model.train()
    logical_batch_size = optimizer.expected_batch_size

    cat_cards = model._module.cat_card if hasattr(model, "_module") else model.cat_card
    bin_cards = model._module.bin_card if hasattr(model, "_module") else model.bin_card

    accum_total_loss = 0.0
    accum_cat_loss = 0.0
    accum_bin_loss = 0.0
    accum_off_loss = 0.0
    sub_batch_count = 0

    log_file = f"{save_path}/training_logs.csv"
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Epoch", "Avg_Total_Loss", "Avg_Cat_Loss", "Avg_Bin_Loss", "Avg_Off_Loss", "Epsilon"])

    global_step = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        log_var_optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            batch = [t.to(device) for t in batch]
            x_cat, x_bin, x_off, cat_missing_mask, bin_missing_mask = batch

            current_mask_pct = torch.rand(1).item() * (mask_pct[1] - mask_pct[0]) + mask_pct[0]
            x_cat_in, m_cat = apply_fixed_count_mask(x_cat, cat_cards, current_mask_pct)
            x_bin_in, m_bin = apply_fixed_count_mask(x_bin, bin_cards, current_mask_pct)

            x_cat_in = x_cat_in.to(device)
            x_bin_in = x_bin_in.to(device)

            pred_cat, pred_bin, pred_off = model(x_cat_in, x_bin_in)

            def get_opacus_safe_loss(pred_list, target_tensor, mask_tensor):
                total_loss_sum = 0
                for i in range(len(pred_list)):
                    per_sample_loss = F.cross_entropy(pred_list[i], target_tensor[:, i], reduction='none')
                    total_loss_sum = total_loss_sum + (per_sample_loss * mask_tensor[:, i]).sum()
                    
                return total_loss_sum / (mask_tensor.sum() + 1e-8)
            
            loss_cat = get_opacus_safe_loss(pred_cat, x_cat, m_cat * cat_missing_mask)
            loss_bin = get_opacus_safe_loss(pred_bin, x_bin, m_bin * bin_missing_mask)

            all_off_preds = torch.stack([logits.gather(1, x_bin[:, i].unsqueeze(1)).squeeze(1) 
                                       for i, logits in enumerate(pred_off)], dim=1)
            off_masks = m_bin * bin_missing_mask
            loss_off = (F.mse_loss(all_off_preds, x_off, reduction='none') * off_masks).sum() / (off_masks.sum() + 1e-8)

            total_loss_dp = (loss_cat * torch.exp(-uncertainty_net.log_var_cat.detach())) + \
                (loss_bin * torch.exp(-uncertainty_net.log_var_bin.detach())) + \
                (loss_off * torch.exp(-uncertainty_net.log_var_off.detach()))


            total_loss_dp.backward()

            s_cat = uncertainty_net.log_var_cat
            s_bin = uncertainty_net.log_var_bin
            s_off = uncertainty_net.log_var_off

            total_loss_logvars = (loss_cat.detach() * torch.exp(-s_cat)) + 0.5 * s_cat + \
                                (loss_bin.detach() * torch.exp(-s_bin)) + 0.5 * s_bin + \
                                (loss_off.detach() * torch.exp(-s_off)) + 0.5 * s_off

            total_loss_logvars.backward()

            with torch.no_grad():
                uncertainty_net.log_var_cat.clamp_(max=1.0)
                uncertainty_net.log_var_bin.clamp_(max=1.0)
                uncertainty_net.log_var_off.clamp_(min=-2.3)

            cur_cat = loss_cat.item()
            cur_bin = loss_bin.item()
            cur_off = loss_off.item()
            cur_total = cur_cat + cur_bin + cur_off
            
            accum_total_loss += cur_total
            accum_cat_loss += cur_cat
            accum_bin_loss += cur_bin
            accum_off_loss += cur_off
            sub_batch_count += 1

            saved_grads = {}
            for p in uncertainty_net.parameters():
                if p.grad is not None:
                    saved_grads[p] = p.grad
                    p.grad = None

            optimizer.step()

            for p, g in saved_grads.items():
                p.grad = g

            if not optimizer._is_last_step_skipped:
                torch.cuda.empty_cache()
                log_var_optimizer.step()
                scheduler.step()

                optimizer.zero_grad()
                log_var_optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    print(f"Epoch {epoch} | Step {global_step} | Total Loss: {total_loss_dp.item():.4f} | Cat Loss: {loss_cat.item():.4f} | Bin Loss: {loss_bin.item():.4f} | Off Loss: {loss_off.item():.4f}")

                    with torch.no_grad():
                        w_cat = torch.exp(-uncertainty_net.log_var_cat).item()
                        w_bin = torch.exp(-uncertainty_net.log_var_bin).item()
                        w_off = torch.exp(-uncertainty_net.log_var_off).item()
                        print(f"Weights -> Cat: {w_cat:.4f} | Bin: {w_bin:.4f} | Off: {w_off:.4f}")

                if global_step % save_interval == 0:
                    epsilon = accountant.get_epsilon(delta=1e-5)
                    checkpoint_path = os.path.join(save_path, "checkpoints", f"model_step_{global_step}.pt")

                    avg_total = accum_total_loss / sub_batch_count
                    avg_cat = accum_cat_loss / sub_batch_count
                    avg_bin = accum_bin_loss / sub_batch_count
                    avg_off = accum_off_loss / sub_batch_count

                    torch.save(model.state_dict(), checkpoint_path)
                    with open(log_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([global_step, epoch, avg_total, avg_cat, avg_bin, avg_off, epsilon])
                    
                    print(f"Step {global_step} (from {sub_batch_count} sub-batches) | Avg Loss: {avg_total:.4f} | Epsilon: {epsilon:.2f}")

                    accum_total_loss, accum_cat_loss, accum_bin_loss, accum_off_loss, sub_batch_count = 0, 0, 0, 0, 0

    final_eps = accountant.get_epsilon(delta=1e-5)
    torch.save(model.state_dict(), os.path.join(save_path, "checkpoints", f"model_final_eps_{final_eps:.2f}.pt"))
    print(f"\nTraining complete. Final ε: {final_eps:.2f}")
        

def save_model(model, path):
    torch.save(model.state_dict(), path)

class OpacusTransformerNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=1, num_channels=num_features)

    def forward(self, x):
        x = x.transpose(1, 2) 
        x = self.gn(x)
        return x.transpose(1, 2)

def replace_layernorm_with_groupnorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            num_features = child.normalized_shape[0]
            setattr(module, name, OpacusTransformerNorm(num_features))
        else:
            replace_layernorm_with_groupnorm(child)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    exp_num = len([d for d in os.listdir(save_path) if d.startswith("exp_")]) + 1
    exp_save_path = os.path.join(save_path, f"exp_{exp_num}")

    os.makedirs(exp_save_path, exist_ok=True)
    os.makedirs(os.path.join(exp_save_path, "checkpoints"), exist_ok=True)

    return exp_save_path

def main(physical_batch_size, logical_batch_size, learning_rate, epsilon, data, config, epochs, save_interval, save_path, device):
    set_seed(42)

    with open(config, 'r') as f:
        fconfig = json.load(f)

    save_path = setup(save_path)

    data_handler = TabularDataHandler(
        csv_path=data,
        config=fconfig,
        batch_size=physical_batch_size
    )

    data_handler.prepare_data() 
    data_handler.save_metadata(os.path.join(save_path, "metadata.joblib"))
    train_loader = data_handler.train_dataloader(drop_last=True)

    model = GenTabularData(
        cat_card=data_handler.get_cardinalities()['cat'], 
        bin_card=data_handler.get_cardinalities()['bin'],
        d_model=256,
        nhead=8
    )
    replace_layernorm_with_groupnorm(model)
    model = ModuleValidator.fix(model)
    model.train()

    uncertainty_net = UncertaintyWeights().to(device)

    errors = ModuleValidator.validate(model, strict=False)
    if not errors:
        print("Model is DP-ready!")

    log_var_optimizer = torch.optim.Adam(uncertainty_net.parameters(), lr=1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=epsilon,
        target_delta=1e-5,
        epochs=epochs,
        max_grad_norm=1.0,
        batch_size=logical_batch_size
    )

    logical_steps_per_epoch = len(train_loader.dataset) // optimizer.expected_batch_size
    total_steps = logical_steps_per_epoch * epochs

    scheduler, _ = get_cosine_warmup_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        base_lr=learning_rate,
        warmup_ratio=0.10, 
        min_lr_ratio=0.05
    )

    train_tabular_model(
        model=model,
        uncertainty_net=uncertainty_net,
        dataloader=train_loader,
        optimizer=optimizer,
        accountant=privacy_engine.accountant,
        log_var_optimizer=log_var_optimizer,
        scheduler=scheduler, 
        mask_pct=[0.3, 0.5],
        epochs=epochs,
        save_interval=save_interval,
        save_path=save_path,
        device=device
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pbs", "--physical_batch_size", type=int, default=128)
    parser.add_argument("-lbs", "--logical_batch_size", type=int, default=4096)
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-4)
    parser.add_argument("-eps", "--epsilon", type=float, default=2.0)
    parser.add_argument("-da", "--data", type=str, default="./folktables_features.csv")
    parser.add_argument("-c", "--config", type=str, default="./config.json")
    parser.add_argument("-e", "--epochs", type=int, default=150)
    parser.add_argument("-si", "--save_interval", type=int, default=1000)
    parser.add_argument("-s", "--save_path", type=str, default="./train")
    parser.add_argument("-d", "--device", type=str, default="cuda")
    return vars(parser.parse_args())
    
if __name__ == '__main__':
    args = parse_args()
    main(**args)