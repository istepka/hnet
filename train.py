import json
import os
from collections import defaultdict
from pprint import pprint

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from data import prepare_tokenized_data
from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig

# HNet imports
from hnet.models.mixer_seq import CausalLMOutput, HNetForCausalLM
from hnet.utils.tokenizers import ByteTokenizer
from hnet.utils.train import load_balancing_loss
from train_helpers import plot_loss_curves, plot_words, prepare_group_params


class TokDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, block_size: int) -> None:
        self.data = torch.from_numpy(np.memmap(path, dtype=np.int32, mode="r"))
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) // self.block_size - 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        i = idx * self.block_size
        x = self.data[i : i + self.block_size + 1]  # +1 for shifting later
        return x


def collate_batch(batch) -> torch.Tensor:
    return torch.stack(batch)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    print("--- Configuration ---")
    pprint(OmegaConf.to_container(cfg, resolve=True))
    print("---------------------")

    # --- 1. Model Setup ---
    device = cfg.training.device if torch.cuda.is_available() else "cpu"

    # Resolve path relative to script
    model_config_path = hydra.utils.to_absolute_path(cfg.model.config_path)
    with open(model_config_path, "r") as f:
        config = json.load(f)

    unique_id = wandb.util.generate_id()
    experiment_dir = cfg.paths.experiment_dir
    data_dir = cfg.paths.data_dir
    expeirment_name = cfg.name
    output_dir = os.path.join(experiment_dir, expeirment_name + "_" + unique_id)
    tokens_dir = os.path.join(data_dir, "tokens", cfg.data.name)
    ckpt_path = cfg.paths.checkpoint_save_path
    ckpt_path = os.path.join(ckpt_path, cfg.name + "_" + unique_id)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    _ = wandb.init(
        entity="istepka-carnegie-mellon-university",
        project="hnet",
        name=expeirment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    wandb.config.update({"model_config": config})
    wandb.define_metric("val/*", summary="mean")

    attn_cfg = AttnConfig(**config.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config.pop("ssm_cfg"))
    hnet_cfg = HNetConfig(**config, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)

    # --- 2. Data Preparation ---
    print(f"Loading tokenizer and dataset '{cfg.data.name}'...")
    input_len = cfg.data.input_len
    tokenizer = ByteTokenizer()

    train_tokens_path = prepare_tokenized_data(
        tokenizer=tokenizer,
        tokens_path=os.path.join(tokens_dir, "train"),
        hf_cache_path=cfg.paths.hf_cache,
        regenerate=cfg.data.regenerate_tokens,
        take_rows=cfg.data[cfg.data.name].train.take_rows,
        split="train",
        **cfg.data[cfg.data.name],
    )
    valid_tokens_path = prepare_tokenized_data(
        tokenizer=tokenizer,
        tokens_path=os.path.join(tokens_dir, "validation"),
        hf_cache_path=cfg.paths.hf_cache,
        regenerate=cfg.data.regenerate_tokens,
        take_rows=cfg.data[cfg.data.name].valid.take_rows,
        split="validation",
        **cfg.data[cfg.data.name],
    )

    train_dataset = TokDataset(train_tokens_path, block_size=input_len)
    valid_dataset = TokDataset(valid_tokens_path, block_size=input_len)
    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        drop_last=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor,
        collate_fn=collate_batch,
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor,
        collate_fn=collate_batch,
    )

    # --- 3. Optimization Setup  ---
    print("Loading model...")
    model = HNetForCausalLM(hnet_cfg, device=device, dtype=torch.bfloat16)
    total_params = sum(p.numel() for p in model.parameters())
    wandb.config.update({"total_params": total_params})
    print(f"Total number of parameters: {total_params / 1e6:.2f}M")

    optimizer = prepare_group_params(model, **cfg.optimizer)
    ce_loss = torch.nn.CrossEntropyLoss()
    alpha = torch.tensor(cfg.training.alpha, dtype=torch.bfloat16, device=device)

    # --- 4. Training Loop  ---
    print("Starting training...")
    model.train()

    experiment_logs = defaultdict(list)
    tq_bar = tqdm(total=cfg.training.total_steps, desc="Total Training Steps")

    step = 0
    finished = False
    processed_tokens = 0
    while not finished:
        metric_aggregator = defaultdict(list)

        for x in train_dl:
            assert isinstance(x, torch.Tensor), "Batch data should be a torch.Tensor"
            x = x.to(device, non_blocking=True, dtype=torch.long)
            inputs = x[:, :-1]
            labels = x[:, 1:]

            optimizer.zero_grad()
            mask = torch.ones(inputs.shape, device=device, dtype=torch.bool)
            output: CausalLMOutput = model(inputs, mask=mask)
            logits = output.logits
            bpred = output.bpred_output

            # Align logits and labels
            ce: torch.Tensor = ce_loss(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )

            lb_loss = torch.tensor(0.0, device=device)
            if len(bpred) > 0:
                for i, stage_i_bpred in enumerate(bpred):
                    # Using cfg.training.n_ratios (from n_experts in notebook)
                    stage_loss = load_balancing_loss(
                        stage_i_bpred, N=cfg.training.n_ratios[i]
                    )
                    lb_loss = lb_loss + stage_loss
                    metric_aggregator[f"train/lb_loss_stage-{i}"].append(
                        stage_loss.item()
                    )
                    wandb.log(
                        {f"train/lb_loss_stage-{i}": stage_loss.item()}, step=step
                    )

            loss = ce + alpha * lb_loss
            loss.backward()
            optimizer.step()

            ppl = torch.exp(ce).item()

            metric_aggregator["ce_loss"].append(ce.item())
            metric_aggregator["lb_loss_total"].append(lb_loss.item())
            metric_aggregator["total_loss"].append(loss.item())
            metric_aggregator["perplexity"].append(ppl)

            step += 1
            tq_bar.update(1)
            processed_tokens += inputs.numel()

            wandb.log(
                {
                    "train/ce_loss": ce.item(),
                    "train/lb_loss_total": lb_loss.item(),
                    "train/total_loss": loss.item(),
                    "train/processed_tokens": processed_tokens,
                    "train/perplexity": ppl,
                },
                step=step,
            )

            if step % cfg.training.log_every == 0:
                for key, values in metric_aggregator.items():
                    avg_value = sum(values) / len(values)
                    experiment_logs[key].append((step, avg_value))
                metric_aggregator = defaultdict(list)  # Clear aggregator

            if step % cfg.training.save_every == 0:
                torch.save(
                    model.state_dict(), os.path.join(ckpt_path, f"step_{step}.pt")
                )

                # Evaluation plots during training
                model.eval()
                with torch.no_grad():
                    paths = plot_words(
                        model,
                        tokenizer,
                        cfg.plotting.plot_sentences,
                        device,
                        save_path=os.path.join(output_dir, f"word_boundaries_s{step}"),
                    )
                    for i, path in enumerate(paths):
                        wandb.log({f"wb/sentence_{i}": wandb.Image(path)}, step=step)
                model.train()

            if step % cfg.training.val_every == 0:
                model.eval()
                with torch.no_grad():
                    for val_x in tqdm(valid_dl, desc="Validation", total=len(valid_dl)):
                        val_x = val_x.to(device, non_blocking=True, dtype=torch.long)
                        inputs = val_x[:, :-1]
                        labels = val_x[:, 1:]

                        mask = torch.ones(inputs.shape, device=device, dtype=torch.bool)
                        output: CausalLMOutput = model(inputs, mask=mask)
                        logits = output.logits
                        bpred = output.bpred_output

                        # Compute cross-entropy loss
                        ce = ce_loss(
                            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                        )

                        brped0_bm = bpred[0].boundary_mask.squeeze().cpu().numpy()
                        brped1_bm = bpred[1].boundary_mask.squeeze().cpu().numpy()
                        l = len(brped0_bm)  # noqa
                        comp0 = l / (brped0_bm == 1).sum()
                        comp1 = l / (brped1_bm == 1).sum()

                        wandb.log(
                            {
                                "val/ce_loss": ce.item(),
                                "val/perplexity": torch.exp(ce).item(),
                                "val/compression_stage0": comp0,
                                "val/compression_stage1": comp1,
                            },
                            step=step,
                        )

                model.train()

            if step >= cfg.training.total_steps:
                finished = True
                break

    torch.save(model.state_dict(), os.path.join(ckpt_path, f"step_{step}.pt"))
    wandb.save(os.path.join(ckpt_path, f"step_{step}.pt"))

    tq_bar.close()
    print("Training finished")

    # --- 5. Plotting  ---
    # Hydra will save these plots to the output directory
    print("Generating plots")
    model.eval()
    plot_loss_curves(
        experiment_logs, save_path=os.path.join(output_dir, "loss_curves.png")
    )
    paths = plot_words(
        model,
        tokenizer,
        cfg.plotting.plot_sentences,
        device,
        save_path=os.path.join(output_dir, "word_boundaries"),
    )
    for i, path in enumerate(paths):
        wandb.log({f"wb/sentence_{i}": wandb.Image(path)})

    print(f"Run complete. Outputs saved to {output_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()
