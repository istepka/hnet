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
from data import prepare_timeseries_tokenized_data
from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig

# HNet imports
from hnet.models.mixer_seq import CausalLMOutput, HNetForCausalLM
from hnet.utils.tokenizers import TSQuantizedTokenizer
from hnet.utils.train import load_balancing_loss
from train_helpers import (
    TokDataset,
    WandbLogger,
    collate_batch,
    plot_token_boundaries_ts,
    prepare_group_params,
)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    print("--- Configuration ---")
    pprint(OmegaConf.to_container(cfg, resolve=True))
    print("---------------------")

    # --- 1. Model Setup ---
    device = cfg.training.device if torch.cuda.is_available() else "cpu"

    use_wandb = cfg.wandb.use
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Resolve path relative to script
    model_config_path = hydra.utils.to_absolute_path(cfg.model.config_path)
    with open(model_config_path, "r") as f:
        config = json.load(f)
    config["vocab_size"] = cfg.model.vocab_size

    unique_id = wandb.util.generate_id()
    experiment_dir = cfg.paths.experiment_dir
    data_dir = cfg.paths.data_dir
    expeirment_name = cfg.name
    output_dir = os.path.join(experiment_dir, expeirment_name + "_" + unique_id)
    tokens_dir = os.path.join(
        data_dir, "tokens", cfg.data.name, cfg.data[cfg.data.name].name
    )
    ckpt_path = cfg.paths.checkpoint_save_path
    ckpt_path = os.path.join(ckpt_path, cfg.name + "_" + unique_id)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if use_wandb:
        run = wandb.init(
            entity="istepka-carnegie-mellon-university",
            project="hnet",
            name=expeirment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb_logger = WandbLogger(use_wandb, run)
        wandb_logger.update_config({"model_config": config})
        wandb.define_metric("val/*", summary="mean")
    else:
        wandb_logger = WandbLogger(use_wandb)

    attn_cfg = AttnConfig(**config.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config.pop("ssm_cfg"))
    hnet_cfg = HNetConfig(**config, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)
    stages = cfg.model.stages

    # --- 2. Data Preparation ---
    print(f"Loading tokenizer and dataset '{cfg.data.name}'...")
    input_len = cfg.data.input_len
    if cfg.model.tokenizer == "ts_quantized":
        tokenizer = TSQuantizedTokenizer(
            vocab_size=cfg.model.vocab_size, quant_type="cdf"
        )
    else:
        raise ValueError(f"Unsupported tokenizer: {cfg.model.tokenizer}")

    tokens_paths = prepare_timeseries_tokenized_data(
        tokenizer=tokenizer,
        tokens_path=tokens_dir,
        hf_cache_path=cfg.paths.hf_cache,
        regenerate=cfg.data.regenerate_tokens,
        **cfg.data[cfg.data.name],
    )

    valid_tokens_path = None
    if isinstance(tokens_paths, str):
        train_tokens_path = tokens_paths
    else:
        train_tokens_path = tokens_paths[0]
        valid_tokens_path = tokens_paths[1]

    train_dataset = TokDataset(
        train_tokens_path, block_size=input_len, dtype=tokenizer.dtype
    )
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
    run_validation = valid_tokens_path is not None
    if run_validation:
        valid_dataset = TokDataset(
            valid_tokens_path, block_size=input_len, dtype=tokenizer.dtype
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
    wandb_logger.update_config({"total_params": total_params})
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
            # print(f"Batch shape: {x.shape}")
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
                    wandb_logger.log(
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

            processed_tokens += inputs.numel()

            wandb_logger.log(
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

            if step % cfg.training.val_every == 0 and run_validation:
                model.eval()
                val_batches_to_plot = cfg.plotting.time_series.batches_to_plot
                with torch.no_grad():
                    _it = 0
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
                        predictions = torch.argmax(logits, dim=-1)

                        brped0_bm = bpred[0].boundary_mask
                        comp0 = brped0_bm.numel() / (brped0_bm == 1).sum()

                        # Compute MSE on decoded values of the forecast
                        decoded = tokenizer.decode(predictions.cpu().numpy())
                        original = tokenizer.decode(labels.cpu().numpy())
                        mse = np.mean((decoded - original) ** 2)
                        mae = np.mean(np.abs(decoded - original))
                        mape = (
                            np.mean(np.abs((decoded - original) / (original + 1e-8)))
                            * 100
                        )

                        _log = {
                            "val/ce_loss": ce.item(),
                            "val/perplexity": torch.exp(ce).item(),
                            "val/compression_stage0": comp0,
                            "val/mse": mse,
                            "val/mae": mae,
                            "val/mape": mape,
                        }

                        if stages == 2:
                            bpred1_bm = bpred[1].boundary_mask
                            comp1 = bpred1_bm.numel() / (bpred1_bm == 1).sum()
                            _log["val/compression_stage1"] = comp1

                        wandb_logger.log(_log, step=step)

                        if _it >= val_batches_to_plot:
                            break

                        paths = plot_token_boundaries_ts(
                            model,
                            tokenizer,
                            inputs,
                            device,
                            save_path=os.path.join(
                                output_dir, f"token_boundaries_step_{step}"
                            ),
                        )
                        for i, path in enumerate(paths):
                            wandb_logger.log(
                                {f"wb/series_{i}": wandb.Image(path)}, step=step
                            )
                        _it += 1
                model.train()

            if step >= cfg.training.total_steps:
                finished = True
                break

            step += 1
            tq_bar.update(1)

    torch.save(model.state_dict(), os.path.join(ckpt_path, f"step_{step}.pt"))
    wandb_logger.save(os.path.join(ckpt_path, f"step_{step}.pt"))

    tq_bar.close()
    print("Training finished")

    # --- 5. Plotting  ---
    # Hydra will save these plots to the output directory
    # print("Generating plots")
    # model.eval()
    # plot_loss_curves(
    #     experiment_logs, save_path=os.path.join(output_dir, "loss_curves.png")
    # )
    # paths = plot_words(
    #     model,
    #     tokenizer,
    #     cfg.plotting.plot_sentences,
    #     device,
    #     save_path=os.path.join(output_dir, "word_boundaries"),
    # )
    # for i, path in enumerate(paths):
    #     wandb_logger.log({f"wb/sentence_{i}": wandb.Image(path)})

    print(f"Run complete. Outputs saved to {output_dir}")

    wandb_logger.finish()


if __name__ == "__main__":
    main()
