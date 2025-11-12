import numpy as np
import json
import torch
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pprint import pprint
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset

# HNet imports
from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import (
    AttnConfig,
    SSMConfig,
    HNetConfig,
)
from hnet.utils.tokenizers import ByteTokenizer
from hnet.utils.train import load_balancing_loss
from train_helpers import plot_loss_curves, plot_words, prepare_group_params


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
        
    experiment_dir = cfg.paths.experiment_dir
    expeirment_name = cfg.name
    output_dir = os.path.join(experiment_dir, expeirment_name)
    os.makedirs(output_dir, exist_ok=True)

    attn_cfg = AttnConfig(**config.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config.pop("ssm_cfg"))
    hnet_cfg = HNetConfig(**config, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)

    print("Loading model...")
    model = HNetForCausalLM(hnet_cfg, device=device, dtype=torch.bfloat16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params / 1e6:.2f}M")

    # --- 2. Data Preparation ---
    print(f"Loading tokenizer and dataset '{cfg.data.name}'...")
    tokenizer = ByteTokenizer()

    # Using 'ag_news' as per your notebook

    dataset = load_dataset(cfg.data.name)
    train_data = dataset["train"]

    all_train_texts = [doc["text"] for doc in train_data]

    encoded_inputs = tokenizer.encode(all_train_texts, add_bos=True, add_eos=True)
    encoded_inputs_ids = [ids["input_ids"] for ids in encoded_inputs]
    input_ids_list = [
        torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        for ids in encoded_inputs_ids
    ]

    # Filter and trim
    input_len = cfg.data.input_len
    training_data = []
    for ids in input_ids_list:
        if ids.shape[1] < input_len:
            continue
        elif ids.shape[1] > input_len:
            ids = ids[:, :input_len]
        training_data.append(ids)

    print(f"Total training samples after filtering: {len(training_data)}")

    # TODO: ConcatDataset is memory-intensive. For large data, this should be a streaming map-style dataset
    training_data_cat = torch.utils.data.ConcatDataset(training_data)
    batched_training_data = torch.utils.data.DataLoader(
        training_data_cat, batch_size=cfg.data.batch_size, shuffle=True
    )

    # --- 3. Optimization Setup  ---
    optimizer = prepare_group_params(model, cfg.optimizer)
    ce_loss = torch.nn.CrossEntropyLoss()

    # --- 4. Training Loop  ---
    print("Starting training...")
    model.train()

    experiment_logs = defaultdict(list)
    tq_bar = tqdm(total=cfg.training.total_steps, desc="Total Training Steps")

    step = 0
    for epoch in range(cfg.training.max_epochs):
        print(f"Epoch {epoch + 1}")
        metric_aggregator = defaultdict(list)

        for input_ids_BxL in batched_training_data:
            optimizer.zero_grad()
            mask = torch.ones(input_ids_BxL.shape, device=device, dtype=torch.bool)

            output = model.forward(input_ids_BxL, mask=mask)
            
            logits_BxLxV: torch.Tensor = output.logits
            bpred = output.bpred_output

            shift_logits = logits_BxLxV[:, :-1, :].contiguous()
            shift_labels = input_ids_BxL[:, 1:].contiguous()

            ce = ce_loss(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            lb_loss = torch.tensor(0.0, device=device)
            if len(bpred) > 0:
                for i, stage_i_bpred in enumerate(bpred):
                    # Using cfg.training.n_ratios (from n_experts in notebook)
                    stage_loss = load_balancing_loss(
                        stage_i_bpred, N=cfg.training.n_ratios[i]
                    )
                    lb_loss = lb_loss + stage_loss
                    metric_aggregator[f"lb_loss_stage-{i}"].append(stage_loss.item())

            loss = ce + cfg.training.alpha * lb_loss

            metric_aggregator["ce_loss"].append(ce.item())
            metric_aggregator["lb_loss_total"].append(
                lb_loss.item() if torch.is_tensor(lb_loss) else lb_loss
            )
            metric_aggregator["total_loss"].append(loss.item())

            loss.backward()
            optimizer.step()

            step += 1
            tq_bar.update(1)

            if step % cfg.training.log_every == 0:
                for key, values in metric_aggregator.items():
                    avg_value = sum(values) / len(values)
                    experiment_logs[key].append((step, avg_value))
                metric_aggregator = defaultdict(list)  # Clear aggregator

            if step >= cfg.training.total_steps:
                break

        if step >= cfg.training.total_steps:
            break

    tq_bar.close()
    print("Training finished.")

    # --- 5. Plotting  ---
    # Hydra will save these plots to the output directory
    print("Generating plots")
    plot_loss_curves(experiment_logs, save_path=os.path.join(output_dir, "loss_curves.png"))
    plot_words(
        model,
        tokenizer,
        cfg.plotting.plot_sentences,
        device,
        save_path=os.path.join(output_dir, "word_boundaries"),
    )
    print(f"Run complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
