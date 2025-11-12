import json
import os
from collections import defaultdict
from pprint import pprint

import hydra
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

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

    experiment_dir = cfg.paths.experiment_dir
    data_dir = cfg.paths.data_dir
    expeirment_name = cfg.name
    output_dir = os.path.join(experiment_dir, expeirment_name)
    tokens_dir = os.path.join(data_dir, "tokens")
    ckpt_path = cfg.paths.checkpoint_save_path
    ckpt_path = os.path.join(ckpt_path, cfg.name)
    os.makedirs(tokens_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
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
    tokens_path = os.path.join(data_dir, f"{cfg.data.name}_tokens.bin")
    input_len = cfg.data.input_len
    tokenizer = ByteTokenizer()

    if not os.path.exists(tokens_path) or cfg.data.regenerate_tokens:
        print("Tokenized data not found, creating...")

        dataset = load_dataset(cfg.data.name)
        all_train_texts = [doc["text"] for doc in dataset["train"]]
        encoded_inputs = tokenizer.encode(all_train_texts, add_bos=True, add_eos=True)

        longs_ids = list()
        for ids in encoded_inputs:
            longs_ids.extend(ids["input_ids"])

        # Save tokenized data to binary file for memory mapping
        np.array(longs_ids, dtype=np.int32).tofile(tokens_path)
        del encoded_inputs
        del longs_ids
        del all_train_texts
        del dataset

    dataset = TokDataset(tokens_path, block_size=input_len)
    batched_training_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        drop_last=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_batch,
    )

    # --- 3. Optimization Setup  ---
    optimizer = prepare_group_params(
        model, cfg.lr_multipliers, cfg.training.base_lr, cfg.training.weight_decay
    )
    ce_loss = torch.nn.CrossEntropyLoss()
    alpha = torch.tensor(cfg.training.alpha, dtype=torch.bfloat16, device=device)

    # --- 4. Training Loop  ---
    print("Starting training...")
    model.train()

    experiment_logs = defaultdict(list)
    tq_bar = tqdm(total=cfg.training.total_steps, desc="Total Training Steps")

    step = 0
    finished = False
    while not finished:
        metric_aggregator = defaultdict(list)

        for x in batched_training_data:
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
                    metric_aggregator[f"lb_loss_stage-{i}"].append(stage_loss.item())

            loss = ce + alpha * lb_loss
            loss.backward()
            optimizer.step()

            metric_aggregator["ce_loss"].append(ce.item())
            metric_aggregator["lb_loss_total"].append(
                lb_loss.item() if torch.is_tensor(lb_loss) else lb_loss
            )
            metric_aggregator["total_loss"].append(loss.item())

            step += 1
            tq_bar.update(1)

            if step % cfg.training.log_every == 0:
                for key, values in metric_aggregator.items():
                    avg_value = sum(values) / len(values)
                    experiment_logs[key].append((step, avg_value))
                metric_aggregator = defaultdict(list)  # Clear aggregator

            if step % cfg.training.save_every == 0:
                torch.save(
                    model.state_dict(), os.path.join(ckpt_path, f"step_{step}.pt")
                )

            if step >= cfg.training.total_steps:
                finished = True
                break

    torch.save(model.state_dict(), os.path.join(ckpt_path, f"step_{step}.pt"))

    tq_bar.close()
    print("Training finished")

    # --- 5. Plotting  ---
    # Hydra will save these plots to the output directory
    print("Generating plots")
    model.eval()
    plot_loss_curves(
        experiment_logs, save_path=os.path.join(output_dir, "loss_curves.png")
    )
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
