import matplotlib.pyplot as plt
import numpy as np
import torch

# HNet imports
from hnet.models.mixer_seq import HNetForCausalLM
from hnet.utils.tokenizers import ByteTokenizer, TSTokenizer
from hnet.utils.train import apply_optimization_params, group_params


def prepare_group_params(
    model: HNetForCausalLM,
    lr_multipliers: list[float],
    base_lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """
    Prepares optimizer parameter groups with different learning rates and weight decays.
    Hardcoded for 2stage HNet model as per the current configuration.
    """
    assert hasattr(model.backbone.main_network, "encoder"), "Model is not S=2"
    assert (
        len(lr_multipliers) == 3
    ), f"Expecting 3 LR multipliers for S=2, got {len(lr_multipliers)}"

    print("Pre-annotating model parameters with LR multipliers...")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        current_lr_mult = lr_multipliers[2]  # Default: main network LR

        # --- Stage 0 (S=0, outermost) ---
        if (
            name.startswith("embeddings")
            or name.startswith("lm_head")
            or name.startswith("backbone.encoder")
            or name.startswith("backbone.decoder")
            or name.startswith("backbone.routing_module")
            or name.startswith("backbone.chunk_layer")
            or name.startswith("backbone.dechunk_layer")
            or name.startswith("backbone.residual_proj")
        ):
            current_lr_mult = lr_multipliers[0]

        # --- Stage 1 (S=1, middle) ---
        elif (
            name.startswith("backbone.main_network.encoder")
            or name.startswith("backbone.main_network.decoder")
            or name.startswith("backbone.main_network.routing_module")
            or name.startswith("backbone.main_network.chunk_layer")
            or name.startswith("backbone.main_network.dechunk_layer")
            or name.startswith("backbone.main_network.residual_proj")
        ):
            current_lr_mult = lr_multipliers[1]

        # --- Main Network (M, innermost) ---
        elif name.startswith("backbone.main_network.main_network"):
            current_lr_mult = lr_multipliers[2]

        apply_optimization_params(param, lr_mult=current_lr_mult)

    gparams = group_params(model)

    print(f"Created {len(gparams)} optimizer param groups:")
    for i, g in enumerate(gparams):
        n = sum(p.numel() for p in g["params"])
        lr_str = g.get("lr_mult", "default")
        wd_str = g.get("weight_decay", "default_wd")
        print(f"  group {i}: params={n / 1e6:.2f}M, lr_mult={lr_str}, wd={wd_str}")

    # Convert lr_mult to absolute lr
    for g in gparams:
        if "lr_mult" in g:
            g["lr"] = base_lr * g.pop("lr_mult")
        if "weight_decay" not in g:
            g["weight_decay"] = weight_decay

    optimizer = torch.optim.AdamW(gparams, lr=base_lr, weight_decay=weight_decay)
    print("\nOptimizer groups are set up and ready.")
    return optimizer


def plot_loss_curves(experiment_logs: dict, save_path: str = "loss_curves.png") -> None:
    """
    Plots the training loss curves.
    """
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

    plt.figure(figsize=(12, 8))
    to_plot = [
        "ce_loss",
        "lb_loss_total",
        "total_loss",
        "lb_loss_stage-0",
        "lb_loss_stage-1",
    ]
    for metric in to_plot:
        if metric in experiment_logs:
            steps, values = zip(*experiment_logs[metric])
            plt.plot(
                steps,
                values,
                label=metric,
                color=colors[to_plot.index(metric) % len(colors)],
                linestyle=linestyles[to_plot.index(metric) % len(linestyles)],
            )

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Losses Over Steps")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss curves to {save_path}")


def plot_words(
    model: HNetForCausalLM,
    tokenizer: ByteTokenizer,
    sentences: list[str],
    device: str,
    save_path: str = "word_boundaries",
) -> list[str]:
    """
    Plots the dynamic chunking boundaries for a sample sentence.
    (From the last cell)
    """
    model.eval()
    paths = []

    for i, sentence in enumerate(sentences):
        encoded = tokenizer.encode([sentence], add_bos=True)
        encoded_ids = encoded[0]["input_ids"]
        test_input_ids_BxL = torch.tensor(
            encoded_ids, dtype=torch.long, device=device
        ).unsqueeze(0)

        print(f'Visualizing: "{sentence}"')
        print(f"Characters: {len(sentence)}, Tokens: {len(encoded_ids)}")

        with torch.no_grad():
            mask = torch.ones(test_input_ids_BxL.shape, device=device, dtype=torch.bool)
            output = model.forward(test_input_ids_BxL, mask=mask)
            bpred = output.bpred_output

            bpred0_bp = bpred[0].boundary_prob.float().squeeze().cpu().numpy()
            bpred0_bp = np.argmax(bpred0_bp, axis=-1)
            brped0_bm = bpred[0].boundary_mask.squeeze().cpu().numpy()
            bpred0_sp = bpred[0].selected_probs.squeeze().float().cpu().numpy()
            bpred1_bp = bpred[1].boundary_prob.squeeze().float().cpu().numpy()
            bpred1_bp = np.argmax(bpred1_bp, axis=-1)
            brped1_bm = bpred[1].boundary_mask.squeeze().cpu().numpy()
            bpred1_sp = bpred[1].selected_probs.squeeze().float().cpu().numpy()

            compression_ratio_stage0 = len(encoded_ids) / (brped0_bm == 1).sum()
            compression_ratio_stage1 = (brped0_bm == 1).sum() / (brped1_bm == 1).sum()
            print(f"Compression Ratio Stage 0: {compression_ratio_stage0:.2f}")
            print(f"Compression Ratio Stage 1: {compression_ratio_stage1:.2f}")

            fig, ax = plt.subplots(figsize=(14, 2.5))

            sentence_vis = "#" + sentence
            tokens = encoded_ids
            ax.scatter(
                range(len(tokens)),
                [1] * len(tokens),
                c="lightblue",
                s=200,
                label="Input Tokens",
                marker="s",
                edgecolors="black",
            )
            for i, char in enumerate(sentence_vis):
                ax.text(i, 1, char, ha="center", va="center", fontsize=8)

            ax.scatter(
                range(len(bpred0_sp)),
                [0.75] * len(bpred0_sp),
                c=bpred0_sp,
                cmap="Reds",
                s=100,
                label="Stage 0 Boundary probabilities",
                marker="s",
            )
            ax.scatter(
                range(len(brped0_bm)),
                [0.5] * len(brped0_bm),
                c=brped0_bm,
                cmap="gray_r",
                s=100,
                label="Stage 0 Boundary mask",
                marker="s",
            )

            indices = np.where(brped0_bm == 1)[0]
            indices_sel = indices[brped1_bm]

            ax.scatter(
                indices,
                [0.25] * len(indices),
                c=bpred1_sp,
                s=100,
                label="Stage 1 Boundary Probabilities",
                marker="s",
                cmap="Reds",
            )
            ax.scatter(
                indices_sel,
                [0.0] * len(indices_sel),
                c="black",
                s=100,
                label="Stage 1 Boundary Mask",
                marker="s",
            )
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_yticklabels(
                [
                    "[Stage 1] mask",
                    "[Stage 1] boundary probabilities",
                    "[Stage 0] mask",
                    "[Stage 0] boundary probabilities",
                    "Input Tokens",
                ]
            )
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            plt.tight_layout()
            path = save_path + f"{i}.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

            paths.append(path)

        print(f"Saved boundary visualizations to {save_path}")
    return paths


def plot_token_boundaries_ts(
    model: HNetForCausalLM,
    tokenizer: TSTokenizer,
    sequences: torch.Tensor,  # shape (num_sequences, sequence_length), dtype torch.long
    device: str,
    save_path: str = "ts_token_boundaries",
) -> list[str]:
    """
    Plots the dynamic chunking boundaries for a sample time series.
    """
    model.eval()
    paths = []

    with torch.no_grad():
        mask = torch.ones(sequences.shape, device=device, dtype=torch.bool)
        output = model.forward(sequences, mask=mask)
        # bpred = output.bpred_output

    sequences_np = sequences.cpu().numpy()
    bpred_s0 = output.bpred_output[0]
    bpred_s1 = output.bpred_output[1]

    for i, encoded_ids in enumerate(sequences_np):
        # Stage 0 boundaries
        bpred0_bp = bpred_s0.boundary_prob[i].float().squeeze(0).cpu().numpy()
        bpred0_bp = np.argmax(bpred0_bp, axis=-1)
        brped0_bm = bpred_s0.boundary_mask[i].squeeze(0).cpu().numpy()
        bpred0_sp = bpred_s0.selected_probs[i].squeeze(0).float().cpu().numpy()
        # Stage 1 boundaries
        bpred1_bp = bpred_s1.boundary_prob[i].squeeze(0).float().cpu().numpy()
        bpred1_bp = np.argmax(bpred1_bp, axis=-1)
        brped1_bm = bpred_s1.boundary_mask[i].squeeze(0).cpu().numpy()
        bpred1_sp = bpred_s1.selected_probs[i].squeeze(0).float().cpu().numpy()

        compression_ratio_stage0 = len(encoded_ids) / (brped0_bm == 1).sum()
        compression_ratio_stage1 = (brped0_bm == 1).sum() / (brped1_bm == 1).sum()

        print(f"Compression Ratio Stage Input -> 0: {compression_ratio_stage0:.2f}")
        print(f"Compression Ratio Stage 0 -> 1: {compression_ratio_stage1:.2f}")
        print(
            f"Total Compression Ratio Input -> 1: {compression_ratio_stage0 * compression_ratio_stage1:.2f}"
        )

        # Print out shapes of everytihng for debugging
        print(f"Encoded ids shape: {encoded_ids.shape}")
        print(
            f"bpred0_bp shape: {bpred0_bp.shape}, brped0_bm shape: {brped0_bm.shape}, bpred0_sp shape: {bpred0_sp.shape}"
        )
        print(
            f"bpred1_bp shape: {bpred1_bp.shape}, brped1_bm shape: {brped1_bm.shape}, bpred1_sp shape: {bpred1_sp.shape}"
        )

        # Plotting decoded time series
        decoded = tokenizer.decode(encoded_ids)

        fig, ax = plt.subplots(figsize=(14, 4))
        time_steps = np.arange(len(decoded))
        ax.plot(time_steps, decoded, label="Time Series", color="lightblue")

        # Stage 0 boundaries
        ax.scatter(
            range(len(bpred0_sp)),
            [decoded.max() * 1.05] * len(bpred0_sp),
            c=bpred0_sp,
            cmap="Reds",
            s=100,
            label="Stage 0 Boundary probabilities",
            marker="s",
        )
        ax.scatter(
            range(len(brped0_bm)),
            [decoded.max() * 1.1] * len(brped0_bm),
            c=brped0_bm,
            cmap="gray_r",
            s=100,
            label="Stage 0 Boundary mask",
            marker="s",
        )

        # Stage 1 boundaries
        indices = np.where(brped0_bm == True)[0]  # noqa
        print(f"indices shape: {indices.shape}, brped1_bm shape: {brped1_bm.shape}")
        print(f"brped0_bm: {brped0_bm.tolist()}")
        print(f"indices: {indices}")
        print(f"brped1_bm: {brped1_bm.tolist()}")
        indices_sel = indices[brped1_bm]
        print(f"indices_sel shape: {indices_sel.shape}")

        ax.scatter(
            indices,
            [decoded.max() * 1.15] * len(indices),
            c=bpred1_sp,
            s=100,
            label="Stage 1 Boundary Probabilities",
            marker="s",
            cmap="Reds",
        )
        ax.scatter(
            indices_sel,
            [decoded.max() * 1.2] * len(indices_sel),
            c="black",
            s=100,
            label="Stage 1 Boundary Mask",
            marker="s",
        )
        ax.set_yticks([])
        ax.set_title(f"Time Series Token Boundaries Visualization for Sequence {i}")
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        path = save_path + f"{i}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

        paths.append(path)

    print(f"Saved boundary visualizations to {save_path}")
    return paths
