from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .batch_qc import load_config as load_preprocessing_config
from .cnn_data import (
    fit_channel_normalization,
    load_shape_window_index,
    materialize_windows,
    participant_group_folds,
    participant_inner_folds,
)
from .cnn_models import ShapeVariantModel
from .run_shape_cnn import _aggregate_probabilities, _metrics


VARIANTS = ("multigroup_mixup", "multigroup_groupdro")
SIZE_TO_INDEX = {size: index for index, size in enumerate((30, 40, 50, 60, 70, 80))}


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _loader(
    raw: np.ndarray,
    selected: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    sizes: np.ndarray,
    labels: np.ndarray,
    participants: np.ndarray,
    shuffle: bool,
    batch_size: int,
) -> DataLoader:
    normalized = ((raw[selected] - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(normalized),
            torch.from_numpy(sizes[selected]),
            torch.from_numpy(labels[selected]),
            torch.from_numpy(participants[selected]),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )


def _initial_group_weights(participants: np.ndarray, group_count: int, device: torch.device) -> torch.Tensor:
    weights = torch.zeros(group_count, dtype=torch.float32, device=device)
    active = torch.as_tensor(np.unique(participants), dtype=torch.long, device=device)
    weights[active] = 1.0 / len(active)
    return weights


def _mixup_permutation(size: torch.Tensor) -> torch.Tensor:
    permutation = torch.arange(len(size), device=size.device)
    for size_value in torch.unique(size):
        members = torch.where(size == size_value)[0]
        permutation[members] = members[torch.randperm(len(members), device=size.device)]
    return permutation


def _groupdro_loss(
    loss_a: torch.Tensor,
    loss_b: torch.Tensor,
    group_a: torch.Tensor,
    group_b: torch.Tensor,
    mixing: float,
    group_weights: torch.Tensor,
    adjustment_rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_count = len(group_weights)
    sums = torch.zeros(group_count, dtype=loss_a.dtype, device=loss_a.device)
    masses = torch.zeros(group_count, dtype=loss_a.dtype, device=loss_a.device)
    mix_a = torch.full_like(loss_a, mixing)
    mix_b = torch.full_like(loss_b, 1.0 - mixing)
    sums.scatter_add_(0, group_a, mix_a * loss_a)
    sums.scatter_add_(0, group_b, mix_b * loss_b)
    masses.scatter_add_(0, group_a, mix_a)
    masses.scatter_add_(0, group_b, mix_b)
    present = masses > 0
    group_losses = sums / masses.clamp_min(1e-8)
    with torch.no_grad():
        group_weights[present] *= torch.exp(
            adjustment_rate * group_losses.detach()[present].float().clamp(max=5.0)
        )
        active = group_weights > 0
        group_weights[active].clamp_(min=1e-8)
        group_weights /= group_weights.sum()
    present_weight = group_weights[present].sum().clamp_min(1e-8)
    objective = (group_weights[present] * group_losses[present]).sum() / present_weight
    return objective, group_losses.detach()


def _train_epoch(
    model: ShapeVariantModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    group_weights: torch.Tensor | None,
    groupdro_rate: float,
) -> tuple[float, float]:
    model.train()
    loss_total = 0.0
    correct_total = 0.0
    sample_total = 0
    for raw_batch, size, shape, participant in loader:
        raw_batch = raw_batch.to(device)
        size = size.to(device)
        shape = shape.to(device)
        participant = participant.to(device)
        permutation = _mixup_permutation(size)
        mixing = float(np.random.beta(0.2, 0.2))
        mixed = mixing * raw_batch + (1.0 - mixing) * raw_batch[permutation]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(mixed, size)
            loss_a = functional.cross_entropy(logits, shape, reduction="none")
            loss_b = functional.cross_entropy(logits, shape[permutation], reduction="none")
            if group_weights is None:
                loss = (mixing * loss_a + (1.0 - mixing) * loss_b).mean()
            else:
                loss, _ = _groupdro_loss(
                    loss_a, loss_b, participant, participant[permutation], mixing,
                    group_weights, groupdro_rate,
                )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        predicted = logits.detach().argmax(dim=1)
        correct_total += float(
            mixing * (predicted == shape).sum().item()
            + (1.0 - mixing) * (predicted == shape[permutation]).sum().item()
        )
        loss_total += float(loss.detach().item()) * len(raw_batch)
        sample_total += len(raw_batch)
    return loss_total / sample_total, correct_total / sample_total


@torch.no_grad()
def _predict(model: ShapeVariantModel, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    for raw_batch, size, _, _ in loader:
        logits, _ = model(raw_batch.to(device), size.to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _file_result(records, indices: np.ndarray, probability: np.ndarray) -> tuple[dict[str, object], list[dict]]:
    truth, predicted, rows = _aggregate_probabilities(records, indices, probability, "file_id")
    return _metrics(truth, predicted, 3), rows


def _participant_metrics(prediction_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for participant in sorted({str(row["participant"]) for row in prediction_rows}):
        selected = [row for row in prediction_rows if str(row["participant"]) == participant]
        result = _metrics(
            np.asarray([int(row["y_true"]) for row in selected]),
            np.asarray([int(row["y_pred"]) for row in selected]),
            3,
        )
        output.append({"participant": participant, **result})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-group validation and participant GroupDRO shape CNN")
    parser.add_argument("--split-config", type=Path, default=Path("configs/shape_grouped_12_train_3_test.json"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/modeling/shape_cnn_multigroup_groupdro_12_3"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--minimum-selection-epoch", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--stability-penalty", type=float, default=0.25)
    parser.add_argument("--groupdro-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fold-limit", type=int)
    args = parser.parse_args()
    if not 1 <= args.minimum_selection_epoch <= args.epochs:
        raise ValueError("minimum-selection-epoch must be between 1 and epochs")

    split_config = json.loads(args.split_config.read_text(encoding="utf-8"))
    split_seed = int(split_config["random_seed"])
    training_seed = split_seed if args.seed is None else args.seed
    test_groups = [fold["test_participants"] for fold in split_config["folds"]]
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    device = torch.device("cuda")
    _, preprocessing = load_preprocessing_config(Path("configs/preprocessing.json"))
    records = load_shape_window_index(
        Path("artifacts/segmentation/segments.csv"), preprocessing.sampling_rate_hz,
        0.5, 0.25, "SPH", ("cuboid", "cylinder", "sphere"), {"pass", "warning"}, None,
    )
    all_indices = np.arange(len(records), dtype=np.int64)
    print(f"materializing {len(records)} windows", flush=True)
    raw, _ = materialize_windows(
        Path("."), records, all_indices, Path("configs/preprocessing.json"),
        ("EDC", "FCR", "FDS", "FCU"), True,
    )
    shape_y = np.asarray([("cuboid", "cylinder", "sphere").index(record.shape) for record in records], dtype=np.int64)
    size_y = np.asarray([SIZE_TO_INDEX[record.size_mm] for record in records], dtype=np.int64)
    participant_names = sorted({record.participant_id for record in records})
    participant_to_index = {name: index for index, name in enumerate(participant_names)}
    participant_y = np.asarray([participant_to_index[record.participant_id] for record in records], dtype=np.int64)
    splits = participant_group_folds(records, test_groups)
    if args.fold_limit is not None:
        splits = splits[:args.fold_limit]
    args.output_root.mkdir(parents=True, exist_ok=True)

    root_summary: dict[str, object] = {
        "protocol": split_config["protocol"],
        "training_seed": training_seed,
        "epochs": args.epochs,
        "minimum_selection_epoch": args.minimum_selection_epoch,
        "inner_folds": args.inner_folds,
        "selection_score": f"mean_macro_f1-{args.stability_penalty}*std_macro_f1",
        "groupdro_rate": args.groupdro_rate,
        "variants": {},
    }
    for variant_position, variant in enumerate(args.variants):
        output = args.output_root / variant
        history_root = output / "history"
        history_root.mkdir(parents=True, exist_ok=True)
        fold_rows: list[dict[str, object]] = []
        prediction_rows: list[dict[str, object]] = []
        use_groupdro = variant == "multigroup_groupdro"
        print(f"starting {variant}", flush=True)
        for fold, (test_group, outer_train, test_idx) in enumerate(splits, 1):
            fold_seed = training_seed + variant_position * 10000 + fold * 100
            inner_folds = participant_inner_folds(
                records, outer_train, args.inner_folds, training_seed + fold * 97
            )
            inner_scores: list[list[float]] = []
            inner_groups: list[str] = []
            for inner_number, (fit_idx, validation_idx, validation_participants) in enumerate(inner_folds, 1):
                _seed(fold_seed + inner_number)
                mean, std = fit_channel_normalization(raw[fit_idx])
                fit_loader = _loader(
                    raw, fit_idx, mean, std, size_y, shape_y, participant_y, True, args.batch_size
                )
                validation_loader = _loader(
                    raw, validation_idx, mean, std, size_y, shape_y, participant_y, False, args.batch_size
                )
                model = ShapeVariantModel("size_specific_mixup").to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                scaler = torch.amp.GradScaler("cuda")
                group_weights = (
                    _initial_group_weights(participant_y[fit_idx], len(participant_names), device)
                    if use_groupdro else None
                )
                history: list[dict[str, object]] = []
                scores: list[float] = []
                for epoch in range(1, args.epochs + 1):
                    training_loss, training_accuracy = _train_epoch(
                        model, fit_loader, optimizer, scaler, device, group_weights, args.groupdro_rate
                    )
                    probability = _predict(model, validation_loader, device)
                    result, _ = _file_result(records, validation_idx, probability)
                    scores.append(float(result["macro_f1"]))
                    history.append({
                        "epoch": epoch,
                        "training_loss": training_loss,
                        "training_mixup_accuracy": training_accuracy,
                        "validation_accuracy": result["accuracy"],
                        "validation_macro_f1": result["macro_f1"],
                        "validation_balanced_accuracy": result["balanced_accuracy"],
                        "group_weight_min": float(group_weights[group_weights > 0].min().item()) if group_weights is not None else "",
                        "group_weight_max": float(group_weights.max().item()) if group_weights is not None else "",
                    })
                _write_rows(history_root / f"fold_{fold:02d}_inner_{inner_number:02d}.csv", history)
                inner_scores.append(scores)
                inner_groups.append(",".join(validation_participants))
                del model, optimizer, scaler, fit_loader, validation_loader, probability
                torch.cuda.empty_cache()

            score_matrix = np.asarray(inner_scores, dtype=np.float64).T
            aggregate_history: list[dict[str, object]] = []
            for epoch_index, values in enumerate(score_matrix, 1):
                mean_score = float(values.mean())
                std_score = float(values.std(ddof=0))
                aggregate_history.append({
                    "epoch": epoch_index,
                    **{f"inner_{i + 1}_macro_f1": float(value) for i, value in enumerate(values)},
                    "mean_macro_f1": mean_score,
                    "std_macro_f1": std_score,
                    "selection_score": mean_score - args.stability_penalty * std_score,
                })
            eligible = aggregate_history[args.minimum_selection_epoch - 1:]
            best = max(eligible, key=lambda row: (float(row["selection_score"]), -int(row["epoch"])))
            best_epoch = int(best["epoch"])
            _write_rows(history_root / f"fold_{fold:02d}_aggregate.csv", aggregate_history)

            _seed(fold_seed)
            full_mean, full_std = fit_channel_normalization(raw[outer_train])
            full_loader = _loader(
                raw, outer_train, full_mean, full_std, size_y, shape_y, participant_y, True, args.batch_size
            )
            test_loader = _loader(
                raw, test_idx, full_mean, full_std, size_y, shape_y, participant_y, False, args.batch_size
            )
            final_model = ShapeVariantModel("size_specific_mixup").to(device)
            final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
            final_scaler = torch.amp.GradScaler("cuda")
            final_group_weights = (
                _initial_group_weights(participant_y[outer_train], len(participant_names), device)
                if use_groupdro else None
            )
            for _ in range(best_epoch):
                _train_epoch(
                    final_model, full_loader, final_optimizer, final_scaler, device,
                    final_group_weights, args.groupdro_rate,
                )
            probability = _predict(final_model, test_loader, device)
            result, rows = _file_result(records, test_idx, probability)
            fold_rows.append({
                "fold": fold,
                "test_group": test_group,
                "inner_validation_groups": "|".join(inner_groups),
                "selected_epoch": best_epoch,
                "selection_mean_macro_f1": best["mean_macro_f1"],
                "selection_std_macro_f1": best["std_macro_f1"],
                "selection_score": best["selection_score"],
                "final_group_weight_min": float(final_group_weights[final_group_weights > 0].min().item()) if final_group_weights is not None else "",
                "final_group_weight_max": float(final_group_weights.max().item()) if final_group_weights is not None else "",
                **result,
            })
            prediction_rows.extend({
                "fold": fold,
                "test_group": test_group,
                "participant": row["participant_id"],
                "file_id": row["group_id"],
                "y_true": row["y_true"],
                "y_pred": row["y_pred"],
                "probabilities": row["probabilities"],
            } for row in rows)
            torch.save({
                "model_state": final_model.state_dict(),
                "variant": variant,
                "selected_epoch": best_epoch,
                "normalization_mean": full_mean,
                "normalization_std": full_std,
                "participant_names": participant_names,
                "group_weights": final_group_weights.cpu() if final_group_weights is not None else None,
            }, output / f"fold_{fold:02d}.pt")
            _write_rows(output / "fold_metrics.csv", fold_rows)
            _write_rows(output / "file_predictions.csv", prediction_rows)
            print(
                f"{variant} fold {fold}/{len(splits)} epoch={best_epoch} "
                f"selection={float(best['selection_score']):.4f} acc={float(result['accuracy']):.4f} "
                f"macro={float(result['macro_f1']):.4f}", flush=True,
            )
            del final_model, final_optimizer, final_scaler, full_loader, test_loader, probability
            torch.cuda.empty_cache()

        participant_rows = _participant_metrics(prediction_rows)
        _write_rows(output / "participant_metrics.csv", participant_rows)
        pooled = _metrics(
            np.asarray([int(row["y_true"]) for row in prediction_rows]),
            np.asarray([int(row["y_pred"]) for row in prediction_rows]),
            3,
        )
        summary = {
            "protocol": split_config["protocol"],
            "training_seed": training_seed,
            "variant": variant,
            "folds": len(splits),
            "file_count": len(prediction_rows),
            "selected_epochs": [int(row["selected_epoch"]) for row in fold_rows],
            "participant_mean_accuracy": float(np.mean([row["accuracy"] for row in participant_rows])),
            "participant_mean_macro_f1": float(np.mean([row["macro_f1"] for row in participant_rows])),
            "participant_mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in participant_rows])),
            "pooled": pooled,
        }
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        root_summary["variants"][variant] = summary
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    (args.output_root / "summary.json").write_text(
        json.dumps(root_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
