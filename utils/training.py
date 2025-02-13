import copy
import torch
import torch.nn as nn
import torch_geometric
import numpy as np
import wandb
import tqdm
from ogb.graphproppred import Evaluator
from typing import Callable


def train_loop(
    model: nn.Module,
    train_data_loader: torch_geometric.data.DataLoader,
    validation_data_loader: torch_geometric.loader.DataLoader,
    test_data_loader: torch_geometric.loader.DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    number_of_epochs: int,
    device: str,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    log_to_wandb: bool = True,
    dataset_name: str = "zinc",
):
    (
        best_validation_score,
        best_validation_score_epoch,
        test_score_on_best_validation,
    ) = get_initial_scores(dataset_name)
    score_update = get_score_update_function(dataset_name)
    score_metric = get_score_metric(dataset_name)

    for epoch in range(1, number_of_epochs + 1):
        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            data_loader=train_data_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
        )

        # Compute train and validation evaluations
        print(f"---> Train loss: {train_loss: .4f}")

        normalized = dataset_name in ["cross-diameter-zinc", "betti-numbers-zinc"]

        validation_score = evaluate(
            model=model,
            data_loader=validation_data_loader,
            metric=score_metric,
            device=device,
            normalized=normalized,
        )

        print(f"---> Validation score: {validation_score: .4f}")

        test_score = evaluate(
            model=model,
            data_loader=test_data_loader,
            metric=score_metric,
            device=device,
            normalized=normalized,
        )
        print(f"---> Test score: {test_score: .4f}")

        (
            best_validation_score,
            best_validation_score_epoch,
            test_score_on_best_validation,
        ) = score_update(
            best_validation_score=best_validation_score,
            best_validation_score_epoch=best_validation_score_epoch,
            test_score_on_best_validation=test_score_on_best_validation,
            validation_score=validation_score,
            epoch=epoch,
            test_score=test_score,
        )

        print(
            f"---> Test score on best validation: {test_score_on_best_validation: .4f}"
        )

        # Update learning rate
        if lr_scheduler is None:
            pass
        else:
            lr_scheduler.step()

        if log_to_wandb:
            wandb.log(
                {
                    "Train loss (epoch)": train_loss,
                    "Validation score (epoch)": validation_score,
                    "Test score": test_score,
                    "Test score on best validation score": test_score_on_best_validation,
                    "Best validation score epoch": best_validation_score_epoch,
                }
            )


def train_epoch(
    model: nn.Module,
    data_loader: torch_geometric.data.DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: str,
):
    model.train()
    model.to(device)
    loss_list = []
    for batch in (pbar := tqdm.tqdm(data_loader, total=len(data_loader))):
        # Load batch to device
        batch = batch.to(device)

        # Zero parameter gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch).view(batch.y.shape)

        # Compute loss
        loss = loss_fn(predictions.to(torch.float32), batch.y.to(torch.float32))

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Log
        pbar.set_description(f"Epoch: {epoch}, batch loss: {loss.item(): .4f}")
        loss_list.append(loss.item())

    return np.mean(loss_list)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: torch_geometric.data.DataLoader,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
    normalized: bool = False,
):
    model.to(device)
    model.eval()

    pred, true = [], []
    for batch in data_loader:
        batch = batch.to(device)
        (
            true.append(batch.y * batch.y_std + batch.y_mean)
            if normalized
            else true.append(batch.y)
        )
        (
            pred.append(model(batch).view(batch.y.shape) * batch.y_std + batch.y_mean)
            if normalized
            else pred.append(model(batch).view(batch.y.shape))
        )

    y_true = torch.cat(true, dim=0)
    y_pred = torch.cat(pred, dim=0)

    return metric(y_true, y_pred).item()


def get_score_metric(dataset_name: str = "zinc"):
    if dataset_name == "zinc":

        def score_metric(y_true: torch.Tensor, y_pred: torch.Tensor):
            y_true = y_true.to(torch.float32)
            y_pred = y_pred.to(torch.float32)
            return nn.L1Loss()(y_true, y_pred)

    elif "ogb" in dataset_name:

        def score_metric(y_true: torch.Tensor, y_pred: torch.Tensor):
            evaluator = Evaluator(name=dataset_name)
            input_dict = {"y_true": y_true, "y_pred": y_pred}
            # Compute the evaluation metric
            result = evaluator.eval(input_dict)
            return list(result.values())[0]

    elif dataset_name in ["cross-diameter-zinc", "betti-numbers-zinc"]:

        def score_metric(y_true: torch.Tensor, y_pred: torch.Tensor):
            y_true = torch.round(y_true).long()
            y_pred = torch.round(y_pred).long()

            return (y_pred == y_true).float().mean()

    return score_metric


def get_initial_scores(dataset_name: str = "zinc"):
    """
    decide how to initialize xcore based on dataset
    """
    if dataset_name in ["zinc", "ogbg-molesol"]:
        best_validation_score = 1000
        best_validation_score_epoch = 0
        test_score_on_best_validation = 1000

    elif dataset_name in ["ogbg-molhiv", "betti-numbers-zinc", "cross-diameter-zinc"]:
        best_validation_score = 0
        best_validation_score_epoch = 0
        test_score_on_best_validation = 0
    return (
        best_validation_score,
        best_validation_score_epoch,
        test_score_on_best_validation,
    )


def get_score_update_function(dataset_name: str = "zinc"):
    """
    decide how to update scores based on dataset
    """
    if dataset_name in ["zinc", "ogbg-molesol"]:

        def score_update(
            best_validation_score,
            best_validation_score_epoch,
            test_score_on_best_validation,
            validation_score,
            epoch,
            test_score,
        ):
            if validation_score < best_validation_score:
                best_validation_score = validation_score
                best_validation_score_epoch = epoch
                test_score_on_best_validation = test_score
            return (
                best_validation_score,
                best_validation_score_epoch,
                test_score_on_best_validation,
            )

    elif dataset_name in ["ogbg-molhiv", "betti-numbers-zinc", "cross-diameter-zinc"]:

        def score_update(
            best_validation_score,
            best_validation_score_epoch,
            test_score_on_best_validation,
            validation_score,
            epoch,
            test_score,
        ):
            if validation_score > best_validation_score:
                best_validation_score = validation_score
                best_validation_score_epoch = epoch
                test_score_on_best_validation = test_score
            return (
                best_validation_score,
                best_validation_score_epoch,
                test_score_on_best_validation,
            )

    return score_update
