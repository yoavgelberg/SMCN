import torch
import torch.nn as nn
import torch.optim
import torch_geometric
import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from typing import Optional
from tqdm import tqdm
from experiments.second_betti_number_prediction.model import (
    build_subcomplex_model,
    build_homp_model,
    build_homp_no_edge_model,
)
from utils.training import train_loop
from utils.utils import build_run_tag
from data.complex_dataset import ComplexDataset, construct_datasets


def set_attribute_as_target(datasets, target_name: str):
    """
    sets the specified target feature as the y value
    """
    for name, dataset in datasets.items():
        datalist = []
        for data in tqdm(dataset):
            target = getattr(data, target_name)
            data.y = target
            datalist.append(data)
        datasets[name] = datalist
    return datasets


def get_dataloaders(
    dataset_root: str,
    original_root: Optional[str],
    construct_complexes: bool,
    batch_size: int,
    min_len: int,
    max_len: int,
    use_subcomplex_features: bool,
    subcomplex_low_rank: int,
    subcomplex_high_rank: int,
    num_workers: int,
    target_name: str,
    zinc_subset: bool = False,
):
    datasets = {
        name: torch_geometric.datasets.ZINC(
            split=name,
            subset=zinc_subset,
            root=original_root + f"/{name}",
        )
        for name in ["train", "val", "test"]
    }

    datasets = (
        construct_datasets(
            root=dataset_root,
            datasets=datasets,
            dim=2,
            min_len=min_len,
            max_len=max_len,
            use_subcomplex_features=use_subcomplex_features,
            subcomplex_low_rank=subcomplex_low_rank,
            subcomplex_high_rank=subcomplex_high_rank,
            compute_betti_numbers=True,
            binary_marking=True,
        )
        if construct_complexes
        else {
            name: ComplexDataset(dataset_root + f"/{name}")
            for name in ["train", "val", "test"]
        }
    )
    datasets = set_attribute_as_target(datasets, target_name)

    dataloaders = {
        name: torch_geometric.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            follow_batch=["x_0", "x_1", "x_2"],
            shuffle=(name == "train"),
        )
        for name, dataset in datasets.items()
    }

    return dataloaders


def get_model(cfg):
    print(f"using {cfg.arch.model_type} model")

    if cfg.arch.model_type == "homp":
        return build_homp_model(
            number_of_layers=cfg.arch.number_of_layers,
            embedding_dim=cfg.arch.embedding_dim,
            device=cfg.device,
            number_of_mlp_layers=cfg.arch.number_of_mlp_layers,
            dropout=cfg.arch.dropout,
            final_dropout=cfg.arch.final_dropout,
            in_dropout=cfg.arch.in_dropout,
        )

    elif cfg.arch.model_type == "subcomplex":
        return build_subcomplex_model(
            embedding_dim=cfg.arch.embedding_dim,
            number_subgraph_layers=cfg.arch.number_subgraph_layers,
            device=cfg.device,
            output_ranks=cfg.arch.output_ranks,
            number_of_mlp_layers=cfg.arch.number_of_mlp_layers,
            dropout=cfg.arch.dropout,
            in_dropout=cfg.arch.in_dropout,
            residual=cfg.arch.residual,
            activation=cfg.arch.activation,
            high_rank=cfg.subcomplex_high_rank,
        )

    elif cfg.arch.model_type == "homp_no_edge":
        return build_homp_no_edge_model(
            number_of_layers=cfg.arch.number_of_layers,
            embedding_dim=cfg.arch.embedding_dim,
            device=cfg.device,
            number_of_mlp_layers=cfg.arch.number_of_mlp_layers,
            dropout=cfg.arch.dropout,
            final_dropout=cfg.arch.final_dropout,
            in_dropout=cfg.arch.in_dropout,
        )

    else:
        raise ValueError("does not support this model type")


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Construct model
    model = get_model(cfg)

    # Print number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    tag = build_run_tag(prefix=cfg.wandb.prefix, attributes=cfg)

    if cfg.wandb.log:
        wandb.init(
            settings=wandb.Settings(start_method="thread"),
            project=cfg.wandb.project_name,
            name=tag,
            config=dict(cfg),
            group=cfg.wandb.group,
        )

    # Get combinatorial complex version of zinc, train, validation and test.
    dataloaders = get_dataloaders(
        dataset_root=cfg.dataset_root,
        original_root=cfg.original_root,
        construct_complexes=cfg.construct_complexes,
        batch_size=cfg.batch_size,
        min_len=cfg.min_len,
        max_len=cfg.max_len,
        use_subcomplex_features=cfg.use_subcomplex_features,
        subcomplex_low_rank=cfg.subcomplex_low_rank,
        subcomplex_high_rank=cfg.subcomplex_high_rank,
        num_workers=cfg.num_workers,
        target_name=cfg.target_name,
        zinc_subset=cfg.zinc_subset,
    )

    # Configure optimizer and learning rate scheduler
    optimizer = instantiate(cfg.optimizer, model.parameters())

    lr_scheduler = instantiate(cfg.lr_scheduler, optimizer=optimizer)

    train_loop(
        model=model,
        train_data_loader=dataloaders["train"],
        validation_data_loader=dataloaders["val"],
        test_data_loader=dataloaders["test"],
        loss_fn=nn.MSELoss(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        number_of_epochs=cfg.number_of_epochs,
        device=cfg.device,
        log_to_wandb=cfg.wandb.log,
        dataset_name="betti-numbers-zinc",
    )

    if cfg.wandb.log:
        wandb.finish()


if __name__ == "__main__":
    main()
