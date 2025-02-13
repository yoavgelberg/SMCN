import torch
import torch.nn as nn
import torch.optim
import torch_geometric
import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from typing import Optional
from utils.training import train_loop
from utils.utils import build_run_tag
from data.complex_dataset import ComplexDataset, construct_datasets
from experiments.molhiv.model import build_homp_model, build_subcomplex_ensemble_model
from ogb.graphproppred import PygGraphPropPredDataset


def get_model(cfg):
    print(f"using {cfg.arch.model_type} model")

    if cfg.arch.model_type == "subcomplex":
        return build_subcomplex_ensemble_model(
            embedding_dim_cin=cfg.arch.embedding_dim_cin,
            embedding_dim_subgraphs=cfg.arch.embedding_dim_subgraphs,
            number_cin_layers=cfg.arch.number_cin_layers,
            number_subgraph_layers=cfg.arch.number_subgraph_layers,
            device=cfg.device,
            output_ranks_cin=[0, 1, 2],
            output_ranks_subgraphs=cfg.arch.output_ranks_subgraphs,
            residual_cin=cfg.arch.residual_cin,
            residual_subgraph=cfg.arch.residual_subgraph,
            number_of_mlp_layers=cfg.arch.number_of_mlp_layers,
            dropout_cin=cfg.arch.dropout_cin,
            dropout_subgraph=cfg.arch.dropout_subgraph,
            in_dropout=cfg.arch.in_dropout,
            subgraph_embedding_aggregation="concatenate",
            subgraph_layer_aggregation="concatenate",
            subgraph_final_aggregation="mean",
            activation=cfg.arch.activation,
            final_dropout=cfg.arch.final_dropout,
            high_rank=cfg.subcomplex_high_rank,
        )

    elif cfg.arch.model_type == "homp":
        return build_homp_model(
            number_of_layers=cfg.arch.number_cin_layers,
            embedding_dim=cfg.arch.embedding_dim_cin,
            device=cfg.device,
            number_of_mlp_layers=cfg.arch.number_of_mlp_layers,
        )

    else:
        raise ValueError("does not support this model type")


def get_dataloaders(
    dataset_root: str,
    original_root: Optional[str],
    construct_complexes: bool,
    batch_size: int,
    max_complex_rank: int,
    min_len: int,
    max_len: int,
    use_subcomplex_features: bool,
    subcomplex_low_rank: int,
    subcomplex_high_rank: int,
    num_workers: int,
):
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=original_root)
    split_idx = dataset.get_idx_split()
    datasets = {name: dataset[split_idx[name]] for name in ["train", "valid", "test"]}
    del dataset

    datasets = (
        construct_datasets(
            root=dataset_root,
            datasets=datasets,
            dim=max_complex_rank,
            min_len=min_len,
            max_len=max_len,
            use_subcomplex_features=use_subcomplex_features,
            subcomplex_low_rank=subcomplex_low_rank,
            subcomplex_high_rank=subcomplex_high_rank,
        )
        if construct_complexes
        else {
            name: ComplexDataset(dataset_root + f"/{name}")
            for name in ["train", "valid", "test"]
        }
    )

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
        max_complex_rank=2,
        min_len=cfg.min_len,
        max_len=cfg.max_len,
        use_subcomplex_features=cfg.use_subcomplex_features,
        subcomplex_low_rank=cfg.subcomplex_low_rank,
        subcomplex_high_rank=cfg.subcomplex_high_rank,
        num_workers=cfg.num_workers,
    )

    # Configure optimizer and learning rate scheduler
    optimizer = instantiate(cfg.optimizer, model.parameters())

    train_loop(
        model=model,
        train_data_loader=dataloaders["train"],
        validation_data_loader=dataloaders["valid"],
        test_data_loader=dataloaders["test"],
        loss_fn=nn.BCEWithLogitsLoss(),
        optimizer=optimizer,
        number_of_epochs=cfg.number_of_epochs,
        device=cfg.device,
        log_to_wandb=cfg.wandb.log,
        dataset_name="ogbg-molhiv",
    )

    if cfg.wandb.log:
        wandb.finish()


if __name__ == "__main__":
    main()
