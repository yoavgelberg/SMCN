import torch
import torch.nn as nn
import torch.optim
import torch_geometric
import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from typing import Optional
from experiments.zinc.model import build_sequential_subcomplex_model, build_homp_model
from utils.training import train_loop
from utils.utils import build_run_tag
from data.complex_dataset import ComplexDataset, construct_datasets


def get_model(cfg):
    if cfg.arch.model_type == "subcomplex":
        model = build_sequential_subcomplex_model(
            cin_embedding_dim=cfg.arch.cin_embedding_dim,
            subcomplex_embedding_dim=cfg.arch.subgraph_embedding_dim,
            number_cin_layers_top=cfg.arch.number_of_top_cin_layers,
            number_cin_layers_bottom=cfg.arch.number_of_bottom_cin_layers,
            number_subgraph_layers=cfg.arch.number_of_sub_complex_layers,
            max_output_rank=cfg.arch.max_output_rank,
            device=cfg.device,
            add_residual=cfg.arch.add_residual,
            use_second_conv=cfg.arch.use_second_conv,
            second_conv_type=cfg.arch.second_conv_type,
            number_of_mlp_layers=cfg.arch.number_of_mlp_layers,
            high_rank=cfg.subcomplex_high_rank,
        )

    elif cfg.arch.model_type == "homp":
        model = build_homp_model(
            number_of_blocks=cfg.arch.number_of_top_cin_layers,
            embedding_dim=cfg.arch.cin_embedding_dim,
            device=cfg.device,
            number_of_mlp_layers=cfg.arch.number_of_mlp_layers,
        )

    else:
        raise (ValueError("model type not supported"))
    return model


def get_dataloaders(
    dataset_root: str,
    zinc_root: Optional[str],
    construct_complexes: bool,
    batch_size: int,
    min_len: int,
    max_len: int,
    use_subcomplex_features: bool,
    subcomplex_low_rank: int,
    subcomplex_high_rank: int,
    num_workers: int,
):
    datasets = {
        name: torch_geometric.datasets.ZINC(
            split=name,
            subset=True,
            root=zinc_root + f"/{name}",
        )
        for name in ["train", "val", "test"]
    }

    datasets = (
        construct_datasets(
            root=dataset_root,
            datasets=datasets,
            min_len=min_len,
            max_len=max_len,
            use_subcomplex_features=use_subcomplex_features,
            subcomplex_low_rank=subcomplex_low_rank,
            subcomplex_high_rank=subcomplex_high_rank,
        )
        if construct_complexes
        else {
            name: ComplexDataset(dataset_root + f"/{name}")
            for name in ["train", "val", "test"]
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
        )

    # Get combinatorial complex version of zinc, train, validation and test.
    dataloaders = get_dataloaders(
        dataset_root=cfg.dataset_root,
        zinc_root=cfg.zinc_root,
        construct_complexes=cfg.construct_complexes,
        batch_size=cfg.batch_size,
        min_len=cfg.min_len,
        max_len=cfg.max_len,
        use_subcomplex_features=cfg.use_subcomplex_features,
        subcomplex_low_rank=cfg.subcomplex_low_rank,
        subcomplex_high_rank=cfg.subcomplex_high_rank,
        num_workers=cfg.num_workers,
    )

    # Configure optimizer and learning rate scheduler
    optimizer = instantiate(cfg.optimizer, model.parameters())
    lr_scheduler = instantiate(cfg.lr_scheduler, optimizer=optimizer)

    train_loop(
        model=model,
        train_data_loader=dataloaders["train"],
        validation_data_loader=dataloaders["val"],
        test_data_loader=dataloaders["test"],
        loss_fn=nn.L1Loss(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        number_of_epochs=cfg.number_of_epochs,
        device=cfg.device,
        log_to_wandb=cfg.wandb.log,
    )

    if cfg.wandb.log:
        wandb.finish()


if __name__ == "__main__":
    main()
