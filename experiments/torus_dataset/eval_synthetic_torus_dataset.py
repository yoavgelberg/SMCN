import torch
import tqdm
from experiments.torus_dataset import synthetic_torus_dataset as synth_dataset
from torch_geometric.loader import DataLoader
from torch.nn import CosineEmbeddingLoss
import hydra
from experiments.torus_dataset.model import build_subcomplex_model, build_homp_model


def get_model(cfg):
    print(f"using {cfg.model_type} model")

    if cfg.model_type == "subcomplex":
        return build_subcomplex_model(
            embedding_dim=cfg.embedding_dim,
            number_cin_layers=cfg.number_cin_layers_for_subcomplex,
            number_subgraph_layers=cfg.number_subgraph_layers,
            device=cfg.device,
            number_of_mlp_layers=cfg.number_of_mlp_layers,
        ).to(cfg.device)

    elif cfg.model_type == "homp":
        return build_homp_model(
            embedding_dim=cfg.embedding_dim,
            number_of_layers=cfg.number_cin_layers_for_homp,
            device=cfg.device,
            number_of_mlp_layers=cfg.number_of_mlp_layers,
        ).to(cfg.device)

    else:
        raise ValueError("does not support this model type")


def get_dataloaders(cfg):
    dataset1 = synth_dataset.TorusDataset(
        root=cfg.root_first_element_dataset,
        min_size=cfg.min_size,
        max_size=cfg.max_size,
        num_components=cfg.num_components,
        number_of_permuted_copies=cfg.number_of_permuted_copies,
        low_rk=cfg.low_rk,
        high_rk=cfg.high_rk,
        first_element_of_pair=True,
    )

    data_loader1 = DataLoader(
        dataset1,
        batch_size=cfg.number_of_permuted_copies,
        shuffle=False,
        follow_batch=[
            "x_0",
            "x_1",
            "x_2",
        ],
    )
    dataset2 = synth_dataset.TorusDataset(
        root=cfg.root_second_element_dataset,
        min_size=cfg.min_size,
        max_size=cfg.max_size,
        num_components=cfg.num_components,
        number_of_permuted_copies=cfg.number_of_permuted_copies,
        low_rk=cfg.low_rk,
        high_rk=cfg.high_rk,
        first_element_of_pair=False,
    )
    data_loader2 = DataLoader(
        dataset2,
        batch_size=cfg.number_of_permuted_copies,
        shuffle=False,
        follow_batch=[
            "x_0",
            "x_1",
            "x_2",
        ],
    )
    return data_loader1, data_loader2


def eval_pairs(cfg, first_complex_batch, second_complex_batch, model):
    print("T2")
    threshold = torch.tensor(cfg.threshold).to(cfg.device)
    T_square = T2_calculation(cfg, first_complex_batch, second_complex_batch, model)
    above_threshold = T_square > threshold
    close_to_threshold = torch.isclose(T_square, threshold, atol=cfg.epsilon_cmp)
    print("above_threshold: ", above_threshold.item())
    print("close_to_threshold: ", close_to_threshold.item())
    return above_threshold, close_to_threshold


def train_on_pair(
    cfg, first_complex_batch, second_complex_batch, model, loss_fn, optimizer
):
    model.train()
    for _ in (pbar := tqdm.trange(cfg.num_epochs)):
        optimizer.zero_grad()
        # get vector per cc
        vec1 = model(first_complex_batch)
        vec2 = model(second_complex_batch)

        # compute loss
        y = torch.tensor([-1] * len(vec1)).to(cfg.device)
        loss = loss_fn(vec1, vec2, y)

        # backprop
        loss.backward()
        optimizer.step()
        # if loss.item() < 1e-5:
        #     return
        pbar.set_description(f"loss on batch: {loss.item()}")


def T2_calculation(cfg, first_complex_batch, second_complex_batch, model):
    with torch.no_grad():

        vec1 = model(first_complex_batch)
        vec2 = model(second_complex_batch)
        batch_dim, output_dim = vec1.shape

        S_epsilon = torch.diag(
            torch.full(size=(output_dim, 1), fill_value=cfg.epsilon_matrix).reshape(-1)
        ).to(cfg.device)

        D = vec1 - vec2
        D_mean = D.mean(dim=0).reshape(-1, 1)
        S = torch.cov(D.t())
        inv_S = torch.linalg.pinv(S + S_epsilon)
        print("T2 score: ", torch.mm(torch.mm(D_mean.T, inv_S), D_mean).item())
        return torch.mm(torch.mm(D_mean.T, inv_S), D_mean)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg):
    dataloader1, dataloader2 = get_dataloaders(cfg)

    model = get_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = CosineEmbeddingLoss(margin=cfg.margin)

    # Record the index of graph pairs which the model could reliably distinguish
    id = 0
    reliable_ids = []
    sucsess_ids = []
    reliability_flag = False
    for first_complex_batch, second_complex_batch in zip(dataloader1, dataloader2):
        print("evaluating on ispmorphic pairs: ", reliability_flag)
        first_complex_batch = first_complex_batch.to(cfg.device)
        second_complex_batch = second_complex_batch.to(cfg.device)

        # train to seperate pairs
        train_on_pair(
            cfg=cfg,
            first_complex_batch=first_complex_batch,
            second_complex_batch=second_complex_batch,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        print("training sucsessfull")

        # evaluate if we managed to reliably seperate pairs
        pair_above_threshold, pair_close_to_threshold = eval_pairs(
            cfg, first_complex_batch, second_complex_batch, model
        )

        if not reliability_flag:
            if pair_above_threshold and not pair_close_to_threshold:
                sucsess_ids.append(id)

        if reliability_flag:
            if (not pair_above_threshold) and (not pair_close_to_threshold):
                reliable_ids.append(id)

        reliability_flag = not reliability_flag
        id += 1

    print("num_sucsessfull: ", len(sucsess_ids))
    print("num_reliable: ", len(reliable_ids))
    return sucsess_ids, reliable_ids


if __name__ == "__main__":
    main()
