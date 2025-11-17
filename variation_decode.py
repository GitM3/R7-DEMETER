import argparse
import os
from typing import Dict, Tuple

import numpy as np
import open3d as o3d
import torch

from representation.graph import PlantGraphFixedTopology
from utils.graph import load_class, load_parent
from utils.pca import NodePCA


def _load_graph(data_folder: str, species: str, sample_name: str) -> Tuple[PlantGraphFixedTopology, Dict[str, NodePCA]]:
    """Load a fitted graph together with the PCA models for the given species."""
    instance_folder = os.path.join(data_folder, species, "instances", sample_name)
    if not os.path.isdir(instance_folder):
        raise FileNotFoundError(f"Could not find instance folder '{instance_folder}'.")

    classes = load_class(os.path.join(instance_folder, "info", "class.txt"))
    parents = load_parent(os.path.join(instance_folder, "info", "parent.txt"))

    pca_models = {
        "stem": NodePCA(path=os.path.join(data_folder, species, "3d_stem_pca.pth")),
        "leaf_deform": NodePCA(path=os.path.join(data_folder, species, "3d_leaf_pca.pth")),
        "leaf_shape": NodePCA(path=os.path.join(data_folder, species, "2d_leaf_pca.pth")),
    }

    plant_graph = PlantGraphFixedTopology(
        classes=classes,
        parents=parents,
        species=species,
        pca_leaf_3d=pca_models["leaf_deform"],
        pca_stem_3d=pca_models["stem"],
        pca_leaf_2d=pca_models["leaf_shape"],
    )
    plant_graph.load(os.path.join(instance_folder, "graph.pkl"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plant_graph.to(device)

    return plant_graph, pca_models


def _sample_coefficients(
    pca_model: NodePCA,
    param: torch.nn.Parameter,
    rng: np.random.Generator,
    scale: float,
    strategy: str,
) -> torch.Tensor:
    """Sample PCA coefficients that match the storage shape of `param`."""
    target_shape = param.shape
    flat_param = param.detach().float().cpu().view(-1).numpy()
    num_coeff = flat_param.size

    coeff_mean = getattr(pca_model, "coeff_mean", None)
    coeff_std = getattr(pca_model, "coeff_std", None)
    if coeff_mean is None or coeff_std is None:
        raise ValueError("PCA model is missing coefficient statistics.")

    base = coeff_mean.detach().cpu().view(-1).numpy()
    std = coeff_std.detach().cpu().view(-1).numpy()
    if base.size != num_coeff or std.size != num_coeff:
        raise ValueError(
            f"PCA configuration mismatch: expected {num_coeff} coefficients, "
            f"got {base.size}."
        )

    if strategy == "perturb":
        base = flat_param

    noise = rng.standard_normal(num_coeff) * scale
    sample = base + noise * std
    sample_tensor = torch.from_numpy(sample).float().to(param.device).view(*target_shape)
    return sample_tensor


def randomize_graph_coefficients(
    plant_graph: PlantGraphFixedTopology,
    pca_models: Dict[str, NodePCA],
    rng: np.random.Generator,
    stem_scale: float,
    leaf_shape_scale: float,
    leaf_deform_scale: float,
    strategy: str,
) -> None:
    """Overwrite the PCA parameters of the graph with random samples."""
    with torch.no_grad():
        for node_id in plant_graph.stem_key:
            param = getattr(plant_graph, f"deform_{node_id}")
            new_value = _sample_coefficients(
                pca_model=pca_models["stem"],
                param=param,
                rng=rng,
                scale=stem_scale,
                strategy=strategy,
            )
            param.copy_(new_value)

        for node_id in plant_graph.leaf_key:
            deform_param = getattr(plant_graph, f"deform_{node_id}")
            new_deform = _sample_coefficients(
                pca_model=pca_models["leaf_deform"],
                param=deform_param,
                rng=rng,
                scale=leaf_deform_scale,
                strategy=strategy,
            )
            deform_param.copy_(new_deform)

            shape_param = getattr(plant_graph, f"shape_{node_id}")
            new_shape = _sample_coefficients(
                pca_model=pca_models["leaf_shape"],
                param=shape_param,
                rng=rng,
                scale=leaf_shape_scale,
                strategy=strategy,
            )
            shape_param.copy_(new_shape)


def save_mesh(mesh: o3d.geometry.TriangleMesh, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    print(f"Writing mesh to {path}")
    o3d.io.write_triangle_mesh(path, mesh, write_ascii=False, compressed=False)


def generate_variations(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    plant_graph, pca_models = _load_graph(args.data_folder, args.species, args.sample_name)
    if args.draw_graph:
        plant_graph.draw_topology()

    base_state = {k: v.detach().clone() for k, v in plant_graph.state_dict().items()}
    for idx in range(args.num_variations):
        plant_graph.load_state_dict(base_state, strict=False)
        randomize_graph_coefficients(
            plant_graph,
            pca_models,
            rng,
            stem_scale=args.stem_scale,
            leaf_shape_scale=args.leaf_shape_scale,
            leaf_deform_scale=args.leaf_deform_scale,
            strategy=args.strategy,
        )

        with torch.no_grad():
            mesh = plant_graph.generate(
                output_format="mesh",
                color=args.color,
                align_global=args.align_global,
            )

        if args.visualize:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
            o3d.visualization.draw_geometries([mesh, axis], mesh_show_back_face=True)

        output_name = f"{args.sample_name}_variation_{idx:02d}.ply"
        save_mesh(mesh, os.path.join(args.output_dir, output_name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random Demeter variations by sampling PCA coefficients."
    )
    parser.add_argument("--data_folder", type=str, default="sample_params")
    parser.add_argument("--species", type=str, default="soybean")
    parser.add_argument("--sample_name", type=str, required=True, help="Base instance to copy topology from.")
    parser.add_argument("--output_dir", type=str, default="variation_output")
    parser.add_argument("--num_variations", type=int, default=1)
    parser.add_argument("--stem_scale", type=float, default=1.0, help="Noise scale for stem PCA coefficients.")
    parser.add_argument("--leaf_deform_scale", type=float, default=1.0, help="Noise scale for 3D leaf PCA coefficients.")
    parser.add_argument("--leaf_shape_scale", type=float, default=1.0, help="Noise scale for 2D leaf PCA coefficients.")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=("mean", "perturb"),
        default="mean",
        help="Sample around the PCA mean or perturb the fitted instance coefficients.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seeds the RNG for repeatable sampling.")
    parser.add_argument("--align_global", action="store_true", help="Align the main stem to the global axis.")
    parser.add_argument("--draw_graph", action="store_true", help="Print plant hierarchy information.")
    parser.add_argument("--visualize", action="store_true", help="Open an interactive viewer for each variation.")
    parser.add_argument("--color", type=str, default="gray", help="Mesh color mode used by PlantGraph.generate.")
    return parser.parse_args()


if __name__ == "__main__":
    generate_variations(parse_args())
