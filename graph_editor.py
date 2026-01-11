#!/usr/bin/env python3
"""
Minimal interactive graph editor for Demeter.

Features:
- Load an existing instance (classes/parents + graph.pkl).
- Add new nodes to the topology (clone parameters from an existing node).
- Generate and visualize: draw topology (matplotlib) and mesh (Open3D), similar to decode.py.

Notes:
- Requires CUDA: current decoding/generation uses .cuda() internally.
- New nodes clone parameters from a chosen source node of the same class.
  This keeps generation stable without bespoke initialization.
"""

import argparse
import os
from typing import Dict

import torch
import open3d as o3d
from easydict import EasyDict as edict

from representation.graph import PlantGraphFixedTopology
from utils.pca import NodePCA
from utils.constant import LEAF_CLASS, STEM_CLASS, FLOWER_CLASS, FRUIT_CLASS
from utils.graph import load_class, load_parent


def _instance_paths(data_folder: str, species: str, sample_name: str):
    base = os.path.join(data_folder, species, "instances", sample_name)
    info = os.path.join(base, "info")
    graph_pkl = os.path.join(base, "graph.pkl")
    return base, info, graph_pkl


def _load_graph_base(data_folder: str, species: str, classes: edict, parents: edict) -> PlantGraphFixedTopology:
    pca_models = {
        "stem": NodePCA(path=os.path.join(data_folder, species, "3d_stem_pca.pth")),
        "leaf_deform": NodePCA(path=os.path.join(data_folder, species, "3d_leaf_pca.pth")),
        "leaf_shape": NodePCA(path=os.path.join(data_folder, species, "2d_leaf_pca.pth")),
    }
    g = PlantGraphFixedTopology(
        classes=classes,
        parents=parents,
        species=species,
        pca_leaf_3d=pca_models["leaf_deform"],
        pca_stem_3d=pca_models["stem"],
        pca_leaf_2d=pca_models["leaf_shape"],
    )
    g.to(torch.device("cuda"))
    return g


def _sample_coefficients(pca_model: NodePCA, param: torch.nn.Parameter, rng, scale: float, strategy: str) -> torch.Tensor:
    """Sample PCA coefficients matching the storage shape of `param`.

    strategy: 'mean' samples around PCA mean; 'perturb' samples around current param.
    """
    target_shape = param.shape
    flat_param = param.detach().float().cpu().view(-1).numpy()
    num_coeff = flat_param.size

    coeff_mean = getattr(pca_model, "coeff_mean", None)
    coeff_std = getattr(pca_model, "coeff_std", None)
    if coeff_mean is None or coeff_std is None:
        raise ValueError("PCA model missing coeff statistics; cannot sample.")

    base = coeff_mean.detach().cpu().view(-1).numpy()
    std = coeff_std.detach().cpu().view(-1).numpy()
    if base.size != num_coeff or std.size != num_coeff:
        raise ValueError(f"PCA size mismatch: expected {num_coeff}, got {base.size}.")

    if strategy == "perturb":
        base = flat_param

    noise = rng.standard_normal(num_coeff) * scale
    sample = base + noise * std
    return torch.from_numpy(sample).float().to(param.device).view(*target_shape)


def _rebuild_with_topology_clone_params(
    g: PlantGraphFixedTopology,
    classes: edict,
    parents: edict,
    clone_map: Dict[str, str],
) -> PlantGraphFixedTopology:
    """
    Rebuild PlantGraphFixedTopology for new topology and clone parameters for new IDs.

    clone_map: mapping new_id -> src_id to clone parameter tensors from.
    """
    device = torch.device("cuda")
    # Snapshot state from existing graph
    sd_old = {k: v.detach().clone().to(device) for k, v in g.state_dict().items()}

    # For each new id, clone relevant keys from src
    for new_id, src_id in clone_map.items():
        # Common keys
        for k in ["scale", "M_quat", "length"]:
            src_key = f"{k}_{src_id}"
            dst_key = f"{k}_{new_id}"
            if src_key in sd_old:
                sd_old[dst_key] = sd_old[src_key].clone()

        # Class-specific
        cls = classes[new_id]
        if cls == STEM_CLASS:
            for k in ["cp_w_local", "thickness", "deform"]:
                src_key = f"{k}_{src_id}"
                dst_key = f"{k}_{new_id}"
                if src_key in sd_old:
                    sd_old[dst_key] = sd_old[src_key].clone()
        elif cls == LEAF_CLASS:
            for k in ["cp_w_local", "shape", "deform"]:
                src_key = f"{k}_{src_id}"
                dst_key = f"{k}_{new_id}"
                if src_key in sd_old:
                    sd_old[dst_key] = sd_old[src_key].clone()
        # flower/fruit not supported in minimal editor; fall back to common params only

    # Rebuild graph with extended topology using existing PCA models from g
    pca_leaf_2d = getattr(getattr(g, "_surface", None), "pca", None)
    if pca_leaf_2d is None:
        raise RuntimeError("Existing graph is missing 2D leaf PCA; cannot rebuild.")

    g_new = PlantGraphFixedTopology(
        classes=classes,
        parents=parents,
        species=g.species,
        pca_leaf_3d=g.pca_leaf_3d,
        pca_stem_3d=g.pca_stem_3d,
        pca_leaf_2d=pca_leaf_2d,
    )
    g_new.to(device)

    # Register missing keys and load
    model_dict = g_new.state_dict()
    for k, v in sd_old.items():
        if k not in model_dict:
            g_new.register_parameter(k, torch.nn.Parameter(v))
    g_new.load_state_dict(sd_old, strict=False)
    return g_new


def _randomize_new_node_pca(
    g: PlantGraphFixedTopology,
    classes: edict,
    new_id: str,
    rng,
    stem_scale: float,
    leaf_shape_scale: float,
    leaf_deform_scale: float,
    strategy: str,
):
    """Randomize PCA params for the new node, keeping articulation cloned."""
    with torch.no_grad():
        if classes[new_id] == STEM_CLASS:
            try:
                p = getattr(g, f"deform_{new_id}")
                new_val = _sample_coefficients(g.pca_stem_3d, p, rng, stem_scale, strategy)
                p.copy_(new_val)
            except Exception as e:
                print(f"Warn: stem PCA sampling failed for {new_id}: {e}")
        elif classes[new_id] == LEAF_CLASS:
            try:
                p_def = getattr(g, f"deform_{new_id}")
                new_def = _sample_coefficients(g.pca_leaf_3d, p_def, rng, leaf_deform_scale, strategy)
                p_def.copy_(new_def)
            except Exception as e:
                print(f"Warn: leaf deform sampling failed for {new_id}: {e}")
            try:
                p_shape = getattr(g, f"shape_{new_id}")
                # 2D leaf PCA stored on surface
                pca_leaf_2d = getattr(getattr(g, "_surface", None), "pca", None)
                if pca_leaf_2d is None:
                    raise ValueError("leaf 2D PCA missing on graph")
                new_shape = _sample_coefficients(pca_leaf_2d, p_shape, rng, leaf_shape_scale, strategy)
                p_shape.copy_(new_shape)
            except Exception as e:
                print(f"Warn: leaf shape sampling failed for {new_id}: {e}")


def _visualize(g: PlantGraphFixedTopology):
    # Graph topology (matplotlib / networkx)
    try:
        g.draw_topology()
    except Exception as e:
        print(f"Warning: draw_topology failed: {e}")

    # Mesh visualization (Open3D)
    with torch.no_grad():
        mesh = g.generate(output_format="mesh", color="gray", align_global=True)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    o3d.visualization.draw_geometries([mesh, axis], mesh_show_back_face=True)


def main():
    parser = argparse.ArgumentParser(description="Minimal plant graph editor")
    parser.add_argument("--data_folder", type=str, default="sample_params")
    parser.add_argument("--species", type=str, default="soybean")
    parser.add_argument("--sample_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--strategy", type=str, default="mean", choices=["mean", "perturb"], help="Randomization strategy for PCA on add.")
    parser.add_argument("--stem_scale", type=float, default=1.0, help="Noise scale for stem PCA on add.")
    parser.add_argument("--leaf_deform_scale", type=float, default=1.0, help="Noise scale for leaf 3D PCA on add.")
    parser.add_argument("--leaf_shape_scale", type=float, default=1.0, help="Noise scale for leaf 2D PCA on add.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemError("CUDA device is required for graph editing (generate() uses .cuda()).")

    base, info, graph_pkl = _instance_paths(args.data_folder, args.species, args.sample_name)
    if not os.path.isfile(graph_pkl):
        raise FileNotFoundError(f"graph.pkl not found under: {base}")

    classes = load_class(os.path.join(info, "class.txt"))
    parents = load_parent(os.path.join(info, "parent.txt"))

    # Init and load parameters
    g = _load_graph_base(args.data_folder, args.species, classes, parents)
    g.load(graph_pkl)

    print("Graph Editor — commands: list, add, generate, graph, help, quit")
    print("add syntax: add <class> <parent_id> [src_id] — PCA for new node is randomized")

    import numpy as np
    rng = np.random.default_rng(args.seed)

    # Interactive loop
    while True:
        try:
            cmd = input("editor> ").strip()
        except EOFError:
            break
        if not cmd:
            continue

        if cmd in ("quit", "q", "exit"):
            break

        if cmd in ("help", "h", "?"):
            print("Commands:\n- list: show nodes (id, class, parent)\n- add <class> <parent_id> [src_id]: add node cloning params from src_id (default: first of same class)\n- generate: visualize mesh + topology\n- graph: draw topology only\n- quit: exit")
            continue

        if cmd == "list":
            print("Nodes (id: class -> parent):")
            # sort by id numeric
            for k in sorted(classes.keys(), key=lambda x: int(x)):
                c = classes[k]
                cname = "stem" if c == STEM_CLASS else ("leaf" if c == LEAF_CLASS else ("flower" if c == FLOWER_CLASS else "fruit"))
                print(f"  {k}: {cname} -> {parents[k]}")
            continue

        if cmd.startswith("add"):
            parts = cmd.split()
            if len(parts) < 3:
                print("Usage: add <class> <parent_id> [src_id]")
                continue
            class_name = parts[1].lower()
            parent_id = parts[2]
            src_id = parts[3] if len(parts) >= 4 else None

            if class_name not in ("stem", "leaf"):
                print("Only 'stem' and 'leaf' supported in minimal editor.")
                continue
            if parent_id not in parents:
                # allow adding under root -1? Only if -1 given
                if parent_id != "-1":
                    print(f"Parent id {parent_id} does not exist. Use 'list' to inspect.")
                    continue

            # resolve class id
            class_id = STEM_CLASS if class_name == "stem" else LEAF_CLASS

            # pick clone source if not provided
            if src_id is None:
                # pick first node of same class
                same_cls = [k for k, v in classes.items() if v == class_id]
                if not same_cls:
                    print(f"No existing node of class {class_name} to clone from. Specify src_id.")
                    continue
                src_id = sorted(same_cls, key=lambda x: int(x))[0]
            else:
                if src_id not in classes:
                    print(f"Source id {src_id} does not exist.")
                    continue
                if classes[src_id] != class_id:
                    print(f"Source id {src_id} class mismatch; must be {class_name}.")
                    continue

            # allocate new id
            new_id_int = max(int(k) for k in classes.keys()) + 1
            new_id = str(new_id_int)

            # update topology dicts
            classes[new_id] = int(class_id)
            parents[new_id] = int(parent_id)

            # rebuild graph with clone mapping (clone articulation), then randomize PCA for the new node
            clone_map = {new_id: src_id}
            g = _rebuild_with_topology_clone_params(g, classes, parents, clone_map)
            _randomize_new_node_pca(
                g,
                classes,
                new_id,
                rng,
                stem_scale=args.stem_scale,
                leaf_shape_scale=args.leaf_shape_scale,
                leaf_deform_scale=args.leaf_deform_scale,
                strategy=args.strategy,
            )
            print(f"Added node {new_id} ({class_name}) with parent {parent_id}, cloned from {src_id}; PCA randomized.")
            continue

        if cmd == "generate":
            _visualize(g)
            continue

        if cmd == "graph":
            try:
                g.draw_topology()
            except Exception as e:
                print(f"draw_topology failed: {e}")
            continue

        print("Unknown command. Type 'help' for options.")


if __name__ == "__main__":
    main()

