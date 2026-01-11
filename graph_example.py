#!/usr/bin/env python3
"""
Graph topology exploration utility for Demeter graphs.

What this script demonstrates:
- How topology is represented: parents/classes live in text files; per-node parameters live in graph.pkl.
- How to inspect an existing instance's topology and parameters.
- How to edit parameters (e.g., scale a node) and save to a new instance folder.
- How to duplicate an existing node into a new node id and attach it in the topology.

Notes:
- This script avoids mesh generation so it runs without a GPU.
- Use decode.py afterwards to visualize meshes from the saved output.
"""

import argparse
import os
import shutil
from typing import Tuple

import torch
from easydict import EasyDict as edict

from representation.graph import PlantGraphFixedTopology
from utils.constant import FLOWER_CLASS, FRUIT_CLASS, LEAF_CLASS, STEM_CLASS
from utils.graph import (find_layers_and_paths, load_class, load_parent,
                         save_class, save_parent)
from utils.pca import NodePCA


def _instance_paths(data_folder: str, species: str, sample_name: str) -> Tuple[str, str, str]:
    base = os.path.join(data_folder, species, "instances", sample_name)
    info = os.path.join(base, "info")
    graph_pkl = os.path.join(base, "graph.pkl")
    return base, info, graph_pkl


def inspect_instance(data_folder: str, species: str, sample_name: str) -> None:
    base, info, graph_pkl = _instance_paths(data_folder, species, sample_name)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Instance folder not found: {base}")
    classes = load_class(os.path.join(info, "class.txt"))
    parents = load_parent(os.path.join(info, "parent.txt"))

    print("== Topology (parents/classes) ==")
    print(f"Num nodes: {len(classes)}")
    root = [k for k, v in parents.items() if v == -1]
    print(f"Root node(s): {root}")
    layers, paths = find_layers_and_paths(parents)
    print(f"Layers: { {i: layers[i] for i in sorted(layers)} }")
    print("Classes (id: class_id):")
    print({k: int(v) for k, v in classes.items()})

    if not os.path.isfile(graph_pkl):
        print(f"graph.pkl not found at {graph_pkl}.")
        return

    print("\n== graph.pkl (state_dict keys preview) ==")
    sd = torch.load(graph_pkl, map_location="cpu", weights_only=True)
    keys = sorted(sd.keys())
    print(f"Total parameters: {len(keys)}")
    for k in keys[:20]:
        print(f"  {k}: {tuple(sd[k].shape)}")
    if len(keys) > 20:
        print("  ...")


def _load_graph_with_pcas(data_folder: str, species: str, classes: edict, parents: edict) -> PlantGraphFixedTopology:
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
    return g


def _prepare_output_instance(src_base: str, out_base: str) -> None:
    # Copy the instance folder structure except graph.pkl (we overwrite)
    if os.path.isdir(out_base):
        shutil.rmtree(out_base)
    shutil.copytree(src_base, out_base)


def scale_node(
    data_folder: str,
    species: str,
    sample_name: str,
    node_id: str,
    scale_multiplier: float,
    out_sample_suffix: str = "scaled",
) -> str:
    """Multiply a node's scale parameter and save to a new instance folder."""
    base, info, graph_pkl = _instance_paths(data_folder, species, sample_name)
    classes = load_class(os.path.join(info, "class.txt"))
    parents = load_parent(os.path.join(info, "parent.txt"))

    if node_id not in classes:
        raise ValueError(f"Node id {node_id} not found in classes.")

    g = _load_graph_with_pcas(data_folder, species, classes, parents)
    g.load(graph_pkl)

    key = f"scale_{node_id}"
    if not hasattr(g, key):
        raise ValueError(f"Parameter {key} not found in graph. Available params include: "
                         f"{[k for k in g.state_dict().keys() if k.startswith('scale_')]}")

    with torch.no_grad():
        p = getattr(g, key)
        p.copy_(p * float(scale_multiplier))

    out_sample = f"{sample_name}_{out_sample_suffix}"
    out_base, out_info, out_graph = _instance_paths(data_folder, species, out_sample)
    _prepare_output_instance(base, out_base)
    # Just in case, persist the same parents/classes
    save_parent(parents, os.path.join(out_info, "parent.txt"))
    save_class(classes, os.path.join(out_info, "class.txt"))
    g.save(out_graph)
    print(f"Saved scaled graph to: {out_base}")
    return out_sample


def duplicate_node(
    data_folder: str,
    species: str,
    sample_name: str,
    src_node_id: str,
    new_node_id: str,
    new_parent_id: str = None,
    out_sample_suffix: str = "duplicated",
) -> str:
    """Duplicate an existing node's parameters into a new node id and attach it in topology.

    The new node will inherit class and parameters; you can change parent via --new_parent_id.
    """
    base, info, graph_pkl = _instance_paths(data_folder, species, sample_name)
    classes = load_class(os.path.join(info, "class.txt"))
    parents = load_parent(os.path.join(info, "parent.txt"))

    if src_node_id not in classes:
        raise ValueError(f"Source node id {src_node_id} not found.")
    if new_node_id in classes:
        raise ValueError(f"New node id {new_node_id} already exists in this instance.")

    # Load the state dict from disk (plain) to duplicate keys easily
    sd = torch.load(graph_pkl, map_location="cpu", weights_only=True)

    # Update topology edicts
    classes = edict({**classes, str(new_node_id): int(classes[src_node_id])})
    parent_to_set = parents[src_node_id] if new_parent_id is None else int(new_parent_id)
    parents = edict({**parents, str(new_node_id): parent_to_set})

    # Duplicate parameter tensors with renamed keys
    def _maybe(src_key: str, dst_key: str):
        if src_key in sd:
            sd[dst_key] = sd[src_key].clone()

    # Common params
    for stem in ["scale", "M_quat", "length"]:
        _maybe(f"{stem}_{src_node_id}", f"{stem}_{new_node_id}")

    # Class-specific params
    if classes[new_node_id] == STEM_CLASS:
        for k in ["cp_w_local", "thickness", "deform"]:
            _maybe(f"{k}_{src_node_id}", f"{k}_{new_node_id}")
    elif classes[new_node_id] == LEAF_CLASS:
        for k in ["cp_w_local", "shape", "deform"]:
            _maybe(f"{k}_{src_node_id}", f"{k}_{new_node_id}")
    elif classes[new_node_id] in (FLOWER_CLASS, FRUIT_CLASS):
        # No class-specific geometric params in the baseline; keep only common ones
        pass

    # Instantiate a graph with updated topology and load the extended state dict
    g = _load_graph_with_pcas(data_folder, species, classes, parents)
    # Allow registering new keys on the fly as PlantGraphFixedTopology.load() does
    # but here we pass the sd directly to load_state_dict
    model_dict = g.state_dict()
    for k, v in sd.items():
        if k not in model_dict:
            g.register_parameter(k, torch.nn.Parameter(v))
    g.load_state_dict(sd, strict=False)

    # Save as a new instance folder
    out_sample = f"{sample_name}_{out_sample_suffix}"
    out_base, out_info, out_graph = _instance_paths(data_folder, species, out_sample)
    _prepare_output_instance(base, out_base)
    save_parent(parents, os.path.join(out_info, "parent.txt"))
    save_class(classes, os.path.join(out_info, "class.txt"))
    g.save(out_graph)
    print(f"Saved duplicated-node graph to: {out_base}")
    return out_sample


def main():
    parser = argparse.ArgumentParser(description="Explore and edit Demeter graph topologies (graph.pkl + info)")
    parser.add_argument("--data_folder", type=str, default="sample_params")
    parser.add_argument("--species", type=str, default="soybean")
    parser.add_argument("--sample_name", type=str, required=True, help="Existing instance name under <data_folder>/<species>/instances/")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("inspect", help="Print topology, layers, and preview graph.pkl parameters")

    p_scale = sub.add_parser("scale_node", help="Multiply a node's scale parameter and save to a new instance")
    p_scale.add_argument("--node_id", type=str, required=True)
    p_scale.add_argument("--mult", type=float, default=1.2)
    p_scale.add_argument("--suffix", type=str, default="scaled")

    p_dup = sub.add_parser("duplicate_node", help="Duplicate a node into a new id and attach it in the topology")
    p_dup.add_argument("--src_node_id", type=str, required=True)
    p_dup.add_argument("--new_node_id", type=str, required=True)
    p_dup.add_argument("--new_parent_id", type=str, default=None, help="Optional parent override for the new node")
    p_dup.add_argument("--suffix", type=str, default="duplicated")

    args = parser.parse_args()

    if args.cmd == "inspect":
        inspect_instance(args.data_folder, args.species, args.sample_name)
    elif args.cmd == "scale_node":
        scale_node(args.data_folder, args.species, args.sample_name, args.node_id, args.mult, args.suffix)
    elif args.cmd == "duplicate_node":
        duplicate_node(args.data_folder, args.species, args.sample_name, args.src_node_id, args.new_node_id, args.new_parent_id, args.suffix)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

