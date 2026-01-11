#!/usr/bin/env python3
"""
Minimal Open3D-based live editor for Demeter graphs.

Features (keyboard):
- H: print help
- [: previous node   ]: next node
- +: increase scale   -: decrease scale
- L: increase length  K: decrease length
- D: duplicate current node (attach to same parent)
- C: toggle color mode (gray/instance)
- S: save to new instance folder (<sample_name>_live)
- Q: quit

Notes:
- Requires CUDA (current PlantGraphFixedTopology uses .cuda() internally).
- Operates entirely in-memory until you press S to save.
"""

import argparse
import os
import shutil
from typing import Dict, List

import numpy as np
import open3d as o3d
import torch
from easydict import EasyDict as edict

from representation.graph import PlantGraphFixedTopology
from utils.graph import load_class, load_parent, save_class, save_parent
from utils.pca import NodePCA
from utils.constant import LEAF_CLASS, STEM_CLASS, FLOWER_CLASS, FRUIT_CLASS


def _cycle_node_id(order: List[str], current_id: str, step: int) -> str:
    """Return the next valid node id while guarding against empty/missing lists."""
    if not order:
        return current_id
    if current_id not in order:
        return order[0]
    idx = (order.index(current_id) + step) % len(order)
    return order[idx]


def _instance_paths(data_folder: str, species: str, sample_name: str):
    base = os.path.join(data_folder, species, "instances", sample_name)
    info = os.path.join(base, "info")
    graph_pkl = os.path.join(base, "graph.pkl")
    return base, info, graph_pkl


def _load_graph(data_folder: str, species: str, classes: edict, parents: edict) -> PlantGraphFixedTopology:
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
    device = torch.device("cuda")
    g.to(device)
    return g


def _node_order_for_edit_id(g: PlantGraphFixedTopology) -> List[str]:
    # Mirror the order used inside generate(): sorted layer keys, iterate nodes, skip flower/fruit
    order = []
    layers = {k: g.layers[k] for k in sorted(g.layers.keys())}
    for _, layer in layers.items():
        for child in layer:
            cid = str(child)
            if g.classes[cid] in (FLOWER_CLASS, FRUIT_CLASS):
                continue
            order.append(cid)
    return order


def _update_camera_focus(vis: o3d.visualization.Visualizer, state: dict, bbox: o3d.geometry.AxisAlignedBoundingBox):
    if bbox is None:
        return
    vc = vis.get_view_control()
    # Cache reference zoom/scene size the first time we focus.
    if "base_zoom" not in state:
         state["base_zoom"] = 1
    diag = float(np.linalg.norm(bbox.get_extent()))
    if diag <= 1e-6:
        diag = 1e-6
    if "scene_diag" not in state:
        state["scene_diag"] = diag
    vc.set_lookat(bbox.get_center().tolist())
    zoom_scale = np.clip(state.get("scene_diag", diag) / diag, 0.25, 4.0)
    target_zoom = float(state.get("base_zoom", 1) * zoom_scale)
    vc.set_zoom(float(np.clip(target_zoom, 0.02, 2.5)))


def _regenerate_mesh(vis: o3d.visualization.Visualizer, state: dict, highlight_id: str):
    g: PlantGraphFixedTopology = state["graph"]
    color = state["color"]
    node_order = state["node_order"]
    # Map node id to edit_id index
    edit_id = node_order.index(highlight_id) if highlight_id in node_order else -1
    with torch.no_grad():
        mesh_dict, _, _, _ = g.generate(
            output_format="instance_mesh_full",
            color=color,
            align_global=True,
            edit_id=edit_id,
        )
    mesh = None
    for geom in mesh_dict.values():
        mesh = geom if mesh is None else mesh + geom
    if mesh is None:
        return
    if state.get("mesh") is not None:
        try:
            vis.remove_geometry(state["mesh"], reset_bounding_box=False)
        except Exception:
            pass
    state["mesh"] = mesh
    # On first add, reset bounding box so the camera near/far planes fit the mesh
    vis.add_geometry(mesh, reset_bounding_box=is_first_mesh)
    vis.update_geometry(mesh)

    if mesh is not None and "scene_diag" not in state:
        bbox = mesh.get_axis_aligned_bounding_box()
        state["scene_diag"] = float(max(np.linalg.norm(bbox.get_extent()), 1e-6))

    focus_bbox = None
    if highlight_id in mesh_dict:
        focus_bbox = mesh_dict[highlight_id].get_axis_aligned_bounding_box()
    elif mesh is not None:
        focus_bbox = mesh.get_axis_aligned_bounding_box()
    if highlight_id != state.get("last_focus_id"):
        _update_camera_focus(vis, state, focus_bbox)
        state["last_focus_id"] = highlight_id

    vis.poll_events()
    vis.update_renderer()


def _duplicate_current_node(state: dict, current_id: str):
    g: PlantGraphFixedTopology = state["graph"]
    data_folder: str = state["data_folder"]
    species: str = state["species"]
    classes = edict(dict(g.classes))
    parents = edict(dict(g.parents))

    # New id is max(existing)+1
    new_id_int = max(int(k) for k in classes.keys()) + 1
    new_id = str(new_id_int)
    classes[new_id] = int(classes[current_id])
    parents[new_id] = int(parents[current_id])

    # Clone parameters from state_dict
    sd = {k: v.detach().clone() for k, v in g.state_dict().items()}

    def _maybe(src_key: str, dst_key: str):
        if src_key in sd:
            sd[dst_key] = sd[src_key].clone()

    for k in ["scale", "M_quat", "length"]:
        _maybe(f"{k}_{current_id}", f"{k}_{new_id}")

    cls = classes[new_id]
    if cls == STEM_CLASS:
        for k in ["cp_w_local", "thickness", "deform"]:
            _maybe(f"{k}_{current_id}", f"{k}_{new_id}")
    elif cls == LEAF_CLASS:
        for k in ["cp_w_local", "shape", "deform"]:
            _maybe(f"{k}_{current_id}", f"{k}_{new_id}")
    else:
        # flower/fruit: keep only common params
        pass

    # Rebuild graph with extended topology
    g_new = _load_graph(data_folder, species, classes, parents)
    model_dict = g_new.state_dict()
    for k, v in sd.items():
        if k not in model_dict:
            g_new.register_parameter(k, torch.nn.Parameter(v))
    g_new.load_state_dict(sd, strict=False)
    state["graph"] = g_new
    state["classes"] = classes
    state["parents"] = parents
    state["node_order"] = _node_order_for_edit_id(g_new)
    return new_id


def _save_instance(state: dict, sample_name: str, out_suffix: str = "live"):
    data_folder: str = state["data_folder"]
    species: str = state["species"]
    g: PlantGraphFixedTopology = state["graph"]
    classes = state["classes"]
    parents = state["parents"]

    src_base, _, _ = _instance_paths(data_folder, species, sample_name)
    out_sample = f"{sample_name}_{out_suffix}"
    out_base, out_info, out_graph = _instance_paths(data_folder, species, out_sample)
    if os.path.isdir(out_base):
        shutil.rmtree(out_base)
    shutil.copytree(src_base, out_base)
    os.makedirs(out_info, exist_ok=True)
    save_class(classes, os.path.join(out_info, "class.txt"))
    save_parent(parents, os.path.join(out_info, "parent.txt"))
    g.save(out_graph)
    print(f"Saved to: {out_base}")


def main():
    parser = argparse.ArgumentParser(description="Open3D live editor for Demeter graphs")
    parser.add_argument("--data_folder", type=str, default="sample_params")
    parser.add_argument("--species", type=str, default="soybean")
    parser.add_argument("--sample_name", type=str, required=True)
    parser.add_argument("--color", type=str, default="gray", help="mesh color mode: gray|instance|blue")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemError("CUDA device is required for interactive editor (graph.generate() uses .cuda()).")

    base, info, graph_pkl = _instance_paths(args.data_folder, args.species, args.sample_name)
    if not os.path.isfile(graph_pkl):
        raise FileNotFoundError(f"graph.pkl not found under: {base}")

    classes = load_class(os.path.join(info, "class.txt"))
    parents = load_parent(os.path.join(info, "parent.txt"))

    g = _load_graph(args.data_folder, args.species, classes, parents)
    g.load(graph_pkl)

    state = {
        "data_folder": args.data_folder,
        "species": args.species,
        "sample_name": args.sample_name,
        "graph": g,
        "classes": edict(dict(classes)),
        "parents": edict(dict(parents)),
        "color": args.color,
        "mesh": None,
        "last_focus_id": None,
    }
    state["node_order"] = _node_order_for_edit_id(g)
    current_id = state["node_order"][0] if state["node_order"] else list(classes.keys())[0]

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Demeter Live Editor", width=1280, height=960)
    # Improve visibility for thin meshes/leaves and avoid back-face culling surprises
    try:
        opt = vis.get_render_option()
        opt.mesh_show_back_face = True
    except Exception:
        pass
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    vis.add_geometry(axis)
    _regenerate_mesh(vis, state, current_id)

    # Try to push camera clipping planes further out to avoid distance clipping
    try:
        ctr = vis.get_view_control()
        if hasattr(ctr, "set_constant_z_far"):
            ctr.set_constant_z_far(1e6)
        if hasattr(ctr, "set_constant_z_near"):
            ctr.set_constant_z_near(1e-4)
    except Exception:
        pass

    def print_help():
        print("\nKeys: [ prev, ] next, +/- scale, K/L length -, +, D duplicate, C color, S save, H help, Q quit\n")
        print(f"Current node: {current_id}\n")

    # Key callbacks
    def cb_prev(vis):
        nonlocal current_id
        order = state["node_order"]
        if not order:
            return False
        current_id = _cycle_node_id(order, current_id, -1)
        print(f"Selected node: {current_id}")
        _regenerate_mesh(vis, state, current_id)
        return False

    def cb_next(vis):
        nonlocal current_id
        order = state["node_order"]
        if not order:
            return False
        current_id = _cycle_node_id(order, current_id, 1)
        print(f"Selected node: {current_id}")
        _regenerate_mesh(vis, state, current_id)
        return False

    def cb_scale_up(vis):
        g = state["graph"]
        key = f"scale_{current_id}"
        if hasattr(g, key):
            with torch.no_grad():
                p = getattr(g, key)
                p.copy_(p * 1.05)
            _regenerate_mesh(vis, state, current_id)
        return False

    def cb_scale_down(vis):
        g = state["graph"]
        key = f"scale_{current_id}"
        if hasattr(g, key):
            with torch.no_grad():
                p = getattr(g, key)
                p.copy_(p / 1.05)
            _regenerate_mesh(vis, state, current_id)
        return False

    def cb_len_inc(vis):
        g = state["graph"]
        key = f"length_{current_id}"
        if hasattr(g, key):
            with torch.no_grad():
                p = getattr(g, key)
                p.copy_(torch.clamp(p + 0.01, 0.0, 1.0))
            _regenerate_mesh(vis, state, current_id)
        return False

    def cb_len_dec(vis):
        g = state["graph"]
        key = f"length_{current_id}"
        if hasattr(g, key):
            with torch.no_grad():
                p = getattr(g, key)
                p.copy_(torch.clamp(p - 0.01, 0.0, 1.0))
            _regenerate_mesh(vis, state, current_id)
        return False

    def cb_duplicate(vis):
        nonlocal current_id
        new_id = _duplicate_current_node(state, current_id)
        current_id = new_id
        print(f"Duplicated node. New id: {new_id}")
        _regenerate_mesh(vis, state, current_id)
        return False

    def cb_color(vis):
        cur = state["color"]
        state["color"] = "instance" if cur == "gray" else ("gray" if cur == "instance" else "gray")
        _regenerate_mesh(vis, state, current_id)
        return False

    def cb_save(vis):
        _save_instance(state, args.sample_name, out_suffix="live")
        return False

    def cb_help(vis):
        print_help()
        return False

    def cb_quit(vis):
        vis.close()
        return True

    # Register callbacks
    vis.register_key_callback(ord('['), cb_prev)
    vis.register_key_callback(ord(']'), cb_next)
    vis.register_key_callback(ord('+'), cb_scale_up)
    vis.register_key_callback(ord('='), cb_scale_up)
    vis.register_key_callback(ord('-'), cb_scale_down)
    vis.register_key_callback(ord('L'), cb_len_inc)
    vis.register_key_callback(ord('K'), cb_len_dec)
    vis.register_key_callback(ord('D'), cb_duplicate)
    vis.register_key_callback(ord('C'), cb_color)
    vis.register_key_callback(ord('S'), cb_save)
    vis.register_key_callback(ord('H'), cb_help)
    vis.register_key_callback(ord('Q'), cb_quit)

    print("Demeter Live Editor — press H for help.")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
