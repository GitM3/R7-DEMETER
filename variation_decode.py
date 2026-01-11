import argparse
import random
import os
import glob
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import open3d as o3d
import torch

from representation.graph import PlantGraphFixedTopology
from utils.constant import FLOWER_CLASS, FRUIT_CLASS, LEAF_CLASS, STEM_CLASS
from utils.graph import load_class, load_parent
from utils.pca import NodePCA
from utils.texturing import attach_uv_to_grid_mesh

NODE_CLASS = 99

CLASS_COLOR_MAP = {
    STEM_CLASS: (0.55, 0.33, 0.1),
    LEAF_CLASS: (0.12, 0.6, 0.2),
    FLOWER_CLASS: (0.9, 0.4, 0.7),
    FRUIT_CLASS: (0.3, 0.6, 0.2),
    NODE_CLASS: (0.7, 0.7, 0.7),
}

CLASS_NAME_MAP = {
    STEM_CLASS: "stem",
    LEAF_CLASS: "leaf",
    FLOWER_CLASS: "flower",
    FRUIT_CLASS: "fruit",
    NODE_CLASS: "node",
}


def _load_graph(
    data_folder: str, species: str, sample_name: str
) -> Tuple[PlantGraphFixedTopology, Dict[str, NodePCA], Dict[str, int]]:
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

    return plant_graph, pca_models, classes


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


def save_mesh(mesh: o3d.geometry.TriangleMesh, path: str, write_uvs: bool = False) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    print(f"Writing mesh to {path}")
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(
        path,
        mesh,
        write_ascii=False,
        compressed=False,
        write_triangle_uvs=write_uvs,
    )


def _merge_with_class_colors(
    meshes: Dict[str, o3d.geometry.TriangleMesh], classes: Dict[str, int]
) -> o3d.geometry.TriangleMesh:
    merged = None
    for node_id, mesh in meshes.items():
        class_idx = classes.get(str(node_id))
        color = CLASS_COLOR_MAP.get(class_idx, (0.7, 0.7, 0.7))
        mesh.paint_uniform_color(color)
        merged = mesh if merged is None else merged + mesh
    return merged


def _export_instance_meshes(
    meshes: Dict[str, o3d.geometry.TriangleMesh],
    classes: Dict[str, int],
    destination: str,
    write_uvs: bool = False,
    file_ext: str = "ply",
) -> None:
    os.makedirs(destination, exist_ok=True)
    for node_id, mesh in meshes.items():
        class_idx = classes.get(str(node_id))
        class_name = CLASS_NAME_MAP.get(class_idx, f"class_{class_idx}")
        class_dir = os.path.join(destination, class_name)
        os.makedirs(class_dir, exist_ok=True)
        filename = f"{node_id}_{class_name}.{file_ext}"
        save_mesh(mesh, os.path.join(class_dir, filename), write_uvs=write_uvs)

def _infer_fruit_texture(obj_path: str) -> str:
    base, _ = os.path.splitext(obj_path)
    mtl_path = f"{base}.mtl"
    if os.path.isfile(mtl_path):
        with open(mtl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("map_kd"):
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        texture_path = parts[1].strip().strip('"')
                        if not os.path.isabs(texture_path):
                            texture_path = os.path.join(os.path.dirname(mtl_path), texture_path)
                        if os.path.isfile(texture_path):
                            return texture_path

    obj_dir = os.path.dirname(obj_path)
    for name in sorted(os.listdir(obj_dir)):
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            return os.path.join(obj_dir, name)
    return ""

def _parse_obj_mesh(path: str) -> o3d.geometry.TriangleMesh:
    vertices = []
    uvs = []
    triangles = []
    triangle_uvs = []
    have_uvs = True

    def _fix_index(idx: int, size: int) -> int:
        return idx - 1 if idx > 0 else size + idx

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vt "):
                parts = line.split()
                if len(parts) >= 3:
                    uvs.append([float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) < 3:
                    continue
                face = []
                face_uv = []
                for token in parts:
                    vals = token.split("/")
                    if not vals[0]:
                        continue
                    vi = _fix_index(int(vals[0]), len(vertices))
                    face.append(vi)
                    if len(vals) > 1 and vals[1]:
                        vti = _fix_index(int(vals[1]), len(uvs))
                        face_uv.append(vti)
                    else:
                        have_uvs = False
                for i in range(1, len(face) - 1):
                    tri = [face[0], face[i], face[i + 1]]
                    triangles.append(tri)
                    if have_uvs and len(face_uv) == len(face):
                        tri_uv = [face_uv[0], face_uv[i], face_uv[i + 1]]
                        for uv_idx in tri_uv:
                            triangle_uvs.append(uvs[uv_idx])
                    else:
                        have_uvs = False

    mesh = o3d.geometry.TriangleMesh()
    if vertices and triangles:
        mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices, dtype=np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))
        if have_uvs and len(triangle_uvs) == len(triangles) * 3:
            mesh.triangle_uvs = o3d.utility.Vector2dVector(
                np.array(triangle_uvs, dtype=np.float64)
            )
        mesh.compute_vertex_normals()
    return mesh


def _load_fruit_mesh(path: str) -> Tuple[o3d.geometry.TriangleMesh, str]:
    if not path:
        raise ValueError("Fruit mesh path is empty.")
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Fruit mesh not found: {path}")
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    if mesh.is_empty():
        mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        mesh = _parse_obj_mesh(path)
    if mesh.is_empty():
        raise ValueError(
            f"Fruit mesh is empty: {path}. "
            "If the OBJ contains quads/ngons, enable_post_processing should triangulate."
        )
    mesh.compute_vertex_normals()
    texture_path = _infer_fruit_texture(path)
    if texture_path:
        image = o3d.io.read_image(texture_path)
        mesh.textures = [image]
    return mesh, texture_path

def _add_random_fruit_meshes(
    meshes: Dict[str, o3d.geometry.TriangleMesh],
    classes: Dict[str, int],
    offsets: Dict[str, torch.Tensor],
    plant_graph: PlantGraphFixedTopology,
    rng: np.random.Generator,
    fruit_per_plant: int,
    fruit_mesh: o3d.geometry.TriangleMesh,
    scale_k: float,
    rotate_x_deg: float,
) -> Tuple[Dict[str, o3d.geometry.TriangleMesh], Dict[str, int]]:
    classes_with_fruit = dict(classes)
    if fruit_per_plant <= 0:
        return meshes, classes_with_fruit

    main_stem = str(plant_graph.main_stem)
    candidates = []
    for node_id in plant_graph.stem_key:
        parent_id = str(plant_graph.parents[str(node_id)])
        grandparent_id = str(plant_graph.parents.get(parent_id, ""))
        if grandparent_id == main_stem:
            candidates.append(node_id)
    if not candidates:
        candidates = [
            node_id for node_id in plant_graph.stem_key if node_id != main_stem
        ]
    if not candidates:
        return meshes, classes_with_fruit

    replace = fruit_per_plant > len(candidates)
    chosen = rng.choice(candidates, size=fruit_per_plant, replace=replace)

    base_mesh = fruit_mesh
    aabb = base_mesh.get_axis_aligned_bounding_box()
    extent = float(np.max(aabb.get_extent()))
    if extent <= 0:
        return meshes, classes_with_fruit

    down_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    rot_x_rad = np.deg2rad(rotate_x_deg)
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rot_x_rad), -np.sin(rot_x_rad)],
            [0.0, np.sin(rot_x_rad), np.cos(rot_x_rad)],
        ]
    )
    total_rotation = rot_x @ down_rotation

    for idx, node_id in enumerate(chosen):
        offset = offsets.get(str(node_id))
        if offset is None:
            continue
        thickness = getattr(plant_graph, f"thickness_{node_id}", None)
        if thickness is None:
            continue
        parent_id = str(plant_graph.parents[str(node_id)])
        parent_thickness = getattr(plant_graph, f"thickness_{parent_id}", None)
        base_radius = thickness
        if parent_thickness is not None:
            base_radius = torch.maximum(base_radius, parent_thickness)
        radius = float(base_radius.detach().cpu().item()) * scale_k
        if radius <= 0.0:
            continue

        fruit = o3d.geometry.TriangleMesh(base_mesh)
        scale = (2.0 * radius) / extent
        fruit.scale(scale, center=(0, 0, 0))
        fruit.rotate(total_rotation, center=(0, 0, 0))
        fruit.translate(offset.detach().cpu().numpy())
        fruit.compute_vertex_normals()

        fruit_key = f"fruit_{node_id}_{idx}"
        meshes[fruit_key] = fruit
        classes_with_fruit[str(fruit_key)] = FRUIT_CLASS

    return meshes, classes_with_fruit

def _add_junction_node_meshes(
    meshes: Dict[str, o3d.geometry.TriangleMesh],
    classes: Dict[str, int],
    offsets: Dict[str, torch.Tensor],
    plant_graph: PlantGraphFixedTopology,
    radius_delta: float,
) -> Tuple[Dict[str, o3d.geometry.TriangleMesh], Dict[str, int]]:
    classes_with_nodes = dict(classes)
    parent_map = plant_graph.parents
    main_stem = plant_graph.main_stem

    for node_id in plant_graph.stem_key:
        if node_id == main_stem:
            continue
        offset = offsets.get(node_id)
        if offset is None:
            continue
        thickness = getattr(plant_graph, f"thickness_{node_id}", None)
        if thickness is None:
            continue
        parent_id = str(parent_map[node_id])
        parent_thickness = getattr(plant_graph, f"thickness_{parent_id}", None)
        base_radius = thickness
        if parent_thickness is not None:
            base_radius = torch.maximum(base_radius, parent_thickness)
            radius = float(base_radius.detach().cpu().item())
            radius *= (1.0 + radius_delta / 100.0)
            if radius <= 0.0:
                continue
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
        sphere.translate(offset.detach().cpu().numpy())
        sphere.compute_vertex_normals()
        node_key = f"node_{node_id}"
        meshes[node_key] = sphere
        classes_with_nodes[str(node_key)] = NODE_CLASS

    return meshes, classes_with_nodes

def generate_variations(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    plant_graph, pca_models, classes = _load_graph(args.data_folder, args.species, args.sample_name)
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
            fruit_per_plant = int(getattr(args, "fruit_per_plant", 0))
            fruit_obj_path = getattr(args, "fruit_obj_path", None)
            fruit_scale_k = float(getattr(args, "fruit_scale_k", 1.0))
            fruit_rotate_x_deg = float(getattr(args, "fruit_rotate_x_deg", 0.0))
            want_fruit = fruit_per_plant > 0 and fruit_obj_path
            want_junction_nodes = args.add_junction_nodes
            want_offsets = want_junction_nodes or want_fruit
            if args.output_type == "mesh":
                if want_offsets:
                    meshes, _, _, offsets = plant_graph.generate(
                        output_format="instance_mesh_full",
                        color=args.color,
                        align_global=args.align_global,
                        junction_sphere_delta=args.junction_sphere_delta,
                    )
                    classes_for_output = dict(classes)
                    if want_junction_nodes:
                        meshes, classes_for_output = _add_junction_node_meshes(
                            meshes,
                            classes_for_output,
                            offsets,
                            plant_graph,
                            radius_delta=args.junction_sphere_delta,
                        )
                    if want_fruit:
                        fruit_mesh, _ = _load_fruit_mesh(fruit_obj_path)
                        meshes, classes_for_output = _add_random_fruit_meshes(
                            meshes,
                            classes_for_output,
                            offsets,
                            plant_graph,
                            rng=rng,
                            fruit_per_plant=fruit_per_plant,
                            fruit_mesh=fruit_mesh,
                            scale_k=fruit_scale_k,
                            rotate_x_deg=fruit_rotate_x_deg,
                        )
                    mesh = _merge_with_class_colors(meshes, classes_for_output)
                else:
                    mesh = plant_graph.generate(
                    output_format="mesh",
                    color=args.color,
                    align_global=args.align_global,
                    junction_sphere_delta=args.junction_sphere_delta,
                    )
                output_name = f"{args.output_prepend}_{args.sample_name}_variation_{idx:02d}.ply"
                save_mesh(mesh, os.path.join(args.output_dir, output_name))
                if args.visualize:
                    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
                    o3d.visualization.draw_geometries([mesh, axis], mesh_show_back_face=True)
            else:
                if want_offsets:
                    meshes, _, _, offsets = plant_graph.generate(
                        output_format="instance_mesh_full",
                        color=args.color,
                        align_global=args.align_global,
                        junction_sphere_delta=args.junction_sphere_delta,
                    )
                else:
                    meshes = plant_graph.generate(
                        output_format="instance_mesh",
                        color=args.color,
                        align_global=args.align_global,
                        junction_sphere_delta=args.junction_sphere_delta,
                    )
                classes_for_output = dict(classes)
                if want_junction_nodes:
                    meshes, classes_for_output = _add_junction_node_meshes(
                        meshes,
                        classes_for_output,
                        offsets,
                        plant_graph,
                        radius_delta=args.junction_sphere_delta,
                    )
                if want_fruit:
                    fruit_mesh, fruit_texture = _load_fruit_mesh(fruit_obj_path)
                    meshes, classes_for_output = _add_random_fruit_meshes(
                        meshes,
                        classes_for_output,
                        offsets,
                        plant_graph,
                        rng=rng,
                        fruit_per_plant=fruit_per_plant,
                        fruit_mesh=fruit_mesh,
                        scale_k=fruit_scale_k,
                        rotate_x_deg=fruit_rotate_x_deg,
                    )
                if args.texturise:
                    if args.leaf_texture is None or not os.path.isdir(args.leaf_texture):
                        raise FileNotFoundError("--texturise requires a valid folder path for --leaf_texture")

                    image_paths = glob.glob(os.path.join(args.leaf_texture, "*.*"))
                    image_paths = [p for p in image_paths if p.lower().endswith((".png", ".jpg", ".jpeg"))]

                    if len(image_paths) == 0:
                        raise FileNotFoundError(f"No image files found in folder: {args.leaf_texture}")

                    for node_id, mesh in meshes.items():
                        class_idx = classes_for_output.get(str(node_id))
                        if class_idx == LEAF_CLASS:
                            chosen_texture_path = random.choice(image_paths)
                            print(f"[TEXTURE] Using random leaf texture: {chosen_texture_path}")

                            img = o3d.io.read_image(chosen_texture_path)
                            attach_uv_to_grid_mesh(
                                mesh,
                                (plant_graph.leaf_w, plant_graph.leaf_h),
                                flip_u=args.leaf_texture_flip_u,
                                flip_v=args.leaf_texture_flip_v,
                                rotate_deg=args.leaf_texture_rotate_deg,
                            )
                            mesh.textures = [img]
                            if len(mesh.triangle_material_ids) == 0:
                                mesh.triangle_material_ids = o3d.utility.IntVector(
                                    np.zeros(len(mesh.triangles), dtype=np.int32)
                                )
                        mesh.compute_vertex_normals()
                if args.output_type == "color_mesh":
                    mesh = _merge_with_class_colors(meshes, classes_for_output)
                    output_name = f"{args.output_prepend}_color_{args.sample_name}_variation_{idx:02d}_color.ply"
                    save_mesh(mesh, os.path.join(args.output_dir, output_name))
                    if args.visualize:
                        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
                        o3d.visualization.draw_geometries([mesh, axis], mesh_show_back_face=True)
                elif args.output_type == "instance_mesh":
                    inst_dir = os.path.join(
                        args.output_dir, f"{args.output_prepend}_{args.sample_name}_variation_{idx:02d}", "instances"
                    )
                    want_obj = bool(args.texturise) or (want_fruit and fruit_texture)
                    # If textured, export OBJ with UVs; otherwise use PLY
                    if want_obj:
                        _export_instance_meshes(meshes, classes_for_output, inst_dir, write_uvs=True, file_ext="obj")
                    else:
                        _export_instance_meshes(meshes, classes_for_output, inst_dir, write_uvs=False, file_ext="ply")
                    if args.visualize:
                        o3d.visualization.draw_geometries(
                            list(meshes.values()), mesh_show_back_face=True
                        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random Demeter variations by sampling PCA coefficients."
    )
    today = datetime.now().strftime("%Y%m%d")
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
    parser.add_argument(
        "--output_type",
        type=str,
        choices=("mesh", "color_mesh", "instance_mesh"),
        default="mesh",
        help="Switch between a merged mesh, merged per-class colors, or separate instance meshes.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seeds the RNG for repeatable sampling.")
    parser.add_argument("--align_global", action="store_true", help="Align the main stem to the global axis.")
    parser.add_argument("--draw_graph", action="store_true", help="Print plant hierarchy information.")
    parser.add_argument("--visualize", action="store_true", help="Open an interactive viewer for each variation.")
    parser.add_argument("--color", type=str, default="gray", help="Mesh color mode passed to PlantGraph.generate.")
    parser.add_argument(
        "--junction_sphere_delta",
        type=float,
        default=0.0,
        help="Percent increase over stem thickness for junction spheres (e.g. 2 = +2%).",
    )
    parser.add_argument(
        "--add_junction_nodes",
        action="store_true",
        help="Export junction spheres as separate 'node' instances (instance_mesh/color_mesh only).",
    )
    parser.add_argument("--texturise", action="store_true", help="Apply a single texture to all leaf instance meshes.")
    parser.add_argument("--leaf_texture", type=str, default=None, help="Path to leaf texture images used when --texturise is set.")
    parser.add_argument("--leaf_texture_flip_u", action="store_true", help="Mirror texture horizontally on leaves.")
    parser.add_argument("--leaf_texture_flip_v", action="store_true", help="Mirror texture vertically on leaves.")
    parser.add_argument("--leaf_texture_rotate_deg", type=int, default=0, choices=[0, 90, 180, 270], help="Rotate leaf texture UVs clockwise (deg).")
    parser.add_argument(
        "--output_prepend",
        type=str,
        default=today,
        help="Prefix prepended to every exported variation (default: today's date, YYYYMMDD).",
    )
    parser.add_argument("--fruit_obj_path", type=str, default=None, help="Path to an OBJ mesh used for fruit instances.")
    parser.add_argument("--fruit_per_plant", type=int, default=0, help="Number of fruit instances to add per plant.")
    parser.add_argument("--fruit_scale_k", type=float, default=1.0, help="Fruit scale multiplier applied to stem thickness.")
    parser.add_argument("--fruit_rotate_x_deg", type=float, default=0.0, help="Additional fruit rotation around X axis (deg).")
    return parser.parse_args()


if __name__ == "__main__":
    generate_variations(parse_args())
