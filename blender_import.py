"""
Utility script for Blender to import a set of Demeter meshes arranged in a grid.

Usage example:
    blender --background --python blender_import.py -- \
        --input_root variation_batches --spacing 2.0
"""

import argparse
import math
import os
import sys

import bpy


def parse_args():
    parser = argparse.ArgumentParser(description="Import color meshes and arrange them in a grid.")
    parser.add_argument("--input_root", type=str, required=True, help="Root folder containing sample/batch subfolders.")
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.5,
        help="Distance between neighboring meshes along the grid axes.",
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=0,
        help="Optional fixed number of columns; defaults to ceil(sqrt(N)).",
    )
    parser.add_argument(
        "--clear_scene",
        action="store_true",
        help="Remove all existing objects before importing the meshes.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional .blend path to save after import; defaults to overwriting the current file.",
    )

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def collect_meshes(root):
    meshes = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Input root '{root}' does not exist.")
    for sample_name in sorted(os.listdir(root)):
        sample_path = os.path.join(root, sample_name)
        if not os.path.isdir(sample_path):
            continue
        for batch_name in sorted(os.listdir(sample_path)):
            batch_path = os.path.join(sample_path, batch_name)
            if not os.path.isdir(batch_path):
                continue
            for file_name in sorted(os.listdir(batch_path)):
                if file_name.lower().endswith(".ply"):
                    meshes.append(
                        {
                            "sample": sample_name,
                            "batch": batch_name,
                            "filename": file_name,
                            "path": os.path.join(batch_path, file_name),
                        }
                    )
    if not meshes:
        raise RuntimeError(f"No .ply files found under '{root}'.")
    return meshes


def import_mesh(entry):
    result = bpy.ops.wm.ply_import(
        filepath=entry["path"],
        import_colors="SRGB",
        forward_axis="Y",
        up_axis="Z",
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"Failed to import {entry['path']}")
    obj = bpy.context.active_object
    obj.name = f"{entry['sample']}_{entry['batch']}_{entry['filename'].rsplit('.', 1)[0]}"
    return obj


def arrange_objects(objs, spacing, cols):
    for idx, obj in enumerate(objs):
        row = idx // cols
        col = idx % cols
        obj.location = (col * spacing, -row * spacing, 0.0)


def main():
    args = parse_args()
    meshes = collect_meshes(args.input_root)

    if args.clear_scene:
        clear_scene()

    cols = args.grid_cols if args.grid_cols > 0 else max(1, math.ceil(math.sqrt(len(meshes))))
    imported_objects = []
    for entry in meshes:
        obj = import_mesh(entry)
        imported_objects.append(obj)
        print(f"Imported {entry['path']} as {obj.name}")

    arrange_objects(imported_objects, args.spacing, cols)
    print(f"Placed {len(imported_objects)} meshes in a {cols}-column grid.")

    save_target = args.save_path or bpy.data.filepath
    if not save_target:
        raise RuntimeError(
            "No target .blend file to save to. Please open an existing .blend or provide --save_path."
        )
    if args.save_path:
        bpy.ops.wm.save_as_mainfile(filepath=args.save_path)
        print(f"Saved arrangement to {args.save_path}")
    else:
        bpy.ops.wm.save_mainfile()
        print(f"Saved arrangement to {save_target}")


if __name__ == "__main__":
    main()
