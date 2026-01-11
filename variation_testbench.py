import argparse
import os
from datetime import datetime

from variation_decode import generate_variations


def build_run_args(
    base_args,
    sample_name,
    sample_output_root,
    batch_idx,
    stem_scale,
    leaf_deform_scale,
    leaf_shape_scale,
    seed,
):
    """Create a Namespace compatible with variation_decode.generate_variations."""
    output_dir = os.path.join(sample_output_root, f"batch_{batch_idx:02d}")
    output_prepend = f"{base_args.name_prefix}_{sample_name}_b{batch_idx:02d}"
    return argparse.Namespace(
        data_folder=base_args.data_folder,
        species=base_args.species,
        sample_name=sample_name,
        output_dir=output_dir,
        num_variations=base_args.variations_per_batch,
        stem_scale=stem_scale,
        leaf_deform_scale=leaf_deform_scale,
        leaf_shape_scale=leaf_shape_scale,
        strategy=base_args.strategy,
        output_type=base_args.output_type,
        seed=seed,
        align_global=base_args.align_global,
        draw_graph=False,
        visualize=base_args.visualize,
        color=base_args.color,
        texturise=base_args.texturise,
        leaf_texture=base_args.leaf_texture,
        leaf_texture_flip_u=base_args.leaf_texture_flip_u,
        leaf_texture_flip_v=base_args.leaf_texture_flip_v,
        leaf_texture_rotate_deg=base_args.leaf_texture_rotate_deg,
        add_junction_nodes=base_args.add_junction_nodes,
        junction_sphere_delta=base_args.junction_sphere_delta,
        output_prepend=output_prepend,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate batches of plant variations with progressively larger perturbations."
    )
    today = datetime.now().strftime("%Y%m%d")
    parser.add_argument("--data_folder", type=str, default="sample_params")
    parser.add_argument("--species", type=str, default="soybean")
    parser.add_argument("--sample_name", type=str, help="Single sample to process. If omitted, run across all available samples.")
    parser.add_argument("--output_root", type=str, default="variation_batches")
    parser.add_argument("--name_prefix", type=str, default=today, help="Prefix for saved batches.")
    parser.add_argument("--variations_per_batch", type=int, default=5)
    parser.add_argument("--num_batches", type=int, default=4)
    parser.add_argument("--stem_scale_start", type=float, default=0.5)
    parser.add_argument("--stem_scale_step", type=float, default=0.25)
    parser.add_argument("--leaf_deform_scale_start", type=float, default=0.5)
    parser.add_argument("--leaf_deform_scale_step", type=float, default=0.25)
    parser.add_argument("--leaf_shape_scale_start", type=float, default=0.5)
    parser.add_argument("--leaf_shape_scale_step", type=float, default=0.25)
    parser.add_argument("--strategy", choices=("mean", "perturb"), default="mean")
    parser.add_argument("--output_type", choices=("mesh", "color_mesh", "instance_mesh"), default="mesh")
    parser.add_argument("--align_global", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--color", type=str, default="gray")
    parser.add_argument("--texturise", action="store_true")
    parser.add_argument("--leaf_texture", type=str, default=None)
    parser.add_argument("--leaf_texture_flip_u", action="store_true")
    parser.add_argument("--leaf_texture_flip_v", action="store_true")
    parser.add_argument("--leaf_texture_rotate_deg", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--add_junction_nodes", action="store_true")
    parser.add_argument("--junction_sphere_delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None, help="Base seed; batches increment this value.")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    instances_dir = os.path.join(args.data_folder, args.species, "instances")
    if args.sample_name:
        sample_names = [args.sample_name]
    else:
        if not os.path.isdir(instances_dir):
            raise FileNotFoundError(f"No instances directory found at {instances_dir}")
        sample_names = sorted(
            entry for entry in os.listdir(instances_dir) if os.path.isdir(os.path.join(instances_dir, entry))
        )
        if not sample_names:
            raise RuntimeError(f"No samples found under {instances_dir}")

    for sample_idx, sample_name in enumerate(sample_names):
        sample_output_root = os.path.join(args.output_root, sample_name)
        os.makedirs(sample_output_root, exist_ok=True)
        seed_offset = 0 if args.seed is None else sample_idx * args.num_batches
        print(f"=== Processing sample '{sample_name}' ===")

        for batch_idx in range(args.num_batches):
            stem_scale = args.stem_scale_start + batch_idx * args.stem_scale_step
            leaf_def_scale = args.leaf_deform_scale_start + batch_idx * args.leaf_deform_scale_step
            leaf_shape_scale = args.leaf_shape_scale_start + batch_idx * args.leaf_shape_scale_step
            batch_seed = None if args.seed is None else args.seed + seed_offset + batch_idx
            batch_args = build_run_args(
                args,
                sample_name=sample_name,
                sample_output_root=sample_output_root,
                batch_idx=batch_idx,
                stem_scale=stem_scale,
                leaf_deform_scale=leaf_def_scale,
                leaf_shape_scale=leaf_shape_scale,
                seed=batch_seed,
            )
            print(
                f"[Sample {sample_name} | Batch {batch_idx}] stem_scale={stem_scale:.2f}, "
                f"leaf_deform_scale={leaf_def_scale:.2f}, leaf_shape_scale={leaf_shape_scale:.2f}"
            )
            generate_variations(batch_args)


if __name__ == "__main__":
    main()
