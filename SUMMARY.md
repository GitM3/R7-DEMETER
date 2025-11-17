# Demeter Repository Summary

High-level summary of what each script/module does, how the reconstruction pipeline fits together, and whether specific features mentioned in the paper are present here.

## Overview
- Goal: Learn a parametric plant model (Demeter) from 3D scans and decode fitted parameters to geometric reconstructions.
- Core building blocks: point cloud preprocessing, semantic and boundary predictions, instance clustering, per-part fitting (stems/leaf surfaces), graph construction, and graph-level fitting/decoding.

## Pipeline At A Glance
1. Normalize raw point cloud: `script_reconstruction/process_data.py` produces `normalized_pcd.pth` and alignment transforms.
2. Predict semantics and boundary distances: run PointTransformer-V3 (in `third_party/PointTransformer_V3/Pointcept`) to produce `normalized_pcd_pred.npy` (semantics) and `normalized_pcd_pred_dist.npy` (inter-cluster distance / boundary score).
3. Build instances and detect boundary points: `script_reconstruction/recon.py` thresholds distance to find boundary points, clusters non-boundary points (DBSCAN), then inpaints by nearest neighbor per semantic.
4. Fit per-part primitives: `recon.py` fits stems as Catmull–Rom curves and leaves as Catmull–Rom surfaces; extracts PCA coefficients.
5. Build plant graph: `representation/build_graph.py` and `representation/graph.py` assemble parts into a hierarchical graph with relative transforms along the main stem; optionally fine-tune to the instances; can generate point clouds/meshes.
6. Decode saved parameters to mesh: `decode.py` loads a saved graph and PCA weights to generate meshes (per-instance/segmented or merged).

## Root
- `readme.md`: Project intro, environment setup, decode + reconstruction instructions, release plan.
- `requirements.txt`: Minimal dependencies for decoding/geometry utilities.
- `decode.py`: Loads PCA weights and a saved per-plant graph to generate meshes. Options to draw graph topology; outputs Open3D mesh(es).
- `LICENSE`, `.gitignore`: License and ignore rules.

## script_reconstruction
- `readme.md`: 3-step reconstruction workflow with screenshots: preprocessing, PointTransformer inference, and reconstruction.
- `process_data.py`:
  - Interactively pick two points on main stem (bottom then top) to define axis.
  - Aligns, recenters, scales the point cloud; saves `normalized_pcd.pth`, `original_aligned.ply`, and transform metadata (`transform.pkl`, `rotation.txt`, `rotation_click.txt`).
  - Visualizes aligned point cloud.
- `recon.py`:
  - Loads normalized point cloud and PointTransformer predictions (`normalized_pcd_pred.npy`, `normalized_pcd_pred_dist.npy`).
  - Boundary detection: thresholds predicted inter-cluster distance to mark boundary points; inpaints assignments to nearest cluster per semantic.
  - Instance clustering: DBSCAN on non-boundary points; semantic-aware inpainting; merges/splits clusters and filters small ones.
  - Graph topology: heuristically determines parent-child relations with `utils.graph.get_guessed_parent` and BFS layers/paths; saves `info/parent.txt` and `info/class.txt`.
  - Per-part fits:
    - Stems: fit `CatmullRomCurve` between geodesic farthest endpoints; optimize length/rotation/thickness; compute Frenet–Serret frames.
    - Leaves: fit `CatmullRomSurface` with 2D leaf PCA shape coefficients, and 3D deformation PCA on mean-shape; supports super-resolution.
  - Builds the connected plant graph with `representation/build_graph.build_plant_graph`, fine-tunes parameters (optional), and writes fitted part clouds (`fit/*.ply`) and params.
  - Caches intermediate dictionaries: `points.pkl`, `instances.pkl`, `semantics.pkl` and segmentation `segmentation.ply`.

## representation
- `primitive.py`:
  - `CatmullRomCurve`: differentiable curve primitive for stems; parameterized by end points, Catmull–Rom controls, quaternion rotation, scalar scale; fit routines for rotation then full curve; thickness estimation; losses for fit/smoothness.
  - `CatmullRomSurface`: differentiable rectangular surface grid for leaves; uses 2D leaf PCA (mean + components + sigma) to form a 2D template; projects to 3D by learned main/sub rotations; supports inversion to recover rotations; fit routines and super-resolution.
  - Utility math: spherical/cartesian conversions, Catmull–Rom evaluation (fast/batched), smoothing terms.
- `graph.py`:
  - `PlantGraphFixedTopology`: container of per-node parameters (stem and leaf) and parent relations; holds PCA models for stems/leaves.
  - Decoding: reconstructs per-node geometry from PCA coefficients and local transforms; composes transforms along parent stem using Frenet frames and quaternion slerp.
  - Output modes: points, per-instance meshes, segmented meshes (stem/leaf), or merged mesh via Open3D; simple color options.
  - Editing hooks: blend weights for stem/leaf shape/deform during generation.
- `build_graph.py`:
  - Builds the graph from per-instance fits: computes relative attachment along parent stems, local frames, and per-node PCA coefficients (stem 3D, leaf 3D deform + 2D shape); stores per-node scales, rotations, and offsets.
  - Optionally fine-tunes the assembled graph against the instance point sets; saves/loads `graph.pkl`; visualizes results.

## utils
- `pcd.py`:
  - Point cloud helpers: DBSCAN segmentation, nearest/farthest queries, KNN/graph-based distances, Frenet–Serret frames, mesh generation from grids, Gram–Schmidt, plane fitting, and simple visualization utilities.
- `graph.py`:
  - Graph utilities: BFS to find layers/paths, synthetic tree generator, geodesic distance helpers, farthest points, parent/class load/save, and nearest-distance calculations.
- `pca.py`:
  - `NodePCA`: thin wrapper around sklearn PCA with Torch buffers; can train on parameter vectors, save/load `.pth`, and encode/decode parameterizations.
  - Used for: stem 3D control points, leaf 3D deformations, leaf 2D shape.
- `rotation_pytorch3d.py`, `rotation_custom.py`: quaternion <-> matrix conversions, spherical/cartesian conversions and helpers.
- `plot.py`: mesh utilities (e.g., cylinders along curves), topology drawing.
- `metrics.py`: basic metric utilities used sporadically.
- `pde.py`: NumPy/Numba grid relaxations and polygon rasterization helpers for boundary conditions (supporting leaf shaping tasks).
- `tool.py`: small geometric helpers (e.g., polygon area).
- `color_print.py`: colored console output.
- `constant.py`: class IDs for nodes (leaf/stem/flower/fruit).

## sample_* and assets
- `sample_params/`: Per-species PCA weights (`2d_leaf_pca.pth`, `3d_stem_pca.pth`, `3d_leaf_pca.pth`) and example instance parameters/graphs.
- `sample_point_cloud/val/...`: Example raw point clouds for reconstruction.
- `assets/`: Figures used in docs (annotation, semantics, boundary detection, segmentation, recon).

## third_party (external baselines/models)
- `third_party/PointTransformer_V3/`:
  - Contains PointTransformer V3 (with Pointcept) configs and code to run inference for plant semantics and inter-cluster distance prediction.
  - Follow `script_reconstruction/readme.md` to place pretrained weights under `Pointcept/exp/...` and run the provided `scripts/test.sh` invocation; results are copied back as `normalized_pcd_pred.npy` and `normalized_pcd_pred_dist.npy`.
- `third_party/CropCraft/`:
  - L-system baseline for reconstruction (`fit_single.py`) supporting soybean/maize; uses the same preprocessing alignment; generates a parametric L-system plant and fits to the input cloud.

## Where The Paper’s Intermediate Steps Live
- Predicting semantics: `third_party/PointTransformer_V3/Pointcept` (inference), consumed by `script_reconstruction/recon.py`.
- Boundary points: `script_reconstruction/recon.py` (thresholding `normalized_pcd_pred_dist.npy`, red overlays in docs), then add back to clusters via nearest neighbor per semantic.
- Building plant graph: `script_reconstruction/recon.py` (topology + fitting) and `representation/build_graph.py` (assemble relative transforms + PCA coeffs) into `representation/graph.py` for decoding.

## Feature Checks (requested)
- Leaf PCA submodule: Present. `utils/pca.NodePCA` with train/encode/decode; used in `representation/primitive.py` and `representation/graph.py`. Pretrained `.pth` files are supplied in `sample_params/<species>/`.
- Learn from 2D leaf scans: Not provided as a training script or dataset in this repo. The `readme.md` marks “learning leaf shape PCA from 2D leaf scanns (TBD)”. The plumbing exists (`NodePCA.train`) to fit PCA if you prepare 2D parameter vectors, but no end-to-end script is included here.
- Transfering textures: Not implemented in Demeter code. Geometry decoding yields colored Open3D meshes with uniform colors; there is no UV unwrapping/material/texture transfer pipeline. Any “texture” or UV mentions in the `third_party/PointTransformer_V3` dataset utilities are unrelated to Demeter’s plant decoding.

## FAQ: Generation and Data
- Q1 — Given a plant graph and parameters, can an infinite number of new plant meshes be generated?
  - Short answer: Yes, for geometry variations with fixed topology. The decoded meshes are continuous functions of PCA coefficients (leaf 2D shape, leaf 3D deform, stem 3D deform) and articulation parameters (per-node scale, rotation quaternion, and attachment length along the parent stem). Since these are real-valued, you can sample continuously to produce unbounded variations.
  - How: Load a `PlantGraphFixedTopology` (e.g., via `decode.py`), then programmatically modify per-node parameters before calling `generate()`:
    - Leaf nodes: `shape_<id>`, `deform_<id>`, `scale_<id>`, `M_quat_<id>`, `length_<id>`.
    - Stem nodes: `deform_<id>`, `thickness_<id>`, `scale_<id>`, `M_quat_<id>`, `length_<id>`.
    - Optional: use `utils/pca.NodePCA`’s `coeff_mean/coeff_std` (if present in loaded `.pth`) to sample plausible coefficients. No built-in random sampler is provided, but sampling `N(coeff_mean, coeff_std)` and clamping to ~±3σ is consistent with how PCA is used elsewhere.
  - Limits: Topology is fixed per instance (“FixedTopology” class). Changing number/arrangement of nodes requires rebuilding `parents/classes/layers` (not provided as a stochastic generator in this repo).

- Q2 — What inputs does reconstruction accept? Can it generate variations?
  - Inputs: A single raw point cloud file (`.ply`) per plant. The pipeline requires interactive selection of two main-stem points.
  - Minimal CLI flow (from repo root):
    1) Preprocess and align
       `python script_reconstruction/process_data.py --point_path sample_point_cloud/val/27_o/pcd.ply`
       Produces: `normalized_pcd.pth`, `original_aligned.ply`, and `transform.pkl` under the same folder.
    2) Semantics + boundary (PointTransformer-V3 under its env)
       - Place pretrained weights under `third_party/PointTransformer_V3/Pointcept/exp/...` as in `script_reconstruction/readme.md`.
       - Run:
         `conda activate pointcept`
         `cd third_party/PointTransformer_V3/Pointcept`
         `sh scripts/test.sh -p python -d soybean3d -c custom3 -n plant3 -g 1 -w model_last`
       - Copy outputs back to the point folder:
         `cp exp/soybean3d/plant3/result/normalized_pcd_pred.npy <your_folder>/`
         `cp exp/soybean3d/plant3/result/normalized_pcd_pred_dist.npy <your_folder>/`
    3) Reconstruction and graph fitting (Demeter env)
       `conda activate demeter`
       `python script_reconstruction/recon.py --data_folder <your_folder> --species soybean`
       Produces: fitted part clouds under `<your_folder>/fit`, graph under `<your_folder>/params/`.
  - Species/weights: Provided PointTransformer weights and instructions target soybean (dataset `soybean3d`). Recon loads PCA weights per species (`sample_params/<species>/...`), but without semantic/boundary predictors for that species you cannot complete Step 2 out-of-the-box.
  - Variations from reconstruction: The reconstruction itself is deterministic for a given input and weights. After reconstructing, you can create variations by modifying the saved plant graph’s parameters (see Q1), then regenerating meshes.

- Q3 — What’s needed to learn from a custom dataset?
  - From code + docs, the repository does not include end-to-end training scripts. Readme marks these as TBD:
    - “building demeter representation from your own annotated 3d point cloud (TBD)”
    - “learning leaf shape PCA from 2D leaf scanns (TBD)”
  - What is present:
    - PCA scaffolding (`utils/pca.NodePCA`) to fit/serialize PCA models for stems/leaves if you can assemble supervised parameter vectors:
      - Stem: 3D control points of the Catmull–Rom curve in canonical local coordinates.
      - Leaf: 2D mean-shape coefficients (for `CatmullRomSurface` template) and 3D deformation coefficients on the mean-shape.
    - Reconstruction/fitting code that can produce such per-part parameterizations from point clouds (once semantics/boundaries/instances are available), which could be aggregated to train PCA.
    - Third-party PointTransformer-V3 code to train/finetune semantic + boundary predictors (but no Demeter-specific training configs for new datasets are documented here).
  - What is missing/unclear:
    - Dataset format/spec for training PointTransformer on plants outside soybean; no provided training configs or scripts tailored to Demeter’s labels.
    - Scripts to export large-scale per-part parameter vectors from many reconstructions and fit PCA models; only the NodePCA utility exists.
    - Any pipeline to learn or standardize species-specific topologies.
  - Practical path (inferred):
    1) Collect point clouds per plant; annotate/train a semantic + boundary predictor (PointTransformer) for your species.
    2) Run reconstruction (Step 1–3) to fit per-plant graphs across your dataset.
    3) Extract and aggregate per-part parameters (stems: curve control points; leaves: 2D template params + inverted rotations + 3D mean-shape grids) and use `NodePCA.train()` to learn PCA; save `.pth` to `sample_params/<species>/`.
    4) Optionally re-fit graphs using the updated PCA bases; iterate.

## Notes and Tips
- Reconstruction depends strongly on input point cloud quality and completeness (as noted in docs).
- Thresholds for boundary detection are species-dependent (e.g., soybean uses a lower threshold); see `recon.py`.
- Caches: Intermediate clustering/semantics are saved in `.pkl`/`.npy` files to speed re-runs.
- Graph save/load: Per-plant fitted parameters live under `.../params/graph.pkl` (and `.../params/info/`). `decode.py` consumes these for mesh generation.

