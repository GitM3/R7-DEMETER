import os
import numpy as np
import open3d as o3d
import torch
from utils.constant import *
from representation.graph import PlantGraphFixedTopology
from utils.pca import NodePCA
from utils.graph import load_parent, load_class
from utils.texturing import attach_uv_to_grid_mesh
    
def decode_params(data_folder:str, sample_name:str, species:str='soybean', **kwargs):

    instance_folder = os.path.join(data_folder, species, 'instances', sample_name)

    # load class annotation
    classes = load_class(os.path.join(instance_folder, 'info','class.txt'))

    # load parent annotation
    parents = load_parent(os.path.join(instance_folder, 'info','parent.txt'))

    # load PCA weights for stem and leaf
    pca_stem_3d = NodePCA(path=os.path.join(data_folder, species, '3d_stem_pca.pth'))
    pca_leaf_3d = NodePCA(path=os.path.join(data_folder, species, '3d_leaf_pca.pth'))
    pca_leaf_2d = NodePCA(path=os.path.join(data_folder, species, '2d_leaf_pca.pth'))
    
    # init plant graph
    plant_graph = PlantGraphFixedTopology(
        classes=classes, parents=parents, species=species,
        pca_leaf_3d=pca_leaf_3d, pca_stem_3d=pca_stem_3d, pca_leaf_2d=pca_leaf_2d
    )
    plant_graph.load(os.path.join(instance_folder, 'graph.pkl'))
    plant_graph.cuda()

    # draw graph structure
    if kwargs.get('draw_graph', True):
        plant_graph.draw_topology()

    texturise = kwargs.get('texturise', False)
    leaf_texture_path = kwargs.get('leaf_texture', None)
    leaf_texture_flip_u = kwargs.get('leaf_texture_flip_u', False)
    # default flip_v=True keeps origin at image top-left; allow override to align texture orientation
    leaf_texture_flip_v = kwargs.get('leaf_texture_flip_v', True)
    leaf_texture_rotate_deg = kwargs.get('leaf_texture_rotate_deg', 0)
    output_mesh_path = kwargs.get('output_mesh', None)

    if texturise:
        if not leaf_texture_path or not os.path.isfile(leaf_texture_path):
            raise FileNotFoundError("--texturise is set but --leaf_texture is missing or not a file")

        with torch.no_grad():
            meshes = plant_graph.generate(output_format='instance_mesh', color='gray', align_global=True)

        # Attach UVs and texture only to leaf meshes
        for node_id, m in list(meshes.items()):
            cls = classes.get(str(node_id))
            if cls == LEAF_CLASS:
                # Add UVs for canonical grid and attach the same texture
                attach_uv_to_grid_mesh(
                    m,
                    (plant_graph.leaf_w, plant_graph.leaf_h),
                    flip_u=leaf_texture_flip_u,
                    flip_v=leaf_texture_flip_v,
                    rotate_deg=leaf_texture_rotate_deg,
                )
                img = o3d.io.read_image(leaf_texture_path)
                m.textures = [img]
                # Ensure material ids exist so Open3D doesn't crash when sampling textures
                if len(m.triangle_material_ids) == 0:
                    m.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(m.triangles), dtype=np.int32))

        # Visualize all instance meshes together
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
        o3d.visualization.draw_geometries(list(meshes.values()) + [axis], mesh_show_back_face=True)

        if output_mesh_path:
            # merge instance meshes and export with UVs for Blender
            merged = None
            for m in meshes.values():
                merged = m if merged is None else merged + m
            merged.compute_vertex_normals()

            # normalize output path and create folders
            mesh_path = output_mesh_path
            base, ext = os.path.splitext(mesh_path)
            if ext == '':
                mesh_dir = mesh_path
                mesh_name = f"{sample_name or 'plant'}.obj"
                mesh_path = os.path.join(mesh_dir, mesh_name)
            else:
                mesh_dir = os.path.dirname(mesh_path)
                if mesh_dir == '':
                    mesh_dir = '.'
            os.makedirs(mesh_dir, exist_ok=True)
            o3d.io.write_triangle_mesh(mesh_path, merged, write_triangle_uvs=True)
            print(f"Saved textured mesh to {mesh_path}")
    else:
        with torch.no_grad():
            # align the plant to global X-axis to make it stand straight
            mesh = plant_graph.generate(output_format='mesh', color='gray', align_global=True)

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
        axis.translate([0, 0, 0])
        o3d.io.write_triangle_mesh("./output.ply",mesh,print_progress=True)
        o3d.visualization.draw_geometries([mesh, axis], mesh_show_back_face=True)
        if output_mesh_path:
            mesh_path = output_mesh_path
            base, ext = os.path.splitext(mesh_path)
            if ext == '':
                mesh_dir = mesh_path
                mesh_name = f"{sample_name or 'plant'}.obj"
                mesh_path = os.path.join(mesh_dir, mesh_name)
            else:
                mesh_dir = os.path.dirname(mesh_path)
                if mesh_dir == '':
                    mesh_dir = '.'
            os.makedirs(mesh_dir, exist_ok=True)
            o3d.io.write_triangle_mesh(mesh_path, mesh, write_triangle_uvs=True)
            print(f"Saved mesh to {mesh_path}")

if __name__ == "__main__":

    # example usage 1
    # use argparse to parse the command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='sample_params', help='path to the folder containing the sample parameters')
    parser.add_argument('--sample_name', type=str, default=None, help='name of the sample to process')
    parser.add_argument('--species', type=str, default='soybean', help='species of the plant')
    parser.add_argument('--draw_graph', action='store_true', help='whether to draw the graph structure')
    parser.add_argument('--texturise', action='store_true', help='apply a single texture to all leaf meshes (proof of concept)')
    parser.add_argument('--leaf_texture', type=str, default=None, help='path to a leaf texture image (png/jpg) used when --texturise is set')
    parser.add_argument('--leaf_texture_flip_u', action='store_true', help='mirror texture horizontally on leaves')
    parser.add_argument('--leaf_texture_flip_v', action='store_true', help='mirror texture vertically on leaves')
    parser.add_argument('--leaf_texture_rotate_deg', type=int, default=0, choices=[0, 90, 180, 270], help='rotate leaf texture UVs clockwise (deg)')
    parser.add_argument('--output_mesh', type=str, default=None, help='path or directory to save the exported mesh (OBJ recommended for Blender). If a directory is given, saves as <sample_name>.obj')
    args = parser.parse_args()

    data_folder = args.data_folder
    sample_name = args.sample_name
    species = args.species

    if sample_name is not None:
        decode_params(
            data_folder,
            sample_name,
            species,
            draw_graph=args.draw_graph,
            texturise=args.texturise,
            leaf_texture=args.leaf_texture,
            leaf_texture_flip_u=args.leaf_texture_flip_u,
            leaf_texture_flip_v=args.leaf_texture_flip_v,
            leaf_texture_rotate_deg=args.leaf_texture_rotate_deg,
            output_mesh=args.output_mesh,
        )
        exit(0)
    
    # example usage 2
    data_folder = 'sample_params'
    decode_params(data_folder, '24_o', 'soybean') # '3_o', '3_i', '4_o', '4_i', '6_o',  '8_i', '24_o', '101_o' etc.
    decode_params(data_folder, '1', 'tobacco')
    decode_params(data_folder, '02', 'rose')
    decode_params(data_folder, '10008da', 'maize')
    decode_params(data_folder, '08', 'ribes')
