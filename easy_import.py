import bpy
import os
import glob

ROOT_GLOB = os.path.expanduser(
    "~/Development/00_Demeter/variation_output/20260111*"
)


def import_all_objs(root_glob):
    roots = sorted(glob.glob(root_glob))
    for root_dir in roots:
        if not os.path.isdir(root_dir):
            continue
        for dirpath, _, filenames in os.walk(root_dir):
            for name in sorted(filenames):
                if not name.lower().endswith(".obj"):
                    continue
                path = os.path.join(dirpath, name)
                print(f"Importing: {path}")
                bpy.ops.wm.obj_import(filepath=path)


bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

import_all_objs(ROOT_GLOB)
