Grease Pencil objects are not fully integrated with Blender 3D meshes. As a result, some issues can appear in scenes that involve Grease Pencil objects in an environment of 3D meshes. Such as:
  - Not being properly affected by lighting.
  - Not being properly affected by DOF.
  - Artifacts when overlapping 3D objects affected by DOF cameras.
  - Not being affected by volumetric effects such as fog.
  - Etc.

A possible approach to this issue is rendering the Grease Pencil object and replacing it in the scene with a plane matching the location, scale, and orientation of the Grease Pencil strokes, with the rendered frames applied as an image sequence texture on the plane. 

File mesh_plane_from_grease_pencil.py contains a script to provide that functionality. To use it, you will only need a scene with an active camera and a Grease Pencil object, and follow these steps:
  1. Open the Scripting workspace in Blender and create a new text file.
  2. Copy the full content of mesh_plane_from_grease_pencil.py into it.
  3. Make sure the proper camera is active and check your staging.
  4. Select the Grease Pencil object. Only the active objective in the scene will be considered, so select only one.
  5. Run the script.

If everything goes well after a few seconds you should see the generated plane.

To track the progress of the script you can use the Blender console. You can access it from Window -> Toggle System Console
