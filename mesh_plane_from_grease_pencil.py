import bmesh
import bpy
import mathutils
import numpy as np
import os


def get_active_camera():
    '''
    Returns the active camera, raises an Exception if no camera was found.
    '''
    active_camera = bpy.context.scene.camera  # returns de active camera
    if active_camera:
        return active_camera
    else:
        raise Exception('No camera found in the scene.') 

def create_gp_camera(gp_object):
    '''
    Create a new camera based on the active camera to frame the given Grease Pencil object tightly.
    Args:
        - gp_object: Grease Pencil object to be framed.
    Returns:
        - New camera framing the Grease Pencil object.
    '''
    active_camera = get_active_camera()
    
    # Create the new camera:
    bpy.ops.object.select_all(action='DESELECT')
    active_camera.select_set(True)
    bpy.context.view_layer.objects.active = active_camera
    bpy.ops.object.duplicate()
    gp_camera = bpy.context.view_layer.objects.active
    gp_camera.name = 'gp_camera_tmp'
    bpy.context.scene.camera = gp_camera

    # Frame the Grease Pencil object:
    bpy.context.view_layer.objects.active = gp_object
    bpy.ops.object.select_all(action='DESELECT')
    gp_object.select_set(True)
    bpy.ops.view3d.camera_to_view_selected()

    # Restore camera:
    bpy.context.scene.camera = active_camera

    return gp_camera

def hide_everything_except_object(target_gp_object):
    for obj in bpy.data.objects:
        obj.hide_render = obj != target_gp_object

def get_render_visibility_data():
    obj_visibility_data = {}
    for obj in bpy.data.objects:
        obj_visibility_data[obj.name] = obj.hide_render
    return obj_visibility_data

def restore_hidden_objects(obj_visibility_data): 
    for obj_name in obj_visibility_data:
        obj = bpy.data.objects.get(obj_name)
        obj.hide_render = False

def collect_render_settings(target_gp_object):
     render_settings = {
        'file_format': bpy.context.scene.render.image_settings.file_format,
        'ffmpeg_format': bpy.context.scene.render.ffmpeg.format,
        'ffmpeg_codec': bpy.context.scene.render.ffmpeg.codec,
        'filepath': bpy.context.scene.render.filepath,
        'fps': bpy.context.scene.render.fps,
        'color_mode': bpy.context.scene.render.image_settings.color_mode,
        'film_transparent': bpy.context.scene.render.film_transparent,
        'use_lights': target_gp_object.data.layers.active.use_lights
    }
     return render_settings

def set_render_settings(target_gp_object):
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.fps = 24                                 # Set frames per second
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True
    target_gp_object.data.layers.active.use_lights = False

def restore_render_settings(render_settings, target_gp_object):
    bpy.context.scene.render.image_settings.file_format = render_settings.get('file_format')
    bpy.context.scene.render.ffmpeg.format =  render_settings.get('ffmpeg_format')
    bpy.context.scene.render.ffmpeg.codec = render_settings.get('ffmpeg_codec')
    bpy.context.scene.render.fps = render_settings.get('fps')
    bpy.context.scene.render.image_settings.color_mode = render_settings.get('color_mode')
    bpy.context.scene.render.film_transparent = render_settings.get('film_transparent')
    target_gp_object.data.layers.active.use_lights = render_settings.get('use_lights')

def restore_camera(scene_camera, gp_camera):
    scene_camera.select_set(True)
    bpy.context.scene.camera = scene_camera
    bpy.data.objects.remove(gp_camera, do_unlink=True)

def export_single_gp_object_animation(gp_camera, output_path):
    '''
    Render the Grease Pencil camera, focused on the target object, and save the frame fles in the output_path.
    '''
    # Set camera:
    bpy.context.scene.camera = gp_camera

    # Execute render:
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print('Rendered image saved at {}'.format(output_path))


def get_selected_gp_object():
    '''
    Get selected object and make sure it is a Grese Pencil object.
    Returns:
        Selected Grease Pencil Object.
    Raises:
        Exception if there is no selected object, or if the selected object is not a Grease Pencil object.
    '''
    selected_obj = bpy.context.object
    if selected_obj is None or selected_obj.type != 'GPENCIL':
        raise Exception('Please select a Grease Pencil object.')
    return selected_obj

def get_output_directory(foldername='script_data'):
    '''
    Return the output directory: a folder with name 'foldername' where the working file is located. 
    '''
    working_directory = bpy.path.abspath('//')
    data_folder_path = os.path.join(working_directory, foldername)
    os.makedirs(data_folder_path, exist_ok=True)
    return data_folder_path

def get_frame_size(frame_path):
    '''
    Given the path of an image, return its width and height.
    '''
    image = bpy.data.images.load(frame_path)
    width = image.size[0]
    height = image.size[1]
    bpy.data.images.remove(image)

    return width, height
    
def create_gp_preview_plane(gp_object):
    '''
    Create a plane and fix the default orientation.
    Args:
        - gp_object: Grease Pencil object the plane will be named after. 
                     We will also place it on the same collections in the outliner. 
    Returns:
        -The resulting plane.
    '''
    name = gp_object.name

    # Create plane
    bpy.ops.mesh.primitive_plane_add(size=1)
    plane = bpy.context.active_object
    plane.name = '{}_mesh'.format(name)

    # Assert collections:
    for collection in gp_object.users_collection:
        collection.objects.link(plane)
    
    # Rotate 90 degrees around the Y axis in Edit Mode:
    rotate_plane_in_edit_mode(plane, 'Y', 90)
    rotate_plane_in_edit_mode(plane, 'X', 90)
    
    return plane

def set_gp_preview_plane_material(name, plane_object, frame_list):
    # Create material
    mat = bpy.data.materials.new(name='{}_preview_mat'.format(name))
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    tex = nodes.new('ShaderNodeTexImage')
    
    # Set up image sequence
    tex.image_user.use_auto_refresh = True
    tex.image_user.frame_duration = len(frame_list)
    
    # Load first image of sequence
    tex.image = bpy.data.images.load(frame_list[0])
    tex.image.alpha_mode = 'STRAIGHT'
    tex.image.source = 'SEQUENCE'
    
    # Connect nodes
    links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(tex.outputs['Alpha'], bsdf.inputs['Alpha'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # Assign material to plane
    plane_object.data.materials.append(mat)


def rotate_plane_in_edit_mode(plane_obj, axis='Y', degrees=90):
    '''
    Rotate the given plane in edit mode.
    Args:
        plane_obj (mesh object): plane to rotate
        axis (string): Axis around which the plane will be rotated ('X', 'Y', or 'Z')
        degrees (int): the number of degrees the plane will be rotated
    Returns:
        The rotated plane.
    '''
    from math import radians

    # Enter Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Get the BMesh representation of the object
    bm = bmesh.from_edit_mesh(plane_obj.data)
    
    # Create a rotation matrix (90 degrees converted to radians) around the Y-axis
    rotation_matrix = mathutils.Matrix.Rotation(radians(degrees), 4, axis)
    
    # Apply the rotation to all vertices
    for vert in bm.verts:
        vert.co = rotation_matrix @ vert.co
    
    # Update the mesh
    bmesh.update_edit_mesh(plane_obj.data)
    
    # Return to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')

    return plane_obj


def calculate_grease_pencil_data(gp_object, frame_number):
    '''
    Calculate the bounding box dimesions (W, H, D) of the given Grease Pencil object. 
    '''    
    # Initialize min/max coordinates
    min_coords = mathutils.Vector((float('inf'), float('inf'), float('inf')))
    max_coords = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
    
    points = []

    # Iterate through all strokes
    gpencil_data = gp_object.data
    for layer in gpencil_data.layers:
        if not layer.hide:  # Skip hidden layers
            for frame_number_check in range(frame_number, 0, -1):  # Decreasing range
                for frame in layer.frames:
                    if frame.frame_number == frame_number_check:
                        for stroke in frame.strokes:
                            for point in stroke.points:
                                # Get the point position in world space
                                local_pos = point.co
                                world_pos = gp_object.matrix_world @ local_pos

                                points.append([world_pos.x, world_pos.y, world_pos.z])
                                
                                # Update bounding box coordinates
                                min_coords.x = min(min_coords.x, world_pos.x)
                                min_coords.y = min(min_coords.y, world_pos.y)
                                min_coords.z = min(min_coords.z, world_pos.z)
                                
                                max_coords.x = max(max_coords.x, world_pos.x)
                                max_coords.y = max(max_coords.y, world_pos.y)
                                max_coords.z = max(max_coords.z, world_pos.z)
                        break
                else:
                    continue
                break

    
    # Transform global bounding box back to local space
    #min_coords = gp_object.matrix_world.inverted() @ min_coords
    #max_coords = gp_object.matrix_world.inverted() @ max_coords

    # Calculate bounding box dimensions
    dimensions = max_coords - min_coords
    
    # Calculate bounding bond center:
    bbox_center = (min_coords + max_coords) / 2

    bbox_data = {'dimensions': dimensions, 'center': bbox_center}

    # Calculate dominant plane:
    dominant_plane_data = {}
    if len(points) > 3:

        # Convert points to a NumPy array
        points_array = np.array(points)

        # Calculate the centroid (mean position)
        centroid = np.mean(points_array, axis=0)

        # Perform PCA using NumPy's covariance matrix and eigenvalue decomposition
        cov_matrix = np.cov(points_array.T)  # Covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # Eigen decomposition

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        # The normal to the plane is the eigenvector with the smallest eigenvalue
        normal_vector = eigenvectors[:, 2]  # Smallest eigenvalue corresponds to the plane normal
        plane_axes = (eigenvectors[:, 0], eigenvectors[:, 1])  # Two dominant directions on the plane

        # Convert results back to mathutils.Vector for Blender compatibility
        centroid = mathutils.Vector(centroid)
        normal = mathutils.Vector(normal_vector)
        axes = (mathutils.Vector(plane_axes[0]), mathutils.Vector(plane_axes[1]))

        dominant_plane_data = {'centroid': centroid, 'normal': normal, 'axes': axes}
   
    return (bbox_data, dominant_plane_data)


def calculate_grease_pencil_bounding_box(gp_object, frame_number):
    '''
    Calculate the bounding box dimesions (W, H, D) of the given Grease Pencil object. 
    '''    
    # Initialize min/max coordinates
    min_coords = mathutils.Vector((float('inf'), float('inf'), float('inf')))
    max_coords = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
    
    # Iterate through all strokes
    gpencil_data = gp_object.data
    for layer in gpencil_data.layers:
        if not layer.hide:  # Skip hidden layers
            for frame_number_check in range(frame_number, 0, -1):  # Decreasing range
                for frame in layer.frames:
                    if frame.frame_number == frame_number_check:
                        for stroke in frame.strokes:
                            for point in stroke.points:
                                # Get the point position in world space
                                local_pos = point.co
                                world_pos = gp_object.matrix_world @ local_pos
                                
                                # Update bounding box coordinates
                                min_coords.x = min(min_coords.x, world_pos.x)
                                min_coords.y = min(min_coords.y, world_pos.y)
                                min_coords.z = min(min_coords.z, world_pos.z)
                                
                                max_coords.x = max(max_coords.x, world_pos.x)
                                max_coords.y = max(max_coords.y, world_pos.y)
                                max_coords.z = max(max_coords.z, world_pos.z)
                        break
                else:
                    continue
                break

    
    # Transform global bounding box back to local space
    #min_coords = gp_object.matrix_world.inverted() @ min_coords
    #max_coords = gp_object.matrix_world.inverted() @ max_coords

    # Calculate bounding box dimensions
    dimensions = max_coords - min_coords

    # Calculate bounding bond center:
    bbox_center = (min_coords + max_coords) / 2

    bbox_data = {'dimensions': dimensions, 'center': bbox_center}
    return bbox_data


def visualize_gp_bounding_box(gp_object, bbox_data):
    """
    Visualize the center and bounding box of a Grease Pencil object.
    Args:
        - gp_object: The Grease Pencil object.
        - bbox_data: Dictionary containing 'center' and 'dimensions' of the bounding box.
        - wireframe: Whether to set the bounding box cube to wireframe mode (default: True).
    """
    center = bbox_data['center']
    dimensions = bbox_data['dimensions']
    current_frame = bpy.context.scene.frame_current

    # Sphere for the center visualization
    sphere_name = f"{gp_object.name}_center_sphere"
    sphere = bpy.data.objects.get(sphere_name)

    if not sphere:
        # Create sphere if it doesn't exist
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=center)
        sphere = bpy.context.object
        sphere.name = sphere_name

        # Assign red material
        mat = bpy.data.materials.new(name="RedMaterial")
        mat.diffuse_color = (1.0, 0.0, 0.0, 1.0)  # RGBA (Red)
        sphere.data.materials.append(mat)
    else:
        # Update sphere location
        sphere.location = center

    # Keyframe the sphere location
    sphere.keyframe_insert(data_path="location", frame=current_frame)

    # Set the sphere as not renderable
    sphere.hide_render = True

    # Cube for the bounding box visualization
    cube_name = f"{gp_object.name}_bounding_box"
    cube = bpy.data.objects.get(cube_name)

    if not cube:
        # Create cube if it doesn't exist
        bpy.ops.mesh.primitive_cube_add(size=1, location=center)
        cube = bpy.context.object
        cube.name = cube_name

        # Make the cube wireframe
        cube.display_type = 'WIRE'
    else:
        # Update cube location and scale
        cube.location = center
        cube.scale = dimensions

    # Keyframe the cube location, rotation, and scale
    cube.keyframe_insert(data_path="location", frame=current_frame)
    cube.keyframe_insert(data_path="rotation_euler", frame=current_frame)
    cube.keyframe_insert(data_path="scale", frame=current_frame)

    # Set the cube as not renderable
    cube.hide_render = True

    # Make the cube wireframe
    cube.display_type = 'WIRE'


def align_plane_to_dominant_plane(plane, target_plane_data):
    """
    Align a plane object to the calculated dominant plane.
    Args:
        - plane: The plane object to align.
        - plane_data: Dictionary containing 'centroid', 'normal', and 'axes' from the dominant plane.
    """
    # Extract plane data
    centroid = target_plane_data['centroid']
    normal = target_plane_data['normal']
    axes = target_plane_data['axes']  # Two orthogonal vectors on the plane

    # New axes:
     # Define the upward Y-axis in world space
    world_y = mathutils.Vector((0, 1, 0))
    # Calculate the new X-axis (orthogonal to the normal and world Y-axis)
    x_axis = normal.cross(world_y).normalized()
    # If the normal is parallel to the world Y-axis, fallback to another direction
    if x_axis.length < 1e-6:  # Check for near-zero vector
        world_x = mathutils.Vector((1, 0, 0))  # Use world X-axis instead
        x_axis = normal.cross(world_x).normalized()
    # Recalculate the new Y-axis to ensure orthogonality
    y_axis = normal.cross(x_axis).normalized()

    # Set the plane's location to the centroid
    plane.location = centroid

    # Create a 3x3 rotation matrix from the plane axes
    rotation_matrix = mathutils.Matrix((
        normal, #axes[0],  # Local X-axis of the plane
        y_axis, #axes[1],  # Local Y-axis of the plane
        x_axis,#normal,   # Local Z-axis of the plane
    )).transposed()  # Transpose to convert axes to column vectors

    # Apply the rotation matrix to the plane
    plane.rotation_euler = rotation_matrix.to_euler()

def create_mesh_plane_on_dominant_plane(gp_object, plane_data, plane_size=1.0):
    """
    Create a mesh plane aligned with the dominant plane of a Grease Pencil object.
    Args:
        - gp_object: The Grease Pencil object.
        - plane_data: Dictionary containing 'centroid', 'normal', and 'axes' from the dominant plane.
        - plane_size: The size of the plane.
    Returns:
        - The created mesh plane object.
    """
    centroid = plane_data['centroid']
    normal = plane_data['normal']
    axes = plane_data['axes']  # Two orthogonal vectors on the plane

    # Create a new mesh plane
    bpy.ops.mesh.primitive_plane_add(size=plane_size, location=centroid)
    plane = bpy.context.object
    plane.name = f"{gp_object.name}_dominant_plane"

    # Rotate 90 degrees around the Y axis in Edit Mode:
    #rotate_plane_in_edit_mode(plane, 'Y', 90)
    #rotate_plane_in_edit_mode(plane, 'X', 90)    

    # Create a rotation matrix from the dominant plane's axes
    rotation_matrix = mathutils.Matrix((
        axes[0],  # Local X-axis of the plane
        axes[1],  # Local Y-axis of the plane
        normal,   # Local Z-axis of the plane
    )).transposed()  # Transpose to match Blender's conventions

    # Apply the rotation matrix to the plane
    plane.rotation_euler = rotation_matrix.to_euler()

    return plane

def transform_preview_plane(plane_obj, gp_obj, frame_path, frame_number):
    '''
    For the given plane we want to:
        - Have the same size as the Grease PEncil object.
        - Be located at the same place as the Grease Pencil object.
        - Face the active camera.
    '''
    (bbox_data, dominant_plane_data) = calculate_grease_pencil_data(gp_obj, frame_number)
    #dominant_plane = create_mesh_plane_on_dominant_plane(gp_obj, dominant_plane_data, 5.0)
    align_plane_to_dominant_plane(plane_obj, dominant_plane_data)

    # visualize_gp_bounding_box(gp_obj, bbox_data)

    # Set plane aspect ratio:
    (width, height) = get_frame_size(frame_path)
    plane_obj.scale.z = 1
    plane_obj.scale.y = width/height

    # Grease Pencil object size:
    gp_obj_dimensions = bbox_data.get('dimensions')
    gp_width = gp_obj_dimensions.y
    gp_height = gp_obj_dimensions.z
    gp_aspect_ratio = gp_width/gp_height
    
    # Plane object size:
    plane_width = plane_obj.scale.y
    plane_height = plane_obj.scale.z
    pln_aspect_ratio = plane_width/plane_height 
    
    if pln_aspect_ratio < gp_aspect_ratio:
        plane_obj.scale.y = gp_width
        plane_obj.scale.z = plane_obj.scale.y / pln_aspect_ratio
    else:
        plane_obj.scale.z = gp_height
        plane_obj.scale.y = plane_obj.scale.z * pln_aspect_ratio 

    # Set new Plane object location:
    gp_obj_center = bbox_data.get('center')
    plane_obj.location = gp_obj_center

    # Set key frame for location, rotation and scale:
    plane_obj.keyframe_insert(data_path="location", frame=frame_number)
    plane_obj.keyframe_insert(data_path="rotation_euler", frame=frame_number)
    plane_obj.keyframe_insert(data_path="scale", frame=frame_number)

    return plane_obj



def main():

    scene_camera = get_active_camera()

    gp_object = get_selected_gp_object()
    
    plane_object = create_gp_preview_plane(gp_object)

    output_directory = get_output_directory()
    frames_output_path = os.path.join(output_directory, gp_object.name)

    render_visibility_data = get_render_visibility_data()
    hide_everything_except_object(gp_object)

    render_settings = collect_render_settings(gp_object)

    set_render_settings(gp_object)

    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end

    frame_list = []
    for frame_number in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame_number)

        gp_camera = create_gp_camera(gp_object)

        frame_output_path = '{}_{}.png'.format(frames_output_path, str(frame_number).zfill(4))
        frame_list.append(frame_output_path)

        export_single_gp_object_animation(gp_camera, frame_output_path)

        restore_camera(scene_camera, gp_camera)
        
        transform_preview_plane(plane_object, gp_object, frame_output_path, frame_number)
    
    set_gp_preview_plane_material(gp_object.name, plane_object, frame_list)

    restore_hidden_objects(render_visibility_data)

    restore_render_settings(render_settings, gp_object)

    #gp_object.hide_viewport = True
    #gp_object.hide_render = True 


if __name__ == '__main__':
    main()
