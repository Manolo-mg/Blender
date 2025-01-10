import bmesh
import mathutils
import numpy as np
import os

from math import radians, degrees, atan, tan

# BLENDER UTILS
import bpy

# TODO: you probably can clean out the dominant plane stuff from the GP data
# TODO: make one single GP camera and keyframe it for the duration of the animation.

def get_active_camera():
    '''
    Returns the active camera, raises an Exception if no camera was found.
    '''
    active_camera = bpy.context.scene.camera  # returns de active camera
    if active_camera:
        return active_camera
    else:
        raise Exception('No camera found in the scene.')

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

def create_plane_with_name(gp_object_name):
    '''
    Create a plane.
    Args:
        - gp_object_name: name of the target Grease Pencil object. 
    Returns:
        -The resulting plane.
    '''
    bpy.ops.mesh.primitive_plane_add(size=1)
    plane_object = bpy.context.active_object
    return plane_object

def insert_plane_in_gp_object_collections(plane_object, gp_object):
    '''
    Insert the given plane object in all the collections the given Grease Pencil object is present.
    '''
    for collection in gp_object.users_collection:
        objects_names = [object.name for object in collection.objects]
        if plane_object.name not in objects_names:
            collection.objects.link(plane_object)

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

def get_output_directory(foldername='script_data'):
    '''
    Return the output directory: a folder with name 'foldername' where the working file is located. 
    '''
    working_directory = bpy.path.abspath('//')
    data_folder_path = os.path.join(working_directory, foldername)
    os.makedirs(data_folder_path, exist_ok=True)
    return data_folder_path

def get_render_visibility_data():
    obj_visibility_data = {}
    for obj in bpy.data.objects:
        obj_visibility_data[obj.name] = obj.hide_render
    return obj_visibility_data

def hide_everything_except_object(object):
    for obj in bpy.data.objects:
        obj.hide_render = obj != object
    bpy.context.view_layer.update()

def restore_hidden_objects(obj_visibility_data): 
    for obj_name in obj_visibility_data:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            obj.hide_render = obj_visibility_data[obj_name]
    bpy.context.view_layer.update()

def get_start_and_end_frames():
    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end
    return(frame_start, frame_end)

def set_current_frame(frame_number):
    bpy.context.scene.frame_set(frame_number)

def create_tmp_camera_from_camera(source_camera, name=None):
    bpy.ops.object.select_all(action='DESELECT')
    source_camera.select_set(True)
    bpy.context.view_layer.objects.active = source_camera
    bpy.ops.object.duplicate()
    tmp_camera = bpy.context.view_layer.objects.active
    if name:
        tmp_camera.name = 'gp_camera_tmp'
    else: 
        tmp_camera.name = '{}_tmp'.format(source_camera.name)

    return tmp_camera

def create_tmp_camera(name=None):
    camera_data = bpy.data.cameras.new(name=name)
    camera_object = bpy.data.objects.new(name, camera_data)
    bpy.context.scene.collection.objects.link(camera_object)

    return camera_object

def frame_object(target_object, camera):
    bpy.context.scene.camera = camera
    bpy.context.view_layer.objects.active = target_object
    bpy.ops.object.select_all(action='DESELECT')
    target_object.select_set(True)
    bpy.ops.view3d.camera_to_view_selected()

def set_scene_camera(camera):
    bpy.context.scene.camera = camera

def export_single_gp_object_animation(camera, output_path):
    '''
    Render the Grease Pencil camera, focused on the target object, and save the frame fles in the output_path.
    '''
    # Set camera:
    bpy.context.scene.camera = camera

    # Execute render:
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print('Rendered image saved at {}'.format(output_path))

def restore_camera(original_camera, current_camera):
    original_camera.select_set(True)
    bpy.context.scene.camera = original_camera
    bpy.data.objects.remove(current_camera, do_unlink=True)

def create_sphere(name, location, frame_number):
    sphere = bpy.data.objects.get(name)

    if not sphere:
        # Create sphere if it doesn't exist
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=location)
        sphere = bpy.context.object
        sphere.name = name

        # Assign red material
        mat = bpy.data.materials.new(name="RedMaterial")
        mat.diffuse_color = (1.0, 0.0, 0.0, 1.0)  # RGBA (Red)
        sphere.data.materials.append(mat)
    else:
        # Update sphere location
        sphere.location = location

    # Keyframe the sphere location
    sphere.keyframe_insert(data_path="location", frame=frame_number)

    # Set the sphere as not renderable
    sphere.hide_render = True

def create_cube(name, location, dimensions, frame_number):
    cube = bpy.data.objects.get(name)

    if not cube:
        # Create cube if it doesn't exist
        bpy.ops.mesh.primitive_cube_add(size=1, location=location)
        cube = bpy.context.object
        cube.name = name

        # Make the cube wireframe
        cube.display_type = 'WIRE'

        # Update cube location and scale
        cube.scale = dimensions
    else:
        # Update cube location and scale
        cube.location = location
        cube.scale = dimensions

    # Keyframe the cube location, rotation, and scale
    cube.keyframe_insert(data_path="location", frame=frame_number)
    cube.keyframe_insert(data_path="rotation_euler", frame=frame_number)
    cube.keyframe_insert(data_path="scale", frame=frame_number)

    # Set the cube as not renderable
    cube.hide_render = True

    # Make the cube wireframe
    cube.display_type = 'WIRE'

def get_frame_size(frame_path):
    '''
    Given the path of an image, return its width and height.
    '''
    image = bpy.data.images.load(frame_path)
    width = image.size[0]
    height = image.size[1]
    bpy.data.images.remove(image)

    return width, height

def create_material(name):
    mat = bpy.data.materials.new(name=name)
    return mat

def collect_render_settings(target_gp_object):
    render_settings = {
        'file_format': bpy.context.scene.render.image_settings.file_format,
        'ffmpeg_format': bpy.context.scene.render.ffmpeg.format,
        'ffmpeg_codec': bpy.context.scene.render.ffmpeg.codec,
        'filepath': bpy.context.scene.render.filepath,
        'fps': bpy.context.scene.render.fps,
        'color_mode': bpy.context.scene.render.image_settings.color_mode,
        'film_transparent': bpy.context.scene.render.film_transparent,
        #'use_lights': target_gp_object.data.use_lights,
        'layers_use_light': {}
    }
    for layer in target_gp_object.data.layers:
        render_settings['layers_use_light'][layer] = layer.use_lights
    return render_settings

def set_render_settings(target_gp_object):
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.fps = 24                                 # Set frames per second
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True
    #target_gp_object.data.use_lights = False
    for layer in target_gp_object.data.layers:
        layer.use_lights = False
    

def restore_render_settings(render_settings, target_gp_object):
    bpy.context.scene.render.image_settings.file_format = render_settings.get('file_format')
    bpy.context.scene.render.ffmpeg.format =  render_settings.get('ffmpeg_format')
    bpy.context.scene.render.ffmpeg.codec = render_settings.get('ffmpeg_codec')
    bpy.context.scene.render.fps = render_settings.get('fps')
    bpy.context.scene.render.image_settings.color_mode = render_settings.get('color_mode')
    bpy.context.scene.render.film_transparent = render_settings.get('film_transparent')
    #target_gp_object.data.use_lights = render_settings.get('use_lights')
    for layer in target_gp_object.data.layers:
        layer.use_lights = render_settings['layers_use_light'][layer]
    
def get_render_aspect_ratio():
    return bpy.context.scene.render.resolution_x / bpy.context.scene.render.resolution_y

def update_view_layer():
    bpy.context.view_layer.update()

# UTILS

def align_object_to_vector(object, normal):
    """
    Align the Z axiz of the given object to the vector.
    
    Args:
        - plane: The plane object to align.
        - target_plane_data: Dictionary containing 'centroid', 'normal', and 'axes' from the dominant plane.
        - camera: The active camera in the scene for view direction.
    """
    # Step 1: Stabilize the axes
    world_up = mathutils.Vector((0, 0, 1))  # World Z-axis as the 'up' direction
    
    # Calculate the new X-axis: perpendicular to the normal and the world-up vector
    x_axis = world_up.cross(normal).normalized()
    
    # If the normal is nearly parallel to the world_up vector, use a fallback
    if x_axis.length < 1e-6:
        fallback_up = mathutils.Vector((1, 0, 0))  # Use world X-axis as fallback
        x_axis = fallback_up.cross(normal).normalized()
    
    # Calculate the new Y-axis: orthogonal to both the normal and the new X-axis
    y_axis = normal.cross(x_axis).normalized()

    # Step 2: Build a rotation matrix
    rotation_matrix = mathutils.Matrix((x_axis, y_axis, normal)).transposed()

    # Step 3: Apply rotation and location to the plane using quaternions
    object.rotation_euler = rotation_matrix.to_euler()

    update_view_layer()

def get_euler_rotation_in_degrees(obj):
    """
    Get the Euler rotation in degrees for the current frame, matching the Blender UI.
    Args:
        - obj: The object to evaluate.
    Returns:
        - A tuple (x, y, z) representing the rotation in degrees.
    """
    # Get the evaluated dependency graph
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = obj.evaluated_get(depsgraph)

    # Extract the world matrix
    world_matrix = evaluated_obj.matrix_world

    # Extract rotation as Euler
    rotation_euler = world_matrix.to_euler('XYZ')  # Ensure consistent order

    # Convert to degrees
    rotation_x_deg = degrees(rotation_euler.x)
    rotation_y_deg = degrees(rotation_euler.y)
    rotation_z_deg = degrees(rotation_euler.z)

    return [rotation_x_deg, rotation_y_deg, rotation_z_deg]

def insert_loc_rot_scale_keyframe(target_object, frame_number):
    target_object.keyframe_insert(data_path="location", frame=frame_number)
    target_object.keyframe_insert(data_path="rotation_euler", frame=frame_number)
    target_object.keyframe_insert(data_path="scale", frame=frame_number)


class MeshPlaneFromGreasePencilObjectFactory(object):

    def __init__(self, grease_pencil_object):
        self.gp_object = grease_pencil_object
        self.gp_data = GreasePencilObjectData(self.gp_object)
        self.scene_camera = get_active_camera()
        self.gp_camera = create_tmp_camera(name='{}_gp_tmp'.format(self.scene_camera.name))
        self.plane_object = self.create_gp_preview_plane('{}_mesh'.format(self.gp_object.name))
        self.plane_data = {}
        self.frame_list = []

    def create_gp_preview_plane(self, name):
        plane_object = create_plane_with_name(name)
        plane_object.name = '{}'.format(name)
        insert_plane_in_gp_object_collections(plane_object, self.gp_object)

        return plane_object


    def create_mesh_plane(self):
        '''
        
        '''
        output_directory = get_output_directory()
        frames_output_path = os.path.join(output_directory, self.gp_object.name)

        render_visibility_data = get_render_visibility_data()
        hide_everything_except_object(self.gp_object)

        render_settings = collect_render_settings(self.gp_object)

        set_render_settings(self.gp_object)

        (frame_start, frame_end) = get_start_and_end_frames()
        #frame_start = 1 
        #frame_end = frame_start

        for frame_number in range(frame_start, frame_end + 1):
            set_current_frame(frame_number)

            self.gp_data.calculate_grease_pencil_data_for_frame(frame_number)
            #self.gp_data.visualize_gp_bounding_box(frame_number)

            self.set_gp_camera(frame_number)

            frame_output_path = '{}_{}.png'.format(frames_output_path, str(frame_number).zfill(4))
            self.frame_list.append(frame_output_path)

            export_single_gp_object_animation(self.gp_camera, frame_output_path)

            self.transform_preview_plane(frame_output_path, frame_number)

        self.set_gp_preview_plane_material()

        restore_camera(self.scene_camera, self.gp_camera)

        restore_hidden_objects(render_visibility_data)

        restore_render_settings(render_settings, gp_object)

        #gp_object.hide_viewport = True
        #gp_object.hide_render = True 


    def set_gp_preview_plane_material(self):
        # Create material
        mat = create_material(name='{}_preview_mat'.format(self.gp_object.name))
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
        tex.image_user.frame_duration = len(self.frame_list)
        
        # Load first image of sequence
        tex.image = bpy.data.images.load(self.frame_list[0])
        tex.image.alpha_mode = 'STRAIGHT'
        tex.image.source = 'SEQUENCE'
        
        # Connect nodes
        links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(tex.outputs['Alpha'], bsdf.inputs['Alpha'])
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        # Assign material to plane
        self.plane_object.data.materials.append(mat)

    def set_gp_camera(self, frame_number):
        '''
        Create a new camera based on the active camera to frame the given Grease Pencil object tightly.
        Args:
            - gp_object: Grease Pencil object to be framed.
        Returns:
            - New camera framing the Grease Pencil object.
        '''

        # Ensure the aspect ratio matches the scene camera's
        self.gp_camera.data.lens = self.scene_camera.data.lens
        self.gp_camera.data.sensor_width = self.scene_camera.data.sensor_width
        self.gp_camera.data.sensor_height = self.scene_camera.data.sensor_height

        # Align the camera to the bounding box dominant plane.
        self.gp_camera.location = self.gp_data.bbox.get(frame_number).get('center')

        normal = self.gp_data.dplane.get(frame_number).get('normal')
        align_object_to_vector(self.gp_camera, normal)

        # Compute view data:
        bbox_y = self.gp_data.bbox.get(frame_number).get('dimensions').y
        bbox_x = self.gp_data.bbox.get(frame_number).get('dimensions').x
        gp_object_width = (bbox_y**2 + bbox_x**2)**0.5
        gp_object_height = self.gp_data.bbox.get(frame_number).get('dimensions').z

        # Compute required distance for the camera to fit the GP object
        horizontal_fov = self.gp_camera.data.angle_x
        horizontal_distance = gp_object_width / (2 * tan(horizontal_fov / 2))
        vertical_fov = self.gp_camera.data.angle_y
        vertical_distance = gp_object_height / (2 * tan(vertical_fov / 2))
        required_distance = max(vertical_distance, horizontal_distance)
        
        # Move the camera back along its local Z-axis
        #camera_direction = gp_camera.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))
        self.gp_camera.location += normal * required_distance

        # NOTE: This should suffice location-wise, but for some reason sometimes 
        # the camera view doesn't match the GP object. We are using the built-in function now, 
        # hoping that, being very close to the final destination, it will give a good result:
        frame_object(self.gp_object, self.gp_camera)

        self.ensure_tmp_camera_faces_gp_object(frame_number)

        insert_loc_rot_scale_keyframe(self.gp_camera, frame_number)

        # Restore scene camera 
        set_scene_camera(self.scene_camera)  


    def ensure_tmp_camera_faces_gp_object(self, frame_number):
        """
        Ensures the camera is facing the GP object's center.
        
        Args:
            - camera: The camera object.
            - gp_center: The world coordinates of the GP object's center.
        """
        # Camera's forward direction (local Z-axis in world coordinates)
        camera_forward = self.gp_camera.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))

        # Vector pointing from the camera to the GP object's center
        object_center = self.gp_data.bbox.get(frame_number).get('center')
        to_gp_center = (object_center - self.gp_camera.location).normalized()

        # Check if the camera is facing the GP object
        if camera_forward.dot(to_gp_center) < 0:
            # Rotate the camera 180° around its local Y-axis to flip it
            self.gp_camera.rotation_euler.rotate_axis("Y", radians(180))
            print("Camera orientation adjusted to face the GP object.")

    def transform_preview_plane(self, frame_path, frame_number):
        '''
        For the given plane we want to:
            - Have the same size as the Grease PEncil object.
            - Be located at the same place as the Grease Pencil object.
            - Face the active camera.
        '''
        align_object_to_vector(self.plane_object, self.gp_data.dplane.get(frame_number).get('normal'))
        #self.fix_orientation_flickering(frame_number)

        # Set plane aspect ratio:
        (width, height) = get_frame_size(frame_path)
        self.plane_object.scale.y = 1
        self.plane_object.scale.x = width/height

        # Grease Pencil object size:
        gp_obj_dimensions = self.gp_data.bbox.get(frame_number).get('dimensions')
        gp_width = (gp_obj_dimensions.y**2 + gp_obj_dimensions.x**2)**0.5
        #gp_width = max(gp_obj_dimensions.y, gp_obj_dimensions.x)
        gp_height = gp_obj_dimensions.z
        gp_aspect_ratio = gp_width/gp_height
        
        # Plane object size:
        plane_width = self.plane_object.scale.x
        plane_height = self.plane_object.scale.y
        pln_aspect_ratio = plane_width/plane_height 
        
        if pln_aspect_ratio < gp_aspect_ratio or (gp_height * pln_aspect_ratio) < gp_width:
            self.plane_object.scale.x = gp_width
            self.plane_object.scale.y = self.plane_object.scale.x / pln_aspect_ratio
        else:
            self.plane_object.scale.y = gp_height
            self.plane_object.scale.x = self.plane_object.scale.y * pln_aspect_ratio 

        # Set new Plane object location:
        gp_obj_center = self.gp_data.bbox.get(frame_number).get('center')
        self.plane_object.location = gp_obj_center

        # Set key frame for location, rotation and scale:
        insert_loc_rot_scale_keyframe(self.plane_object, frame_number)

        self.plane_data[frame_number] = get_euler_rotation_in_degrees(self.plane_object)


    def fix_orientation_flickering(self, frame_number):
        '''
        Ocassionally when appliying the rotation matrix the plane might be flipped. 
        Here we check if there was flip and we correct it in that case.
        '''
        if not frame_number-1 in self.plane_data:
            return

        [current_x_degrees, current_y_degrees, current_z_degrees] = get_euler_rotation_in_degrees(self.plane_object)
        [prev_x_degrees, prev_y_degrees, prev_z_degrees] = self.plane_data.get(frame_number-1)

        delta_z = abs(current_z_degrees - prev_z_degrees)

        if 179 <= delta_z  <= 270:
            print('Flip detected. Correcting')
            self.plane_object.rotation_euler.z += radians(180)
            #plane_obj.keyframe_insert(data_path="rotation_euler", frame=frame_number)

    

class GreasePencilObjectData(object):
    '''
    This class gets a Grease Pencil object and stores a dictionary with the following structure:
    {
        frame_number(int): {
            'bbox': {
                'dimensions': (vector)  - Dimensions in the x, y, z axis of the bounding box of the GP object
                'center': (vector)  - Center of the bounding box.
            }
            'dplane':{
                'centroid': (vector) - Centroid of the cominant plane of the Grease Pencil object
                'normal': (vector) - Normal vector of the dominant plane of the GP object.
                'axes': (tuple of two vectors) - Vecotrs aligned with the other two axes.
            }
            
        }
    '''
    def __init__(self, grease_pencil_object):
        self.gp_object = grease_pencil_object
        self.bbox = {}
        self.dplane = {}

    def _calculate_bounding_box_data_at_frame(self, frame_number):
        # Initialize min/max coordinates
        min_coords = mathutils.Vector((float('inf'), float('inf'), float('inf')))
        max_coords = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
        
        points = []

        # Iterate through all strokes
        gpencil_data = self.gp_object.data
        for layer in gpencil_data.layers:
            if not layer.hide:  # Skip hidden layers
                for frame_number_check in range(frame_number, 0, -1):  # Decreasing range
                    for frame in layer.frames:
                        if frame.frame_number == frame_number_check:
                            for stroke in frame.strokes:
                                for point in stroke.points:
                                    # Get the point position in world space
                                    local_pos = point.co
                                    world_pos = self.gp_object.matrix_world @ local_pos

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

        # Calculate bounding box dimensions
        dimensions = max_coords - min_coords
        
        # Calculate bounding bond center:
        bbox_center = (min_coords + max_coords) / 2

        # Save data
        self.bbox[frame_number]['dimensions'] = dimensions
        self.bbox[frame_number]['center'] = bbox_center

        return points
    
    def _calculate_dominant_plane_data_at_frame(self, frame_number, points):
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

            # Save data
            self.dplane[frame_number]['centroid'] = centroid
            self.dplane[frame_number]['normal'] = normal
            self.dplane[frame_number]['axes'] = axes

    def calculate_grease_pencil_data_for_frame(self, frame_number):
        '''
        Calculate the bounding box dimesions (W, H, D) of the given Grease Pencil object,
        as well the dominant plane data () 
        '''
        self.bbox[frame_number] = {}
        points = self._calculate_bounding_box_data_at_frame(frame_number)
        self.dplane[frame_number] = {}
        self._calculate_dominant_plane_data_at_frame(frame_number, points)

    def visualize_gp_bounding_box(self, frame_number):
        """
        Visualize the center and bounding box of a Grease Pencil object.
        Args:
            - gp_object: The Grease Pencil object.
            - bbox_data: Dictionary containing 'center' and 'dimensions' of the bounding box.
            - wireframe: Whether to set the bounding box cube to wireframe mode (default: True).
        """
        center = self.bbox.get(frame_number).get('center')
        dimensions = self.bbox.get(frame_number).get('dimensions')

        # Sphere for the center visualization
        sphere_name = '{}_center_sphere'.format(gp_object.name)
        create_sphere(sphere_name, center, frame_number)

        # Cube for the bounding box visualization
        cube_name = '{}_bounding_box'.format(gp_object.name)
        create_cube(cube_name, center, dimensions, frame_number)



if __name__ == '__main__':
    gp_object = get_selected_gp_object()
    plane_from_gp_factory = MeshPlaneFromGreasePencilObjectFactory(gp_object)
    plane_from_gp_factory.create_mesh_plane()
