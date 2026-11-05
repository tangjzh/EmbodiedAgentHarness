# Code block 0
import numpy as np

# Pick the salad dressing
pick_success = pick_object("salad_dressing", z_approach=0.2, lift_height=0.3)

# Place it in the basket
if pick_success:
    place_object("basket", z_offset=0.3)

# Code block 1
import numpy as np

# Get 3D points of the salad dressing
res = get_object_3d_points_and_masks_from_language("salad dressing bottle")
points = res["points_3d"]
# Filter out points that are too close to the floor
points = points[points[:, 2] > 0.01]

if len(points) > 0:
    # Find the center using median to be robust to outliers
    center = np.median(points, axis=0)
    
    # Find the top of the bottle
    top_z = np.max(points[:, 2])
    
    # Grasp position: center x, center y, slightly below top z to grasp the neck/body
    grasp_pos = np.array([center[0], center[1], top_z - 0.05])
    
    # Top-down grasp orientation
    quat = np.array([0.0, 0.0, 1.0, 0.0])
    
    # Execute grasp
    open_gripper()
    goto_pose(grasp_pos, quat, z_approach=0.2)
    goto_pose(grasp_pos, quat)
    close_gripper()
    
    # Lift the object
    lift_pos = grasp_pos.copy()
    lift_pos[2] += 0.3
    goto_pose(lift_pos, quat)
    
    # Place in basket
    basket_pos, _ = get_object_pose("basket")
    if basket_pos is not None:
        place_pos = np.array([basket_pos[0], basket_pos[1], basket_pos[2] + 0.3])
        goto_pose(place_pos, quat)
        open_gripper()
        
        # Return to home position
        goto_home_joint_position()

# Code block 2
import numpy as np

def pick_and_place():
    # Try to get grasp pose for the salad dressing (black bottle)
    pos, quat = sample_grasp_pose("salad_dressing")
    if pos is None:
        pos, quat = sample_grasp_pose("black bottle")
        
    if pos is not None:
        open_gripper()
        goto_pose(pos, quat, z_approach=0.2)
        goto_pose(pos, quat)
        close_gripper()
        
        lift_pos = pos.copy()
        lift_pos[2] += 0.3
        goto_pose(lift_pos, quat)
        
        basket_pos, _ = get_object_pose("basket")
        if basket_pos is not None:
            place_pos = basket_pos.copy()
            place_pos[2] += 0.3
            goto_pose(place_pos, quat)
            open_gripper()
            return True
    return False

if not pick_and_place():
    # Fallback to 3D points and OBB if sample_grasp_pose fails
    res = get_object_3d_points_and_masks_from_language("black bottle")
    points = res["points_3d"]
    if len(points) > 0:
        # Filter out points too close to the floor
        points = points[points[:, 2] > 0.01]
        if len(points) > 0:
            obb = get_oriented_bounding_box_from_3d_points(points)
            center = obb["center"]
            R = obb["R"]
            extents = obb["extent"]
            
            # Find the longest axis of the bottle
            longest_idx = np.argmax(extents)
            longest_axis = R[:, longest_idx]
            
            # Construct grasp rotation:
            # Z-axis points down
            z_axis = np.array([0.0, 0.0, -1.0])
            # Y-axis (gripper closing direction) should be perpendicular to the bottle's longest axis
            y_axis = np.cross(z_axis, longest_axis)
            if np.linalg.norm(y_axis) > 1e-3:
                y_axis = y_axis / np.linalg.norm(y_axis)
            else:
                y_axis = np.array([0.0, 1.0, 0.0])
            x_axis = np.cross(y_axis, z_axis)
            
            grasp_R = np.column_stack([x_axis, y_axis, z_axis])
            quat = rot_to_quat(grasp_R)
            
            grasp_pos = center.copy()
            # Ensure we don't hit the floor
            grasp_pos[2] = max(0.03, grasp_pos[2])
            
            open_gripper()
            goto_pose(grasp_pos, quat, z_approach=0.2)
            goto_pose(grasp_pos, quat)
            close_gripper()
            
            lift_pos = grasp_pos.copy()
            lift_pos[2] += 0.3
            goto_pose(lift_pos, quat)
            
            basket_pos, _ = get_object_pose("basket")
            if basket_pos is not None:
                place_pos = basket_pos.copy()
                place_pos[2] += 0.3
                goto_pose(place_pos, quat)
                open_gripper()

goto_home_joint_position()

# Code block 3
import numpy as np

def pick_and_place(obj_name):
    res = get_object_3d_points_and_masks_from_language(obj_name)
    points = res["points_3d"]
    if len(points) > 0:
        points = points[points[:, 2] > 0.01]
        if len(points) > 0:
            center = np.median(points, axis=0)
            top_z = np.max(points[:, 2])
            
            # Grasp slightly below the top to get a good grip on the body
            grasp_pos = np.array([center[0], center[1], top_z - 0.06])
            quat = np.array([0.0, 0.0, 1.0, 0.0]) # Top-down grasp
            
            open_gripper()
            goto_pose(grasp_pos, quat, z_approach=0.2)
            goto_pose(grasp_pos, quat)
            close_gripper()
            
            lift_pos = grasp_pos.copy()
            lift_pos[2] += 0.3
            goto_pose(lift_pos, quat)
            
            basket_pos, _ = get_object_pose("basket")
            if basket_pos is not None:
                place_pos = basket_pos.copy()
                place_pos[2] += 0.3
                goto_pose(place_pos, quat)
                open_gripper()
                return True
    return False

# The salad dressing is likely the orange bottle
if not pick_and_place("orange bottle"):
    # Fallback to black bottle if orange bottle fails
    pick_and_place("black bottle")

goto_home_joint_position()

# Code block 4
import numpy as np

def pick_and_place(obj_name):
    res = get_object_3d_points_and_masks_from_language(obj_name)
    points = res["points_3d"]
    if len(points) > 0:
        points = points[points[:, 2] > 0.005]
        if len(points) > 0:
            obb = get_oriented_bounding_box_from_3d_points(points)
            center = obb["center"]
            R = obb["R"]
            extents = obb["extent"]
            
            # Find the longest axis of the bottle
            longest_idx = np.argmax(extents)
            longest_axis = R[:, longest_idx]
            
            # Construct grasp rotation:
            # Z-axis points down
            z_axis = np.array([0.0, 0.0, -1.0])
            # Y-axis (gripper closing direction) should be perpendicular to the bottle's longest axis
            y_axis = np.cross(z_axis, longest_axis)
            if np.linalg.norm(y_axis) > 1e-3:
                y_axis = y_axis / np.linalg.norm(y_axis)
            else:
                y_axis = np.array([0.0, 1.0, 0.0])
            x_axis = np.cross(y_axis, z_axis)
            
            grasp_R = np.column_stack([x_axis, y_axis, z_axis])
            quat = rot_to_quat(grasp_R)
            
            grasp_pos = center.copy()
            # Ensure we don't hit the floor
            grasp_pos[2] = max(0.02, grasp_pos[2])
            
            open_gripper()
            goto_pose(grasp_pos, quat, z_approach=0.2)
            goto_pose(grasp_pos, quat)
            close_gripper()
            
            lift_pos = grasp_pos.copy()
            lift_pos[2] += 0.3
            goto_pose(lift_pos, quat)
            
            basket_pos, _ = get_object_pose("basket")
            if basket_pos is not None:
                place_pos = basket_pos.copy()
                place_pos[2] += 0.3
                goto_pose(place_pos, quat)
                open_gripper()
                return True
    return False

if not pick_and_place("salad dressing"):
    pick_and_place("orange bottle")

goto_home_joint_position()

# Code block 5
import numpy as np

# Try to pick the orange bottle (salad dressing) using the built-in pick_object
success = pick_object("orange bottle", z_approach=0.2, lift_height=0.3)

if success:
    place_object("basket", z_offset=0.3)
else:
    # Fallback to manual grasp calculation using OBB
    res = get_object_3d_points_and_masks_from_language("orange bottle")
    points = res["points_3d"]
    if len(points) > 0:
        # Filter out points that are too close to the floor
        points = points[points[:, 2] > 0.005]
        if len(points) > 0:
            obb = get_oriented_bounding_box_from_3d_points(points)
            center = obb["center"]
            R = obb["R"]
            extents = obb["extent"]
            
            # Find the longest axis of the bottle
            longest_idx = np.argmax(extents)
            longest_axis = R[:, longest_idx]
            
            # Construct grasp rotation:
            # Z-axis points down
            z_axis = np.array([0.0, 0.0, -1.0])
            # Y-axis (gripper closing direction) should be perpendicular to the bottle's longest axis
            y_axis = np.cross(z_axis, longest_axis)
            if np.linalg.norm(y_axis) > 1e-3:
                y_axis = y_axis / np.linalg.norm(y_axis)
            else:
                y_axis = np.array([0.0, 1.0, 0.0])
            x_axis = np.cross(y_axis, z_axis)
            
            grasp_R = np.column_stack([x_axis, y_axis, z_axis])
            quat = rot_to_quat(grasp_R)
            
            grasp_pos = center.copy()
            # Ensure we don't hit the floor
            grasp_pos[2] = max(0.02, grasp_pos[2])
            
            open_gripper()
            goto_pose(grasp_pos, quat, z_approach=0.2)
            goto_pose(grasp_pos, quat)
            close_gripper()
            
            # Lift the object
            lift_pos = grasp_pos.copy()
            lift_pos[2] += 0.3
            goto_pose(lift_pos, quat)
            
            # Place in basket
            basket_pos, _ = get_object_pose("basket")
            if basket_pos is not None:
                place_pos = basket_pos.copy()
                place_pos[2] += 0.3
                goto_pose(place_pos, quat)
                open_gripper()

goto_home_joint_position()