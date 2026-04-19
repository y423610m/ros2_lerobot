import numpy as np

def rpy_to_matrix(roll, pitch, yaw):
    # Rotation matrices around each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # R = Rz * Ry * Rx
    return Rz @ Ry @ Rx


def matrix_to_rpy(R):
    # Extract pitch
    pitch = np.arcsin(-R[2, 0])

    # Handle numerical issues near singularities if needed
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw  = np.arctan2(R[1, 0], R[0, 0])

    return roll, pitch, yaw


def compose_rpy(rpy1, rpy2):
    R1 = rpy_to_matrix(*rpy1)
    R2 = rpy_to_matrix(*rpy2)

    # Apply R1 then R2
    R = R2 @ R1

    return matrix_to_rpy(R)


# Example
rpy1 = (0, 1.53, 0)
rpy2 = (-0.77, 0, 0)

result = compose_rpy(rpy1, rpy2)
print(result)