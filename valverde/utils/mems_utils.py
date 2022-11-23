import numpy as np


def getindex(mesh, value, spacing) -> float:
    """Grab index of specifed value within a mesh"""

    # Check if value is already in mesh
    if value in mesh:
        index = np.where(mesh == value)[0][0]

    else:
        index = np.argmin(abs(mesh - value))

    return index
