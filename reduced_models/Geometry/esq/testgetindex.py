import numpy as np


def getindex(mesh, value, spacing):
    """Find index in mesh for or closest to specified value

    Function finds index corresponding closest to 'valu' in 'mesh'. The spacing
    parameter should be enough to the range [value-spacing, value+spacing]
    captures values in the mesh.

    Parameters
    ----------
    mesh : ndarray
        1D array that will be used to find entry closest to value
    value : float
        This is the number that is searched for in mesh.
    spacing : float
        Dictates the range of values that will fall into the region holding the
        desired value in mesh.

    Returns
    -------
    index : int
        Index for value in mesh i.e. mesh[index] = value. This will be the
        closest entry to value.
    """

    # Check if value is already in mesh
    if value in mesh:
        return np.where(mesh == value)[0][0]
    else:
        pass

    # Create array of possible indices
    indices = np.where((mesh > (value - spacing)) & (mesh < (value + spacing)))[0]

    # Compute differences of the indexed mesh-value with desired value
    difference = []
    for index in indices:
        diff = np.sqrt((mesh[index] ** 2 - value**2) ** 2)
        difference.append(diff)

    # Smallest element will be the index closest to value in indices
    i = np.argmin(difference)
    index = indices[i]

    return index


spacing = 0.1
mylist = [i * spacing for i in range(1, 21, 1)]
correct_index = 3

smallest_value = min(mylist) - abs(max(mylist))
mylist[correct_index] = smallest_value
mesh = np.array(mylist)
assert np.argmin(mesh) == 3

index = getindex(mesh, smallest_value, 2 * spacing)

print(index == correct_index)
