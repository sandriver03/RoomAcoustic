import numpy
import itertools


def roomImag(roomdim, sourcecoor, wallorder):
    """
    use image source method to simulation reflection sources
    :param roomdim: dimensions of the room walls (list: length, width, height)
    :param sourcecoor: coordinates of the source (list: x, y, z) measured from the bottom,
                       front, left corner of the room
    :param wallorder: order of reflections to consider
    :return:
        order       : order of the reflections
        floor_order : order of the floor reflections
        ceil_order  : order of the ceiling reflections
        coordinates : coordinates of the mirror sources (list: x, y, z, order of the reflections)
    """
    if isinstance(roomdim, numpy.ndarray) and len(roomdim.shape) == 2:
        roomdim = roomdim.ravel()
    if isinstance(sourcecoor, numpy.ndarray) and len(sourcecoor.shape) == 2:
        sourcecoor = sourcecoor.ravel()
    Lx, Ly, Lz = roomdim
    Sx, Sy, Sz = sourcecoor
    Ix = []
    Iy = []
    Iz = []
    order = []
    floor_order = []
    ceil_order = []
    # positions of images
    # for (l, m, n) in itertools.product(range(-wallorder,wallorder+1), repeat=3):
    for l in range(-wallorder, wallorder+1):
        for m in range(-wallorder, wallorder+1):
            for n in range(-wallorder, wallorder+1):
                # x-dimension
                Ix.append(Sx + 2 * l * Lx)
                Ix.append(Sx + 2 * l * Lx)
                Ix.append(Sx + 2 * l * Lx)
                Ix.append(Sx + 2 * l * Lx)
                Ix.append(-Sx + 2 * l * Lx)
                Ix.append(-Sx + 2 * l * Lx)
                Ix.append(-Sx + 2 * l * Lx)
                Ix.append(-Sx + 2 * l * Lx)
                # y-dimension
                Iy.append(Sy + 2 * m * Ly)
                Iy.append(Sy + 2 * m * Ly)
                Iy.append(-Sy + 2 * m * Ly)
                Iy.append(-Sy + 2 * m * Ly)
                Iy.append(Sy + 2 * m * Ly)
                Iy.append(Sy + 2 * m * Ly)
                Iy.append(-Sy + 2 * m * Ly)
                Iy.append(-Sy + 2 * m * Ly)
                # z-dimension
                Iz.append(Sz + 2 * n * Lz)
                Iz.append(-Sz + 2 * n * Lz)
                Iz.append(Sz + 2 * n * Lz)
                Iz.append(-Sz + 2 * n * Lz)
                Iz.append(Sz + 2 * n * Lz)
                Iz.append(-Sz + 2 * n * Lz)
                Iz.append(Sz + 2 * n * Lz)
                Iz.append(-Sz + 2 * n * Lz)

    for x, y, z in zip(Ix, Iy, Iz):
        if x >= 0:
            n_order = numpy.floor(x/Lx)
        else:
            n_order = numpy.ceil(-x/Lx)
        if y >= 0:
            n_order = n_order + numpy.floor(y/Ly)
        else:
            n_order = n_order + numpy.ceil(-y/Ly)
        if z >= 0:
            n_order = n_order + numpy.floor(z/Lz)
            n_floor_order = numpy.floor(numpy.floor(z/Lz)/2)
            n_ceil_order = numpy.ceil(numpy.floor(z/Lz)/2)
        else:
            n_order = n_order + numpy.ceil(-z/Lz)
            n_floor_order = numpy.ceil(numpy.ceil(-z/Lz)/2)
            n_ceil_order = numpy.floor(numpy.ceil(-z/Lz)/2)
        order.append(int(n_order))
        floor_order.append(int(n_floor_order))
        ceil_order.append(int(n_ceil_order))

    coordinates = numpy.stack([Ix, Iy, Iz], axis=-1)
    return order, floor_order, ceil_order, coordinates


if __name__ == '__main__':
    roomdim = [4, 3, 2]
    sourcecoor = [1, 1, 2]
    wallorder = 1
    order, floor_order, ceil_order, coordinates = roomImag(roomdim, sourcecoor, wallorder)
    print(coordinates)

