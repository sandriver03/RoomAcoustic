from roomimage import roomImag
import numpy as np


def _reshape_coors(coors):
    """
    convert coordinates into correct form
    :param coors:
    :return:
    """
    if not isinstance(coors, np.ndarray):
        coors = np.array(coors, dtype=np.float64)
    if len(coors.shape) == 1:
        coors = coors.reshape(1, -1)
    assert coors.shape[1] == 3, "coordinates must be n-by-3"
    return coors


def car2pol(cartesian):
    """
    converts data from cartesian coordinates into spatial hearing polar coordinates (vertical polar)
    :param cartesian: [x, y, z]
    :return: vertical polar [azim, elev, dist]
    """
    cartesian = _reshape_coors(cartesian)

    vertical_polar = np.zeros_like(cartesian)
    xy = cartesian[:, 0] ** 2 + cartesian[:, 1] ** 2
    vertical_polar[:, 0] = np.rad2deg(np.arctan2(cartesian[:, 1], cartesian[:, 0]))
    vertical_polar[vertical_polar[:, 0] < 0, 0] += 360
    vertical_polar[:, 1] = 90 - np.rad2deg(np.arctan2(np.sqrt(xy), cartesian[:, 2]))
    vertical_polar[:, 2] = np.sqrt(xy + cartesian[:, 2] ** 2)
    return vertical_polar


def pol2car(vertical_polar):
    """
    converts data from spatial hearing polar coordinates (vertical polar) into cartesian coordinates
    :param vertical_polar: vertical polar [azim, elev, dist]
    :return:
    """
    vertical_polar = _reshape_coors(vertical_polar)

    cartesian = np.zeros_like(vertical_polar)
    azimuths = np.deg2rad(vertical_polar[:, 0])
    elevations = np.deg2rad(90 - vertical_polar[:, 1])
    r = vertical_polar[:, 2].mean()  # get radii of sound sources
    cartesian[:, 0] = r * np.cos(azimuths) * np.sin(elevations)
    cartesian[:, 1] = r * np.sin(elevations) * np.sin(azimuths)
    cartesian[:, 2] = r * np.cos(elevations)
    return cartesian


def validate_speaker_loc(speaker_loc, room_size):
    """
    check if the speaker location makes sense
    currently check if the speaker is inside the room
    :param speaker_loc: rows of [x, y, z] in room coordinate
    :param room_size: [Lx, Ly, Lz]; room starts at [0, 0, 0]
    :return:
    """
    assert np.all(speaker_loc > 0) and np.all(speaker_loc < room_size), \
        "speaker must located inside the room"


def validate_listener_loc(listener_loc, room_size):
    """
    check if the listener location makes sense
    currently check if the listener is inside the room
    :param listener_loc: rows of [x, y, z] in room coordinate
    :param room_size: [Lx, Ly, Lz]; room starts at [0, 0, 0]
    :return:
    """
    assert np.all(listener_loc > 0) and np.all(listener_loc < room_size), \
        "listener must located inside the room"


def simRoom(room_size, listener_loc, speaker_direct, sim_order=2, source_num_max=50):
    # room always starts at [0, 0, 0] and goes to positive
    # listener location in [x, y, z]
    # speaker direction in polar coordinate relative to listener

    room_size = _reshape_coors(room_size)
    listener_loc = _reshape_coors(listener_loc)
    speaker_direct = _reshape_coors(speaker_direct)
    # validate listener loc
    validate_listener_loc(listener_loc, room_size)
    # convert speaker location to cartesian, with reference to room coordinate
    spk_cart = pol2car(speaker_direct) + listener_loc
    # make sure the speaker location actually makes sense
    validate_speaker_loc(spk_cart, room_size)
    # calculate source images
    order, floor_order, ceil_order, img_locs = roomImag(room_size, spk_cart, sim_order)

    # source images are in cartesian, with reference to room coordinate
    # transform it into vertical polar with reference to listener
    img_loc_pol = car2pol(img_locs - listener_loc)
    if source_num_max is None:
        source_num_max = len(order)

    # sort according to distance; get the index used to sort
    s_idx = np.argsort(img_loc_pol[:, 2])
    # choose N closest sources
    img_loc_pol = img_loc_pol[s_idx[:source_num_max]]
    order = np.array(order)[s_idx[:source_num_max]]
    floor_order = np.array(floor_order)[s_idx[:source_num_max]]
    ceil_order = np.array(ceil_order)[s_idx[:source_num_max]]

    return img_loc_pol, order, floor_order, ceil_order


if __name__ == '__main__':
    roomdim = [4, 3, 3]
    listener_loc = [1, 1, 1.5]
    speaker_di = [45, 0, 1.4]   # [azim, elev, dist]
    wallorder = 2
    source_pos, order, floor_order, ceil_order = \
        simRoom(roomdim, listener_loc, speaker_di, wallorder)
