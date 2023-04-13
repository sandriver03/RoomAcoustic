import slab
import numpy as np
from copy import deepcopy


def pick_hrtfs(source_locs, HRTF_dataset, interp=False):
    """
    pick HRTFs from a database (e.g. KEMAR) according to [azim, elev] positions
    TODO: assuming sources at distance of 1.4 m; maybe there is a better way, or HRTF dataset at different distances
    :param source_locs: n-by-2 or n-by-3 np array, [azim, elev, dist]
    :param HRTF_dataset: e.g. slab.HRTF.kemar()
    :param interp: if use interpolation if exact match not found
    :return: indexes of corresponding HRTFs in the database
    """
    # enforce distance to be 1.4 m
    locs = deepcopy(source_locs)
    if source_locs.shape[1] == 2:
        dist = 1.4 * np.ones((source_locs.shape[0], 1))
        locs = np.hstack([source_locs, dist])
    elif source_locs.shape[1] == 3:
        locs[:, 2] = 1.4
    else:
        raise ValueError('data shape of source locations not recognized')

        # get the hrtf indexes
    hrtf_idx = []
    for pos in locs:
        # try find the exact match
        pos_tc = np.array(pos, dtype=HRTF_dataset.sources.cartesian.dtype)
        idx = np.where((HRTF_dataset.sources.vertical_polar == pos_tc).all(axis=1))[0]
        if idx.size > 0:
            # find match
            hrtf_idx.append(idx[0])
        else:
            # first get nearest neighbor
            cart_allpos = HRTF_dataset.sources.cartesian
            cart_target = HRTF_dataset._vertical_polar_to_cartesian(np.array(pos).reshape(-1, 3))
            distances = np.sqrt(((cart_target - cart_allpos) ** 2).sum(axis=1))
            idx_nearest = np.argmin(distances)
            hrtf_idx.append(idx_nearest)

    return hrtf_idx


def walltrns(HRIR, total_order, floor_order, sample_rate=44100.):
    """
    transforms HRIR simulating a wall reflection; assuming simple model -2dB per wall
    reflection, independant of frequency
    # TODO: maybe get a database of different wall/floor coefficients
    # TODO: ceiling reflection seems is ignored
    :param HRIR: original head-related impulse response catalog (anechoic environment)
    :param total_order: order of total reflections
    :param floor_order: order of floor reflections
    :param sample_rate: HRIR sample rate, Hz
    :return:
    """
    floorcoef = np.array([0.6921,    0.0523,    0.0612,    0.0020,    0.0071,    0.0071,    0.0162,
                          0.0176,    0.0187,    0.0152,    0.0117,    0.0075,    0.0045,    0.0025,
                          0.0014,    0.0008,    0.0004,    0.0002,    0.0001,   -0.0000,   -0.0001,
                          -0.0002,   -0.0002,   -0.0003,   -0.0003,   -0.0003,  -0.0003,   -0.0002,
                          -0.0002,   -0.0001,   -0.0001,   -0.0001,   -0.0001,  -0.0001,   -0.0001,
                          -0.0001,   -0.0001,   -0.0001,   -0.0000,   -0.0000,   0.0000,    0.0000,
                          0.0000,    0.0000,    0.0001,    0.0001,    0.0001,    0.0001,    0.0001,
                          0.0001,    0.0001,    0.0001,    0.0001,    0.0001,    0.0001,    0.0001,
                          0.0001,    0.0001,    0.0001,    0.0001,    0.0001,    0.0001,    0.0001,
                          0.0001,    0.0001,    0.0001,    0.0001,    0.0001])

    wall_coef = np.array([0.2655,    0.3718,    0.0672,   -0.0008,    0.0259,    0.0207,    0.0195,
                          0.0141,    0.0120,    0.0090,    0.0082,    0.0068,    0.0064,    0.0057,
                          0.0051,    0.0046,    0.0041,    0.0037,    0.0033,    0.0030,    0.0027,
                          0.0024,    0.0022,    0.0019,    0.0017,    0.0016,    0.0014,    0.0013,
                          0.0012,    0.0010,    0.0009,    0.0009,    0.0008,    0.0007,    0.0006,
                          0.0006,    0.0005,    0.0005,    0.0004,    0.0004,    0.0004,    0.0003,
                          0.0003,    0.0003,    0.0002,    0.0002,    0.0002,    0.0002,    0.0001,
                          0.0001,    0.0001])

    HRIR_data = deepcopy(HRIR)
    HRIR_l = HRIR_data[:, 0]
    HRIR_r = HRIR_data[:, 1]

    for n in range(total_order - floor_order):
        HRIR_l = np.convolve(HRIR_l, wall_coef)
        HRIR_r = np.convolve(HRIR_r, wall_coef)
    for n in range(floor_order):
        HRIR_l = np.convolve(HRIR_l, floorcoef)
        HRIR_r = np.convolve(HRIR_r, floorcoef)

    return HRIR_data


def reverb_time(room_size, absorp_coefs=(0.1, ), sound_speed=344):
    """
    use Sabine formula to calculate reverberation time for a given room
    assuming it is the same across all frequencies
    :param room_size: [x, y, z] in meters
    :param absorp_coefs: absorption coefficients; [wall, [floor, [ceiling]]]; m^-2
    :param sound_speed: m/s
    :return: reverberation time, s
    """
    vol = room_size[0] * room_size[1] * room_size[2]
    wall_surface = (room_size[0] * room_size[2]) * 2 + (room_size[1] * room_size[2]) * 2
    floor_surface = room_size[0] * room_size[1]
    ceil_surface = floor_surface
    # calculate sabins
    if len(absorp_coefs) == 3:
        sa = wall_surface * absorp_coefs[0] + floor_surface * absorp_coefs[1] + \
             ceil_surface * absorp_coefs[2]
    elif len(absorp_coefs) == 2:
        sa = wall_surface * absorp_coefs[0] + 2 * floor_surface * absorp_coefs[1]
    elif len(absorp_coefs) == 1:
        sa = (wall_surface + 2 * floor_surface) * absorp_coefs[0]
    else:
        raise ValueError("absorption coefficients not understood")

    # sabine formula
    return 24 * np.log(10) / sound_speed * vol / sa


def filtnoise(noise_data, sample_rate):
    """
    generate log-powered scaled noise, both low-pass and high-pass
    :param noise_data: white noise data, assume 1d
    :param sample_rate: Hz
    :return: 2 copy of noise_data, high- and low- passed
    """
    # fft params
    fft_length = 2 ** (len(noise_data) - 1).bit_length()
    f = sample_rate * (np.arange(fft_length / 2) + 1) / fft_length
    fr = f[::-1]
    ff = np.hstack([f, fr])

    # log-powered filter in frequency domain
    s = (np.log2(ff) - np.log2(ff[0])) / (np.log2(ff[int(len(ff)/2)]) - np.log2(ff[0]))

    # convolution (multiplication in frequency domain)
    nd_fft = np.fft.fft(noise_data, fft_length)
    ndf_high = np.fft.ifft(nd_fft * s).real[:len(noise_data)]
    ndf_low = np.fft.ifft(nd_fft * (1 - s)).real[:len(noise_data)]

    return ndf_low, ndf_high


def reverb_tail(t_reverb, sample_rate):
    """
    generates reverberation tail y from noise sample
    :param t_reverb: reverberation time, s
    :param sample_rate: Hz
    :return:
    """
    # TODO: where are those coming from?
    RTlow, RThigh = 0.8 * t_reverb, t_reverb

    # Determine reverberation envelope
    RT = max(RTlow, RThigh)
    t = np.arange(RT * sample_rate * 2) / sample_rate
    envLow = np.exp(-t/RTlow * 60/20 * np.log(10))
    envHigh = np.exp(-t/RThigh * 60/20 * np.log(10))
    x = np.random.randn(len(envHigh), 2)

    # generate filtered noise
    xl1, xh1 = filtnoise(x[:, 0], sample_rate)
    xl2, xh2 = filtnoise(x[:, 1], sample_rate)

    # shape noise with envelope
    y = np.zeros((len(envHigh), 2))
    y[:, 0] = xl1 * envLow
    y[:, 1] = xl2 * envLow
    y[:, 0] = y[:, 0] + xh1 * envHigh
    y[:, 1] = y[:, 1] + xh2 * envHigh

    return y


def echoHRTF(room_size, source_locs, total_orders, floor_orders,
             sample_rate=44100, sound_velocity=344):
    """
    calculate HRIR/HRTF for the binaural room response
    TODO: add some kwargs to control the parameters for the funtion reverb_time
    TODO: right now slab.HRTF.kemar() is used as the only HRTF dataset. could make the HRTF dataset as an input so it can be easily modified
    :param room_size: [x, y, z] in meters
    :param source_locs: see simRoom
    :param total_orders: see simRoom
    :param floor_orders: see simRoom
    :param sample_rate: Hz
    :param sound_velocity: m/s
    :return: HRIR for given source, room and listener position
    """
    # source location, orders are from simRoom
    delays = source_locs[:, 2] / sound_velocity
    r0 = min(delays)
    a = max(delays) * sample_rate

    # get HRIRs from slab
    HRTFs = slab.HRTF.kemar()
    assert HRTFs.samplerate == sample_rate, "HRTFs sample rate need to match specified sample rate"
    filter_taps = HRTFs.data[0].n_taps

    echo_HRIR = np.zeros((int(filter_taps + a + 1000), 2))
    # construct echo HRIR from chosen sources
    # pick corresponding HRTF
    hrtf_idx = pick_hrtfs(source_locs[:, :2], HRTFs)
    for idx, sr in enumerate(source_locs):
        onset = int(max(np.ceil(delays[idx] * sample_rate), 1))
        # calculate attenuated HRIRs
        transHRIR = walltrns(HRTFs[hrtf_idx[idx]].data, total_orders[idx],
                             floor_orders[idx], sample_rate)
        # offset
        transHRIR_length = len(transHRIR[:, 1])
        offset = onset + transHRIR_length - 1
        # update total HRIRs
        echo_HRIR[onset:(onset + transHRIR.shape[0]), :] = \
            echo_HRIR[onset:(onset + transHRIR.shape[0]), :] + transHRIR * (r0 / delays[idx])

    # simulate reverb tail
    # calculate reverb time from room size
    t_reverb = reverb_time(room_size)
    y = reverb_tail(t_reverb, sample_rate)
    peak = np.max(np.abs(transHRIR * (r0 / delays[-1])))

    # combine calculated sources and simulated tail
    final_HRIR = np.zeros((onset + len(y), 2))
    final_HRIR[onset:, :] = y * peak
    final_HRIR[:offset] = final_HRIR[:offset] + echo_HRIR[:offset]

    return final_HRIR


if __name__ == '__main__':
    from simRoom import simRoom
    from scipy.io import wavfile
    import slab

    # scenario
    roomdim = [4, 3, 3]
    listener_loc = [1, 1, 1.5]
    speaker_di = [45, 0, 1.4]   # [azim, elev, dist] w.r.t. listener
    wallorder = 2

    # image source method
    source_pos, order, floor_order, ceil_order = \
        simRoom(roomdim, listener_loc, speaker_di, wallorder)

    # calculate HRIR
    HRIR = echoHRTF(roomdim, source_pos, order, floor_order)

    # testing
    wav_file = 'f_potatos.wav'
    fs, sig = wavfile.read(wav_file)

    bsig = slab.Binaural(sig, fs)
    filt = slab.Filter(HRIR, fs)
    # filter is too large, try only part of it
    # TODO: slab filter implementation need to be modified to accomandate this
    filt.data = filt.data[:15000]
    res = filt.apply(bsig)
