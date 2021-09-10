import numba
import numpy as np
import awkward as ak


def reshape_awkward(array, offset):
    """
    Function which reshapes array according to a list of offsets. Only
    works for a single jagged layer.

    Args:
        array: Flatt array which should be jagged.
        offset: Length of subintervals


    Returns:
        res: awkward1.ArrayBuilder object.
    """
    res = ak.ArrayBuilder()
    _reshape_awkward(array, offset, res)
    return res.snapshot()


@numba.njit
def _reshape_awkward(array, offsets, res):
    start = 0
    end = 0
    for o in offsets:
        end += o
        res.begin_list()
        for value in array[start:end]:
            res.real(value)
        res.end_list()
        start = end


def awkward_to_flat_numpy(array):
    if len(array) == 0:
        return np.array([])
    return (ak.to_numpy(ak.flatten(array)))


@numba.njit
def mulit_range(offsets):
    res = np.zeros(np.sum(offsets), dtype=np.int32)
    i = 0
    for o in offsets:
        res[i:i+o] = np.arange(0, o, dtype=np.int32)
        i += o
    return res

@numba.njit
def offset_range(offsets):
    """
    Computes range of constant event ids while in same offset. E.g.
    for an array [1], [1,2,3], [5] this function yields [0, 1, 1, 1, 2].

    Args:
        offsets (ak.array): jagged array offsets.

    Returns:
        np.array: Indicies.
    """
    res = np.zeros(np.sum(offsets), dtype=np.int32)
    i = 0
    for ind, o in enumerate(offsets):
        res[i:i+o] = ind
        i += o
    return res


def ak_num(array, **kwargs):
    """
    awkward.num() wrapper also for work in empty array
    :param array: Data containing nested lists to count.
    :param kwargs: keywords arguments for awkward.num().
    :return: an array of integers specifying the number of elements
        at a particular level. If array is empty, return empty.
    """
    if len(array) == 0:
        return ak.from_numpy(np.array([], dtype='int64'))
    return ak.num(array, **kwargs)

def calc_dt(result):
    """
    Calculate dt, the time difference from the initial data in the event
    With empty check
    :param result: Including `t` field
    :return dt: Array like
    """
    if len(result) == 0:  # Empty
        return np.array([])
    dt = result['t'] - result['t'][:, 0]  # if result is empty, Error
    return dt


def apply_time_offset(result, dt):
    """
    Apply time offset with empty check
    :param result: np.structured_array
    :param dt: Array timing offset for each events
    :return: result with timing offsets for each events
    """
    if len(result) == 0:
        return result
    return result['t'][:, :] + dt
