import numba
import numpy as np
import awkward as ak


def awkwardify_df(df):
    if 'evtid' in df.keys():
        df_eventid_key = 'evtid'
    else:
        df_eventid_key = 'eventid'

    _, evt_offsets = np.unique(df[df_eventid_key], return_counts=True)

    dictionary = {"x": reshape_awkward(df["x"].values, evt_offsets),
                  "y": reshape_awkward(df["y"].values, evt_offsets),
                  "z": reshape_awkward(df["z"].values, evt_offsets),
                  "x_pri": reshape_awkward(df["x_pri"].values, evt_offsets),
                  "y_pri": reshape_awkward(df["y_pri"].values, evt_offsets),
                  "z_pri": reshape_awkward(df["z_pri"].values, evt_offsets),

                  "t": reshape_awkward(df["t"].values, evt_offsets),
                  "ed": reshape_awkward(df["ed"].values, evt_offsets),
                  "PreStepEnergy": reshape_awkward(df["PreStepEnergy"].values, evt_offsets),
                  "PostStepEnergy": reshape_awkward(df["PostStepEnergy"].values, evt_offsets),
                  "type": reshape_awkward(np.array(df["type"], dtype=str), evt_offsets),
                  "trackid": reshape_awkward(np.array(df["trackid"].values, dtype=int), evt_offsets),
                  "parenttype": reshape_awkward(np.array(df["parenttype"], dtype=str), evt_offsets),
                  "parentid": reshape_awkward(np.array(df["parentid"].values, dtype=int), evt_offsets),
                  "creaproc": reshape_awkward(np.array(df["creaproc"], dtype=str), evt_offsets),
                  "edproc": reshape_awkward(np.array(df["edproc"], dtype=str), evt_offsets),
                  "evtid": reshape_awkward(np.array(df[df_eventid_key].values, dtype=int), evt_offsets),
                  }

    if 'r' in df.keys():
        dictionary["r"] = reshape_awkward(df["r"].values, evt_offsets)

    return ak.Array(dictionary)


def reshape_awkward(array, offset):
    """
    Function which reshapes an array of strings or numbers according
    to a list of offsets. Only works for a single jagged layer.

    Args:
        array: Flatt array which should be jagged.
        offset: Length of subintervals


    Returns:
        res: awkward1.ArrayBuilder object.
    """
    res = ak.ArrayBuilder()
    if (array.dtype == int) or (array.dtype == np.float64) or (array.dtype == np.float32):
        _reshape_awkward_number(array, offset, res)
    else:
        _reshape_awkward_string(array, offset, res)
    return res.snapshot()


@numba.njit
def _reshape_awkward_number(array, offsets, res):
    """
    Function which reshapes an array of numbers according
    to a list of offsets. Only works for a single jagged layer.

    Args:
        array: Flatt array which should be jagged.
        offsets: Length of subintervals
        res: awkward1.ArrayBuilder object

    Returns: 
        res: awkward1.ArrayBuilder object
    """
    start = 0
    end = 0
    for o in offsets:
        end += o
        res.begin_list()
        for value in array[start:end]:
            res.real(value)
        res.end_list()
        start = end


def _reshape_awkward_string(array, offsets, res):
    """
    Function which reshapes an array of strings according
    to a list of offsets. Only works for a single jagged layer.

    Args:
        array: Flatt array which should be jagged.
        offsets: Length of subintervals
        res: awkward1.ArrayBuilder object

    Returns: 
        res: awkward1.ArrayBuilder object
    """
    start = 0
    end = 0
    for o in offsets:
        end += o
        res.begin_list()
        for value in array[start:end]:
            res.string(value)
        res.end_list()
        start = end


def awkward_to_flat_numpy(array):
    if len(array) == 0:
        return ak.to_numpy(array)
    return (ak.to_numpy(ak.flatten(array)))


@numba.njit
def mulit_range(offsets):
    res = np.zeros(np.sum(offsets), dtype=np.int32)
    i = 0
    for o in offsets:
        res[i:i + o] = np.arange(0, o, dtype=np.int32)
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
        res[i:i + o] = ind
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
        return ak.from_numpy(np.empty(0, dtype='int64'))
    return ak.num(array, **kwargs)


def calc_dt(result):
    """
    Calculate dt, the time difference from the initial data in the event
    With empty check
    :param result: Including `t` field
    :return dt: Array like
    """
    if len(result) == 0:
        return np.empty(0)
    dt = result['t'] - result['t'][:, 0]
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
