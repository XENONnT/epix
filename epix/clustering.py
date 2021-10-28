import numpy as np
import pandas as pd
import numba
import awkward as ak
from .common import reshape_awkward
from sklearn.cluster import DBSCAN


def find_cluster(interactions, cluster_size_space, cluster_size_time):
    """
    Function which finds cluster within a event.

    Args:
        x (pandas.DataFrame): Subentries of event must contain the
            fields, x,y,z,time
        cluster_size_space (float): Max spatial distance between two points to
            be inside a cluster [cm].
        cluster_size_time (float): Max time distance between two points to be 
            inside a cluster [ns].
    
    Returns:
        awkward.array: Adds to interaction a cluster_ids record.
    """
    # TODO is there a better way to get the df?
    df = []
    for key in ['x', 'y', 'z', 'ed', 't']:
        df.append(ak.to_pandas(interactions[key], anonymous=key))
    df = pd.concat(df, axis=1)

    if df.empty:
        # TPC interaction is empty
        return interactions

    # Splitting into individual events and apply time clustering:
    groups = df.groupby('entry')

    df["time_cluster"] = np.concatenate(groups.apply(lambda x: simple_1d_clustering(x.t.values, cluster_size_time)))

    # Splitting into individual events and time cluster and apply space clustering space:
    df['cluster_id'] = np.zeros(len(df.index), dtype=np.int)

    for evt in df.index.get_level_values(0).unique():
        _df_evt = df.loc[evt]
        _t_clusters = _df_evt.time_cluster.unique()
        add_to_cluster = 0

        for _t in _t_clusters:
            _cl = _find_cluster(_df_evt[_df_evt.time_cluster == _t], cluster_size_space=cluster_size_space)
            df.loc[(df.time_cluster == _t)&(df.index.get_level_values(0)==evt), 'cluster_id'] = _cl + add_to_cluster            
            add_to_cluster = max(_cl) + add_to_cluster + 1

    ci = df.loc[:, 'cluster_id'].values
    offsets = ak.num(interactions['x'])
    interactions['cluster_ids'] = reshape_awkward(ci, offsets)
    
    return interactions


@numba.jit(nopython=True)
def simple_1d_clustering(data, scale):
    """
    Function to cluster one dimensional data.

    Args:
        data (numpy.array): one dimensional array to be clusterd
        scale (float): Max distance between two points to
            be inside a cluster.
    
    Returns:
        clusters_undo_sort (np.array): Cluster Labels
    """
    
    idx_sort = np.argsort(data)
    idx_undo_sort = np.argsort(idx_sort)

    data_sorted = data[idx_sort]

    diff = data_sorted[1:] - data_sorted[:-1]
    
    clusters = [0]
    c = 0
    for value in diff:
        if value <= scale:
            clusters.append(c)
        elif value > scale:
            c = c + 1
            clusters.append(c)

    clusters_undo_sort = np.array(clusters)[idx_undo_sort]
    
    return clusters_undo_sort


def _find_cluster(x, cluster_size_space):
    """
    Function which finds cluster within a event.
    Args:
        x (pandas.DataFrame): Subentries of event must contain the
            fields, x,y,z,time
        cluster_size_space (float): Max distance between two points to
            be inside a cluster [cm].
    Returns:
        functon: to be used in groupby.apply.
    """
    db_cluster = DBSCAN(eps=cluster_size_space, min_samples=1)
    xprime = x[['x', 'y', 'z']].values 
    return db_cluster.fit_predict(xprime)


def cluster(inter, classify_by_energy=False):
    """
    Function which clusters the found clusters together.

    To cluster events a weighted mean is computed for time and position.
    The individual interactions are weighted by their energy.
    The energy of clustered interaction is given by the total sum.
    Events can be classified either by the first interaction in time in the
    cluster or by the highest energy deposition.

    Args:
        inter (awkward.Array): Array containing at least the following
            fields: x,y,z,t,ed,cluster_ids, type, parenttype, creaproc,
            edproc.

    Kwargs:
        classify_by_energy (bool): If true events are classified
            according to the properties of the highest energy deposit
            within the cluster. If false cluster is classified according
            to first interaction.

    Returns:
        awkward.Array: Clustered events with nest conform
            classification.
    """

    if len(inter) == 0:
        result_cluster_dtype = [('x', 'float64'),
                                ('y', 'float64'),
                                ('z', 'float64'),
                                ('t', 'float64'),
                                ('ed', 'float64'),
                                ('nestid', 'int64'),
                                ('A', 'int64'),
                                ('Z', 'int64')]
        return ak.from_numpy(np.empty(0, dtype=result_cluster_dtype))
    # Sort interactions by cluster_ids to simplify looping
    inds = ak.argsort(inter['cluster_ids'])
    inter = inter[inds]

    # TODO: Better way to do this with awkward?
    x = inter['x']
    y = inter['y']
    z = inter['z']
    ed = inter['ed']
    time = inter['t']
    ci = inter['cluster_ids']
    types = inter['type']
    parenttype = inter['parenttype']
    creaproc = inter['creaproc']
    edproc = inter['edproc']

    # Init result and cluster:
    res = ak.ArrayBuilder()
    _cluster(x, y, z, ed, time, ci,
             types, parenttype, creaproc, edproc,
             classify_by_energy, res)
    return res.snapshot()


@numba.njit
def _cluster(x, y, z, ed, time, ci,
             types, parenttype, creaproc, edproc,
             classify_by_energy, res):
    # Loop over each event
    nevents = len(ed)
    for ei in range(nevents):
        # Init a new list for clustered interactions within event:
        res.begin_list()

        # Init buffers:
        ninteractions = len(ed[ei])
        x_mean = 0
        y_mean = 0
        z_mean = 0
        t_mean = 0
        ed_tot = 0

        current_ci = 0  # Current cluster id
        i_class = 0  # Index for classification (depends on users requirement)
        # Set classifier start value according to user request, interactions
        # are classified either by
        if classify_by_energy:
            # Highest energy
            classifier_max = 0
        else:
            # First interaction
            classifier_max = np.inf

        # Loop over all interactions within event:
        for ii in range(ninteractions):
            if current_ci != ci[ei][ii]:
                # Cluster Id has changed compared to previous interaction,
                # hence we have to write out our result and empty the buffer,
                # but first classify event:
                A, Z, nestid = classify(types[ei][i_class],
                                        parenttype[ei][i_class],
                                        creaproc[ei][i_class],
                                        edproc[ei][i_class])

                # Write result, simple but extensive with awkward...
                _write_result(res, x_mean, y_mean, z_mean,
                              ed_tot, t_mean, A, Z, nestid)

                # Update cluster id and empty buffer
                current_ci = ci[ei][ii]
                x_mean = 0
                y_mean = 0
                z_mean = 0
                t_mean = 0
                ed_tot = 0

                # Reset classifier:
                if classify_by_energy:
                    classifier_max = 0
                else:
                    classifier_max = np.inf

            # We have to gather information of current cluster:
            e = ed[ei][ii]
            t = time[ei][ii]
            x_mean += x[ei][ii] * e
            y_mean += y[ei][ii] * e
            z_mean += z[ei][ii] * e
            t_mean += t * e
            ed_tot += e

            if classify_by_energy:
                # In case we want to classify the event by energy.
                if e > classifier_max:
                    i_class = ii
                    classifier_max = e
            else:
                # or by first arrival time:
                if t < classifier_max:
                    i_class = ii
                    classifier_max = t

        # Before we are done with this event we have to classify and
        # write the last interaction
        A, Z, nestid = classify(types[ei][i_class],
                                parenttype[ei][i_class],
                                creaproc[ei][i_class],
                                edproc[ei][i_class])
        _write_result(res, x_mean, y_mean, z_mean,
                      ed_tot, t_mean, A, Z, nestid)

        res.end_list()


infinity = np.iinfo(np.int16).max
classifier_dtype = [(('Interaction type', 'types'), np.dtype('<U30')),
                    (('Interaction type of the parent', 'parenttype'), np.dtype('<U30')),
                    (('Creation process', 'creaproc'), np.dtype('<U30')),
                    (('Energy deposit process', 'edproc'), np.dtype('<U30')),
                    (('Atomic mass number', 'A'), np.int16),
                    (('Atomic number', 'Z'), np.int16),
                    (('Nest Id for qunata generation', 'nestid'), np.int16)]
classifier = np.zeros(7, dtype=classifier_dtype)
classifier['types'] = ['None', 'neutron', 'alpha', 'None','None', 'gamma', 'e-']
classifier['parenttype'] = ['None', 'None', 'None', 'Kr83[9.405]','Kr83[41.557]', 'None', 'None']
classifier['creaproc'] = ['None', 'None', 'None', 'None', 'None','None', 'None']
classifier['edproc'] = ['ionIoni', 'hadElastic', 'None', 'None','None', 'None', 'None']
classifier['A'] = [0, 0, 4, infinity, infinity, 0, 0]
classifier['Z'] = [0, 0, 2, 0, 0, 0, 0]
classifier['nestid'] = [0, 0, 6, 11, 11, 7, 8]


@numba.njit
def classify(types, parenttype, creaproc, edproc):
    for c in classifier:
        m = 0
        m += (c['types'] == types) or (c['types'] == 'None')
        m += (c['parenttype'] == parenttype) or (c['parenttype'] == 'None')
        m += (c['creaproc'] == creaproc) or (c['creaproc'] == 'None')
        m += (c['edproc'] == edproc) or (c['edproc'] == 'None')

        if m == 4:
            return c['A'], c['Z'], c['nestid']

    # If our data does not match any classification make it a nest None type
    # TODO: fix me
    return infinity, infinity, 12


@numba.njit
def _write_result(res, x_mean, y_mean, z_mean,
                  ed_tot, t_mean, A, Z, nestid):
    """
    Helper to write result into record array.
    """
    res.begin_record()
    res.field('x')
    res.real(x_mean / ed_tot)
    res.field('y')
    res.real(y_mean / ed_tot)
    res.field('z')
    res.real(z_mean / ed_tot)
    res.field('t')
    res.real(t_mean / ed_tot)
    res.field('ed')
    res.real(ed_tot)
    res.field('nestid')
    res.integer(nestid)
    res.field('A')
    res.integer(A)
    res.field('Z')
    res.integer(Z)
    res.end_record()
