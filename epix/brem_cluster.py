import numpy as np
import pandas as pd
import uproot
import time
import os
import re
import awkward as ak
import collections


class BremClustering():

    def __init__(self):
        self.lineage_dict = {'trackid': 0, 'parentid': 1, 'type': 2, 'creaproc': 3, 'edproc': 4, 'PreStepEnergy': 5}

        self.min_dE = 1e-6
        self.max_lineage_depth = 1000  # safeguard for the lineage loop

        g4_keys = ['evtid', 'trackid', 'parentid', 'parenttype',
                   'ed', 'edproc', 'type', 'creaproc',
                   'PreStepEnergy', 'PostStepEnergy',
                   'x', 'y', 'z', 'x_pri', 'y_pri', 'z_pri', 't']
        self.agg_beta = {x: 'first' for x in g4_keys}
        self.agg_beta['PostStepEnergy'] = 'last'
        self.agg_beta['ed'] = 'sum'
        self.agg_beta['dE'] = 'sum'

        self.agg_brem = {x: 'last' for x in g4_keys}
        self.agg_brem['PreStepEnergy'] = 'first'

        self.agg_escaped = {x: 'last' for x in g4_keys}

    def get_clean_gamma_df(self, df):
        # In the final gamma_df we will remove compt steps since they will be taken
        # into account in the electron_df. We keep them now to see all steps
        # in the lineages of gammas during correction calculation.
        gamma_df = df[(df.type == 'gamma') & ((df.edproc == 'phot') | (df.edproc == 'compt'))]
        # Exclude gamma from e- via eIoni since they are accounted in the e- eDep
        # and their max energy is 35 keV
        gamma_df = gamma_df[gamma_df.creaproc != 'eIoni']
        gamma_df = gamma_df[gamma_df.dE > self.min_dE]
        return gamma_df

    def get_clean_electron_df(self, df):
        em_df = df[(df.type == 'e-')
                   & ((df.creaproc == 'phot') | (df.creaproc == 'conv') | (df.creaproc == 'compt'))]
        em_df = em_df[em_df.dE > self.min_dE]
        return em_df.groupby(['trackid'], as_index=False).aggregate(self.agg_beta)

    def get_clean_positron_df(self, df):
        ep_df = df[(df.type == 'e+')]
        ep_df = ep_df[ep_df.edproc != 'annihil']
        ep_df = ep_df[ep_df.dE > self.min_dE]
        return ep_df.groupby(['trackid'], as_index=False).aggregate(self.agg_beta)

    def get_brem_event_df(self, event_df):
        brem_df = event_df[(event_df.type == 'gamma') & (event_df.creaproc == 'eBrem')]
        brem_df = brem_df[brem_df.edproc != 'Transportation']
        brem_df = brem_df.groupby('trackid', as_index=False).aggregate(self.agg_brem)

        brem_df['corrected'] = False

        return brem_df

    def get_escaped_event_df(self, event_df):
        # Before we clean the event_df, we check if there're any particles
        # that escaped LXe volume in the event

        # Remove nu/anti_nu
        event_df_no_nu = event_df[~(event_df.type.str.contains('nu'))]

        escaped_df = event_df_no_nu.groupby('trackid', as_index=False).aggregate(self.agg_escaped)
        # escaped_df = escaped_df[escaped_df.edproc=='Transportation']
        escaped_df = escaped_df[escaped_df.PostStepEnergy > self.min_dE]

        escaped_df['corrected'] = False

        return escaped_df

    def get_clean_event_df(self, event_df):
        _df = event_df[(event_df.edproc != 'Transportation')]  # ignore due to zero deposited energy
        _df = _df[(_df.edproc != 'Scintillation')]  # ignore last step due to zero particle energy
        _df = _df[(_df.edproc != 'Rayl')]  # ignore due to zero deposited energy

        _df['dE'] = _df.PreStepEnergy - _df.PostStepEnergy

        clean_df = pd.DataFrame()
        clean_df = pd.concat((clean_df, self.get_clean_gamma_df(_df)))
        clean_df = pd.concat((clean_df, self.get_clean_electron_df(_df)))
        clean_df = pd.concat((clean_df, self.get_clean_positron_df(_df)))

        clean_df['corr'] = 0.0
        clean_df['dE_corr'] = 0.0

        return clean_df

    def get_tracks_array(self, clean_event_df):
        df = clean_event_df.groupby(['trackid'], as_index=False).aggregate(
            {k: 'first' for k in self.lineage_dict.keys()})
        return df.to_numpy()

    def get_lineage(self, data, parentid):
        datalist = []

        def get_node(data, parentid):
            newdata = []
            for d in data:
                if d[self.lineage_dict['parentid']] == parentid:
                    newdata.append(d)
            for d in newdata:
                datalist.append(d)
                get_node(data, d[self.lineage_dict['trackid']])

        get_node(data, parentid)
        return np.array(datalist)

    def get_reverse_lineage(self, data, trackid):
        datalist = []
        current_trackid = trackid
        current_parentid = -1

        for i in range(self.max_lineage_depth):
            try:
                _track = np.array([None for k in self.lineage_dict.values()])
                for k in self.lineage_dict.keys():
                    _track[self.lineage_dict[k]] = data.loc[(data.trackid == current_trackid)][k].to_numpy()[0]
                if current_trackid != trackid:
                    datalist.append(_track)
                current_trackid = _track[self.lineage_dict['parentid']]
            except KeyError:
                break
            except IndexError:
                break

        return np.array(datalist)

    def get_gamma_df(self, event_df,
                     tracks_arr, clean_event_df,
                     phot_gamma_correction=False,
                     verbose=False):
        phot_gamma_df = clean_event_df[clean_event_df.type == 'gamma']
        phot_gamma_df = phot_gamma_df[(phot_gamma_df.edproc == 'phot') | (phot_gamma_df.edproc == 'eIoni')]

        if not phot_gamma_correction:
            phot_gamma_df = phot_gamma_df[phot_gamma_df.creaproc != 'phot']

        phot_gamma_df['dE_corr'] = phot_gamma_df['dE'] + phot_gamma_df['corr']
        return phot_gamma_df

    def filter_electron_tracks(self, tracks_arr, clean_event_df,
                               exclude_e_from_phot_gamma=True):
        ## exclude_e_from_phot_gamma: remove e- which come from the lineage of the gamma created by phot

        non_e_arr = tracks_arr[tracks_arr[:, self.lineage_dict['type']] != 'e-']

        electron_arr = tracks_arr[tracks_arr[:, self.lineage_dict['type']] == 'e-']
        electron_arr = electron_arr[electron_arr[:, self.lineage_dict['creaproc']] != 'eIoni']

        # Remove phot-e since their eDep are already included in the phot-gamma eDep
        electron_arr = electron_arr[electron_arr[:, self.lineage_dict['creaproc']] != 'phot']

        if exclude_e_from_phot_gamma:
            # Remove e- which come from the lineage of the gamma created by phot
            # since we disregard these type of gamma in gamma_df
            phot_gamma_trackid = clean_event_df[
                (clean_event_df.type == 'gamma') & (clean_event_df.creaproc == 'phot')].trackid.unique()
            e_from_phot_gamma_trackid = []
            for g in phot_gamma_trackid:
                lineage = self.get_lineage(tracks_arr, g)
                if len(lineage) > 0:
                    e_from_phot_gamma_trackid.extend(np.unique(lineage[:, self.lineage_dict['trackid']]))

            e_from_phot_gamma_trackid = list(set(e_from_phot_gamma_trackid))
            electron_arr = electron_arr[
                ~np.isin(electron_arr[:, self.lineage_dict['trackid']], e_from_phot_gamma_trackid)]

        clean_tracks_arr = np.concatenate((electron_arr, non_e_arr))

        return clean_tracks_arr

    def remove_e_from_eIoni_gamma(self, event_df, e_df):
        e_from_eIoni_gamma = []

        for e_trackid in e_df.trackid.unique():
            e_rlin = self.get_reverse_lineage(event_df, trackid=e_trackid)
            for step in e_rlin:
                try:
                    if step[self.lineage_dict['type']] == 'gamma' and step[self.lineage_dict['creaproc']] == 'eIoni':
                        e_from_eIoni_gamma.append(e_trackid)
                        break
                except IndexError:
                    print('IndexError')

        e_df = e_df[~e_df.trackid.isin(e_from_eIoni_gamma)]
        return e_df

    def get_e_df(self, event_df,
                 tracks_arr, clean_event_df, e_type,
                 exclude_e_from_phot_gamma=True,
                 exclude_e_from_eIoni_gamma=True,
                 verbose=False):
        clean_tracks_arr = self.filter_electron_tracks(tracks_arr, clean_event_df, exclude_e_from_phot_gamma)
        e_tracks = clean_tracks_arr[clean_tracks_arr[:, self.lineage_dict['type']] == e_type]
        e_trackid = np.unique(e_tracks[:, self.lineage_dict['trackid']])
        e_df = clean_event_df[clean_event_df.trackid.isin(e_trackid)]

        if exclude_e_from_eIoni_gamma:
            e_df = self.remove_e_from_eIoni_gamma(event_df, e_df)

        try:
            if e_type == 'e-':
                e_df = e_df[(e_df.creaproc == 'compt') | (e_df.creaproc == 'conv') | (e_df.parentid == 0)]
            e_df['dE_corr'] = e_df['dE'] + e_df['corr']
        except AttributeError:
            pass

        return e_df

    def correct_for_escaped(self, track_df, escaped_df, escaped_rlineage_dict, mode):
        corrected_escaped_trackid = []

        for escaped_trackid, escaped_rlineage in escaped_rlineage_dict.items():
            for step in escaped_rlineage:
                if np.isin(step[self.lineage_dict['trackid']], track_df.trackid.to_numpy()):
                    try:
                        _corr = escaped_df[escaped_df.trackid == escaped_trackid].PostStepEnergy.to_numpy()[0]
                        track_df.loc[track_df.trackid == step[self.lineage_dict['trackid']], 'corr'] += -1.0 * _corr
                        track_df.loc[track_df.trackid == step[self.lineage_dict['trackid']], 'dE_corr'] += -1.0 * _corr
                        escaped_df.loc[escaped_df.trackid == escaped_trackid, 'corrected'] = True
                        corrected_escaped_trackid.append(escaped_trackid)
                        break
                    except IndexError:
                        print('IndexError')

        for corr_trackid in corrected_escaped_trackid:
            escaped_rlineage_dict.pop(corr_trackid, None)

        return track_df, escaped_rlineage_dict

    def escaped_secondaries_correction(self, event_df, escaped_df,
                                       gamma_df, e_df, p_df,
                                       brem_gamma_added_to_corr, ):
        # Check if we have some of the eBrem gamma which have already been accounted for
        # among the escaped secondaries
        escaped_df.loc[(escaped_df.trackid.isin(brem_gamma_added_to_corr)), 'corrected'] = True

        # Check if we have compt-e- which are already in the e- table
        # among the escaped secondaries
        escaped_df.loc[(escaped_df.trackid.isin(e_df.trackid)), 'corrected'] = True

        # Check if we have gamma created by annihil or RDB
        # among the escaped secondaries
        escaped_df.loc[(escaped_df.creaproc == 'annihil'), 'corrected'] = True
        escaped_df.loc[(escaped_df.creaproc == 'RadioactiveDecayBase'), 'corrected'] = True

        # Add reverse lineages of the escaped secondaries
        esc_rlin_dict = {}

        for escaped_trackid in escaped_df[~escaped_df.corrected].trackid.to_numpy():
            esc_rlin_dict[escaped_trackid] = self.get_reverse_lineage(event_df, trackid=escaped_trackid)

            # Correction for the escaped secondaries, starting with compt e-
        e_df, esc_rlin_dict = self.correct_for_escaped(e_df, escaped_df, esc_rlin_dict, mode='e-')
        p_df, esc_rlin_dict = self.correct_for_escaped(p_df, escaped_df, esc_rlin_dict, mode='e+')
        gamma_df, esc_rlin_dict = self.correct_for_escaped(gamma_df, escaped_df, esc_rlin_dict, mode='gamma')

        return gamma_df, e_df, p_df

    def correct_for_brem(self, track_df, brem_df, brem_rlineage_dict, mode):
        corrected_brem_trackid = []

        for brem_trackid, brem_rlineage in brem_rlineage_dict.items():
            for step in brem_rlineage:
                if np.isin(step[self.lineage_dict['trackid']], track_df.trackid.to_numpy()):
                    try:
                        _corr = brem_df[brem_df.trackid == brem_trackid].PreStepEnergy.to_numpy()[0]
                        track_df.loc[track_df.trackid == step[self.lineage_dict['trackid']], 'corr'] += -1.0 * _corr
                        track_df.loc[track_df.trackid == step[self.lineage_dict['trackid']], 'dE_corr'] += -1.0 * _corr
                        brem_df.loc[brem_df.trackid == brem_trackid, 'corrected'] = True
                        corrected_brem_trackid.append(brem_trackid)
                        break
                    except IndexError:
                        print('IndexError')

        for corr_trackid in corrected_brem_trackid:
            brem_rlineage_dict[corr_trackid] = []

        return track_df, brem_rlineage_dict

    def brem_gamma_correction(self, event_df, brem_df,
                              gamma_df, e_df, p_df):
        # Add reverse lineages of the Brem gamma
        brem_rlin_dict = {}

        for brem_trackid in brem_df[~brem_df.corrected].trackid.to_numpy():
            brem_rlin_dict[brem_trackid] = self.get_reverse_lineage(event_df, trackid=brem_trackid)

            # Correction for the Brem gamma, starting with compt e-
        e_df, brem_rlin_dict = self.correct_for_brem(e_df, brem_df, brem_rlin_dict, mode='e-')
        p_df, brem_rlin_dict = self.correct_for_brem(p_df, brem_df, brem_rlin_dict, mode='e+')
        gamma_df, brem_rlin_dict = self.correct_for_brem(gamma_df, brem_df, brem_rlin_dict, mode='gamma')

        brem_gamma_added_to_corr = [k for k in brem_rlin_dict.keys() if len(brem_rlin_dict[k]) == 0]

        return gamma_df, e_df, p_df, brem_gamma_added_to_corr

    def check_dE_corr(self, gamma_df, e_df, p_df, event_df):
        # Positron table - check if there was a local eDep during annihilation step;
        # add it as a correction

        for p_id in p_df.trackid.unique():
            _df = event_df[(event_df.trackid == p_id)]
            _df = _df[_df.edproc == 'annihil']

            try:
                if len(_df) > 0:
                    _corr = _df.ed.sum()
                    p_df.loc[p_df['trackid'] == p_id, 'corr'] += _corr
                    p_df.loc[p_df['trackid'] == p_id, 'dE_corr'] += _corr
            except IndexError:
                print('IndexError')

        # Safeguard - check if there are any negative dE_corr values in the output
        for df in [gamma_df, e_df, p_df]:
            df.loc[df['dE_corr'] < self.min_dE, 'dE_corr'] *= 0.0

        return gamma_df, e_df, p_df

    def cluster(self, event_tracks_df, verbose=False):
        gamma_df = pd.DataFrame([])
        e_df = pd.DataFrame([])
        p_df = pd.DataFrame([])
        event_df = event_tracks_df.copy()

        event_empty = False
        event_types = event_df.type.unique()
        if 'gamma' not in event_types \
                and 'e-' not in event_types \
                and 'e+' not in event_types:
            event_empty = True

        if not event_empty:
            event_df['dE'] = event_df.PreStepEnergy - event_df.PostStepEnergy

            escaped_df = self.get_escaped_event_df(event_df)
            brem_df = self.get_brem_event_df(event_df)
            clean_event_df = self.get_clean_event_df(event_df)
            event_tracks_array = self.get_tracks_array(clean_event_df)

            gamma_df = self.get_gamma_df(event_df=event_df,
                                         tracks_arr=event_tracks_array,
                                         clean_event_df=clean_event_df,
                                         verbose=verbose)
            e_df = self.get_e_df(event_df=event_df,
                                 tracks_arr=event_tracks_array,
                                 clean_event_df=clean_event_df,
                                 e_type='e-',
                                 verbose=verbose)
            p_df = self.get_e_df(event_df=event_df,
                                 tracks_arr=event_tracks_array,
                                 clean_event_df=clean_event_df,
                                 e_type='e+',
                                 verbose=verbose)

            gamma_df, e_df, p_df, \
                brem_gamma_added_to_corr = self.brem_gamma_correction(event_df, brem_df,
                                                                      gamma_df, e_df, p_df)
            gamma_df, e_df, p_df = self.escaped_secondaries_correction(event_df, escaped_df,
                                                                       gamma_df, e_df, p_df,
                                                                       brem_gamma_added_to_corr)
            gamma_df, e_df, p_df = self.check_dE_corr(gamma_df, e_df, p_df, event_df)

        return gamma_df, e_df, p_df
