import numpy as np
import pandas as pd
import uproot
import time
import os
import re
import awkward as ak
import collections


class BetaDecayClustering():

    def __init__(self):
        pass

    def cluster(self, event_tracks_df):
        betagamma_df = event_tracks_df[
            ((event_tracks_df.type == 'e-') & (event_tracks_df.creaproc == 'RadioactiveDecayBase'))
            | ((event_tracks_df.type == 'gamma') & (event_tracks_df.creaproc == 'RadioactiveDecayBase'))]
        betagamma_df = betagamma_df[
            ~((betagamma_df.edproc == "Rayl") | (betagamma_df.edproc == "Transportation"))]

        betagamma_df['ed'] = betagamma_df.PreStepEnergy - betagamma_df.PostStepEnergy
        betagamma_df.loc[betagamma_df.type == 'e-', 'ed'] = betagamma_df.loc[
            betagamma_df.type == 'e-', 'PreStepEnergy']
        betagamma_df = betagamma_df[betagamma_df.ed > 1e-6]

        beta_trackid = []
        for ind, row in betagamma_df.iterrows():
            if row['type'] == 'gamma':
                pass
            elif row['type'] == 'e-':
                if row['trackid'] in beta_trackid:
                    betagamma_df.drop(ind, inplace=True)
                else:
                    beta_trackid.append(row['trackid'])
            else:
                betagamma_df.drop(ind, inplace=True)

        return betagamma_df
