import scipy as scp
import collections
import uproot
import itertools
import numpy as np

class NVetoUtils():
    @staticmethod
    def get_nv_pmt_qe(pmt_json_dict, pmt_ch, photon_ev):
        wvl = (1e9 * (scp.constants.c * scp.constants.h) / (photon_ev * 1.60218e-19))

        nv_pmt_qe_wavelength = np.array(pmt_json_dict['nv_pmt_qe_wavelength'])
        nv_pmt_qe = pmt_json_dict['nv_pmt_qe']

        wvl_index = np.abs(nv_pmt_qe_wavelength - wvl).argmin()

        return nv_pmt_qe[str(pmt_ch)][wvl_index]

    @staticmethod
    def get_nv_hits(ttree, pmt_nv_json_dict, nveto_dtype,
                    SPE_Resolution=0.35, SPE_ResThreshold=0.5,
                    max_coin_time_ns=500.0, batch_size=10000):
        
        
        hits_dict = {'event_id': [], 'time': [],'channel': []}
        num_hits=0
        for events_iterator in uproot.iterate(ttree,
                                          ['eventid', 'pmthitID', 'pmthitTime', 'pmthitEnergy'],
                                          step_size=batch_size,
                                          ioutputtype=collections.namedtuple):
            for eventid_evt, pmthitID_evt, pmthitTime_evt, \
                pmthitEnergy_evt in zip(getattr(events_iterator, 'eventid'),
                                        getattr(events_iterator, 'pmthitID'),
                                        getattr(events_iterator, 'pmthitTime'),
                                        getattr(events_iterator, 'pmthitEnergy')):
                hit_list = []
                hit_coincidence = 0

                for _time, _id, _energy in zip(pmthitTime_evt, pmthitID_evt, pmthitEnergy_evt):
                    if _id >= 2000 and _id < 2120:

                        qe = 1e-2 * NVetoUtils.get_nv_pmt_qe(pmt_nv_json_dict, _id, _energy)

                        pe = np.random.binomial(1, qe, 1)[0]
                        if pe < 0.5:
                            continue

                        pe_res = np.random.normal(1.0, SPE_Resolution, 1)
                        if pe_res >= SPE_ResThreshold:
                            hit_list.append([_time * 1e9, _id])

                hit_array = np.array(hit_list)
                pmt_coincidence_dict = None
                hit_array_coincidence = np.array([])

                if hit_array.shape[0] > 0:
                    t0 = hit_array[:, 0].min()
                    tf = t0 + max_coin_time_ns
                    hit_array_coincidence = hit_array[hit_array[:, 0] < tf]
                if len(hit_array_coincidence)>0:
                    hits_dict['event_id'].append([eventid_evt]*len(hit_array_coincidence))
                    hits_dict['time'].append(hit_array_coincidence[:,0])
                    hits_dict['channel'].append(hit_array_coincidence[:,1])
                    num_hits+=len(hit_array_coincidence)
                    
        result = np.zeros(num_hits,dtype=nveto_dtype)
        result['event_id']=list(itertools.chain.from_iterable(hits_dict['event_id']))
        result['channel']=list(itertools.chain.from_iterable(hits_dict['channel']))
        result['time']=list(itertools.chain.from_iterable(hits_dict['time']))
        result['endtime']=result['time']+1

