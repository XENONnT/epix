import strax
import straxen
from straxen import pax_file, InterpolatingMap, get_resource, get_config_from_cmt

import wfsim
from wfsim.load_resource import load_config

import epix
import pandas as pd
import numpy as np
from numpy import int64, float64
from copy import deepcopy
import numba

import scipy as scp
from scipy.interpolate import interp1d

import collections
import uproot
import json

import itertools
from immutabledict import immutabledict

#Numba and classes still are not a match made in heaven
@numba.njit
def _merge_these_clusters(s2_area1, z1, s2_area2, z2):
    sensitive_volume_ztop=0 #it's the ground mesh, the top liquid level is at 2.7; // mm
    max_s2_area = max(s2_area1,s2_area2)
    if max_s2_area>5000:
        SeparationDistanceIntercept = 0.00024787 * 5000. + 3.4056346550312973
        SeparationDistanceSlope = 5.5869678412887262e-07 * 5000. + 0.0044792968
    else:
        SeparationDistanceIntercept = \
          0.00024787 * max_s2_area + 3.4056346550312973
        SeparationDistanceSlope = \
          5.5869678412887262e-07 * max_s2_area + 0.0044792968
    SeparationDistance = \
        SeparationDistanceIntercept - \
        SeparationDistanceSlope * (-sensitive_volume_ztop +\
            (z1 + z2) * 0.5)
    return z1-z2<SeparationDistance
    

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

                pmt_in_coincidence = []
                pe_per_pmt = []

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


class Helpers():
    @staticmethod
    def assignOrder(order):
        def do_assignment(to_func):
            to_func.order = order
            return to_func

        return do_assignment

    @staticmethod
    def average_spe_distribution(spe_shapes):
        uniform_to_pe_arr = []
        for ch in spe_shapes.columns[1:]:  # skip the first element which is the 'charge' header
            if spe_shapes[ch].sum() > 0:
                # mean_spe = (spe_shapes['charge'].values * spe_shapes[ch]).sum() / spe_shapes[ch].sum()
                scaled_bins = spe_shapes['charge'].values  # / mean_spe
                cdf = np.cumsum(spe_shapes[ch]) / np.sum(spe_shapes[ch])
            else:
                # if sum is 0, just make some dummy axes to pass to interpolator
                cdf = np.linspace(0, 1, 10)
                scaled_bins = np.zeros_like(cdf)

            grid_cdf = np.linspace(0, 1, 2001)
            grid_scale = interp1d(cdf, scaled_bins,
                                  bounds_error=False,
                                  fill_value=(scaled_bins[0], scaled_bins[-1]))(grid_cdf)

            uniform_to_pe_arr.append(grid_scale)
        spe_distribution = np.mean(uniform_to_pe_arr, axis=0)
        return spe_distribution
    
    @staticmethod
    @numba.njit
    def macro_cluster_events(instructions):
        for ix1 in range(len(instructions)):
            if instructions[ix1]['type']!=2:
                continue
            for ix2 in range(1,len(instructions[ix1:])+1):
                if instructions[ix1+ix2]['type']!=2:
                    continue
                if instructions[ix1]['event_number']!=instructions[ix1+ix2]['event_number']:
                    break
                if _merge_these_clusters(instructions[ix1]['amp'],instructions[ix1]['z'],
                                         instructions[ix1+ix2]['amp'],instructions[ix1+ix2]['z']):
                    instructions[ix1+ix2]['x'] = (instructions[ix1]['x']+instructions[ix1+ix2]['x'])*0.5
                    instructions[ix1+ix2]['y'] = (instructions[ix1]['y']+instructions[ix1+ix2]['y'])*0.5
                    instructions[ix1+ix2]['z'] = (instructions[ix1]['z']+instructions[ix1+ix2]['z'])*0.5
                    instructions[ix1+ix2]['amp'] = int((instructions[ix1]['amp']+instructions[ix1+ix2]['amp'])*0.5)
                    instructions[ix1]['amp']=-1 #flag to throw this instruction away later

    @staticmethod
    def get_s1_area_with_spe(spe_distribution, num_photons):
        s1_area_spe = []
        for n_ph in num_photons:
            s1_area_spe.append(np.sum(spe_distribution[
                                          (np.random.random(n_ph) * len(spe_distribution)).astype(int64)]))
        return np.array(s1_area_spe)

    @staticmethod
    def get_s1_light_yield(n_photons, positions, s1_light_yield_map, config):
        return wfsim.S1.get_n_photons(n_photons=n_photons,
                                           positions=positions,
                                           s1_light_yield_map=s1_light_yield_map,
                                           config=config)

    @staticmethod
    def get_s2_light_yield(positions, config, resource):
        return wfsim.S2.get_s2_light_yield(positions=positions,
                                           config=config,
                                           resource=resource)



    @staticmethod
    def get_s2_charge_yield(n_electron, positions, z_obs, config, resource):
        return wfsim.S2.get_electron_yield(n_electron=n_electron,
                                           positions=positions,
                                           z_obs=z_obs,
                                           config=config,
                                           resource=resource)
    
    
# class Resource():
#     def __init__(self, config) -> None:
#         self.resource=load_resource(config)
#         self._update_resource(self,config)

#     def _update_resource(self, config):
#         '''Loads needed configs to call wfsim. We need s1/s2 light yield maps,
#         spe distibutions and corrections maps'''
#         self.run_id = '1000'  # ? I just put this to something for the cmt
#         self.s1_map = InterpolatingMap(
#             get_resource(config['s1_relative_lce_map']))
#         self.s2_map = InterpolatingMap(
#             get_resource(get_config_from_cmt(self.run_id, config['s2_xy_correction_map'])))

#         map_data = straxen.get_resource(config['s1_pattern_map'], fmt='pkl')
#         self.s1_pattern_map = straxen.InterpolatingMap(map_data)
#         # self.s1_pattern_map = make_mapf(config['s1_pattern_map'], fmt='pkl')
#         lymap = deepcopy(self.s1_pattern_map)
#         lymap.data['map'] = np.sum(lymap.data['map'][:][:][:], axis=3, keepdims=True)
#         lymap.__init__(lymap.data)
#         self.s1_light_yield_map = lymap

#         map_data = straxen.get_resource(config['s2_pattern_map'], fmt='pkl')
#         self.s2_pattern_map = straxen.InterpolatingMap(map_data)
#         # self.s2_pattern_map = make_map(config['s2_pattern_map'], fmt='pkl')
#         lymap = deepcopy(self.s2_pattern_map)
#         lymap.data['map'] = np.sum(lymap.data['map'][:][:], axis=2, keepdims=True)
#         lymap.__init__(lymap.data)
#         self.s2_light_yield_map = lymap

#         self.photon_area_distribution = Helpers.average_spe_distribution(
#             get_resource(config['photon_area_distribution'], fmt='csv'))
#         self.nveto_pmt_qe = json.loads(get_resource(config['nv_pmt_qe']))


class GenerateEvents():
    '''Class to hold all the stuff to be applied to data.
    The functions will be grouped together and executed by Simulator'''

    def __init__(self) -> None:
        pass

    @staticmethod
    @Helpers.assignOrder(0)
    def smear_positions(instructions, config, **kwargs):
        instructions['x'] = np.random.normal(instructions['x'], config['xy_resolution'])
        instructions['y'] = np.random.normal(instructions['y'], config['xy_resolution'])
        instructions['z'] = np.random.normal(instructions['z'], config['z_resolution'])

        instructions['alt_s2_x'] = np.random.normal(instructions['alt_s2_x'], config['xy_resolution'])
        instructions['alt_s2_y'] = np.random.normal(instructions['alt_s2_y'], config['xy_resolution'])
        instructions['alt_s2_z'] = np.random.normal(instructions['alt_s2_z'], config['z_resolution'])

    @staticmethod
    @Helpers.assignOrder(1)
    def make_s1(instructions, config, resource):
        xyz = np.vstack([instructions['x'], instructions['y'], instructions['z']]).T

        num_ph = instructions['s1_area'].astype(int64)
        n_photons = Helpers.get_s1_light_yield(n_photons=num_ph,
                                               positions=xyz,
                                               s1_light_yield_map=resource.s1_light_yield_map,
                                               config=config) * config['dpe_fraction']
        instructions['s1_area'] = Helpers.get_s1_area_with_spe(resource.photon_area_distribution,
                                                               n_photons.astype(int64))

        num_ph = instructions['alt_s1_area'].astype(int64)
        alt_n_photons = Helpers.get_s1_light_yield(n_photons=num_ph,
                                                   positions=xyz,
                                                   s1_light_yield_map=resource.s1_light_yield_map,
                                                   config=config) * config['dpe_fraction']
        instructions['alt_s1_area'] = Helpers.get_s1_area_with_spe(resource.photon_area_distribution,
                                                                   alt_n_photons.astype(int64))

    @staticmethod
    @Helpers.assignOrder(2)
    def make_s2(instructions, config, resource):
        '''Call functions from wfsim to drift electrons. Since the s2 light yield implementation is a little bad how to do that?
        Make sc_gain factor 11 to large to correct for this? Also whats the se gain variation? Lets take sqrt for now'''
        xy = np.array([instructions['x'], instructions['y']]).T
        alt_xy = np.array([instructions['alt_s2_x'], instructions['alt_s2_y']]).T

        n_el = instructions['s2_area'].astype(int64)
        n_electron = Helpers.get_s2_charge_yield(n_electron=n_el,
                                                 positions=xy,
                                                 z_obs=instructions['z'],
                                                 config=config,
                                                 resource=resource)

        n_el = instructions['alt_s2_area'].astype(int64)
        alt_n_electron = Helpers.get_s2_charge_yield(n_electron=n_el,
                                                     positions=alt_xy,
                                                     z_obs=instructions['z'],
                                                     config=config,
                                                     resource=resource)

        sc_gain = Helpers.get_s2_light_yield(positions=xy,
                                             config=config,
                                             resource=resource)
        sc_gain_sigma = np.sqrt(sc_gain)

        instructions['s2_area'] = n_electron * np.random.normal(sc_gain, sc_gain_sigma) * config['dpe_fraction']
        instructions['drift_time'] = -instructions['z'] / config['drift_velocity_liquid']

        instructions['alt_s2_area'] = alt_n_electron * np.random.normal(sc_gain, sc_gain_sigma)
        instructions['alt_s2_drift_time'] = -instructions['alt_s2_z'] / config['drift_velocity_liquid']

    @staticmethod
    @Helpers.assignOrder(3)
    def correction_s1(instructions, resource, **kwargs):
        event_positions = np.vstack([instructions['x'], instructions['y'], instructions['z']]).T
        instructions['cs1'] = instructions['s1_area'] / resource.s1_map(event_positions)

        event_positions = np.vstack([instructions['x'], instructions['y'], instructions['z']]).T
        instructions['alt_cs1'] = instructions['alt_s1_area'] / resource.s1_map(event_positions)

    @staticmethod
    @Helpers.assignOrder(4)
    def correction_s2(instructions, config, resource):
        lifetime_corr = np.exp(instructions['drift_time'] / config['electron_lifetime_liquid'])

        alt_lifetime_corr = (np.exp((instructions['alt_s2_drift_time']) / config['electron_lifetime_liquid']))

        # S2(x,y) corrections use the observed S2 positions
        s2_positions = np.vstack([instructions['x'], instructions['y']]).T
        alt_s2_positions = np.vstack([instructions['alt_s2_x'], instructions['alt_s2_y']]).T

        instructions['cs2'] = (instructions['s2_area'] * lifetime_corr / resource.s2_map(s2_positions))
        instructions['alt_cs2'] = (
                instructions['alt_s2_area'] * alt_lifetime_corr / resource.s2_map(alt_s2_positions))

        alt_s2_nan = instructions['alt_s2_area'] < 1e-6
        instructions['alt_s2_x'][alt_s2_nan] = 0.0
        instructions['alt_s2_y'][alt_s2_nan] = 0.0
        instructions['alt_s2_z'][alt_s2_nan] = 0.0
        instructions['alt_s2_drift_time'][alt_s2_nan] = 0.0


class Simulator():
    '''Simulator class for epix to go from  epix instructions to fully processed data'''

    def __init__(self, instructions_epix, config, resource):
        self.config = config
        self.ge = GenerateEvents()
        self.resource=resource
        self.simulation_functions = sorted(
            # get the functions of GenerateEvents in order to run through them all
            [getattr(self.ge, field) for field in dir(self.ge)
             if hasattr(getattr(self.ge, field), "order")
             ],
            # sort them by their order
            key=(lambda field: field.order)
        )
        self.instructions_epix = instructions_epix
         
    def cluster_events(self, ):
        # Events have more than 1 s1/s2. Here we throw away all of them except the largest 2
        # Take the position to be that of the main s2
        # And strax wants some start and endtime, so we make something up
        
        #First do the macro clustering. Clustered instructions will be flagged with amp=-1
        #so we can safely through those out
        Helpers.macro_cluster_events(self.instructions_epix)
        self.instructions_epix=self.instructions_epix[self.instructions_epix['amp']!=-1]
        
        event_numbers = np.unique(self.instructions_epix['event_number'])
        ins_size = len(event_numbers)
        instructions = np.zeros(ins_size, dtype=StraxSimulator.dtype['events_tpc'])

        for ix in range(ins_size):
            i = instructions[ix]
            inst = self.instructions_epix[self.instructions_epix['event_number'] == event_numbers[ix]]
            inst_s1 = inst[inst['type'] == 1]
            inst_s2 = inst[inst['type'] == 2]

            s1 = np.argsort(inst_s1['amp'])
            s2 = np.argsort(inst_s2['amp'])

            if len(s1) < 1 or len(s2) < 1:
                continue

            i['s1_area'] = inst_s1[s1[-1]]['amp']
            if len(s1) > 1:
                i['alt_s1_area'] = inst_s1[s1[-2]]['amp']

            i['s2_area'] = inst_s2[s2[-1]]['amp']
            if len(s2) > 1:
                i['alt_s2_area'] = inst_s2[s2[-2]]['amp']
                i['alt_s2_x'] = inst_s2[s2[-2]]['x']
                i['alt_s2_y'] = inst_s2[s2[-2]]['y']
                i['alt_s2_z'] = inst_s2[s2[-2]]['z']

            i['x'] = inst_s2[s2[-1]]['x']
            i['y'] = inst_s2[s2[-1]]['y']
            i['z'] = inst_s2[s2[-1]]['z']
            i['time'] = ix * 1000
            i['endtime'] = ix * 1000 + 1

        instructions = instructions[~((instructions['time'] == 0) & (instructions['endtime'] == 0))]
        self.instructions = instructions

    def simulate(self, ):
        for func in self.simulation_functions:
            func(instructions=self.instructions,
                 config=self.config,
                 resource=self.resource)

    def run_simulator(self, ):
        self.cluster_events()
        self.simulate()

        return self.instructions


@strax.takes_config(
    strax.Option('detector', default='XENONnT', help='Detector model'),
    strax.Option('g4_file', help='G4 file to simulate'),
    strax.Option('epix_config', default=dict(), help='Configuration file for epix', ),
    strax.Option('configuration_files', default=dict(), help='Files required for simulating'),
    strax.Option('fax_config', help='Fax configuration to load'),
    strax.Option('fax_config_overrides', help='Fax configuration to override',default=None),
    strax.Option('xy_resolution', default=5, help='xy position resolution (cm)'),
    strax.Option('z_resolution', default=1, help='xy position resolution (cm)'),
    strax.Option('dpe_fraction', default=1, help="double photo electron emission probabilty. \
                                                 Should be 1 since it's included in the LCE maps"),
    strax.Option('nv_spe_resolution', default=0.35, help='nveto something?'),
    strax.Option('nv_spe_res_threshold', default=0.5, help='some nveto threshold?'),
    strax.Option('nv_max_coin_time_ns', default=500.0, help='maximum coincidence time'),
    
)
class StraxSimulator(strax.Plugin):
    provides = ('events_tpc','events_nveto')
    depends_on = ()
    data_kind = immutabledict(zip(provides, provides))
    dtype=dict(events_tpc=[('time', np.int64),
             ('endtime', np.int64),
             ('s1_area', np.float),
             ('s2_area', np.float),
             ('cs1', np.float),
             ('cs2', np.float),
             ('alt_s1_area', np.float),
             ('alt_s2_area', np.float),
             ('alt_cs1', np.float),
             ('alt_cs2', np.float),
             ('x', np.float),
             ('y', np.float),
             ('z', np.float),
             ('alt_s2_x', np.float),
             ('alt_s2_y', np.float),
             ('alt_s2_z', np.float),
             ('drift_time', np.float),
             ('alt_s2_drift_time', np.float)],
          events_nveto=[('time',np.float),
                               ('endtime',np.float),
                               ('event_id',np.int),
                               ('channel',np.int),])
    
    def load_config(self):
        """First load the config through wfsim, then we add some things we'd like to have"""
        self.resource = load_config(self.config)
        
        self.resource.s1_map = InterpolatingMap(
            get_resource(self.config['configuration_files']['s1_relative_lce_map']))
        self.resource.s2_map = InterpolatingMap(
            get_resource(self.config['configuration_files']['s2_xy_correction_map']))
        
        self.resource.nv_pmt_qe=straxen.get_resource(self.config['configuration_files']['nv_pmt_qe'], fmt='json')
        self.resource.photon_area_distribution = Helpers.average_spe_distribution(
            get_resource(self.config['configuration_files']['photon_area_distribution'], fmt='csv'))
        

    def setup(self, ):
        overrides=self.config['fax_config_overrides']
        self.config.update(straxen.get_resource(self.config['fax_config'],fmt='json'))
        if overrides is not None:
            self.config.update(overrides)
        self.load_config()
        
    def get_nveto_data(self, ):
        file_tree, _ = epix.io._get_ttree(self.config['epix_config']['path'],
                                          self.config['epix_config']['file_name'])
        nveto_df = None
        if 'pmthitID' in file_tree.keys():
            nv_hits = NVetoUtils.get_nv_hits(ttree=file_tree,
                                             pmt_nv_json_dict=self.resource.nv_pmt_qe,
                                             nveto_dtype=self.dtype['events_nveto'],
                                             SPE_Resolution=self.config['nv_spe_resolution'], 
                                             SPE_ResThreshold=self.config['nv_spe_res_threshold'],
                                             max_coin_time_ns=self.config['nv_max_coin_time_ns'], 
                                             batch_size=10000)
        return nv_hits

    def get_epix_instructions(self, ):
        detector = epix.init_detector('xenonnt', '')
        epix_config=deepcopy(self.config['epix_config'])
        epix_config['detector_config'] = detector

        outer_cylinder = getattr(epix.detectors, 'xenonnt')
        _, outer_cylinder = outer_cylinder()
        epix_config['outer_cylinder'] = outer_cylinder

        epix_ins = epix.run_epix.main(epix_config, return_wfsim_instructions=True)
        return epix_ins

    def compute(self):
        simulated_data_nveto = self.get_nveto_data()
        epix_instructions = self.get_epix_instructions()
        self.Simulator = Simulator(instructions_epix=epix_instructions,
                                   config=self.config,
                                   resource=self.resource)
        simulated_data = self.Simulator.run_simulator()
        
        simulated_data_chunk=self.chunk(
                           start=simulated_data['time'][0],
                           end=simulated_data['endtime'][-1],
                           data=simulated_data,
                           data_type='events_tpc')
        
        #write empty chunk if nveto data isn't there
        if simulated_data_nveto==None:
            simulated_data_nveto_chunk=self.chunk(
                           start=0,
                           end=1,
                           data=simulated_data_nveto,
                           data_type='events_nveto')
        else:
            simulated_data_nveto_chunk=self.chunk(
                           start=simulated_data_nveto['time'][0],
                           end=np.max(simulated_data_nveto['endtime']),
                           data=simulated_data_nveto,
                           data_type='events_nveto')
        
        
        return {'events_tpc':simulated_data_chunk,
                'events_nveto':simulated_data_nveto_chunk}
    
    def is_ready(self, chunk):
        # For this plugin we'll smash everything into 1 chunk, should be oke
        return True if chunk==0 else False
    
    def source_finished(self):
        return True

