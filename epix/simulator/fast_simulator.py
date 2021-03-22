from straxen.common import get_resource
from epix.simulator import helpers
import numpy as np
import wfsim
import epix
import strax
from straxen import pax_file

class GenerateEvents():
    '''Class to hold all the stuff to be applied to the data. 
    The functions will be grouped together and executed by Simulator'''
    def __init__(self) -> None:
        pass

    @staticmethod
    @helpers.assignOrder(0)
    def MakeS1(instructions,config, resource):
        xyz = instructions['x'],instructions['y'],instructions['z']
        n_photons = wfsim.core.S1.get_n_photons(
                                    n_photons=instructions['s1_area'],
                                    positions=xyz,
                                    s1_light_yield_map=resource.s1_light_yield_map,
                                    config=config) * config['dpe_fraction']
        
        alt_n_photons = wfsim.core.S1.get_n_photons(
                                    n_photons=instructions['alt_s1_area'],
                                    positions=xyz,
                                    s1_light_yield_map=resource.s1_light_yield_map,
                                    config=config) * config['dpe_fraction']
                                    
        instructions['s1_area'] = np.sum(resource.spe_distibution[np.random.random(n_photons)]],axis=1)
        instructions['alt_s1_area'] = np.sum(resource.spe_distibution[np.random.random(alt_n_photons)]],axis=1)
        
    @staticmethod
    @helpers.ssignOrder(1)
    def MakeS2(instructions,config,resource):
        '''Call functions from wfsim to drift electrons. Since the s2 light yield implementation is a little bad how to do that?
        Make sc_gain factor 11 to large to correct for this? Also whats the se gain variation? Lets take sqrt for now'''
        positions = instructions['x'],instructions['y']

        sc_gain = wfsim.core.S2.get_s2_light_yield(positions=positions,
                                          config=config,
                                          resource=resource)
        sc_gain_sigma = np.sqrt(sc_gain)

        n_electron = wfsim.core.S2.get_electron_yield(n_electron=instructions['s2_area'],
                                             z_obs=instructions['z'],
                                             config=config)

        alt_n_electron = wfsim.core.S2.get_electron_yield(n_electron=instructions['alt_s2_area'],
                                             z_obs=instructions['z'],
                                             config=config)

        instructions['s2_area'] = n_electron * np.random.normal(sc_gain,sc_gain_sigma) * config['dpe_fraction']
        instructions['alt_s2_area'] = alt_n_electron * np.random.normal(sc_gain,sc_gain_sigma)

        instructions['drift_time'] = instructions['z']/config['electron_drift_velocity']
        instructions['alt_s2_drift_time']= instructions['alt_z']/config['electron_drift_velocity']

    @staticmethod
    @helpers.assignOrder(2)
    def SmearPositions(instructions,config,**kwargs):
        instructions['x'] = np.random.normal(instructions['x'],p=config['xy_resolution'])
        instructions['y'] = np.random.normal(instructions['y'],p=config['xy_resolution'])
        instructions['z'] = np.random.normal(instructions['z'],p=config['z_resolution'])
        instructions['alt_x'] = np.random.normal(instructions['alt_x'],p=config['xy_resolution'])
        instructions['alt_y'] = np.random.normal(instructions['alt_y'],p=config['xy_resolution'])
        instructions['alt_z'] = np.random.normal(instructions['alt_z'],p=config['z_resolution'])

    @staticmethod
    @helpers.assignOrder(3)
    def CorrectionS1(instructions, resource, **kwargs):
        event_positions = np.vstack([instructions['x'], instructions['y'], instructions['z']]).T
        instructions['cs1'] = instructions['s1_area'] / resource.s1_map(event_positions),
        instructions['alt_cs1'] = instructions['alt_s1_area'] / resource.s1_map(event_positions),

    @staticmethod
    @helpers.assignOrder(4)
    def CorrectionS2(instructions, config,resource):

        lifetime_corr = np.exp(instructions['drift_time'] / config['elife'])
        alt_lifetime_corr = (
            np.exp((instructions['alt_s2_drift_time'])
                   / config['elife']))

        # S2(x,y) corrections use the observed S2 positions
        s2_positions = np.vstack([instructions['s2_x'], instructions['s2_y']]).T
        alt_s2_positions = np.vstack([instructions['alt_s2_x'], instructions['alt_s2_y']]).T

        instructions['cs2']=(instructions['s2_area'] * lifetime_corr
                 / resource.s2_map(s2_positions)),
        instructions['alt_cs2']=(instructions['alt_s2_area'] * alt_lifetime_corr
                     / resource.s2_map(alt_s2_positions))


class  Simulator():
    '''Simulator class for epix to go from  epix instructions to fully processed data'''

    def __init__(self,instructions_epix,config):
        self.config=config
        self.ge = GenerateEvents()

        self.simulation_functions = sorted(
                    #get the functions of GenerateEvents in order to run through them all
                    [getattr(self.ge, field) for field in dir(x)
                    if hasattr(getattr(self.ge, field), "order")
                    ],
                    #sort them by their order
                    key = (lambda field: field.order)
                    )
        self.instructions_epix=instructions_epix

    def cluster_events(self,):
        #Events have more than 1 s1/s2. Here we throw away all of them except the largest 2
        #Take the position to be that of the main s2
        #And strax wants some start and endtime, so we make something up
        instructions = np.zeros(self.epix_instructions['event_number'][-1]+1,dtype=StraxSimulator.dtype)
        for ix in np.unique(self.instructions_epix['event_number']):
            i=instructions[ix]
            inst =  self.instructions_epix[self.instructions_epix['event_number']==ix]
            s1 = np.argsort(inst[inst['type']==1]['amp'])
            i['s1_area']=inst[s1[0]]['amp']
            if len(s1)>1:
                i['alt_s1_area']=inst[s1[1]]['amp']
            s2 = np.argsort(inst[inst['type']==2]['amp'])
            i['s2_area']=inst[s2[0]]['amp']
            i['x']= inst[s2[0]]['x']
            i['y']= inst[s2[0]]['y']
            if len(s2)>1:
                i['alt_s2_area']=inst[s2[1]]['amp']
                i['alt_s2_x']=inst[s2[1]]['x']
                i['alt_s2_y']=inst[s2[1]]['y']
                i['alt_s2_z']=inst[s2[1]]['z']

            i['time']=ix*1000
            i['endtime']=ix*1000+1
        self.instructions=instructions

    def simulate(self,):
        for func in self.simulation_functions:
            func(instructions=self.instructions,
                 config=self.config,
                 resource=self.resource)

    def run_simulator(self,):
        self.cluster_events()
        self.simulate()
        return self.instructions
        

@strax.takes_config(
    strax.Option('g4_file',help='G4 file to simulate'),
    strax.Option('epix_config',default=dict(),help='Configuration file for epix',),
    strax.Option('configuration_files',default=dict(),help='Files required for simulating'),
    strax.Option('fax_config',default='fax_config_nt_low_field.json',help='Fax configuration to load'),
    strax.Option('xy_resolution',default=5,help='xy position resolution (cm)'),
    strax.Option('z_resolution',default=1,help='xy position resolution (cm)'),
)
class StraxSimulator(strax.Plugin):
    provides = 'events_full_simmed'
    depends_on=()
    dtype=[('time',np.int64),
           ('endtime',np.int64),
           ('s1_area',np.float),
           ('s2_area',np.float),
           ('cs1',np.float),
           ('cs2',np.float),
           ('alt_s1_area',np.float),
           ('alt_s2_area',np.float),
           ('alt_cs1',np.float),
           ('alt_cs2',np.float),
           ('x',np.float),
           ('y',np.float),
           ('z',np.float)
           ('alt_s2_x',np.float),
           ('alt_s2_y',np.float),
           ('alt_s2_z',np.float),
           ('drift_time',np.float),
           ('alt_s2_drift_time',np.float)]
        
    def setup(self,):
        resource_config=dict()
        resource_config['s1_relative_lce_map']=pax_file('XENON1T_s1_xyz_lce_true_kr83m_SR1_pax-680_fdc-3d_v0.json')
        resource_config['s2_xy_correction_map']=pax_file('XENON1T_s2_xy_ly_SR1_v2.2.json')
        resource_config['photon_area_distribution']= 'XENONnT_spe_distributions_20210305.csv',
        resource_config['s1_pattern_map']= 'XENONnT_s1_xyz_patterns_LCE_corrected_qes_MCva43fa9b_wires.pkl',
        resource_config['s2_pattern_map']= 'XENONnT_s2_xy_patterns_LCE_corrected_qes_MCva43fa9b_wires.pkl',

        self.resource=helpers.Resource(resource_config)
        self.config.update(get_resource(self.config['fax_config']))

    def compute(self,):
        epix_config = get_resource(self.config['epix_config'])
        epix_instructions = epix.run_epix.main(self.config['g4_file'],epix_config)
        self.Simulator=Simulator(instructions=epix_instructions,
                                 config=self.config,
                                 resource=self.resource)
        simulated_data = self.Simulator.run_simulator()

        return simulated_data

    def is_ready(self,):
        #For this plugin we'll smash everything into 1 chunk, should be oke
        True
