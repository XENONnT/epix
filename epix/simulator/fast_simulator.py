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
    def SmearPositions(instructions, config, **kwargs):
        instructions['x'] = np.random.normal(instructions['x'], config['xy_resolution'])
        instructions['y'] = np.random.normal(instructions['y'], config['xy_resolution'])
        instructions['z'] = np.random.normal(instructions['z'], config['z_resolution'])
        instructions['alt_s2_x'] = np.random.normal(instructions['alt_s2_x'], config['xy_resolution'])
        instructions['alt_s2_y'] = np.random.normal(instructions['alt_s2_y'], config['xy_resolution'])
        instructions['alt_s2_z'] = np.random.normal(instructions['alt_s2_z'], config['z_resolution'])

    @staticmethod
    @helpers.assignOrder(1)
    def MakeS1(instructions, config, resource):
        # print(resource.photon_area_distribution)
        xyz = np.vstack([instructions['x'], instructions['y'], instructions['z']]).T

        num_ph = instructions['s1_area'].astype(int64)
        n_photons = helpers.Get_S1_LY(n_photons=num_ph,
                                      positions=xyz,
                                      s1_light_yield_map=resource.s1_light_yield_map,
                                      config=config) * config['dpe_fraction']

        num_ph = instructions['alt_s1_area'].astype(int64)
        alt_n_photons = helpers.Get_S1_LY(n_photons=instructions['alt_s1_area'],
                                          positions=xyz,
                                          s1_light_yield_map=resource.s1_light_yield_map,
                                          config=config) * config['dpe_fraction']

        # instructions['s1_area'] = np.sum(resource.photon_area_distribution[np.random.random(n_photons)],axis=1)
        # instructions['alt_s1_area'] = np.sum(resource.photon_area_distribution[np.random.random(alt_n_photons)],axis=1)
        instructions['s1_area'] = np.sum(n_photons)
        instructions['alt_s1_area'] = np.sum(alt_n_photons)

    @staticmethod
    @helpers.assignOrder(2)
    def MakeS2(instructions, config, resource):
        '''Call functions from wfsim to drift electrons. Since the s2 light yield implementation is a little bad how to do that?
        Make sc_gain factor 11 to large to correct for this? Also whats the se gain variation? Lets take sqrt for now'''
        xy = np.vstack([instructions['x'], instructions['y']]).T

        sc_gain = helpers.Get_S2_LY(positions=xy,
                                    config=config,
                                    resource=resource)
        sc_gain_sigma = np.sqrt(sc_gain)

        n_el = instructions['s2_area'].astype(int64)
        n_electron = wfsim.core.S2.get_electron_yield(n_electron=n_el,
                                                      z_obs=instructions['z'],
                                                      config=config)

        n_el = instructions['alt_s2_area'].astype(int64)
        alt_n_electron = wfsim.core.S2.get_electron_yield(n_electron=n_el,
                                                          z_obs=instructions['z'],
                                                          config=config)

        instructions['s2_area'] = n_electron * np.random.normal(sc_gain, sc_gain_sigma) * config['dpe_fraction']
        instructions['alt_s2_area'] = alt_n_electron * np.random.normal(sc_gain, sc_gain_sigma)

        instructions['drift_time'] = instructions['z'] / config['drift_velocity_liquid']
        instructions['alt_s2_drift_time'] = instructions['alt_s2_z'] / config['drift_velocity_liquid']

    @staticmethod
    @helpers.assignOrder(3)
    def CorrectionS1(instructions, resource, **kwargs):
        event_positions = np.vstack([instructions['x'], instructions['y'], instructions['z']]).T
        instructions['cs1'] = instructions['s1_area'] / resource.s1_map(event_positions)
        instructions['alt_cs1'] = instructions['alt_s1_area'] / resource.s1_map(event_positions)

    @staticmethod
    @helpers.assignOrder(4)
    def CorrectionS2(instructions, config, resource):
        lifetime_corr = np.exp(instructions['drift_time'] / config['electron_lifetime_liquid'])
        alt_lifetime_corr = (np.exp((instructions['alt_s2_drift_time']) / config['electron_lifetime_liquid']))

        # S2(x,y) corrections use the observed S2 positions
        s2_positions = np.vstack([instructions['x'], instructions['y']]).T
        alt_s2_positions = np.vstack([instructions['alt_s2_x'], instructions['alt_s2_y']]).T

        instructions['cs2'] = (instructions['s2_area'] * lifetime_corr / resource.s2_map(s2_positions))
        instructions['alt_cs2'] = (instructions['alt_s2_area'] * alt_lifetime_corr / resource.s2_map(alt_s2_positions))

class Simulator():
    '''Simulator class for epix to go from  epix instructions to fully processed data'''

    def __init__(self, instructions_epix, config, resource):
        self.config = config
        self.ge = GenerateEvents()
        self.resource = resource
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
        ins_size = np.unique(self.instructions_epix['event_number'])[-1] + 1
        instructions = np.zeros(ins_size, dtype=StraxSimulator.dtype)

        for ix in np.unique(self.instructions_epix['event_number']):
            i = instructions[ix]
            inst = self.instructions_epix[self.instructions_epix['event_number'] == ix]

            s1 = np.argsort(inst[inst['type'] == 1]['amp'])
            s2 = np.argsort(inst[inst['type'] == 2]['amp'])

            if len(s1) < 1 or len(s2) < 1:
                continue

            i['s1_area'] = inst[inst['type'] == 1][s1[-1]]['amp']
            if len(s1) > 1:
                i['alt_s1_area'] = inst[inst['type'] == 1][s1[-2]]['amp']

            i['s2_area'] = inst[inst['type'] == 2][s2[-1]]['amp']
            if len(s2) > 1:
                i['alt_s2_area'] = inst[inst['type'] == 2][s2[-2]]['amp']
                i['alt_s2_x'] = inst[inst['type'] == 2][s2[-2]]['x']
                i['alt_s2_y'] = inst[inst['type'] == 2][s2[-2]]['y']
                i['alt_s2_z'] = inst[inst['type'] == 2][s2[-2]]['z']

            i['x'] = inst[inst['type'] == 2][s2[-1]]['x']
            i['y'] = inst[inst['type'] == 2][s2[-1]]['y']
            i['z'] = inst[inst['type'] == 2][s2[-1]]['z']
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
    strax.Option('fax_config', default='fax_config_nt_design.json', help='Fax configuration to load'),
    strax.Option('xy_resolution', default=5, help='xy position resolution (cm)'),
    strax.Option('z_resolution', default=1, help='xy position resolution (cm)'),
)
class StraxSimulator(strax.Plugin):
    provides = 'events_full_simmed'
    depends_on = ()
    dtype = [('time', np.int64),
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
             ('alt_s2_drift_time', np.float)]

    def setup(self, ):
        self.resource = Resource(self.config)
        # self.config.update(get_resource(self.config['fax_config']))

    def get_epix_instructions(self, ):
        detector = epix.init_detector('xenonnt', '')
        self.config['epix_config']['detector_config'] = detector

        outer_cylinder = getattr(epix.detectors, 'xenonnt')
        _, outer_cylinder = outer_cylinder()
        self.config['epix_config']['outer_cylinder'] = outer_cylinder

        return epix.run_epix.main(self.config['epix_config'], return_wfsim_instructions=True)

    def compute(self, ):
        epix_instructions = self.get_epix_instructions()
        self.config['fax_config']['detector'] = self.config['detector']
        self.config['fax_config']['dpe_fraction'] = 0.5
        self.config['fax_config']['xy_resolution'] = self.config['xy_resolution']
        self.config['fax_config']['z_resolution'] = self.config['z_resolution']
        self.Simulator = Simulator(instructions_epix=epix_instructions,
                                   config=self.config['fax_config'],
                                   resource=self.resource)
        simulated_data = self.Simulator.run_simulator()

        return simulated_data

    def is_ready(self, chunk):
        # For this plugin we'll smash everything into 1 chunk, should be oke
        return True