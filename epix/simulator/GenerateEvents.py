from .helpers import Helpers
import numpy as np

class GenerateEvents():
    '''Class to hold all the stuff to be applied to data.
    The functions will be grouped together and executed by Simulator'''

    @staticmethod
    @Helpers.assignOrder(0)
    def smear_positions(instructions, config, **kwargs):
        """Take initial positions and apply gaussian smearing with some resolution to get the measured position
            :params: instructions, numpy array with instructions of events_tpc dtype
            :params: config, dict with configuration values for resolution
        """
        instructions['x'] = np.random.normal(instructions['x'], config['xy_resolution'])
        instructions['y'] = np.random.normal(instructions['y'], config['xy_resolution'])
        instructions['z'] = np.random.normal(instructions['z'], config['z_resolution'])

        instructions['alt_s2_x'] = np.random.normal(instructions['alt_s2_x'], config['xy_resolution'])
        instructions['alt_s2_y'] = np.random.normal(instructions['alt_s2_y'], config['xy_resolution'])
        instructions['alt_s2_z'] = np.random.normal(instructions['alt_s2_z'], config['z_resolution'])

    @staticmethod
    @Helpers.assignOrder(1)
    def make_s1(instructions, config, resource):
        """Build the s1s. Takes number of quanta and calcultes the (alt) s1 area using wfsim
            :params: instructions, numpy array with instructions of events_tpc dtype
            :params: config, dict with configuration values for resolution
            :params: resource, instance of wfsim Resource class
        """
        xyz = np.vstack([instructions['x'], instructions['y'], instructions['z']]).T

        num_ph = instructions['s1_area'].astype(np.int64)
        n_photons = Helpers.get_s1_light_yield(n_photons=num_ph,
                                               positions=xyz,
                                               s1_light_yield_map=resource.s1_light_yield_map,
                                               config=config) * config['dpe_fraction']
        instructions['s1_area'] = Helpers.get_s1_area_with_spe(resource.photon_area_distribution,
                                                               n_photons.astype(np.int64))

        num_ph = instructions['alt_s1_area'].astype(np.int64)
        alt_n_photons = Helpers.get_s1_light_yield(n_photons=num_ph,
                                                   positions=xyz,
                                                   s1_light_yield_map=resource.s1_light_yield_map,
                                                   config=config) * config['dpe_fraction']
        instructions['alt_s1_area'] = Helpers.get_s1_area_with_spe(resource.photon_area_distribution,
                                                                   alt_n_photons.astype(np.int64))

    @staticmethod
    @Helpers.assignOrder(2)
    def make_s2(instructions, config, resource):
        """Call functions from wfsim to drift electrons. Since the s2 light yield implementation is a little bad how to do that?
            Make sc_gain factor 11 to large to correct for this? Also whats the se gain variation? Lets take sqrt for now
            :params: instructions, numpy array with instructions of events_tpc dtype
            :params: config, dict with configuration values for resolution
            :params: resource, instance of wfsim Resource class
        """
        xy = np.array([instructions['x'], instructions['y']]).T
        alt_xy = np.array([instructions['alt_s2_x'], instructions['alt_s2_y']]).T

        n_el = instructions['s2_area'].astype(np.int64)
        n_electron = Helpers.get_s2_charge_yield(n_electron=n_el,
                                                 positions=xy,
                                                 z_obs=instructions['z'],
                                                 config=config,
                                                 resource=resource)

        n_el = instructions['alt_s2_area'].astype(np.int64)
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
        """"
            Calculates (alt)cs1. Method taken from CorrectedAreas in straxen
            :params: instructions, numpy array with instructions of events_tpc dtype
            :params: config, dict with configuration values for resolution
            :params: resource, instance of wfsim Resource class
        """
        event_positions = np.vstack([instructions['x'], instructions['y'], instructions['z']]).T
        instructions['cs1'] = instructions['s1_area'] / resource.s1_map(event_positions)

        event_positions = np.vstack([instructions['x'], instructions['y'], instructions['z']]).T
        instructions['alt_cs1'] = instructions['alt_s1_area'] / resource.s1_map(event_positions)

    @staticmethod
    @Helpers.assignOrder(4)
    def correction_s2(instructions, config, resource):
        """"
            Calculates (alt)cs2. Method taken from CorrectedAreas in straxen
            :params: instructions, numpy array with instructions of events_tpc dtype
            :params: config, dict with configuration values for resolution
            :params: resource, instance of wfsim Resource class
        """
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