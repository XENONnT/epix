from .helpers import Helpers
import numpy as np

class GenerateEvents:
    """Class to hold all the stuff to be applied to data.
    The functions will be grouped together and executed by Simulator"""

    @staticmethod
    @Helpers.assign_order(0)
    def make_drift_time(i, config, resource):
        """
        Calculate drift_time and alt_s2_interaction_drift_time
            :params: i, numpy array with instructions of events_tpc dtype
            :params: config, dict with configuration values for resolution
            :params: resource, instance of wfsim Resource class
        """
        for alt_s2, alt_s2_int in [('', ''), ('alt_s2_', 'alt_s2_interaction_')]:
            i[f'{alt_s2_int}drift_time'] = Helpers.get_drift_time(i[f'{alt_s2}z_true'],
                                                                  np.array([i[f'{alt_s2}x_true'], i[f'{alt_s2}y_true']]).T,
                                                                  config, resource)

    @staticmethod
    @Helpers.assign_order(1)
    def get_true_polar_coordinates(i):
        for alt_s2, alt in [('', '', ''), ('alt_s2_', 'alt_')]:
            i[f'{alt_s2}r_true'] = np.sqrt(i[f'{alt_s2}x_true'] ** 2 + i[f'{alt_s2}y_true'] ** 2)
            i[f'{alt_s2}theta_true'] = np.arctan2(i[f'{alt_s2}y_true'], i[f'{alt_s2}x_true'])

    @staticmethod
    @Helpers.assign_order(2)
    def get_naive_positions(i, config, resource):
        """
        Get the uncorrected observed positions s2_x, s2_y in gas gap based on true position
        as well as z_naive from drift time
            :params: i, numpy array with instructions of events_tpc dtype
            :params: config, dict with configuration values for resolution
            :params: resource, instance of wfsim Resource class
        """
        v_drift = config['drift_velocity_liquid'] # cm/ns
        for alt_s2_interaction, alt_s2, alt in [('', '', ''), ('alt_s2_interaction_', 'alt_s2_', 'alt_')]:
            i[f'{alt_s2}r_naive'] = resource.fd_comsol(np.array([i[f'{alt_s2}r_true'], i[f'{alt_s2}z_true']]).T,
                                          map_name='r_distortion_map')
            i[f'{alt}s2_x'] = i[f'{alt_s2}r_naive'] * np.cos(i[f'{alt_s2}theta_true'])
            i[f'{alt}s2_y'] = i[f'{alt_s2}r_naive'] * np.sin(i[f'{alt_s2}theta_true'])

            i[f'{alt_s2}z_naive'] = -(i[f'{alt_s2_interaction}drift_time'] - config['drift_time_gate']) * v_drift  # in cm


    @staticmethod
    @Helpers.assign_order(3)
    def smear_positions(i, config):
        """Take initial positions and apply gaussian smearing with some resolution to get the measured position
            :params: instructions, numpy array with instructions of events_tpc dtype
            :params: config, dict with configuration values for resolution
        """
        # TODO check if this makes sense. Can we get the resolutions from somewhere?
        for alt_s2, alt in [('', '', ''), ('alt_s2_', 'alt_')]:
            i[f'{alt}s2_x'] = np.random.normal(i[f'{alt}s2_x'], config['xy_resolution'])
            i[f'{alt}s2_y'] = np.random.normal(i[f'{alt}s2_y'], config['xy_resolution'])
            i[f'{alt_s2}z_naive'] = np.random.normal(i[f'{alt_s2}z_naive'], config['z_resolution'])

    @staticmethod
    @Helpers.assign_order(4)
    def get_corrected_positions(i, resource):
        """
        Apply FDC to observed positions s2_x, s2_y to get x, y
            :params: instructions, numpy array with instructions of events_tpc dtype
            :params: config, dict with configuration values for resolution
            :params: resource, instance of wfsim Resource class
        """
        for alt_s2, alt, fdc in [('','', ''), ('alt_s2_', 'alt_', '_fdc')]:
            i[f'{alt_s2}r_field_distortion_correction'] = resource.fdc_map(np.array([i[f'{alt}s2_x'], i[f'{alt}s2_y'],
                                                                                     i[f'{alt_s2}z_naive']]).T)
            i[f'{alt_s2}r{fdc}'] = i[f'{alt_s2}r_naive'] + i[f'{alt_s2}r_field_distortion_correction']
            i[f'{alt_s2}x{fdc}'] = i[f'{alt_s2}r{fdc}']*np.cos(i[f'{alt_s2}theta_true'])
            i[f'{alt_s2}y{fdc}'] = i[f'{alt_s2}r{fdc}']*np.sin(i[f'{alt_s2}theta_true'])
            i[f'{alt_s2}theta'] = np.arctan2(i[f'{alt_s2}y{fdc}'], i[f'{alt_s2}x{fdc}'])

            i[f'{alt_s2}z_dv_corr'] = resource.fd_comsol(np.array([i[f'{alt_s2}r_true'], i[f'{alt_s2}z_true']]).T,
                                                         map_name='z_distortion_map')
            i[f'{alt_s2}z'] = i[f'{alt_s2}z_naive']  # Following straxen z = z_naive for now.



    @staticmethod
    @Helpers.assign_order(5)
    def make_s1(i, config, resource):
        """Build the s1s. Takes number of quanta and calculates the (alt) s1 area using wfsim
            :params: i, numpy array with instructions of events_tpc dtype
            :params: config, dict with configuration values for resolution
            :params: resource, instance of wfsim Resource class
        """
        xyz = np.vstack([i['x_true'], i['y_true'], i['z_true']]).T

        num_ph = i['s1_area'].astype(np.int64)
        
        # Here a WFsim function is called which remove the dpe
        # we have to introduce it again in fast simulator
        n_photons = Helpers.get_s1_light_yield(n_photons = num_ph,
                                               positions = xyz,
                                               s1_lce_map = resource.s1_lce_correction_map,
                                               config = config) * (1 + config['p_double_pe_emision']) 
        
        i['s1_area'] = Helpers.get_s1_area_with_spe(resource.mean_photon_area_distribution,
                                                               n_photons.astype(np.int64))

    @staticmethod
    @Helpers.assign_order(6)
    def make_s2_area(i, config, resource):
        """
        Call functions from wfsim to drift electrons. Since the s2 light yield implementation is a little bad how to do that?
        Make sc_gain factor 11 too large to correct for this? Also, what's the se gain variation? Let's take sqrt for now
        :params: i, numpy array with instructions of events_tpc dtype
        :params: config, dict with configuration values for resolution
        :params: resource, instance of wfsim Resource class
        """
        for alt_s2, alt in [('', ''), ('alt_s2_', 'alt_')]:
            xy_true = np.array([i[f'{alt_s2}x_true'], i[f'{alt_s2}y_true']]).T
            xy = np.array([i[f'{alt_s2}x'], i[f'{alt_s2}y']]).T
            n_el = i[f'{alt}s2_area'].astype(np.int64)
            n_electron = Helpers.get_s2_charge_yield(n_electron = n_el,
                                                     xy = xy_true,
                                                     z = i[f'{alt_s2}z_true'],
                                                     config = config,
                                                     resource = resource)
            # Here a WFsim function is called which removes the dpe
            # we have to introduce it again in fast simulator
            sc_gain = Helpers.get_s2_light_yield(positions=xy,
                                                 config=config,
                                                 resource=resource)
            sc_gain *= (1 + config['p_double_pe_emision']) # intentional typo
            i[f'{alt}s2_area'] = n_electron * sc_gain


    @staticmethod
    @Helpers.assign_order(7)
    def correction_s1(i, resource):
        """
            Calculates cs1. Method taken from CorrectedAreas in straxen
            :params: i, numpy array with instructions of events_tpc dtype
            :params: config, dict with configuration values for resolution
            :params: resource, instance of wfsim Resource class
        """

        event_positions = np.vstack([i['x_true'], i['y_true'], i['z_true']]).T

        # Where does 0.15 come from ? Mean of s1_lce_correction_map in -130 < z < -20 and r < 50
        # the_map = []
        # for xyz in resource.s1_lce_correction_map.coordinate_system:
        #     r = np.sqrt(xyz[0]**2 + xyz[1]**2)
        #     if (-130 < xyz[2] < -20) & ( r < 50):
        #         the_map.append(np.squeeze(resource.s1_lce_correction_map(xyz))[0])
        # print(np.mean(the_map))

        i['cs1'] = i['s1_area'] / (resource.s1_lce_correction_map(event_positions)[:, 0]/0.1581797073725071)

    @staticmethod
    @Helpers.assign_order(8)
    def correction_s2(i, config, resource):
        """"
            Calculates (alt)cs2. Method taken from CorrectedAreas in straxen
            :params: i, numpy array with instructions of events_tpc dtype
            :params: config, dict with configuration values for resolution
            :params: resource, instance of wfsim Resource class
        """
        for alt_s2_interaction, alt in [('', ''), ('alt_s2_interaction_', 'alt_')]:
            lifetime_corr = np.exp(i[f'{alt_s2_interaction}drift_time'] / config['electron_lifetime_liquid'])

            # S2(x,y) corrections use the observed S2 positions
            xy = np.vstack([i[f'{alt}s2_x'], i[f'{alt}s2_y']]).T
            alt_s2_positions = np.vstack([i['alt_s2_x'], i['alt_s2_y']]).T

            # Why S2 does not need the same treatment of S1 ?
            i[f'{alt}cs2'] = (i[f'{alt}s2_area'] * lifetime_corr / resource.s2_correction_map(xy))

        #alt_s2_nan = i['alt_s2_area'] < 1e-6
        #i['alt_s2_x'][alt_s2_nan] = 0.0
        #i['alt_s2_y'][alt_s2_nan] = 0.0
        #i['alt_s2_z'][alt_s2_nan] = 0.0
        #i['alt_s2_drift_time'][alt_s2_nan] = 0.0
