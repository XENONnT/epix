import wfsim
from scipy.interpolate import interp1d
import numpy as np
import numba


# Numba and classes still are not a match made in heaven
@numba.njit
def _merge_these_clusters(s2_area1, z1, s2_area2, z2):
    sensitive_volume_ztop = 0  # it's the ground mesh, the top liquid level is at 2.7; // mm
    max_s2_area = max(s2_area1, s2_area2)
    if max_s2_area > 5000:
        SeparationDistanceIntercept = 0.00024787 * 5000. + 3.4056346550312973
        SeparationDistanceSlope = 5.5869678412887262e-07 * 5000. + 0.0044792968
    else:
        SeparationDistanceIntercept = \
            0.00024787 * max_s2_area + 3.4056346550312973
        SeparationDistanceSlope = \
            5.5869678412887262e-07 * max_s2_area + 0.0044792968
    SeparationDistance = \
        SeparationDistanceIntercept - \
        SeparationDistanceSlope * (-sensitive_volume_ztop + \
                                   (z1 + z2) * 0.5)
    return z1 - z2 < SeparationDistance

@numba.njit
def _merge_these_clusters_nt_res(s2_area1, z1, s2_area2, z2):
    sensitive_volume_ztop = 0  # [cm]
    SeparationDistance = 1.6  # [cm], the worst case from [[weiss:analysis:he:zresoultion_zdependence]]
    return np.abs(z1 - z2) < 0.01 # SeparationDistance

# @numba.njit
def _merge_these_clusters_nt_res_jaron(s2_area1, z1, s2_area2, z2, tree):
    tree = pickle.load(open('/dali/lgrandi/jgrigat/s2_separation/s2_separation_decision_tree_fast_sim.p', 'rb+'))
    return bool(tree.predict([[z1, z2-z1, s2_area1, s2_area2]]))

class Helpers():
    @staticmethod
    def assignOrder(order):
        """This is a trick to have the calculation function be executed in a particular order"""

        def do_assignment(to_func):
            to_func.order = order
            return to_func

        return do_assignment

    @staticmethod
    def average_spe_distribution(spe_shapes):
        """We take the spe distribution from all channels and take the average to be our spe distribution to draw photon areas from"""
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
            # For Inverse_transform_sampling methode
            grid_scale = interp1d(cdf, scaled_bins,
                                  bounds_error = False,
                                  fill_value = (scaled_bins[0], scaled_bins[-1]))(grid_cdf)

            uniform_to_pe_arr.append(grid_scale)
        spe_distribution = np.mean(uniform_to_pe_arr, axis=0)
        return spe_distribution

    @staticmethod
    @numba.njit
    def macro_cluster_events(instructions):
        """Loops over all instructions, checks if it's an s2 and if there is another s2 within the same event
            within the macro cluster distance, if it is they are merged."""
        for ix1, _ in enumerate(instructions):
            if instructions[ix1]['type'] != 2:
                continue
            for ix2 in range(1, len(instructions[ix1:]) + 1):
                if instructions[ix1 + ix2]['type'] != 2:
                    continue
                if instructions[ix1]['event_number'] != instructions[ix1 + ix2]['event_number']:
                    break
                # _nt_res
                if _merge_these_clusters_nt_res_jaron(instructions[ix1]['amp'], instructions[ix1]['z'],
                                                      instructions[ix1 + ix2]['amp'], instructions[ix1 + ix2]['z']):
                    instructions[ix1 + ix2]['x'] = (instructions[ix1]['x'] + instructions[ix1 + ix2]['x']) * 0.5
                    instructions[ix1 + ix2]['y'] = (instructions[ix1]['y'] + instructions[ix1 + ix2]['y']) * 0.5
                    instructions[ix1 + ix2]['z'] = (instructions[ix1]['z'] + instructions[ix1 + ix2]['z']) * 0.5
                    instructions[ix1 + ix2]['amp'] = int((instructions[ix1]['amp'] + instructions[ix1 + ix2]['amp']))
                    instructions[ix1]['amp'] = -1  # flag to throw this instruction away later
                    break

    @staticmethod
    def get_s1_area_with_spe(spe_distribution, num_photons):
        """
            :params: spe_distribution, the spe distribution to draw photon areas from
            :params: num_photons, number of photons to draw from spe distribution
        """
        s1_area_spe = []
        for n_ph in num_photons:
            s1_area_spe.append(np.sum(spe_distribution[
                                          (np.random.random(n_ph) * len(spe_distribution)).astype(np.int64)]))

        return np.array(s1_area_spe)

    @staticmethod
    def get_s1_light_yield(n_photons, positions, s1_lce_map, config):
        """See WFsim.s1.get_n_photons"""
        return wfsim.S1.get_n_photons(n_photons = n_photons,
                                      positions = positions,
                                      s1_lce_correction_map = s1_lce_map,
                                      config = config)

    @staticmethod
    def get_s2_light_yield(positions, config, resource):
        """See WFsim.s2.get_s2_light_yield"""
        return wfsim.S2.get_s2_light_yield(positions = positions,
                                           config = config,
                                           resource = resource)

    @staticmethod
    def get_s2_charge_yield(n_electron, positions, z_obs, config, resource):
        """See wfsim.s2.get_electron_yield"""
        return wfsim.S2.get_electron_yield(n_electron = n_electron,
                                           xy_int = positions,
                                           z_int = z_obs,
                                           config = config,
                                           resource = resource)
