import numpy as np
from straxen import InterpolatingMap, get_resource, get_config_from_cmt
from wfsim.load_resource import make_map
from copy import deepcopy
from scipy.interpolate import interp1d
import numpy as np

@staticmethod
def assignOrder(order):
    # @decorator
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
def Get_S1_LY(n_photons, positions, s1_light_yield_map, config):
    if config['detector'] == 'XENONnT':
        ly = np.squeeze(s1_light_yield_map(positions),
                        axis=-1) / (1 + config['p_double_pe_emision'])
    elif config['detector'] == 'XENON1T':
        ly = s1_light_yield_map(positions)
        ly *= config['s1_detection_efficiency']

    n_photons = np.random.binomial(n=n_photons.astype(int64), p=ly)
    return n_photons


@staticmethod
def Get_S2_LY(positions, config, resource):
    if config['detector'] == 'XENONnT':
        sc_gain = np.squeeze(resource.s2_light_yield_map(positions), axis=-1) \
                  * config['s2_secondary_sc_gain']
    elif config['detector'] == 'XENON1T':
        sc_gain = resource.s2_light_yield_map(positions) \
                  * config['s2_secondary_sc_gain']
    return sc_gain

class Resource():
    def __init__(self, config) -> None:
        self._load_resource(config['configuration_files'])

    def _load_resource(self, config):
        '''Loads needed configs to call wfsim. We need s1/s2 light yield maps,
        spe distibutions and corrections maps'''
        self.run_id = '1000'  # ? I just put this to something for the cmt
        self.s1_map = InterpolatingMap(
            get_resource(config['s1_relative_lce_map']))
        self.s2_map = InterpolatingMap(
            get_resource(get_config_from_cmt(self.run_id, config['s2_xy_correction_map'])))

        map_data = straxen.get_resource(config['s1_pattern_map'], fmt='pkl')
        self.s1_pattern_map = straxen.InterpolatingMap(map_data)
        # self.s1_pattern_map = make_map(config['s1_pattern_map'], fmt='pkl')
        lymap = deepcopy(self.s1_pattern_map)
        lymap.data['map'] = np.sum(lymap.data['map'][:][:][:], axis=3, keepdims=True)
        lymap.__init__(lymap.data)
        self.s1_light_yield_map = lymap

        map_data = straxen.get_resource(config['s2_pattern_map'], fmt='pkl')
        self.s2_pattern_map = straxen.InterpolatingMap(map_data)
        # self.s2_pattern_map = make_map(config['s2_pattern_map'], fmt='pkl')
        lymap = deepcopy(self.s2_pattern_map)
        lymap.data['map'] = np.sum(lymap.data['map'][:][:], axis=2, keepdims=True)
        lymap.__init__(lymap.data)
        self.s2_light_yield_map = lymap

        self.photon_area_distribution = helpers.average_spe_distribution(
            get_resource(config['photon_area_distribution'], fmt='csv'))