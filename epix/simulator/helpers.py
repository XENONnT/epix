import numpy as np
from straxen import InterpolatingMap, get_resource, get_config_from_cmt
from wfsim.load_resource import make_map
from copy import deepcopy
from scipy.interpolate import interp1d
import numpy as np

def assignOrder(order):
  @decorator
  def do_assignment(to_func):
    to_func.order = order
    return to_func
  return do_assignment

class Foo():

  @assignOrder(1)
  def bar(self):
    print "bar"

  @assignOrder(2)
  def foo(self):
    print "foo"

  #don't decorate functions you don't want called
  def __init__(self):
    #don't call this one either!
    self.egg = 2

def average_spe_distibution(spe_shapes):
    uniform_to_pe_arr = []
    for ch in spe_shapes.columns[1:]:  # skip the first element which is the 'charge' header
        if spe_shapes[ch].sum() > 0:
            # mean_spe = (spe_shapes['charge'].values * spe_shapes[ch]).sum() / spe_shapes[ch].sum()
            scaled_bins = spe_shapes['charge'].values # / mean_spe
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
    spe_distribution=np.mean(uniform_to_pe_arr,axis=0)
    return spe_distribution


class Resource():
    def __init__(self,config) -> None:
        self._load_resource(config)        
        
    def _load_resource(self, config):
        '''Loads needed configs to call wfsim. We need s1/s2 light yield maps, 
        spe distibutions and corrections maps'''
        self.run_id='1000'#? I just put this to something for the cmt
        self.s1_map = InterpolatingMap(
                get_resource(config['s1_relative_lce_map']))
        self.s2_map = InterpolatingMap(
                get_resource(get_config_from_cmt(self.run_id, self.config['s2_xy_correction_map'])))

        self.s1_pattern_map = make_map(config['s1_pattern_map'], fmt='pkl')
        lymap = deepcopy(self.s1_pattern_map)
        lymap.data['map'] = np.sum(lymap.data['map'][:][:][:], axis=3, keepdims=True)
        lymap.__init__(lymap.data)
        self.s1_light_yield_map = lymap

        self.s2_pattern_map = make_map(config['s2_pattern_map'], fmt='pkl')
        lymap = deepcopy(self.s2_pattern_map)
        lymap.data['map'] = np.sum(lymap.data['map'][:][:], axis=2, keepdims=True)
        lymap.__init__(lymap.data)
        self.s2_light_yield_map = lymap

        self.photon_area_distribution = average_spe_distibution(get_resource(config['photon_area_distribution'], fmt='csv'))


        
