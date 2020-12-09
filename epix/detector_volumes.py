import inspect
import numba
import numpy as np
import awkward1 as ak

from epix.common import *

class SensitiveVolume:
    #def __init__(self, name, vol_id, roi, e_field, xe_properties, create_S2):
    def __init__(self, name, vol_id, roi):
        """
        Sensitive detector volume for which S1 and/or S2 signals should be
        generated.
        
        Args:
            name (str): Name of the volume.
            vol_id (int): Id of the volume, must be unique!
            roi (function): Function which takes x, y, z as arguments.
                Must return True or False. Must be numba njit-able.
        """
        
        self.name = name
        self.volume_id = vol_id   
        self.roi = roi
        #TODO: Check xe_properties and create_S2....
        #self.e_field = e_field
        #self.xe_properties = xe_properties
        #self.create_S2 = create_S2
        # TODO: Fix arginspect with jitted functions...
#         self._is_valid()
        
    def _is_valid(self):
        """
        Function which tests if the inputs are valid.
        """
        # Test vol_id:
        assert isinstance(self.vol_id, int), ('The volume id vol_id must be an ' 
                                              f'integer, but {self.vol_id} was '
                                              'given.')
        assert vol_id > 0, ('The volume id vol_id must be greater zero, '
                            f'but {self.vol_id} was given.')
        
        
        # Test if ROI function is defined properly:
        assert callable(self.roi), ('roi must be a callable function '
                                    'which depends on x,y,z.')
        
        args = inspect.getfullargspec(self.roi).args
        m = np.all(np.isin(['x', 'y', 'z'], args))
        m = m & (len(args) == 3)
        assert m, ('Wrong arguments for roi. Expected arguments: '
                   f'"x", "y" and "z" but {args} were given.')
        
        # Testing the electric field:
        #if not (callable(self.e_field) or 
        #        isinstance(self.e_field, (int, float))
        #       ):
        #    raise ValueError('e_field must be either a function or '
        #                     'a constant!')
        
        #if callable(self.e_field):
        #    args = inspect.getfullargspec(self.e_field).args
        #    m = np.all(np.isin(['x', 'y', 'z'], args))
        #    m = m & (len(args) == 3)
        #    assert m, ('Wrong arguments for e_field. Expected arguments: '
        #               f'"x", "y" and "z" but {args} were given.')

@numba.njit
def in_cylinder(x, y, z, min_z, max_z, max_r):
    """
    Function which checks if a given set of coordinates is within the
    boundaries of the specified cylinder. 
    """
    r = np.sqrt(x**2 + y**2) 
    m = r < max_r
    m = m & (z < max_z)
    m = m & (z >= min_z)
    return m

@numba.njit()
def clyinder_tpc(x, y, z): 
    return in_cylinder(x, y, z, z_cathode, z_gate_mesh, sensitive_volume_radius)
tpc = SensitiveVolume('tpc', 1, clyinder_tpc)

@numba.njit()
def bc(x, y, z): 
    return in_cylinder(x, y, z, z_bottom_pmts, z_cathode, sensitive_volume_radius)
below_cathode = SensitiveVolume('below_cathode', 2, bc)

def in_sensitive_volume(events, sensitive_volumes):
    """
    Function which identifies which events are inside sensitive volumes.
    """
    for ind, vol in enumerate(sensitive_volumes):
        res = ak.ArrayBuilder()
        res = _inside_sens_vol(events['x'], 
                               events['y'], 
                               events['z'], 
                               vol.roi, 
                               vol.volume_id, 
                               res)
        if ind:
            # Now we add the other results, but first test if 
            # volumes overlap.
            m = ak.any((result > 0) & (res == vol.volume_id))
            assert not m, (f'The volume {vol.name} is overlapping with'
                        ' an other volume!')   
            result = result + res.snapshot()
        else:
            # First result initates the array
            result = res.snapshot()
    return result

@numba.njit()
def _inside_sens_vol(xp, yp, zp, roi, vol_id, res):
    nevents = len(xp)
    for i in range(nevents):
        # Loop over all events
        res.begin_list()
        nint = len(xp[i]) 
        if nint:
            for j in range(nint):
                # Loop over all interactions within an event.
                if roi(x=xp[i][j], y=yp[i][j], z=zp[i][j]):
                    res.integer(vol_id)
                else:
                    res.integer(0)
        res.end_list()
    return res