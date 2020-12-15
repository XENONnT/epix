import inspect
import numba
import numpy as np
import awkward1 as ak

from epix.common import *


class SensitiveVolume:
    def __init__(self, name, vol_id, roi, e_field, create_S2, xe_density=2.862):
        """
        Sensitive detector volume for which S1 and/or S2 signals should be
        generated.
        
        Args:
            name (str): Name of the volume.
            vol_id (int): Id of the volume, must be unique and greater
                zero. Zero is reserved for event which are not in any
                volume.
            roi (function): Function which takes x, y, z as arguments.
                Must return True or False. Must be numba njit-able.
            e_field (function): Interpolated efield map associated
                with volume.
            create_S2 (bool): Indicates if electrons should be generated
                for wfsim.

        Kwargs:
            xe_properties (dict): Dictionary of additional xenon
                properties which should be forwarded to nest.
        """
        self.name = name
        self.volume_id = vol_id   
        self.roi = roi

        self.e_field = e_field
        self.xe_density = xe_density
        self.create_S2 = create_S2
        self._is_valid()
        
    def _is_valid(self):
        """
        Function which tests if the inputs are valid.
        """
        # Test vol_id:
        assert isinstance(self.volume_id, int), ('The volume id vol_id must be an ' 
                                                 f'integer, but {self.volume_id} was '
                                                  'given.')
        assert self.volume_id > 0, ('The volume id vol_id must be greater zero, '
                                    f'but {self.volume_id} was given.')
        
        
        # Test if ROI function is defined properly:
        assert callable(self.roi), ('roi must be a callable function '
                                    'which depends on x,y,z.')
        
        # Testing the electric field:
        if not (callable(self.e_field) or
                isinstance(self.e_field, (int, float))):
            raise ValueError('e_field must be either a function or '
                             'a constant!')
        
        if callable(self.e_field):
            args = inspect.getfullargspec(self.e_field).args
            m = np.all(np.isin(['x', 'y', 'z'], args))
            m = m & (len(args) == 3)
            assert m, ('Wrong arguments for e_field. Expected arguments: '
                       f'"x", "y" and "z" but {args} were given.')
        # Cannot add a specific if **kwargs are valid properties. Cannot
        # inspect nestpy functions.

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


def in_sensitive_volume(events, sensitive_volumes):
    """
    Function which identifies which events are inside sensitive volumes.

    Stores volume ids and xenon densities.

    Args:
        events (ak.records): Awkward record of the interactions.
        sensitive_volumes (dict): Dictionary of the different volumes
            defined via the SensitiveVolume class.

    Returns:
        ak.array: Awkward array containing the event ids.

    #TODO: Also add efields here
    """
    for ind, vol in enumerate(sensitive_volumes.values()):
        res = ak.ArrayBuilder()
        res = _inside_sens_vol(events['x'], 
                               events['y'], 
                               events['z'], 
                               vol.roi, 
                               vol.volume_id,
                               vol.xe_density,
                               res)
        if ind:
            # Now we add the other results, but first test if 
            # volumes overlap. Only possible if we explicitly loop
            # over everything. This reduces performance but adds layer of
            # safety.
            m = ak.any((result > 0) & (res == vol.volume_id))
            if np.any(m):
                overlapping_id = result[m][0]
                # Get volume name:
                name = [vol.name for vol in sensitive_volumes if vol.volume_id == overlapping_id][0]
                raise ValueError(f'The volume {vol.name} is overlapping with'
                                 f' volume {name}!')
            result = result + res.snapshot()  # Only works since events should be in two different volumes
        else:
            # First result initiates the array
            result = res.snapshot()
    return result

@numba.njit()
def _inside_sens_vol(xp, yp, zp, roi, vol_id, vol_density, res):
    nevents = len(xp)
    for i in range(nevents):
        # Loop over all events
        res.begin_list()
        nint = len(xp[i]) 
        if nint:
            for j in range(nint):
                # Loop over all interactions within an event.
                res.begin_record()
                if roi(x=xp[i][j], y=yp[i][j], z=zp[i][j]):
                    res.field('xe_density')
                    res.real(vol_density)
                    res.field('vol_id')
                    res.integer(vol_id)
                else:
                    res.field('vol_id')
                    res.integer(0)
                    res.field('xe_density')
                    res.real(0)
                res.end_record()
        res.end_list()
    return res

def get_volume_properties():
    """
    Function which adds properties

    :return:
    """
    raise NotImplementedError