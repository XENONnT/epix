import inspect
import numba
import numpy as np
import awkward as ak
import epix


def init_detector(detector_name, config_file):
    """
    Function which initializes the detector. Reads in config and
    overwrites default settings.

    :param detector_name: Detector as defined in detector.py.
    :param config_file: File to be used to customize the default
        settings e.g. electric field.

    :returns: A list of volumes which represent the detector.
    """
    if not hasattr(epix.detectors, detector_name):
        raise ValueError('The specified "detector_name" must be defined in "epix.detectors". '
                         f'Was not able to find {detector_name}.')

    # Getting default volume object by name:
    volumes = getattr(epix.detectors, detector_name)
    volumes, _ = volumes()

    if config_file:
        # Load config and overwrite default settings for volumes.
        config = epix.io.load_config(config_file)

        # Update default setting with new settings:
        for volume_name, options in config.items():
            if volume_name in volumes:
                default_options = volumes[volume_name]
            else:
                raise ValueError(f'Cannot find "{volume_name}" among the volumes to be initialized.'
                                 f' Valid volumes are: {[k for k in volumes.keys()]}')
            for key, values in options.items():
                default_options[key] = values

    # Now loop over default options and init volumes:
    detector = []
    args_volumes = inspect.signature(epix.SensitiveVolume).parameters.keys()
    for volume_name, default_options in volumes.items():
        if not default_options['to_be_stored']:
            # This volume should be skipped
            continue

        kwargs = {}
        for option_key, option_value in default_options.items():
            if option_key not in args_volumes:
                # This option is not needed to init the volume
                continue

            if option_key == 'electric_field' and isinstance(option_value, str):
                # Special case we have to init the efield first:
                efield = epix.MyElectricFieldHandler(option_value)
                option_value = lambda x, y, z: efield.get_field(x, y, z,
                                                                outside_map=default_options['efield_outside_map']
                                                                )
            kwargs[option_key] = option_value

        detector.append(epix.SensitiveVolume(name=volume_name, **kwargs))

    # Test if volume name and id are unique:
    names = [volume.name for volume in detector]
    ids = [volume.volume_id for volume in detector]
    if len(np.unique(names)) != len(names) or len(np.unique(ids)) != len(ids):
        raise ValueError(f'Detector volumes must have unique names and ids! Found {names} and {ids}.')

    return detector


class SensitiveVolume:
    def __init__(self, name, volume_id, roi, electric_field, create_S2, xe_density=2.862):
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
        self.volume_id = volume_id
        self.roi = roi

        self.electric_field = electric_field
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
        if not (callable(self.electric_field) or
                isinstance(self.electric_field, (int, float))):
            raise ValueError('e_field must be either a function or '
                             'a constant!')

        if callable(self.electric_field):
            args = inspect.getfullargspec(self.electric_field).args
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
    
    Args:
        x,y,z: Coordinates of the interaction
        min_z: Inclusive lower z boundary
        max_z: Exclusive upper z boundary
        max_r: Exclusive radial boundary
    """
    r = np.sqrt(x ** 2 + y ** 2)
    m = r < max_r
    m = m & (z < max_z)
    m = m & (z >= min_z)
    return m


def in_sensitive_volume(events, sensitive_volumes):
    """
    Function which identifies which events are inside sensitive volumes.

    Further, tests if sensitive volumes overlap. Assigns volume id, and
    xenon density to interactions.

    Args:
        events (ak.records): Awkward record of the interactions.
        sensitive_volumes (dict): Dictionary of the different volumes
            defined via the SensitiveVolume class.

    Returns:
        ak.array: Awkward array containing the event ids.
    """
    if len(events) == 0:
        res_det_dtype = [('xe_density', 'float64'),
                         ('vol_id', 'int64'),
                         ('create_S2', 'bool'),
                         ]
        return ak.from_numpy(np.empty(0, dtype=res_det_dtype))

    for ind, vol in enumerate(sensitive_volumes):
        res = ak.ArrayBuilder()
        res = _inside_sens_vol(events['x'],
                               events['y'],
                               events['z'],
                               vol.roi,
                               vol.volume_id,
                               vol.xe_density,
                               vol.create_S2,
                               res)
        if ind:
            # Now we add the other results, but first test if 
            # volumes overlap. Only possible if we explicitly loop
            # over everything. This reduces performance but adds layer of
            # safety.
            m = (result['vol_id'] > 0) & (res['vol_id'] == vol.volume_id)
            if ak.any(m):
                overlapping_id = result[m][0]
                # Get volume name:
                name = [vol.name for vol in sensitive_volumes if vol.volume_id == overlapping_id][0]
                raise ValueError(f'The volume {vol.name} is overlapping with'
                                 f' volume {name}!')
            new_results = res.snapshot()
            for field in result.fields:
                # Workaround since we cannot sum up records-arrays anymore
                result[field] = result[field] + new_results[field]
        else:
            # First result initiates the array
            result = res.snapshot()
    return result


@numba.njit()
def _inside_sens_vol(xp, yp, zp, roi, vol_id, vol_density, create_S2, res):
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
                    res.field('create_S2')
                    res.boolean(create_S2)
                else:
                    res.field('vol_id')
                    res.integer(0)
                    res.field('xe_density')
                    res.real(0)
                    res.field('create_S2')
                    res.boolean(False)
                res.end_record()
        res.end_list()
    return res
