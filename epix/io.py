import warnings
import configparser

SUPPORTED_OPTION = {'to_be_stored': 'getboolean',
                    'electric_field': ('getfloat', 'get'),
                    'create_S2': 'getboolean',
                    'xe_density': 'getfloat',
                    'efield_outside_map': 'getfloat',
                    }


def load_config(config_file_path):
    """
    Loads config file and returns dictionary.

    :param config_file_path:
    :return: dict
    """
    config = configparser.ConfigParser()
    config.read(config_file_path)
    sections = config.sections()
    if not len(sections):
        raise ValueError(f'Cannot load sections from config file "{config_file_path}".' 
                         'Have you specified a wrong file?')
    settings = {}
    for s in sections:
        options = {}
        c = config[s]
        for key in c.keys():
            if key not in SUPPORTED_OPTION:
                warnings.warn(f'Option "{key}" of section {s} is not supported'
                              ' and will be ignored.')
                continue
            # Get correct get method to convert string input:
            if key == 'electric_field':
                # Electric field is a bit more complicated can be
                # either a float or string:
                try:
                    getter = getattr(c, SUPPORTED_OPTION[key][0])
                    options[key] = getter(key)
                except ValueError:
                    getter = getattr(c, SUPPORTED_OPTION[key][1])
                    options[key] = getter(key)
            else:
                try:
                    getter = getattr(c, SUPPORTED_OPTION[key])
                    options[key] = getter(key)
                except Exception as e:
                    raise ValueError(f'Cannot load "{key}" from section "{s}" in config file.') from e

        settings[s] = options
    return settings