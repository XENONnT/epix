from scipy import interpolate
import numpy as np

def clean_separation(n_events, MaxDelay):
    """
    Function to generate evenly spaced event times.

    Args:
        n_events (int): Number of events 
        MaxDelay (float): Time difference between events. Should be large enought to 
            prevent event pile-up.

    Returns:
        dt (numpy.array): Array containing the start times of the events.
    """
    
    dt = np.arange(0, n_events)+np.arange(0, n_events)/10
    dt *= MaxDelay

    return dt

def times_from_fixed_rate(rate, n_events):
    """
    Function to generate event times with a fixed rate.

    The event times are drawn from a uniform distribution.
    For higher rates pile-up is possible. The normalization
    is achived by a variable overall simulation length in time.

    !Rate normalization is only valid for one simualtion job!

    Args:
        rate (int or float): Mean event rate in Hz. 
        n_events (int): Number of events

    Returns:
        event_times (numpy.array): Array containing the start times of the events.
    """
    
    simulation_time = n_events/np.float(rate) #seconds
    simulation_time_nanoseconds = simulation_time*1e9
    
    event_times = np.sort(np.random.uniform(low=0, high=simulation_time_nanoseconds, size=n_events))
    
    return event_times


def times_from_variable_rate(variable_event_rate, times, n_events):
    """
    Function to generate event times with a variable rate.

    Event times are drawn from a given distribution. The distribution
    is interpolated and used in _sample_random_numbers_from_distribution.

    !Normalization not working at the moment. Events are distributed
    according to the given distribution but the absolute rates are only
    achived when the right number of events is simulated!

    Args:
        variable_event_rate (np.array): Array containing the event rates in Hz over time.
        times (np.array): Corresponding times to the given rates in [s].
        n_events (int): Number of events

    Returns:
        event_times (np.array): Array containing the start times of the events.
    """
    
    times_nanoseconds = times*1e9
    
    variable_rate_interpolation = interpolate.interp1d(times_nanoseconds, variable_event_rate)
    
    event_times = _sample_random_numbers_from_distribution(times_nanoseconds[0],
                                                          times_nanoseconds[-1],
                                                          n_events,
                                                          variable_rate_interpolation,
                                                          max(variable_event_rate)+1
                                                         )
    
    return np.sort(event_times)

def _sample_random_numbers_from_distribution(start_time, stop_time, n_events, interp, max_rate):
    """
    Function to sample random numbers from a given distribution.

    Args:
        start_time (float): lower limit to the sampled numbers
        stop_time (float): uper limit to the sampled numbers
        n_events (int): Number of events
        interp (function): Function that takes a time as argument and returns a rate.
            Here: scipy interpolate of the given distribution
        max_rate (float or int): upper limit for the dummy random number. Must be larger
            than the highest output value of interp

    Returns:
        sampled_rndm_times (np.array): Array containing the start times of the events.
    """
    
    n_work = 100*n_events
    rndm_times = np.random.uniform(low=start_time, high=stop_time, size=n_work)
    rndm_dummy = np.random.uniform(low=0, high=max_rate, size=n_work)
    
    cut_function = interp(rndm_times)
    
    sampled_rndm_times = rndm_times[rndm_dummy<=cut_function]
    sampled_rndm_times = np.random.choice(sampled_rndm_times, replace=False, size=n_events)
    
    return sampled_rndm_times