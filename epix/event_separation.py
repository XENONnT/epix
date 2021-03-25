import numpy as np

def times_for_clean_separation(n_events, MaxDelay):
    """
    Function to generate evenly spaced event times.

    Args:
        n_events (int): Number of events 
        MaxDelay (float): Time difference between events. Should be large enough to 
            prevent event pile-up.

    Returns:
        dt (numpy.array): Array containing the start times of the events.
    """
    
    dt = np.arange(1, n_events+1)+np.arange(1, n_events+1)/10
    dt *= MaxDelay

    return dt

def times_from_fixed_rate(rate, n_events, n_simulated_events, offset=0):
    """
    Function to generate event times with a fixed rate.

    The event times are drawn from a uniform distribution.
    For higher rates pile-up is possible. The normalization
    is achieved by a variable overall simulation length in time.

    !Rate normalization is only valid for one simualtion job!

    Args:
        rate (int or float): Mean event rate in Hz. 
        n_events (int): Number of events in file
        n_events_simulated (int): True number of events which were
            simulated.
        offset (int or float): Time offset to shift all event times to larger values.

    Returns:
        event_times (numpy.array): Array containing the start times of the events.
    """
    
    simulation_time = n_simulated_events/rate
    simulation_time *= 1e9
    
    event_times = np.sort(np.random.uniform(low=offset, high=simulation_time+offset, size=n_events))
    
    return event_times
