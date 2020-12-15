from scipy import interpolate
import numpy as np

def clean_separation(n_events, MaxDelay):
    
    dt = np.arange(0, n_events) + np.arange(0, n_events) / 10
    dt *= MaxDelay
    
    return dt

def times_from_fixed_rate(rate, n_events):
    
    simulation_time = n_events/rate #seconds
    simulation_time_nanoseconds = simulation_time*1e9
    
    event_times = np.sort(np.random.uniform(low = 0, high = simulation_time_nanoseconds, size = n_events))
    
    return event_times


def times_from_variable_rate(variable_event_rate, times, n_events):
    
    times_nanoseconds = times*1e9
    
    variable_rate_interpolation = interpolate.interp1d(times_nanoseconds, variable_event_rate)
    
    event_times = sample_random_numbers_from_distribution(times_nanoseconds[0],
                                                          times_nanoseconds[-1],
                                                          n_events,
                                                          variable_rate_interpolation,
                                                          max(variable_event_rate)+1
                                                         )
    
    return np.sort(event_times)

def sample_random_numbers_from_distribution(start_time, stop_time, n, interp, max_rate):
    
    n_work = 100*n
    rndm_times = np.random.uniform(low = start_time, high = stop_time, size = n_work)
    rndm_dummy = np.random.uniform(low = 0, high = max_rate, size = n_work)
    
    cut_function = interp(rndm_times)
    
    sampled_rndm_times = rndm_times[rndm_dummy<=cut_function]
    sampled_rndm_times = np.random.choice(sampled_rndm_times, replace = False, size = n)
    
    return sampled_rndm_times