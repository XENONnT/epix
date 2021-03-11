import numpy as np
import numba 
import wfsim
import epix
import strax

class GenerateEvents():
    '''Class to hold all the stuff to be applied to the data. 
    The functions will be grouped together and executed by Simulator'''

    @staticmethod
    @numba.njit(cache=True,)
    def MakeS1(instructions,
                      s1_light_yield=1,
                      data):
        #Draw number of photons from poisson distribution with p the light yield
        wfsim.core.S1.get_n_photons(positions=xyz,
                                    s1_light_yield_map=...
                                    config=...)
        for ix,inst in enumerate(instructions):
            data[ix]['s1_area']= np.random.binomial(inst['amp'], p=light_yield)
        
    @staticmethod
    @numba.njit(cache=True)
    def MakeS2(instructions,
                        drift_velocity_liquid,
                        drift_time_gate,
                        electron_lifetime_liquid,
                        electron_extraction_yield,
                        data):
        #Something electron_livetime, drift_time for a delay and electron extraction efficiency(~1)
        
        drift_time_mean = - instructions['z'] / drift_velocity_liquid+drift_time_gate

        # Absorb electrons during the drift
        electron_lifetime_correction = np.exp(- 1 * drift_time_mean / electron_lifetime_liquid)
        cy = electron_extraction_yield * electron_lifetime_correction
        cy = np.clip(cy, a_min = 0, a_max = 1)

        for ix,inst in enumerate(instructions):
            n_electron = np.random.binomial(n=inst['amp'], p=cy)
            data[ix]['s2_area'] = np.random.gaussian(n=n_electron,p=se_gain)

    @staticmethod
    def CorrectionS1(data):
        m=data['type'==1]
        data[m]['cs1'] = data[m]#Whatever the correction factors are

    @staticmethod
    def CorrectionS2(data):
        m=data['type'==2]
        data[m]['cs1'] = data[m]#Whatever the correction factors are

    @staticmethod
    def SmearPositions(instructions,xy_resolution,z_resolution,data):
        for ix, inst in enumerate(instructions):
            data[ix]['x'] = np.random.normal(inst['x'],p=xy_resolution)
            data[ix]['y'] = np.random.normal(inst['y'],p=xy_resolution)
            data[ix]['z'] = np.random.normal(inst['z'],p=z_resolution)
            

class  Simulator():
    '''Simulator class for epix to go from  epix instructions to fully processed data'''

    def __init__(self,instructions_epix):
        self.ge = GenerateEvents
        self.instructions_epix=instructions_epix

    def cluster_events(self,):
        #Events have more than 1 s1/s2. Here we throw away all of them except the largest 2
        #Take the position to be that of the main s2
        #And strax wants some start and endtime, so we make something up
        instructions = np.zeros(self.epix_instructions['event_number'][-1],dtype=StraxSimulator.dtype)
        for ix in np.unique(self.instructions_epix['event_number']):
            i=instructions[ix]
            inst =  self.instructions_epix[self.instructions_epix['event_number']==ix]
            s1 = np.argsort(inst[inst['type']==1]['amp'])
            i['s1_area']=inst[s1[0]]['amp']
            if len(s1)>1:
                i['alt_s1_area']=inst[s1[1]]['amp']
            s2 = np.argsort(inst[inst['type']==2]['amp'])
            i['s2_area']=inst[s2[0]]['amp']
            if len(s2)>1:
                i['alt_s2_area']=inst[s2[1]]['amp']
            i['x']= inst[s2[0]]['x']
            i['y']= inst[s2[0]]['y']
            i['time']=ix*1000
            i['endtime']=ix*1000+1
        self.instructions=instructions
        print(self.instructions)

    def simulate(self,):
        #So this guy will take the instructions, fire the other functions to do all the (sort of ) smearings
        #Do large aray multiplication and returns data. Currently I assume there to be 1 S1 and 1 S2 per event!
        pass

    def run_simulator(self,):
        self.cluster_events()
        self.simulate()

        return self.simulated_data
        

@strax.takes_config(
    strax.Option('g4_file',help='G4 file to simulate'),
    strax.Option('epix_config',default=dict(),help='Configuration file for epix')
)
class StraxSimulator(strax.Plugin):
    provides = 'events_full_simmed'
    depends_on=()
    dtype=[('time',np.int64),
           ('endtime',np.int64),
           ('s1_area',np.float),
           ('s2_area',np.float),
           ('cs1',np.float),
           ('cs2',np.float),
           ('alt_s1_area',np.float),
           ('alt_s2_area',np.float),
           ('alt_cs1',np.float),
           ('alt_cs2',np.float),
           ('x',np.float),
           ('y',np.float)]

    def compute(self,):
        epix_instructions = epix.run_epix.main(self.config['g4_file'])
        self.Simulator=Simulator(instructions=epix_instructions)
        simulated_data = self.Simulator.run_simulator()

        return simulated_data

    def is_ready(self,)
        #For this plugin we'll smash everything into 1 chunk, should be oke
        
