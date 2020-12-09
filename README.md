# epix

or **E**lectron and **P**hoton **I**nstructions generator for **X**ENON

The job of epix is to load XENONnT Geant4 MC data and produce inputs for wfsim in csv format, using nestpy for the quanta generation and DBSCAN for the clustering of the individual steps.

## Instructions

Usable on top of the MC environment, i.e., before running it do something like

    module load singularity && singularity shell --bind /cvmfs/ --bind /project2/ --bind /dali /project2/lgrandi/xenonnt/singularity-images/xenonnt-montecarlo-development.simg.

The keyword arguments (see mc/epix/bin/run_epix.py) are:

- `--InputFile` (only one needed for the code to run): path + filename
- `--EntryStop`: how many entries from the ROOT file you want to process; defaulted to all
- `--MicroSeparation`: DBSCAN clustering distance in mm; defaulted to 0.05
- `--MicroSeparationTime`: clustering time in ns; defaulted to 10
- `--TagClusterBy`: decide if you tag the cluster (particle type, energy depositing process) according to first interaction in it (time) or most energetic (energy); defaulted to time
- `--Efield`: it can be a string, pointing to an electric field map in the format specified, or a float, which will assume this uniform field in V/cm; defaulted to 200
- `--MaxDelay`: time after which we cut the rest of the event in ns; defaulted to 1e7
- `--Timing`: boolean to tell epix if you want timing outputs (defaulted to false)
- `--OutputPath`: in case
