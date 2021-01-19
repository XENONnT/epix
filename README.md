# epix

**E**lectron and **P**hoton **I**nstructions generator for **X**ENON

The job of epix is to load XENONnT Geant4 MC data and produce inputs for wfsim, using nestpy for the quanta generation and DBSCAN for the clustering of the individual steps.

## Installation

Assuming you are working on top of our montecarlo_environment (`/project2/lgrandi/xenonnt/singularity-images/xenonnt-montecarlo-development.simg`):
```
git clone https://github.com/XENONnT/epix
cd epix
python3 setup.py develop --user
```

## Usage

Only an input file is needed to run epix:
```
bin/run_epix --InputFile <path_and_filename>
```
The other keyword arguments are:
| Argument | Description | Default |
|--------------|------------------------|---|
| `--EntryStop`  | How many entries from the ROOT file you want to process | all |
| `--MicroSeparation`  | DBSCAN clustering distance (mm) | `0.05` |
| `--MicroSeparationTime`  | Clustering time (ns) | `10` |
| `--TagClusterBy`  | decide if you tag the cluster (particle type, energy depositing process) according to first interaction in it (`time`) or most energetic (`energy`) | `time` |
| `--Efield`  | It can be a string, pointing to an electric field map in the format specified, or a float, which will assume this uniform field (V/cm) | `200` |
| `--MaxDelay`  | Time after which we cut the rest of the event (ns) | `1e7` |
| `--EventRate`  | Event rate for event separation, `-1` for clean simulations; a rate > 0 (Hz) for random spacing; or a csv file containing time (s) and rate (Hz) for a not fixed rate | `-1` |
| `--Timing`  | Boolean to tell epix if you want timing outputs | `false` |
| `--OutputPath`  | Output file path | Same directory as input file |
