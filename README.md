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
| `--Detector`  | Detector to be used. Has to be defined in epix.detectors | `XENONnT` |
| `--DetectorConfig`  | Config file to overwrite default detector settings | in epix.detectors |
| `--EntryStop`  | How many entries from the ROOT file you want to process | all |
| `--MicroSeparation`  | DBSCAN clustering distance (mm) | `0.05` |
| `--MicroSeparationTime`  | Clustering time (ns) | `10` |
| `--TagClusterBy`  | decide if you tag the cluster (particle type, energy depositing process) according to first interaction in it (`time`) or most energetic (`energy`) | `time` |
| `--MaxDelay`  | Time after which we cut the rest of the event (ns) | `1e7` |
| `--EventRate`  | Event rate for event separation, `-1` for clean simulations or a rate > 0 (Hz) for random spacing | `-1` |
| `--Debug`  | Tell epix if you want timing outputs | `false` |
| `--OutputPath`  | Output file path | Same directory as input file |
