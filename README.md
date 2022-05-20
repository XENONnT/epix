# epix

[![PyPI version shields.io](https://img.shields.io/pypi/v/epix.svg)](https://pypi.python.org/pypi/epix/)
[![CodeFactor](https://www.codefactor.io/repository/github/xenonnt/epix/badge)](https://www.codefactor.io/repository/github/xenonnt/epix)

**E**lectron and **P**hoton **I**nstructions generator for **X**ENON

The job of epix is to load XENONnT Geant4 MC data and produce inputs for wfsim, using nestpy for the quanta generation and DBSCAN for the clustering of the individual steps.

## Installation

With all requirements fulfilled (e.g., on top of the [XENONnT montecarlo_environment](https://github.com/XENONnT/montecarlo_environment)):
```
pip install epix
```
or install in editable mode from source:
```
git clone https://github.com/XENONnT/epix
pip install -e epix
```

## Usage

Only an input file is needed to run epix:
```
bin/run_epix --InputFile <path_and_filename>
```
The other keyword arguments are:

| Argument | Description | Default |
| :------------- | :------------- | :------------- |
| `--Detector`  | Detector to be used. Has to be defined in epix.detectors | `XENONnT` |
| `--DetectorConfigOverride`  | Config file to overwrite default epix.detectors settings; see examples in the `configs` folder | in epix.detectors |
| `--CutOnEventid`  | If selected, the next two arguments act on the G4 event id, and not the entry number (default) | `false` |
| `--EntryStart`  | First event to be read | 0 |
| `--EntryStop`  | How many entries from the ROOT file you want to process | all |
| `--MicroSeparation`  | DBSCAN clustering distance (mm) | `0.005` |
| `--MicroSeparationTime`  | Clustering time (ns) | `10` |
| `--TagClusterBy`  | decide if you tag the cluster (particle type, energy depositing process) according to first interaction in it (`time`) or most energetic (`energy`) | `energy` |
| `--MaxDelay`  | Time after which we cut the rest of the event (ns) | `1e7` |
| `--SourceRate`  | Event rate for event separation<br /> - `0` for no time shift (G4 time remains)<br /> - `-1` for clean time shift between events<br /> - `>0` (Hz) for random spacing | `0` |
| `--CutNeutron` | Add if you want to filter only nuclear recoil events (maximum ER energy deposit 10 keV) | `false` |
| `--JobNumber`  | Job number in full chain simulation. Offset is computed as `JobNumber * n_simulated_events/SourceRate`, where `n_simulated_events` is read from file. | `0` |
| `--OutputPath`  | Output file path | Same directory as input file |
| `--Debug`  | Tell epix if you want timing outputs | `false` |

