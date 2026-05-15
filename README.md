# datadescriptor-enose-plif

Data analysis pipeline for combined e-nose (electronic nose) and PLIF (Planar Laser-Induced Fluorescence) sensor data. Provides tools for loading, processing, analyzing, and visualizing synchronized e-nose and PLIF measurements for gas concentration estimation.

## Overview

This project enables the fusion of high-frequency e-nose resistive metal-oxide (MOx) sensor data with 2D PLIF concentration imaging. It supports time alignment, dead-time compensation, concentration estimation via multiple methods, and comprehensive visualization.

### Key Features

- **Multi-strategy data loading** — native PLIF (~20 Hz), interpolated, or user-defined common-rate resampling
- **Time alignment** — dead-time compensation between heterogeneous e-nose (1000 Hz) and PLIF (20 Hz) sensors
- **Concentration estimation** — raw conductance, derivative-based, and supervised deconvolution methods
- **SNR analysis** — signal-to-noise ratio computation for PLIF data quality assessment
- **Visualization** — publication-quality figures for concentration estimation, interpolation, temperature, and more

## Repository Structure

```
├── scripts/              # Analysis and plotting scripts
│   ├── explore_*.py      # Data exploration utilities
│   ├── f1b_*.py          # Heatmap generation
│   ├── f2c_*.py          # E-nose position analysis
│   ├── f2d_*.py          # PLIF interpolation
│   ├── f3*_*.py          # Temperature, SNR, PLIF analysis
│   ├── f4_c_estimation.py# Concentration estimation pipeline
│   └── t3_*.py           # Dynamic range / retention score
├── src/                  # Core algorithms
│   ├── deadtime_compensation.py   # Time-alignment engine
│   └── supervised_deconv.py       # Supervised deconvolution model
├── utils/                # Shared utilities
│   ├── combined_io.py    # E-nose/PLIF CSV loading, resampling, export
│   ├── constants.py      # Experiment configs, sensor mappings, bounds
│   ├── plotting.py       # Visualization helpers
│   ├── workflow.py       # Pipeline orchestration
│   └── misc.py           # Metrics and utility functions
├── notebooks/            # Interactive Jupyter analysis
├── data/                 # Data storage
├── figs/                 # Generated publication figures
```

## Supported Experiments

| Experiment | Location | Speed (cm/s) |
|------------|----------|--------------|
| r56 – r61  | LC       | 10 – 20      |
| r70 – r71  | LR       | 10           |
| r75 – r76  | LU       | 10           |

Each experiment maps a PLIF camera view (LC, LR, LU) to four sensing zones (A, B, C, D), each associated with a specific MOx gas sensor channel.

## Installation

Requires Python 3.9+ and Anaconda: 

```
conda create --file environment.yml

```

## Quick Start

### Get datasets
Download [datasets](https://doi.org/10.3929/ethz-c-000782611) and save in an appropriate location.

### Change paths to dataset
In `utils/constants.py`, change `SSD_PLIF_FOLDER` and `SSD_COMBINED_FOLDER` to where your data is located.


### Inspect a combined dataset

```python
from utils.combined_io import inspect_combined_csv

inspect_combined_csv("56")  # experiment r56
```

### Load with different sampling strategies

```python
from utils.combined_io import load_combined_dataset

# Full interpolated CSV (e-nose time base)
full = load_combined_dataset("56", strategy="interpolated")

# Native PLIF frames only (~20 Hz)
native = load_combined_dataset("56", strategy="plif_native")

# Common-rate resampling at 100 Hz
common = load_combined_dataset(
    "56",
    strategy="common_rate",
    target_rate_hz=100.0,
    plif_source="native",
)
```

### Run concentration estimation pipeline

```python
from scripts.f4_c_estimation import process_experiment

process_experiment("56", window=(120, 130))
```

### Run the full experiment workflow

```bash
python scripts/explore_combined_plif_enose.py
python scripts/f4_c_estimation.py
```

## Pipeline Overview

The concentration estimation workflow consists of three stages:

1. **Data Loading & Alignment** — Load combined e-nose/PLIF CSV data and align signals using `DeadTimeCompensator`, which synchronizes the 1000 Hz e-nose conductance with the 20 Hz PLIF concentration ground truth.

2. **Concentration Estimation** — Three methods are available:
   - **Raw conductance** — Direct resistance-to-conductance conversion
   - **Derivative-based** — Time derivative of conductance (smoothed with Savitzky-Golay filter)
   - **Supervised deconvolution** — A learned model that deconvolves the sensor dynamics

3. **Evaluation** — Metrics (correlation, RMSE, etc.) are computed and comparison plots are generated showing ground-truth PLIF against each estimation method.

## Configuration

Edit `utils/constants.py` to customize:

- **`ENSENSE_BOUNDS`** — Camera ROI coordinates for each location (LC, LR, LU)
- **`PLIF_USEFUL_BOUNDS`** — Sub-region coordinates for each zone (A, B, C, D)
- **`MOX_PLIF_BOUND_RELATIONSHIPS`** — Mapping of MOx sensor IDs to PLIF zones
- **`LOCATIONS`** — Experiment-to-location mapping
- **`SPEEDS`** — Flow speed for each experiment
- **`PLIF_REFS`** — Reference experiment for each location
- **Sampling frequencies** — `PLIF_SAMPLING_FREQUENCY` (20 Hz), `MOX_SAMPLING_FREQUENCY` (1000 Hz)

## Scripts Reference

| Script | Description |
|--------|-------------|
| `explore_combined_plif_enose.py` | Inspect and visualize combined e-nose/PLIF data with different sampling strategies |
| `explore_raw_plif.py` | Explore raw PLIF HDF5 data |
| `f1b_heatmap.py` | Generate heatmaps |
| `f1d_enose_sample.py` | Visualize e-nose sample data |
| `f2c_enose_position.py` | Determine e-nose position relative to PLIF field of view |
| `f2d_interpolation.py` | PLIF spatial interpolation |
| `f3b_mox_temperature.py` | MOx sensor temperature analysis |
| `f3c_plif_snr.py` | PLIF signal-to-noise ratio computation |
| `f3d_mox_plif.py` | Combined MOx/PLIF visualization |
| `f4_c_estimation.py` | **Main concentration estimation pipeline** |
| `t3_dynamicrangeretentionscore.py` | Dynamic range and retention score computation |

## Output

Scripts generate figures in the `figs/` directory and processed examples in `processed_examples/`. Figure naming follows the convention:

```
r{exp_id}_{location}_{zone}_{analysis_type}.{png|svg}
```

Examples:
- `r56_LC_A_concentration_estimation.png` — Concentration estimation for experiment r56, zone A
- `r56_LC_plif_SNR.png` — PLIF SNR for experiment r56
- `r56_LC_temperature.png` — Temperature analysis for experiment r56

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Nik Dennler, Human-centered Sensing Lab — ETH Zürich


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Citation

If you use this software in your research, please cite the following paper:

```bibtex
@article{dennler2026_odoursensingturbulentplumes,
      title={Odour sensing in turbulent plumes with high-speed electronic nose and non-invasive ground truth}, 
      author={Nik Dennler and Elle Stark and Saimon Collaku and Lars Larson and André van Schaik and Michael Schmuker and John Crimaldi and Andreas T. Güntner and Aaron True},
      year={2026},
      journal={arXiv preprint},
      eprint={2604.19626},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      doi = {10.48550/arXiv.2604.19626}
}
```
