

<p align="center">
  <img src="./images/BlackTelperion_Logo.png" alt="BlackTelperion Logo" width="500"/>
</p>

# BlackTelperion

BlackTelperion is Black Square's specialized proprietary library for the processing and analysis of hyperspectral and multispectral imagery across multiple platforms. Named after Telperion, the silver tree that illuminated the world with its radiant light in Tolkien's legendarium, our library transforms spectral information into clear, meaningful data. Designed for remote sensing applications, BlackTelperion offers robust capabilities for capturing, processing, and interpreting spectral data from various platforms.

## Purpose

BlackTelperion serves as a unified framework for processing spectral imagery from various sources:
- Sentinel-2 (ESA)
- ASTER (NASA/METI)
- ENMAP (DLR)
- Hyspex (Our hyperspectral camera)
- Additional platforms (planned for future integration)

## Repository Structure

```
BlackTelperion/
├── analyse/                        # Spectral analysis methods
│   ├── dtree.py                    # Decision tree analysis
│   ├── indices.py                  # Spectral indices computation
│   ├── mwl.py                      # Minimum wavelength mapping
│   ├── sam.py                      # Spectral Angle Mapper
│   ├── supervised.py               # Supervised classification
│   ├── unmixing.py                 # Spectral unmixing
│   └── unsupervised.py             # Unsupervised classification
├── correct/                        # Data correction methods
│   ├── illumination/               # Illumination correction
│   │   ├── occlusion.py            # Occlusion correction
│   │   ├── path.py                 # Path radiance estimation
│   │   └── reflection.py           # Reflection correction
│   ├── detrend.py                  # Polynomial detrending
│   ├── equalize.py                 # Equalization
│   └── panel.py                    # Panel-based calibration
├── filter/                         # Filtering and processing
│   ├── combine.py                  # Image combination/stacking
│   ├── dimension_reduction.py      # PCA / MNF transforms
│   ├── sample.py                   # Sampling utilities
│   ├── segment.py                  # Segmentation and tiling
│   └── tpt.py                      # Turning point transform
├── io/                             # Input/output operations
│   ├── create_spectral_cubes.py    # Spectral cube creation
│   ├── headers.py                  # Header file I/O
│   ├── images.py                   # Image I/O
│   └── libraries.py                # Spectral library I/O
├── reference/                      # Reference spectra and features
│   ├── features/                   # Absorption feature definitions
│   ├── spectra/                    # Reference spectral data
│   │   ├── custom/                 # User-defined reference spectra
│   │   ├── pvc/                    # PVC calibration targets
│   │   └── spectralon/             # Spectralon calibration targets
│   └── generate.py                 # Synthetic data generation
├── utils/                          # Utility functions
│   ├── metadata.py                 # Metadata handling
│   └── visualization.py            # Visualization helpers
├── blackcollection.py              # Collection of hyperspectral objects
├── blackdata.py                    # Base class for spectral data
├── blackfeature.py                 # Spectral feature classes
├── blackheader.py                  # Header/metadata class
├── blackimage.py                   # Hyperspectral image class
├── blacklibrary.py                 # Spectral library class
└── multiprocessing.py              # Parallel processing utilities
```

## Getting Started

### Prerequisites

- Python 3.9+
- GDAL/OGR
- NumPy
- Xarray
- Rasterio
- Scikit-learn
- PyTorch (for deep learning components)

### Installation

For team members:

```bash
# Clone the repository
git clone git@github.com:blacksquare/BlackTelperion.git

# Install in development mode
cd BlackTelperion
pip install -e .
```

## Documentation

API documentation is generated from docstrings using [Sphinx](https://www.sphinx-doc.org/).

### Generating the docs

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build HTML docs
make -C docs_source/ html
```

The generated documentation will be available at `docs/html/index.html`.

### Regenerating API module files

If you add new modules or subpackages, regenerate the `.rst` files:

```bash
sphinx-apidoc -o docs_source/source/api/ BlackTelperion/ -f -e -M --implicit-namespaces
```

Then rebuild with `make -C docs_source/ html`.

## Basic Usage

```python

#To be defined

```

## Development Guidelines

### Adding New Processing Modules

1. Create a new class in the appropriate platform-specific directory
2. Inherit from `BaseProcessingBox`
3. Implement the required methods
4. Add unit tests

### Creating New Pipelines

Pipelines combine multiple processing boxes to create end-to-end workflows. See `base_pipeline.py` for the interface definition.

## Contributing

All team members are encouraged to contribute. Please follow these steps:

1. Create a feature branch from `develop`
2. Implement your changes with appropriate tests
3. Submit a pull request
4. Request review from a team member

## Internal Use Only

This repository and its contents are proprietary to Black Square and intended for internal use only. Do not share access or distribute code outside the organization.

## Contact

For questions or issues:
- Create an issue in this repository
- Contact the Spectral Processing Team

## Future Development

- Additional satellite platforms integration
- Deep learning-based feature extraction
- Cloud-optimized processing pipelines
