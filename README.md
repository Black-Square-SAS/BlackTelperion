

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
├── io/
│   └── create_spectral_cubes.py  # Image readers for different platforms
├── core/
│   ├── image.py                  # Base image class for data representation
│   └── factory.py                # ImageFactory for creating appropriate objects
├── platforms/
│   ├── base_platform.py          # Base class with common methods
│   ├── enmap.py                  # EnMAP platform implementation
│   ├── aster.py                  # ASTER platform implementation 
│   ├── sentinel.py               # Sentinel-2 implementation
│   └── hyspex.py                 # Hyspex implementation
├── products/
│   ├── base_product.py           # Base class for derived products
│   ├── spectral_index.py         # Specialized class for indices
│   ├── classification.py         # Classifications and thematic maps
│   └── mask.py                   # Binary/categorical masks
├── utils/
│   ├── metadata.py               # Metadata handling utilities
│   └── visualization.py          # Visualization helpers
└── examples/                     # Example usage scripts
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
make -C docs/ html
```

The generated documentation will be available at `docs/build/html/index.html`.

### Regenerating API module files

If you add new modules or subpackages, regenerate the `.rst` files:

```bash
sphinx-apidoc -o docs/source/api/ BlackTelperion/ -f -e -M --implicit-namespaces
```

Then rebuild with `make -C docs/ html`.

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
