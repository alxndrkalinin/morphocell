# morphocell

morphocell is a Python library for morphometric analysis of multidimensional
bioimages with CUDA acceleration. It was created to streamline common
morphology workflows on large 3D microscopy data. The library provides tools
for deconvolution, segmentation (via Cellpose integration) and feature
extraction, all accessible from a simple Python API. By leveraging GPU enabled
operations where possible, morphocell offers substantial speed ups over purely
CPU based approaches.

## Getting started

### Dependencies
* Python >=3.10
* Ubuntu 18.04
* CUDA 11.2-12.1
* Optional: TensorFlow/Flowdec for deconvolution, Cellpose for segmentation

### Installation
The project can be installed from source as follows:

```bash
git clone https://github.com/alxndrkalinin/morphocell.git
cd morphocell
pip install -e .[all]
```

### Testing
Run the code style and type checks using

```bash
ruff check .
ruff format --check .
mypy --ignore-missing-imports morphocell/
```

### Contributing
Contributions and bug reports are welcome. Please open an issue or pull request
on GitHub.

## Usage
See the example notebooks in `examples/notebooks/` for demonstrations of
deconvolution and segmentation workflows.

## Citation
If you use morphocell in your research, please cite it:

```bibtex
@misc{morphocell,
  author       = {Alexandr Kalinin},
  title        = {morphocell: morphometric analysis of nD bioimages with CUDA support},
  year         = {2025},
  howpublished = {\url{https://github.com/alxndrkalinin/morphocell}}
}
```
