# Tide

## Summary

This project enables to reconstruct the tide signal anywhere, anytime, based on the FES2014 tide harmonic constituents.
Surge calculation can be performed too, comparing calculated tide with tide gauge measurements. It has been tested with python 3.6.

![run_cell](https://github.com/fournifl/Tide/blob/main/figures_ipynb/amplitude_in_m2.png)


## Installation
Install pyfes library in a conda environment, and the few libraries specified in requirements file.

```
conda install -c fbriol fes
```

## Usage
You can have a look at the jupyter notebook on how to calculate tide (maree_fes.ipynb), or use the following:
```
python fes_compute_tide.py fes.ini
```



