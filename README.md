

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/f5a0f780ba7c4755af5910ff359ffe6f)](https://app.codacy.com/app/lento234/ragnarok?utm_source=github.com&utm_medium=referral&utm_content=lento234/ragnarok&utm_campaign=Badge_Grade_Dashboard)
[![Build Status](https://travis-ci.com/lento234/ragnarok.svg?branch=master)](https://travis-ci.com/lento234/ragnarok)
[![CircleCI](https://circleci.com/gh/lento234/ragnarok.svg?style=svg)](https://circleci.com/gh/lento234/ragnarok)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/lento234/ragnarok/master)

# Ragnarok

Ragnarok is an open-source python library for solving lattice boltzmann method. 



<p align="center">
    <img src="https://github.com/lento234/ragnarok/blob/master/media/doublyperiod_shearlayer_animation.gif" width="600" height="400" alt="Doubly-periodic shear-layer" />
</p>    


## Installation

via pip (local):

```
$ pip install -e ragnarok
```

via pip (directly from git repo):

```
$ pip install git+https://github.com/lento234/ragnarok.git
```

## Tutorials
A short overview on how to run cases, see `examples/`...

### 1D shock-tube flow

A 1D shock-tube navier-stokes problem:

```
$ python shock_tube.py
```

### 2D lid-driven cavity flow

A 2D cavity flow problem:

```
$ python cavity.py
```

## Jupyter notebooks

* 1D Advection-diffusion problem: Gaussian step [![notebook](https://img.shields.io/badge/launch-Jupyter%20Notebook-red.svg)](http://nbviewer.jupyter.org/github/lento234/ragnarok/blob/master/examples/advectiondiffusion1D/gaussian_step.ipynb)
* 2D Navier-Stokes problem: Doubly-periodic shear layer [![notebook](https://img.shields.io/badge/launch-Jupyter%20Notebook-red.svg)](http://nbviewer.jupyter.org/github/lento234/ragnarok/blob/master/examples/navierstokes2D/doublyperiodic_shearlayer.ipynb)

## Features

- [x] 1D advection-diffusion
- [x] 1D Navier-Stokes
- [x] 2D Navier-Stokes
- [x] Numba kernels
- [ ] CUDA kernels
- [ ] Entropic LBM
- [ ] MRT LBM
- [ ] Multiphase problems
- [ ] Thermal LBM
- [ ] 3D problems

## Dependencies

* python 3.7
* numpy >= 1.13.3
* matplotlib >= 2.1.0
* numba >= 0.35.0
* gr >= 1.0.1 


## About

* Personal homepage: <http://www.manickathan.ch>
* Lento Manickathan <lento.manickathan@gmail.com>
