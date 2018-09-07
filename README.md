

[![Build Status](https://travis-ci.com/lento234/ragnarok.svg?branch=master)](https://travis-ci.com/lento234/ragnarok)
[![CircleCI](https://circleci.com/gh/lento234/ragnarok.svg?style=svg)](https://circleci.com/gh/lento234/ragnarok)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/lento234/ragnarok/master)

# Ragnarok

Ragnarok is a python library for solving lattice boltzmann method. The code is open-source and developed by Lento Manickathan.

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

- Doubly-periodic shear layer [![notebook](https://img.shields.io/badge/launch-Jupyter%20Notebook-ff69b4.svg)](http://nbviewer.jupyter.org/github/lento234/ragnarok/blob/master/examples/navierstokes2D/doublyperiodic_shearlayer.ipynb)

## Dependencies

* python 3.7
* numpy >= 1.13.3
* matplotlib >= 2.1.0
* numba >= 0.35.0
* gr >= 1.0.1 


## About

* Personal homepage: <http://www.manickathan.ch>
* Lento Manickathan <lento.manickathan@gmail.com>
