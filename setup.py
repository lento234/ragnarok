
from distutils.core import setup
from Cython.Build import cythonize
import os

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

def check_dependencies():
    install_requires = []
    try:
        import numpy
    except ImportError:
        install_requires.append('numpy')
    try:
        import matplotlib
    except ImportError:
        install_requires.append('matplotlib')
	try:
		import numba
	except ImportError:
		install_requires.append('numba')
	try:
		import gr
	except ImportError:
		install_requires.append('gr')
    return install_requires

setup(
    name='ragnarok',
    version='0.1',
    author='Lento Manickathan',
    author_email='lento.manickathan@gmail.com',
    description='Lattice boltzmann library',
    license='GPL3',
    install_requires=check_dependencies(),
    packages =['ragnarok'])
