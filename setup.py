
from distutils.core import setup
from Cython.Build import cythonize
import os

setup(
    name='ragnarok',
    version='1.0',
    author='Lento Manickathan',
    author_email='lento.manickathan@gmail.com',
    description='Lattice boltzmann library',
    license='GPL3',
    install_requires=['numpy','matplotlib','numba','gr'],
    packages =['ragnarok'])
