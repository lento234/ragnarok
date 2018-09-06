
#from distutils.core import setup
#from Cython.Build import cythonize
#import os
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='ragnarok',
    version='1.0',
    author='Lento Manickathan',
    author_email='lento.manickathan@gmail.com',
    description='Lattice boltzmann library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/lento234/ragnarok',
    license='GNU GPLv3',
    install_requires=['numpy','matplotlib','numba','gr'],
    packages =['ragnarok'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)
