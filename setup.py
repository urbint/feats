"""
   setuptools-based setup module

"""

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

from codecs import open
import os
import logging
import subprocess
import sys

here = os.path.abspath(os.path.dirname(__file__))
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)


required_libraries=[
    "numpy>=1.13.3",
    "pandas==0.22.0",
    "matplotlib==2.1.2",
    "scikit-learn>=0.19.1",
    "jupyter>=1.0.0",
    "tables>=3.4.2",
    "html2text>=2018.1.9",
    "scipy>=1.0.1",
    "gensim>=3.4.0",
    "python-dateutil==2.6.1",
    "annoy>=1.11.5",
    "joblib>=0.11",
    "seaborn>=0.9.0",
]

setup(
    name="feats",
    version="0.1.0",
    description="tools for making features from dataframes",
    url="https://github.com/urbint/feats",
    install_requires=required_libraries
)
