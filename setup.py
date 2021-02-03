 
from os import path
from setuptools import setup

BASE_DIR = path.dirname(path.abspath(__file__))
CHANGELOG_PATH = path.join(BASE_DIR, "CHANGELOG")

with open(CHANGELOG_PATH, "r") as f:
    version = f.readline()
    version = version.split()
    version = version[1][1:-1]

setup(
    version=version,
)