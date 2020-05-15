import os
import sys
from setuptools import setup, find_packages

install_requires = [
    "numpy",
    "scipy",
    "pygsp",
    "matplotlib",
    "sortedcontainers",
    "networkx"
]

version_py = os.path.join(os.path.dirname(__file__), "graph_coarsening", "version.py")
version = open(version_py).read().strip().split("=")[-1].replace('"', "").strip()

readme = open("README.md").read()

setup(
    name="graph_coarsening",
    version=version,
    description="graph_coarsening",
    author="Andreas Loukas",
    author_email="andreas.loukas@epfl.ch",
    packages=find_packages(),
    license="Apache License 2.0",
    install_requires=install_requires,
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/loukasa/graph-coarsening",
    download_url="https://github.com/loukasa/graph-coarsening/archive/v{}.tar.gz".format(
        version
    ),
    keywords=["big-data", "networks",],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Framework :: Jupyter",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
