"""
Setup script for package

Joel Nicolow, Coastal Research Collaborative, March 2025
"""

from setuptools import setup, find_packages

setup(
    name="geedownload",  # Package name (matches repo)
    version="0.1",
    packages=find_packages(),  # Automatically finds `geeutils/`
    install_requires=[
        "numpy",
        "rasterio",
        "requests",
        "geojson",
        "GDAL",
        "earthengine-api"
    ],  # Add dependencies if needed
)
