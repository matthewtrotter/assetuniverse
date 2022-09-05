from setuptools import setup, find_packages

__version__ = "0.1.0"

setup(
    name="assetuniverse",
    version=__version__,
    description="Historical asset return downloader for multiple assets (an asset universe).",
    python_requires=">=3.6",
    install_requires=["yfinance", "pandas", "pandas_datareader", "numpy", "plotly", "pytest", "ib_insync", "appdirs", "arrow", "diskcache"],
    packages=find_packages(exclude=['tests*']),
    zip_safe=False,
)