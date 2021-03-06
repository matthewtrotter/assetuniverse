from setuptools import setup, find_packages

setup(
    name="assetuniverse",
    version="4.1.3",
    description="Historical asset return downloader for multiple assets (an asset universe).",
    python_requires=">=3.6",
    install_requires=["yfinance", "pandas", "pandas_datareader", "numpy", "plotly", "pytest", "ib_insync", "appdirs", "diskcache"],
    packages=find_packages(exclude=['tests*']),
    zip_safe=False,
)