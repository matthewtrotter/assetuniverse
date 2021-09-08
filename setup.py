from setuptools import setup, find_packages

setup(
    name="assetuniverse",
    version="2.0.1",
    description="Historical asset return downloader for multiple assets (an asset universe).",
    python_requires=">=3.5",
    install_requires=["yfinance", "pandas", "pandas_datareader", "numpy", "plotly", "dash", "quandl", "alpha_vantage"],
    packages=find_packages(),
    zip_safe=False,
)