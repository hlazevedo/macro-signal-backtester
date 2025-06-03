from setuptools import setup, find_packages

setup(
    name="macro-signal-backtester",
    version="0.1.0",
    author="Your Name",
    description="A quantitative macro signal backtesting framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "yfinance>=0.2.0",
        "fredapi>=0.5.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "scipy>=1.9.0",
    ],
    python_requires=">=3.8",
)