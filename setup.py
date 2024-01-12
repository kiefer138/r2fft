from setuptools import setup, find_packages

setup(
    name="r2fft",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "matplotlib==3.8.2",
        "numpy==1.26.2",
        "scipy==1.11.4",
        "graphviz==0.20.1",
    ],
)
