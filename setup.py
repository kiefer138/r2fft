from setuptools import setup, find_packages

setup(
    name="r2fft",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'contourpy==1.2.0',
        'cycler==0.12.1',
        'fonttools==4.45.1',
        'graphviz==0.20.1',
        'iniconfig==2.0.0',
        'kiwisolver==1.4.5',
        'matplotlib==3.8.2',
        'numpy==1.26.2',
        'packaging==23.2',
        'Pillow==10.1.0',
        'pluggy==1.3.0',
        'pyparsing==3.1.1',
        'python-dateutil==2.8.2',
        'scipy==1.11.4',
        'six==1.16.0',
        'typing_extensions==4.8.0',
    ],
)
