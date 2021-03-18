from setuptools import setup, find_packages

setup(
    name='ravml',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "numpy==1.20.1",
        "scikit-learn==0.24.1",
        "pandas==1.2.3",
        "matplotlib==3.3.4"
    ],
    dependency_links=[
        "https://github.com/ravenprotocol/ravcom.git",
        "https://github.com/ravenprotocol/ravop.git",
    ]
)
