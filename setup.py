from setuptools import setup, find_packages

setup(
    name='ravml',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "numpy==1.20.1",
        "git+git://github.com/ravenprotocol/ravcom.git",
        "git+git://github.com/ravenprotocol/ravop.git",
        "scikit-learn==0.24.1",
        "pandas==1.2.3",
        "matplotlib==3.3.4"
    ],
)
