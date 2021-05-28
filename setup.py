from setuptools import setup, find_packages

setup(
    name='ravml',
    version='0.1-alpha',
    packages=find_packages(),
    install_requires=[
        "numpy==1.20.1",
        "scikit-learn==0.24.1",
        "pandas==1.2.3",
        "matplotlib==3.3.4",
        "scikit-learn"
    ],
    dependency_links=[
        "https://github.com/ravenprotocol/ravcom.git@0.1-alpha",
        "https://github.com/ravenprotocol/ravop.git@0.1-alpha",
    ]
)
