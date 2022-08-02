from pathlib import Path

from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='ravml',
    version='0.5',
    license='MIT',
    author="Raven Protocol",
    author_email='kailash@ravenprotocol.com',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ravenprotocol/ravdl',
    keywords='Ravml, machine learning library, algorithms',
    install_requires=[
        "numpy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "ravop",
        "python-dotenv"
    ]
)