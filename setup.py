from setuptools import setup, find_packages

setup(
    name='pylat',
    version='0.1.1',
    url='https://github.com/MaginnGroup/PyLAT',
    author='MaginnGroup',
    author_email='ed@nd.edu',
    description=' Python LAMMPS Analysis Tools.',
    packages=find_packages(),
    install_requires=['numpy', 'scipy'],
    scripts=['PyLAT.py']
)
