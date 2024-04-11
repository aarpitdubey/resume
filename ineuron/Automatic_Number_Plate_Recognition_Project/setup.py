import setuptools # type: ignore

VERSION = '0.0.1'
PROJECT_NAME = 'Automatic Number Plate Recognition'
AUTHOR_NAME = 'Arpit Dubey'
DESCRIPTION = 'An application (or) a project which can recognize the number plates of cars'

setuptools.setup(
    version=VERSION,
    PROJECT_NAME=PROJECT_NAME,
    author=AUTHOR_NAME,
    description=DESCRIPTION,
    package_dir= {"":"src"},
    packages=setuptools.find_packages(where='src')
)
