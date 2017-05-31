"""For pip."""

from setuptools import setup, find_packages
from numbskull.version import __version__

setup(
    name='numbskull',
    version=__version__,
    description='sample away',
    packages=find_packages(),
    install_requires=[],
    entry_points={
        'console_scripts': [
            'numbskull = numbskull.numbskull:main',
        ],
    },
)
