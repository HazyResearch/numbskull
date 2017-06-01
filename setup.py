"""For pip."""

from setuptools import setup, find_packages

exec(open('numbskull/version.py').read())
setup(
    name='numbskull',
    version=__version__,
    description='sample away',
    packages=find_packages(),
    install_requires=['futures'],
    entry_points={
        'console_scripts': [
            'numbskull = numbskull.numbskull:main',
        ],
    },
)
