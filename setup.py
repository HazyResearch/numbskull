from setuptools import setup, find_packages

setup(
    name='numbskull',
    version='0.0',
    description='sample away',
    packages=find_packages(),
    install_requires=[],
    entry_points = {
        'console_scripts': [
            'numbskull = numbskull.numbskull:main',
        ],
    },
)
