from setuptools import find_packages, setup

with open('requirements.pip') as f:
    requirements = f.read().splitlines()

setup(
    name='czi-image-tools',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements
)
