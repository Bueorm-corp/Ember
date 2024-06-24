from setuptools import setup, find_packages

setup(
    name='ember',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'requests'
    ],
    author='Bueorm',
    author_email='support.bueorm@proton.me',
    description='A library for using Transformer models',
    url='https://github.com/BueormLLC/LDM-base',
)
