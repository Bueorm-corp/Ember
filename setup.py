# setup.py

from setuptools import setup, find_packages

setup(
    name="Ember",
    version="0.0.1",
    packages=find_packages(),
    package_data={
        'Ember': ['*.h5'],
    },
    install_requires=[
        'tensorflow>=2.0.0'
    ],
    author="Bueorm",
    author_email="support.bueorm@proton.me",
    description="Una librería para cargar y utilizar modelos de inteligencia artificial (echos por Bueorm)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tuusuario/ember",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
