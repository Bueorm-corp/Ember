from setuptools import setup, find_packages

setup(
    name="ember",
    version="0.0.1",
    description="A unified interface for multiple AI model providers",
    author="Bueorm",
    author_email="support.bueorm@proton.me",
    url="https://github.com/BueormLLC/Ember",
    packages=find_packages(),
    install_requires=[
        "openai", 
        "google-generativeai",
        "Pillow",
        "cohere",
        "anthropic",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
