from setuptools import setup, find_packages

setup(
    name="vagen",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gym-sokoban",
    ],
    python_requires=">=3.10",
)