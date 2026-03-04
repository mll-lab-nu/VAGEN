from setuptools import setup, find_packages

setup(
    name="vagen",
    version="26.2.5",
    packages=find_packages(),
    install_requires=[
        "gym-sokoban",
        "gymnasium",
        "gymnasium[toy-text]",
        "uvicorn<0.41",
        # tos dependencies
        "numpy",
        "matplotlib",
        "scipy",
        "pillow",
        "tqdm",
        "imageio",
        "omegaconf",
    ],
    python_requires=">=3.10",
)
