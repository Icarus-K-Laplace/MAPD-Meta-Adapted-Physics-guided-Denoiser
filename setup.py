from setuptools import setup, find_packages

setup(
    name="mapd-ultimate",
    version="2.0.0",
    description="Meta-Adaptive Physics-Guided Denoising Framework",
    author="[Your Name]",
    packages=find_packages(),
    install_requires=[
        "numpy", "opencv-python", "torch", "scipy", "numba", "scikit-image"
    ]
)
