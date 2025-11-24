from setuptools import setup, find_packages

setup(
    name="minilearn",
    version="0.1.0",
    description="A minimal machine learning library for educational purposes",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.1.0',
        'scikit-learn>=0.23.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
    ],
    python_requires='>=3.8',
)