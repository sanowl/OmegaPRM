from setuptools import setup, find_packages

setup(
    name="omega_prm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
        "numpy>=1.24.0",
        "pytest>=7.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="OmegaPRM: Process Reward Model Implementation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/omega_prm",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)