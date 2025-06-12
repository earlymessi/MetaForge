from setuptools import setup, find_packages

setup(
    name="metaforge",
    version="1.0.1",
    description="Metaheuristic and Reinforcement Learning Solvers for the Job Shop Scheduling Problem (JSSP)",
    author="Mageed Ghaleb",
    author_email="your.email@example.com",
    url="https://github.com/mageed-ghaleb/metaforge",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "matplotlib",
        "torch",
        "seaborn",
        "pandas",
        "requests",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
