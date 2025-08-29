from setuptools import setup, find_packages

setup(
    name="causal-impact",                  # Package name
    version="0.1.0",                                 # Initial version
    description="To measure the impact from strategy launches",    # Short description
    author="Peijun Xu",
    author_email="ivanxu259@gmail.com",
    url="https://github.com/bard259/causal-impact",  # GitHub URL
    packages=find_packages(),                        # Automatically find package folders
    install_requires=[
        "numpy", "pandas", "causalimpact"                    # Add dependencies here
    ],
    packages=find_packages(exclude=("tests", "docs", "examples")),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
    ],
    extras_require={
        # If you prefer WillianFuks' repo in Colab, install that separately.
        # Here we use the PyPI name when available.
        "impact": ["pycausalimpact>=0.2.7"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],,
)
