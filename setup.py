from setuptools import setup, find_packages

setup(
    name="QM_FFT_Analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "finufft>=2.0.0",
        "plotly>=5.0.0",
        "scipy>=1.7.0",
        "tqdm>=4.65.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for analyzing quantum mechanical data using FFT techniques",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/QM_FFT_Analysis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 