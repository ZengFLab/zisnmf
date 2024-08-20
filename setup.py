from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='zisnmf',
    version='0.1',
    description='Zero-inflated Supervised Non-negative Matrix Factorization',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Feng Zeng',
    author_email='zengfeng@xmu.edu.cn',
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)