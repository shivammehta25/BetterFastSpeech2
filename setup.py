#!/usr/bin/env python
import os

import numpy
from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as readme_file:
    README = readme_file.read()

cwd = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(cwd, "fs2", "VERSION")) as fin:
    version = fin.read().strip()

setup(
    name="fs2",
    version=version,
    description="I got pissed at all the other implementations for not working, so I made my own.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Shivam Mehta",
    author_email="shivam.mehta25@gmail.com",
    url="https://shivammehta25.github.io/Matcha-TTS",
    install_requires=[str(r) for r in open(os.path.join(os.path.dirname(__file__), "requirements.txt"))],
    include_dirs=[numpy.get_include()],
    include_package_data=True,
    packages=find_packages(exclude=["tests", "tests/*", "examples", "examples/*"]),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            # "matcha-data-stats=matcha.utils.generate_data_statistics:main",
            # "matcha-tts=matcha.cli:cli",
            # "matcha-tts-app=matcha.app:main",
        ]
    },
    # ext_modules=cythonize(exts, language_level=3),
    python_requires=">=3.9.0",
)
