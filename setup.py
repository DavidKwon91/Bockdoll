#%%
import os
import setuptools

import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="bockdoll",
    version="1.0.0",
    author="David Kwon",
    author_email="husiew140@gmail.com",
    description="Some utilities to be used",
    packages=setuptools.find_packages(),
)
