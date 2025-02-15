import os

import pkg_resources
from setuptools import setup, find_packages

# with open('README.rst') as f:
#     long_description = f.read()
long_description = "Cross lingual Information Retrieval for NMT"

setup(
    name='clir',
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],

    long_description=long_description,
    long_description_content_type='text/markdown',

    author='Maxime Bouthors',
    author_email='maxime.bouthors@isir.upmc.fr',
    url='',

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering"
    ],
)
