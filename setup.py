"""Install MophoCell."""
from setuptools import setup, find_packages

NAME = "morphocell"
DESCRIPTION = "Morphomeric analysis of 3D cell images"
URL = "https://github.com/alxndrkalinin/morphocell"
AUTHOR = "Alexandr Kalinin"
EMAIL = "alxndrkalinin@gmail.com"
REQUIRES_PYTHON = ">=3.6.0"
LICENSE = "MIT"

INSTALL_REQUIRES = ["numpy>=1.17.0", "scikit-image>=0.16.1"]
EXTRAS_REQUIRE = {"decon": ["flowdec>=1.1.0"], "frc": ["frc", "miplib>=1.0.6"]}
EXCLUDE_FROM_PACKAGES = ["examples"]


def get_version():
    """Extract current version from __init__.py."""
    with open("morphocell/__init__.py", encoding="utf-8") as fid:
        for line in fid:
            if line.startswith("__version__"):
                VERSION = line.strip().split()[-1][1:-1]
                break
    return VERSION


def get_long_description():
    """Extract long description from README.md."""
    with open("README.md", encoding="utf-8") as f:
        LONG_DESCRIPTION = f.read()
    return LONG_DESCRIPTION


setup(
    name=NAME,
    version=get_version(),
    description=DESCRIPTION,
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author=AUTHOR,
    license=LICENSE,
    url=URL,
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    python_requires=REQUIRES_PYTHON,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
