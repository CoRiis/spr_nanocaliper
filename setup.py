#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fh:
        return fh.read()


def get_version():
    with io.open(
        join(dirname(__file__), "src/dsi23/__init__.py"),
        encoding="utf-8",
    ) as vf:
        for line in vf.readlines():
            if "__version__" in line:
                return line.split("=")[1].strip().replace('"', "")


test_deps = [
    "pytest",
    "pytest-cov",
    "pytest-mock",
]
extras = {
    "test": test_deps,
}

setup(
    name="dsi23",
    version=get_version(),
    description="Surface plasmon resonance fitter",
    long_description="%s\n%s"
    % (
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
            "", read("README.md")
        ),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.rst")),
    ),
    author="Novo Nordisk",
    author_email="NCIR@novonordisk.com",
    url="https://sc216.corp.novocorp.net/DSI-DR/dsi-projects/dsi23_sprfitter.git",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Utilities",
        "Private :: Do Not Upload",
    ],
    project_urls={
        "Documentation": "https://datalab.corp.novocorp.net/docs/libraries/dsi23/latest/",
        "Changelog": "https://datalab.corp.novocorp.net/docs/libraries/dsi23/latest/changelog.html",
    },
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires=">=3.6, !=2.*, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*",
    #install_requires=[
    #    "novopy",
    #],
    tests_require=test_deps,
    extras_require=extras,
)
