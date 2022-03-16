#!/usr/bin/env python

import setuptools

extras_require = {
    "markdown": ["myst_parser", "docutils>=0.16"],
    "docs": [
        "myst_parser",
        "docutils>=0.16"
    ],
}

setuptools.setup(
    name="sphinxcontrib-docker",
    setup_requires=['pbr'],
    pbr=True,
)
