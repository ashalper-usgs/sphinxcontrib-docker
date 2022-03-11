"""
    sphinxcontrib.docker
    ~~~~~~~~~~~~~~~~~~~~

    A Sphinx extension for documenting Docker files.

y    :license: BSD, see LICENSE for details.
"""

import pbr.version
from docutils import nodes
from docutils.parsers.rst import Directive

if False:
    # For type annotations
    from typing import Any, Dict  # noqa
    from sphinx.application import Sphinx  # noqa

__version__ = pbr.version.VersionInfo('sphinxcontrib.docker').version_string()


class Docker(Directive):

    def run(self):
        paragraph_node = nodes.paragraph(text='Hello World!')
        return [paragraph_node]


def setup(app):
    app.add_directive("helloworld", Docker)

    return {
        'version': __version__,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
}
