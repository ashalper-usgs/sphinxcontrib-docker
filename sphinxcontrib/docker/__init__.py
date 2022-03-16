"""
    sphinxcontrib.docker
    ~~~~~~~~~~~~~~~~~~~~

    A Sphinx extension for documenting Docker files.

    :license: BSD, see LICENSE for details.
"""

import pbr.version

from sphinx.application import Sphinx
from docutils import nodes
from docutils.parsers.rst import Directive
from typing import Dict
    
if False:
    # For type annotations
    from typing import Any
    from sphinx.application import Sphinx  # noqa

__version__ = pbr.version.VersionInfo('sphinxcontrib.docker').version_string()


class Docker(Directive):

    def run(self):
        paragraph_node = nodes.paragraph(text='Hello World!')
        return [paragraph_node]


def setup(app: Sphinx):
    app.require_sphinx("3")

    app.add_config_value("docker_sources", app.srcdir, "env", [str, dict])
    app.add_config_value("docker_comment_markup", "", "env", [str])
    
    app.add_directive("helloworld", Docker)

    return {
        'version': __version__,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
}
