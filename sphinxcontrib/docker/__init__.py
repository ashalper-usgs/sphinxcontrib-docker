"""
    sphinxcontrib.docker
    ~~~~~~~~~~~~~~~~~~~~

    A Sphinx extension for documenting Docker files.

    :license: BSD, see LICENSE for details.
"""

import sys
import pbr.version

from docutils import nodes
from docutils.parsers.rst import Directive
from pathlib import Path
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.environment import BuildEnvironment
from sphinx.errors import SphinxError
from sphinx.highlighting import PygmentsBridge
from sphinx.util.logging import getLogger
from typing import TYPE_CHECKING, Dict, Union


__version__ = pbr.version.VersionInfo('sphinxcontrib.docker').version_string()
log = getLogger(__name__)
DOMAIN_NAME = "docker"

class Docker(Directive):

    def run(self):
        paragraph_node = nodes.paragraph(text='Hello World!')
        return [paragraph_node]

class SphinxDockerError(SphinxError):
    category = "Sphinx-Docker error"

def setup(app: Sphinx):
    app.require_sphinx("3")

    app.add_config_value("docker_sources", app.srcdir, "env", [str, dict])
    app.add_config_value("docker_comment_markup", "", "env", [str])

    from sphinxcontrib.docker.markup import DockerDomain

    app.add_domain(DockerDomain)
            
    app.add_directive("helloworld", Docker)

    return {
        'version': __version__,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
}


def get_config_docker_comment_markup(env: BuildEnvironment) -> str:
    configured = env.config.docker_comment_markup

    return str(configured)


def get_config_docker_sources(env: BuildEnvironment) -> Dict[str, Path]:
    configured = env.config.docker_sources

    if not configured or not isinstance(configured, (str, Dict, Path)):
        raise SphinxDockerError(
            t__(
                "No Docker sources were configured in conf.py. "
                "Please provide a value to 'docker_sources'."
            )
        )

    if isinstance(configured, (str, Path)):
        configured = {Path(configured).name: configured}

    for name in configured:
        path = Path(configured[name])
        if not path.is_absolute():
            path = Path(env.project.srcdir, path)
        configured[name] = path

    return configured


def get_env(app_or_env: Union[Sphinx, BuildEnvironment]) -> BuildEnvironment:
    if isinstance(app_or_env, BuildEnvironment):
        return app_or_env
    if isinstance(app_or_env.env, BuildEnvironment):
        return app_or_env.env
    raise SphinxDockerError("Build environment not ready.")


def get_app(app_or_env: Union[Sphinx, BuildEnvironment]) -> Sphinx:
    if isinstance(app_or_env, BuildEnvironment):
        return app_or_env.app
    return app_or_env


def get_builder(app_or_env: Union[Sphinx, BuildEnvironment]) -> Builder:
    app = get_app(app_or_env)
    if isinstance(app.builder, Builder):
        return app.builder

    raise SphinxDockerError("Builder not ready.")


def get_highlighter(app_or_env: Union[Sphinx, BuildEnvironment]) -> PygmentsBridge:
    builder = get_builder(app_or_env)

    if hasattr(builder, "highlighter"):
        return builder.highlighter  # type: ignore

    raise SphinxDockerError("Unsupported builder.")
