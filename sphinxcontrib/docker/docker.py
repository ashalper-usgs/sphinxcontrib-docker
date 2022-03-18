from __future__ import annotations

import re
import sys
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from sphinx.environment import BuildEnvironment
from sphinx.util.logging import getLogger

from sphinxcontrib.docker import SphinxDockerError, get_config_docker_sources
from sphinxcontrib.docker.code import CodePosition, CodeSpan
from sphinxcontrib.docker.i18n import t__

if TYPE_CHECKING or sys.version_info < (3, 8, 0):
    from typing_extensions import Protocol
else:
    from typing import Protocol

log = getLogger(__name__)

_Dockerfile = TypeVar("_Dockerfile", bound="Dockerfile")


class Dockerfile(NamedTuple):
    """
    Represent a Dockerfile.
    """

    root_name: str
    root_path: Path
    name: str
    path: Path
    fullname: str

    @classmethod
    def from_dockerfile_path(
        cls: Type[_Dockerfile],
        env: BuildEnvironment,
        dockerfile_path: Union[str, Path] = "",
    ) -> _Dockerfile:
        if not dockerfile_path:
            return cls.from_dockerfile_parts(env)
        else:
            root, *rest = Path(dockerfile_path).parts
            return cls.from_dockerfile_parts(env, root, "/".join(rest))

    @classmethod
    def from_dockerfile_parts(
        cls: Type[_Dockerfile],
        env: BuildEnvironment,
        root_dockerfile_name: Optional[str] = None,
        subdockerfile: Union[str, Path] = "",
    ) -> _Dockerfile:
        root = cls._find_root(env, root_dockerfile_name)

        if not subdockerfile:
            return root

        path = Path(subdockerfile)
        if not path.is_absolute():
            path = root.path.joinpath(path)

        name = str(path.relative_to(root.path))
        fullname = f"{root.name}/{name}"

        return cls(root.name, root.path, name, path.resolve(), fullname)

    @classmethod
    def _find_root(
        cls: Type[_Dockerfile],
        env: BuildEnvironment,
        root_dockerfile_name: Optional[str] = None,
    ) -> _Dockerfile:
        sources = get_config_docker_sources(env)
        if len(sources) == 1:
            root_name, root_path = next(iter(sources.items()))
        elif root_dockerfile_name is None:
            raise SphinxDockerError(
                t__("Can't determine the proper docker source to use.")
            )
        else:
            try:
                root_name = root_dockerfile_name
                root_path = sources[root_name]
            except KeyError as e:
                raise SphinxDockerError(
                    t__(
                        "Unknown Docker source '%s'. "
                        "Please review 'docker_sources' in conf.py."
                    )
                    % root_dockerfile_name
                ) from e
        return cls(
            root_name,
            root_path.resolve(),
            root_name,
            root_path.resolve(),
            root_name,
        )

    def __str__(self) -> str:
        return self.fullname


class InstrSignature(Protocol):
    # TODO: description of Dockerfile instruction grammar in docstring
    # below?
    """
    Signature of a Dockerfile instruction.

    This does not consider anything regarding an instruction's body,
    only its signature.
    """

    @property
    def type(self) -> DockerBlockType:
        """
        The Docker object type (``resource``, ``from``, etc).
        """
        ...

    @property
    def labels(self) -> List[str]:
        """
        The object's labels, the amount of which depends on the :attr:`~type`.
        """
        ...

    def regex(self) -> Union[str, re.Pattern[str]]:
        """
        A regex that will reliably match the right code signature in a line.

        See also:
            The :func:`~regex` implementation that covers most cases.
        """
        ...

    def __repr__(self) -> str:
        """
        A string representation of
        Returns:

        """
        ...

    def __str__(self) -> str:
        """
        A string representation of
        Returns:

        """
        ...


# TODO: will need to be modified
def regex(self: InstrSignature) -> Union[str, re.Pattern[str]]:
    """
    Return a regex that matches the signature reliably within a module.

    The regex will match a Dockerfile instruction:

    *   starting with 0 or more whitespaces,
    *   followed by an identifier without quotes
        (:attr:`sphinxcontrib.docker.docker.InstrSignature.type`),
    *   followed by 1 or more non-newline whitespaces,
    *   followed by the signature's labels
        (:attr:`sphinxcontrib.docker.docker.InstrSignature.labels`),
        possibily between double quotes `"` and
        separated by 1 or more non-newline whitespaces,
    *   followed by 1 or more non-newline whitespaces,
    *   followed by the `{` character and
    *   ending with the newline character.

    Args:
        self:
            The instruction block signature object we want to find in
            Dockerfile.

    Returns:
        Compiled regular expression.
    """
    # Use a double negation to match non-newline whitespace characters.
    # That is
    #   - Not
    #     - one of
    #       - Non-whitespace
    #       - Carriage return character
    #       - New line character
    non_newline_whitespace = r"[^\S\r\n]"

    signature_regex_inner = rf"{non_newline_whitespace}+".join(
        [
            # the type do not have quotes
            self.type.value,
            *[
                # labels can be between double quotes
                rf"(\"{label}\"|{label})"
                for label in self.labels
            ],
            # end with the block opening bracket
            "{.*",
        ]
    )

    # Enclose with accepted leading and trailing whitespace characters
    signature_regex = rf"^\s*{signature_regex_inner}$"
    return signature_regex


def _repr(self: InstrSignature) -> str:
    """
    Make a valid Python code string that create an equivalent instance.
    """
    labels_within_quotes = [f"'{label}'" for label in self.labels]
    return (
        f"{self.__class__.__name__}("
        f"'{self.type}', "
        f"{', '.join(labels_within_quotes)}"
        f")"
    )


def _str(self: InstrSignature) -> str:
    return f"{'.'.join(self.labels)}"


def make_identifier(
    signature: InstrSignature, dockerfile: Optional[Dockerfile] = None
) -> str:
    """
    Create an URL friendly identifier string from a signature.

    This notion of identifier string does not exist in Docker per se,
    since we do factor in the definituon's module.

    The below example illustrates how the identifier is built.

    Example:
        Given the following definitions within the module "mod":

        .. code-block:: docker

            # in submodule "mod", in any file

            resource "null_resource" "some-name" {
            }

            data "null_data" "other-name" {
            }

        The identifiers would be

        .. code-block:: text

            mod/resource-null_resource.some-name

            mod/data-null_data.other-name

        >>> resource = DockerResourceSignature("null", "resource", "some-name")
        >>> make_identifier(resource)
        'resource-null_resource.some-name'

    Args:
        signature:
            The instruction block signature for the definition we
            create an identifier.
        module:
            An optional module name.

    Returns:
        URL friendly identifier.
    """
    base_identifier = f"{signature.type.value}-{'.'.join(signature.labels)}"

    if dockerfile:
        return f"{dockerfile.name}/{base_identifier}"
    else:
        return base_identifier


class DockerBlockType(Enum):
    FROM = "from"
    LABEL = "label"
    RUN = "run"


class InstrDefinition(NamedTuple):
    """
    Define Dockerfile instruction object.

    We use this because many definitions could have identical
    signatures.
    """

    signature: InstrSignature
    """
    Identify a definition, the signature is the definition *header*.
    """

    file: Path
    """
    Dockerfile where this definition is found.
    """

    doc_code: CodeSpan
    """
    Dockerfile code section where this is documented.
    """

    signature_code: CodeSpan
    """
    Dockerfile code section where this is defined.
    """

    body_code: CodeSpan
    """
    Dockerfile code section where this is defined.
    """

    usages: Set[str]
    """
    Document name where this definition is referenced.
    """

    @property
    def code(self) -> CodeSpan:
        """
        Return where to find the whole code of this definition.

        We consider that the docstring, signature and body are contiguous.
        """
        return CodeSpan(
            self.doc_code.start_position, self.body_code.end_position
        )


class DockerResourceSignature(NamedTuple):
    provider: str
    kind: str
    name: str

    @property
    def type(self) -> DockerBlockType:
        return DockerBlockType.RESOURCE

    @property
    def labels(self) -> List[str]:
        return [f"{self.provider}_{self.kind}", self.name]

    regex = regex  # type: ignore # Assigning methods is unsupported by mypy

    __repr__ = _repr  # type: ignore # Assigning methods is unsupported by mypy

    __str__ = _str  # type: ignore # Assigning methods is unsupported by mypy


class DockerFromSignature(NamedTuple):
    provider: str
    kind: str
    name: str

    @property
    def type(self) -> DockerBlockType:
        return DockerBlockType.FROM

    @property
    def labels(self) -> List[str]:
        return [f"{self.provider}_{self.kind}", self.name]

    regex = regex  # type: ignore # Assigning methods in NamedTuple is unsupported by mypy

    __repr__ = _repr  # type: ignore # Assigning methods in NamedTuple is unsupported by mypy

    __str__ = _str  # type: ignore # Assigning methods is unsupported by mypy


class DockerfileSignature(NamedTuple):
    name: str

    @property
    def type(self) -> DockerBlockType:
        return DockerBlockType.DOCKERFILE

    @property
    def labels(self) -> List[str]:
        return [self.name]

    regex = regex  # type: ignore # Assigning methods is unsupported by mypy

    __repr__ = _repr  # type: ignore # Assigning methods is unsupported by mypy

    __str__ = _str  # type: ignore # Assigning methods is unsupported by mypy


class DockerLabelSignature(NamedTuple):
    provider: str
    kind: str
    name: str

    @property
    def type(self) -> DockerBlockType:
        return DockerBlockType.LABEL

    @property
    def labels(self) -> List[str]:
        return [f"{self.provider}_{self.kind}", self.name]

    regex = regex  # type: ignore # Assigning methods is unsupported by mypy

    __repr__ = _repr  # type: ignore # Assigning methods is unsupported by mypy

    __str__ = _str  # type: ignore # Assigning methods is unsupported by mypy


class DockerRunSignature(NamedTuple):
    provider: str
    kind: str
    name: str

    @property
    def type(self) -> DockerBlockType:
        return DockerBlockType.RUN

    @property
    def labels(self) -> List[str]:
        return [f"{self.provider}_{self.kind}", self.name]

    regex = regex  # type: ignore # Assigning methods is unsupported by mypy

    __repr__ = _repr  # type: ignore # Assigning methods is unsupported by mypy

    __str__ = _str  # type: ignore # Assigning methods is unsupported by mypy


_DockerStore = TypeVar("_DockerStore", bound="DockerStore")


class DockerStore:
    def __init__(self, data: Dict[Any, Any]) -> None:
        """
        Private constructor. Use :meth:`~from_build_env` instead.

        Args:
            data: Existing data, probably read from cache.
        """
        self.data: Dict[Dockerfile, DockerfileData] = data

    @classmethod
    def initial_data(cls: Type[_DockerStore]) -> Dict[Any, Any]:
        return defaultdict(DockerfileData.new)

    def register(
        self,
        dockerfile: Dockerfile,
        signature: InstrSignature,
        docname: str,
    ) -> InstrDefinition:
        """
        Register a definition signature for a given module.

        This signals that a definition is documented. It makes sure we know
        about this instruction definition within the local cache, then
        registers the docname in its known documentation usages.

        Args:
            module:
                The module where this definition is found.  Multiple
                definitions should be unique within a module.
            signature:
                A signature identifies a definition.
            docname:
                The document where this signature is documented.

        Raises:
            sphinxcontrib.docker.SphinxDockerError: No definition could
                be found matching the signature within this module.

        Returns
            The registered definition.
        """
        hcl_definition = self.data[dockerfile].find_definition(dockerfile, signature)
        hcl_definition.usages.add(docname)
        return hcl_definition

    def purge_usage(self, usage_source: str) -> None:  # noqa
        for dockerfile in self.data.values():
            if not isinstance(dockerfile, Dockerfile):
                continue
            for signature in self.data[dockerfile].definitions.keys():
                entry = self.data[dockerfile].definitions[signature]
                if usage_source in entry.usages:
                    entry.usages.remove(usage_source)
                if not entry.usages:
                    del self.data[dockerfile].definitions[signature]

    def get_code(self, df_file: Path) -> List[str]:
        dockerfile = self.get_dockerfile(df_file)
        return self.data[dockerfile].get_code(df_file)

    def get_dockerfile(self, df_file: Path) -> Dockerfile:
        dockerfile_path = df_file.parent
        for dockerfile in self.data:
            if dockerfile.path == dockerfile_path:
                return dockerfile
        else:
            raise SphinxDockerError(f"No Dockerfile found at '{dockerfile_path}'.")

    def get_definitions(
        self,
        dockerfile: Optional[Dockerfile] = None,
        df_file: Optional[Path] = None,
    ) -> Dict[InstrSignature, InstrDefinition]:
        dockerfile = self.get_dockerfile(df_file) if df_file else dockerfile

        def gen_definitions(
            dockerfile: Optional[Dockerfile],
        ) -> Iterator[Tuple[InstrSignature, InstrDefinition]]:
            if dockerfile:
                yield from self.data[dockerfile].definitions.items()
            else:
                for dockerfile in self.data:
                    yield from self.data[dockerfile].definitions.items()

        def condition(entry: InstrDefinition) -> bool:
            return entry.file == df_file if df_file else True

        return {
            signature: definition
            for signature, definition in gen_definitions(dockerfile)
            if condition(definition)
        }

    def get_dockerfile_files(self) -> List[Tuple[Dockerfile, Path]]:
        return sorted(
            [
                (dockerfile, filepath)
                for dockerfile in self.data
                for filepath in self.data[dockerfile].get_documented_files()
            ]
        )

    def get_documented_files(
        self, dockerfile: Optional[Dockerfile] = None
    ) -> Set[Path]:
        def gen_files() -> Iterator[Path]:
            if dockerfile:
                yield from self.data[dockerfile].get_documented_files()
            else:
                for dockerfile_data in self.data.values():
                    yield from dockerfile_data.get_documented_files()

        return set(df_file for df_file in gen_files())

    def get_documentation(self, definition: InstrDefinition) -> List[str]:
        start_line = definition.doc_code.start_position.line
        end_line = definition.doc_code.end_position.line

        code_with_doc = self.get_code(definition.file)[start_line:end_line]

        documentation = extract_docstring_from_comment(code_with_doc)
        return documentation


_DockerfileData = TypeVar("_DockerfileData", bound="DockerfileData")


class DockerfileData(NamedTuple):
    """
    What we store in the build environment for a given Dockerfile.

    Here is a JSON-like representation:

    .. code-block:: text

        {
            code: {
                Path: [
                    # lines of code
                ]
            },
            definitions: {
                HclBlockSignature: HclSignatureDefinition
            }
        }
    """

    code: Dict[Path, List[str]]
    """
    Cache of raw instruction code indexed by file path.
    """

    definitions: Dict[InstrSignature, InstrDefinition]
    """
    Found block definitions indexed by their signature.
    """

    @classmethod
    def new(cls: Type[_DockerfileData]) -> _DockerfileData:
        return cls(defaultdict(list), dict())

    # TODO: might not be necessary
    def find_definition(
        self, dockerfile: Dockerfile, signature: InstrSignature
    ) -> InstrDefinition:
        """
        Look for a Docker definition in all HCL files of a Dockerfile.

        We use an internal cache that is pickled with the Sphinx environment.
        If the cache misses, we parse the code to find the definition.

        Args:
            signature:
                A signature identifies a definition.

        Raises:
            sphinxcontrib.docker.SphinxDockerError: No definition could
                be found matching the signature within this Dockerfile.

        Returns:
            The found definition.
        """
        log.debug(f"Looking for definition of {repr(signature)}.")
        try:
            hcl_definition = self.definitions[signature]
            log.debug(f"Found definition of {repr(signature)} in cache.")
        except KeyError:
            for df_file in dockerfile.path.glob("Dockerfile"):
                hcl_definition = self._find_definition(signature, df_file)  # type: ignore
                if not hcl_definition:
                    continue
                log.debug(f"Caching definition of {repr(signature)}.")
                self.definitions[signature] = hcl_definition
                return hcl_definition
            else:
                raise SphinxDockerError(
                    f"Definition not found for {repr(signature)} in Dockerfile {dockerfile}."
                )
        return hcl_definition

    def get_definitions(
        self, df_file: Optional[Path] = None
    ) -> Dict[InstrSignature, InstrDefinition]:
        """
        Make a mapping of known HCL definitions.

        Args:
            df_file:
                Optionally filter for a specific file.

        Returns:
            The returned dictionary is created on each call, thus removing
            items from it won't remove them from this store.
        """

        def condition(entry: InstrDefinition) -> bool:
            return entry.file == df_file if df_file else True

        return {
            signature: entry
            for signature, entry in self.definitions.items()
            if condition(entry)
        }

    def get_documentation(self, signature: InstrSignature) -> List[str]:
        definition = self.definitions[signature]

        start_line = definition.doc_code.start_position.line
        end_line = definition.doc_code.end_position.line
        code_with_doc = self.get_code(definition.file)[start_line:end_line]

        documentation = extract_docstring_from_comment(code_with_doc)
        return documentation

    def get_code(self, df_file: Path) -> List[str]:
        if df_file not in self.code:
            log.debug(f"Putting code from {df_file} in cache.")
            self.code[df_file] = df_file.read_text().splitlines()
        raw_code = self.code[df_file]
        return raw_code

    def get_documented_files(self) -> Set[Path]:
        return set(entry.file for entry in self.definitions.values())

    def _find_definition(
        self, signature: InstrSignature, df_file: Path
    ) -> Optional[InstrDefinition]:
        log.debug(f"Looking for definition of {repr(signature)} in files.")
        raw_code = self.get_code(df_file)
        log.debug(f"Looking for definition of {repr(signature)} in {df_file}.")
        found_code = self._find_definition_code(signature, raw_code)
        if not found_code:
            return None

        doc_code, signature_code, body_code = found_code
        log.debug(f"Found definition of {repr(signature)} in code.")
        hcl_definition = InstrDefinition(
            signature,
            df_file,
            doc_code,
            signature_code,
            body_code,
            set(),
        )
        return hcl_definition

    def _find_definition_code(
        self, signature: InstrSignature, lines: List[str]
    ) -> Optional[Tuple[CodeSpan, CodeSpan, CodeSpan]]:
        signature_code = self._lookup_signature(signature, lines)

        if not signature_code:
            return None

        return self._find_whole_definition_code(signature_code, lines)

    def _find_whole_definition_code(
        self, signature_code: CodeSpan, lines: List[str]
    ) -> Tuple[CodeSpan, CodeSpan, CodeSpan]:
        # We found the signature, now let's try to climb up to the
        # last signature in order to find include a comment
        log.debug("Looking for beginning of definition, including comment.")
        begining_line = self._find_begining_of_documented_code(
            signature_code.start_position.line, lines
        )
        doc_code = CodeSpan(
            CodePosition(begining_line, 0),
            CodePosition(
                signature_code.start_position.line,
                signature_code.start_position.column,
            ),
        )
        log.debug(f"Found beginning of definition on line {begining_line}.")
        log.debug("Looking for definition body.")
        end_line = self._find_end_of_block(
            signature_code.start_position.line + 1, lines
        )
        body_code = CodeSpan(
            CodePosition(
                signature_code.end_position.line,
                signature_code.end_position.column,
            ),
            CodePosition(end_line, 0),
        )
        log.debug(f"Definition body ends on line {end_line}.")
        return doc_code, signature_code, body_code

    def _lookup_signature(
        self, signature: InstrSignature, lines: List[str]
    ) -> Optional[CodeSpan]:
        signature_regex = signature.regex()
        for i, line_of_code in enumerate(lines):
            if re.match(signature_regex, line_of_code):
                log.debug(f"Found signature on line {i}.")
                signature_code = CodeSpan(
                    CodePosition(
                        i, len(line_of_code) - len(line_of_code.lstrip())
                    ),
                    CodePosition(i, len(line_of_code.rstrip())),
                )
                return signature_code
        else:
            # We could not find 'signature' in this code.
            return None

    def _find_begining_of_documented_code(
        self, start_line: int, lines: List[str]
    ) -> int:
        inspected_line = start_line - 1
        within_multiline_comment = False
        while inspected_line > 0:
            current_line = lines[inspected_line].strip()
            if current_line.startswith("*/"):
                log.debug(
                    f"Found the end of a multiline comment on line {inspected_line}."
                )
                within_multiline_comment = True
                inspected_line -= 1
                continue
            if within_multiline_comment and current_line.startswith("/*"):
                # we found the begining of a '/* ... */' comment.
                # That marks the start of the docstring, thus the end of
                # this search.
                log.debug(
                    f"Found the beginning of a multiline comment on line {inspected_line}."
                )
                return inspected_line
            if (
                current_line.startswith("#")
                or current_line.startswith("//")
                or within_multiline_comment
            ):
                # We found a single line comment.
                # Continue upward
                log.debug(
                    f"Found a single line comment on line {inspected_line}."
                )
                inspected_line -= 1
                continue
            # We found something else:
            # - empty line
            # - something that is not a comment
            log.debug(f"Found a non-comment line on line {inspected_line}.")
            return inspected_line + 1
        # We got to the top of the file. That's ok.
        log.debug("Went up to start of file.")
        return 0

    def _find_end_of_block(  # noqa
        self, start_line_within_block: int, all_lines: List[str]
    ) -> int:
        inspected_line = start_line_within_block
        # Within a block, there is one brace in the stack
        curly_brace_balance = 1

        while inspected_line < len(all_lines):
            log.debug(f"Looking for block delimiters on line {inspected_line}.")
            log.debug(f"Delimiter stack is {curly_brace_balance}")
            inspected_line_of_code = all_lines[inspected_line]
            for character in inspected_line_of_code:
                if character == "{":
                    curly_brace_balance += 1
                elif character == "}":
                    curly_brace_balance -= 1
                if curly_brace_balance == 0:
                    return inspected_line
            inspected_line += 1

        raise SphinxDockerError(
            f"Found end of file when expecting closing scope opened before "
            f"line {start_line_within_block}"
        )


def extract_docstring_from_comment(  # noqa
    code_with_doc: List[str],
) -> List[str]:
    found_some_docs = False
    absolute_doc_indent = 0
    documentation = []
    for line in code_with_doc:
        if not found_some_docs:
            match = re.match(r"^[/\*\s#]* (?P<content>\S+.*)", line)
            if not match:
                continue
            content = match.group("content")
            absolute_doc_indent = len(line) - len(content)
            found_some_docs = True
        else:
            content = line[absolute_doc_indent:]
        if content and re.match(r"^[/\*\s#]+$", content):
            content = None

        if content is not None:
            documentation.append(content)
    return documentation
