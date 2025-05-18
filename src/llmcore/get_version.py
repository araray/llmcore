import pathlib
import tomllib


def _get_version_from_pyproject() -> str:
    """
    Read and return the 'version' from pyproject.toml.
    - Supports PEP 621 [project] tables or Poetry [tool.poetry].
    """
    project_root = pathlib.Path(__file__).parent.parent.parent
    pyproject = project_root / "pyproject.toml"
    content = pyproject.read_bytes()
    data = tomllib.loads(str(content))
    # PEP 621:
    if "project" in data and "version" in data["project"]:
        return data["project"]["version"]
    # Poetry legacy:
    if "tool" in data and "poetry" in data["tool"]:
        return data["tool"]["poetry"]["version"]
    raise RuntimeError("Version not found in pyproject.toml")
