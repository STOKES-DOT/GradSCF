"""Production code must not import or call external chemistry engines."""
import ast
from pathlib import Path
import tomllib


def test_production_has_no_external_chemistry_imports_or_molecule_calls():
    violations = []
    for base in (Path("src/gradscf"), Path("tools"), Path("examples")):
        for path in base.rglob("*.py"):
            if any(p in {"build", "__pycache__"} for p in path.parts):
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute) and node.func.attr in {"intor", "with_range_coulomb", "with_rinv_origin"}:
                        violations.append((str(path), node.lineno, node.func.attr))
                    if ((isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
                            or (isinstance(node.func, ast.Name) and node.func.id == "__import__")):
                        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                            modules = [node.args[0].value]
                for name in modules:
                    if name.split(".")[0] in {"pyscf", "gpu4pyscf"}:
                        violations.append((str(path), node.lineno, name))
    assert not violations, violations


def test_external_chemistry_dependency_is_test_only():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    for dependency in project["dependencies"]:
        assert "pyscf" not in dependency.lower()
    for group, dependencies in project.get("optional-dependencies", {}).items():
        if group != "comparison-tests":
            assert not any("pyscf" in d.lower() for d in dependencies), group
    assert any(d.startswith("pyscf") for d in project["optional-dependencies"]["comparison-tests"])
