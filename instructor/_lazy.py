from __future__ import annotations

import importlib
import importlib.util
from typing import Any, Union

LazyAttr = tuple[str, str | None]
LazyExportSpec = Union[tuple[str, str], tuple[str, str, str | None]]
OptionalExportSpec = Union[
    tuple[str, tuple[str, ...], str],
    tuple[str, tuple[str, ...], str, str | None],
]


def build_lazy_exports(*specs: LazyExportSpec) -> tuple[dict[str, LazyAttr], list[str]]:
    exports: dict[str, LazyAttr] = {}
    names: list[str] = []
    for spec in specs:
        export_name, module_name, *rest = spec
        attr_name = rest[0] if rest else export_name
        exports[export_name] = (module_name, attr_name)
        names.append(export_name)
    return exports, names


def register_optional_exports(
    exports: dict[str, LazyAttr],
    names: list[str],
    *specs: OptionalExportSpec,
) -> None:
    for spec in specs:
        export_name, required_modules, module_name, *rest = spec
        if all(
            importlib.util.find_spec(package) is not None
            for package in required_modules
        ):
            attr_name = rest[0] if rest else export_name
            exports[export_name] = (module_name, attr_name)
            names.append(export_name)


def resolve_lazy_attr(
    package_name: str,
    namespace: dict[str, Any],
    exports: dict[str, LazyAttr],
    name: str,
) -> Any:
    if name not in exports:
        raise AttributeError(f"module '{package_name}' has no attribute '{name}'")

    module_name, attr_name = exports[name]
    module = importlib.import_module(module_name, package_name)
    value = module if attr_name is None else getattr(module, attr_name)
    namespace[name] = value
    return value
