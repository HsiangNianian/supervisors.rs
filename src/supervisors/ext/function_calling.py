"""Function Calling extension.

Provides a :class:`FunctionCallingExtension` that lets users register
custom tool functions with optional JSON-Schema-style specifications.
The agent can then invoke tools by name.

Example::

    from supervisors.ext.function_calling import FunctionCallingExtension, tool

    fc = FunctionCallingExtension()

    @fc.tool(description="Add two numbers")
    def add(a: int, b: int) -> int:
        return a + b

    agent.use(fc)
    result = fc.call_tool("add", a=1, b=2)
    assert result == 3
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from supervisors.ext import Extension

if TYPE_CHECKING:
    pass


class ToolSpec:
    """Specification for a registered tool.

    Attributes:
        name: Tool name.
        func: The callable implementation.
        description: Human-readable description.
        parameters: Optional JSON-Schema-style parameter descriptions.
    """

    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.func = func
        self.description = description
        self.parameters = parameters or {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the spec to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def __repr__(self) -> str:
        return f"ToolSpec(name={self.name!r}, description={self.description!r})"


class FunctionCallingExtension(Extension):
    """Extension that manages a registry of callable tools.

    Tools are registered with :meth:`register_tool` or the :meth:`tool`
    decorator and invoked with :meth:`call_tool`.
    """

    name: str = "function_calling"

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    # -- tool registration ---------------------------------------------------

    def register_tool(
        self,
        func: Callable[..., Any],
        *,
        name: Optional[str] = None,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ToolSpec:
        """Register *func* as a callable tool.

        Parameters:
            func: The callable to register.
            name: Tool name (defaults to ``func.__name__``).
            description: Human-readable description.
            parameters: Optional JSON-Schema-style parameter spec.  If not
                provided, a basic spec is auto-generated from the function
                signature.

        Returns:
            The created :class:`ToolSpec`.
        """
        tool_name = name or func.__name__
        if parameters is None:
            parameters = self._auto_parameters(func)
        spec = ToolSpec(
            name=tool_name,
            func=func,
            description=description or (func.__doc__ or "").strip(),
            parameters=parameters,
        )
        self._tools[tool_name] = spec
        return spec

    def tool(
        self,
        func: Optional[Callable[..., Any]] = None,
        *,
        name: Optional[str] = None,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Decorator to register a function as a tool.

        Can be used with or without arguments::

            @ext.tool
            def my_func(x: int) -> int: ...

            @ext.tool(description="Do something")
            def my_func(x: int) -> int: ...
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register_tool(
                fn, name=name, description=description, parameters=parameters
            )
            return fn

        if func is not None:
            return decorator(func)
        return decorator

    # -- tool invocation -----------------------------------------------------

    def call_tool(self, name: str, **kwargs: Any) -> Any:
        """Invoke the tool identified by *name* with the given *kwargs*.

        Raises:
            KeyError: If no tool with *name* is registered.
        """
        if name not in self._tools:
            raise KeyError(f"No tool registered with name '{name}'")
        return self._tools[name].func(**kwargs)

    # -- introspection -------------------------------------------------------

    def list_tools(self) -> List[ToolSpec]:
        """Return a list of all registered tool specs."""
        return list(self._tools.values())

    def get_tools_spec(self) -> List[Dict[str, Any]]:
        """Return JSON-serialisable specs for all registered tools."""
        return [spec.to_dict() for spec in self._tools.values()]

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _auto_parameters(func: Callable[..., Any]) -> Dict[str, Any]:
        """Generate a basic parameter spec from *func*'s signature."""
        sig = inspect.signature(func)
        props: Dict[str, Any] = {}
        for pname, param in sig.parameters.items():
            entry: Dict[str, str] = {"type": "string"}
            annotation = param.annotation
            if annotation is not inspect.Parameter.empty:
                if annotation is int:
                    entry["type"] = "integer"
                elif annotation is float:
                    entry["type"] = "number"
                elif annotation is bool:
                    entry["type"] = "boolean"
                elif annotation is str:
                    entry["type"] = "string"
            props[pname] = entry
        return {"type": "object", "properties": props}


__all__ = ["FunctionCallingExtension", "ToolSpec"]
