"""Skills extension.

A :class:`SkillsExtension` manages a collection of named *skills* — small
reusable behaviours that an agent can invoke.  Each skill is a callable
that takes the agent and the incoming message as arguments.

Example::

    from supervisor.ext.skills import SkillsExtension

    skills = SkillsExtension()

    @skills.skill
    def summarise(agent, msg):
        return f"Summary of: {msg.content[:20]}..."

    agent.use(skills)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from supervisor.ext import Extension

if TYPE_CHECKING:
    from supervisor._core import Message
    from supervisor.agent import Agent


class Skill:
    """Descriptor for a registered skill.

    Attributes:
        name: Unique skill name.
        func: The callable implementation.
        description: Human-readable description.
    """

    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        description: str = "",
    ) -> None:
        self.name = name
        self.func = func
        self.description = description

    def __repr__(self) -> str:
        return f"Skill(name={self.name!r})"


class SkillsExtension(Extension):
    """Extension that manages reusable skills.

    Skills are registered via :meth:`register_skill` or the :meth:`skill`
    decorator and invoked with :meth:`invoke`.
    """

    name: str = "skills"

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    # -- registration --------------------------------------------------------

    def register_skill(
        self,
        func: Callable[..., Any],
        *,
        name: Optional[str] = None,
        description: str = "",
    ) -> Skill:
        """Register *func* as a skill.

        Parameters:
            func: Callable with signature ``(agent, msg) -> Any``.
            name: Skill name (defaults to ``func.__name__``).
            description: Human-readable description.
        """
        skill_name = name or func.__name__
        sk = Skill(
            name=skill_name,
            func=func,
            description=description or (func.__doc__ or "").strip(),
        )
        self._skills[skill_name] = sk
        return sk

    def skill(
        self,
        func: Optional[Callable[..., Any]] = None,
        *,
        name: Optional[str] = None,
        description: str = "",
    ) -> Any:
        """Decorator to register a function as a skill.

        Usage::

            @skills.skill
            def greet(agent, msg):
                return f"Hello, {msg.sender}!"
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register_skill(fn, name=name, description=description)
            return fn

        if func is not None:
            return decorator(func)
        return decorator

    # -- invocation ----------------------------------------------------------

    def invoke(self, skill_name: str, agent: "Agent", msg: "Message") -> Any:
        """Invoke the skill identified by *skill_name*.

        Raises:
            KeyError: If no skill with that name is registered.
        """
        if skill_name not in self._skills:
            raise KeyError(f"No skill registered with name '{skill_name}'")
        return self._skills[skill_name].func(agent, msg)

    # -- introspection -------------------------------------------------------

    def list_skills(self) -> List[Skill]:
        """Return all registered skills."""
        return list(self._skills.values())


__all__ = ["SkillsExtension", "Skill"]
