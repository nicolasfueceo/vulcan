from pathlib import Path

import jinja2

from src.core.database import get_db_schema_string

_prompt_dir = Path(__file__).parent.parent / "prompts"

# Get the database schema once at startup and add it to the Jinja environment globals
_db_schema = get_db_schema_string()

_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_prompt_dir),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)
_jinja_env.globals["db_schema"] = _db_schema


def load_prompt(template_name: str, **kwargs) -> str:
    """
    Loads and renders a Jinja2 template from the prompts directory.

    Args:
        template_name: The name of the template file (e.g., 'agents/strategist.j2').
        **kwargs: The context variables to render the template with.

    Returns:
        The rendered prompt as a string.
    """
    template = _jinja_env.get_template(template_name)
    return template.render(**kwargs)
