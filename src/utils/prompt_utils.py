import logging
from pathlib import Path

import jinja2

from src.core.database import get_db_schema_string

logger = logging.getLogger(__name__)

_prompt_dir = Path(__file__).parent.parent / "prompts"

# Initialize the Jinja environment
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_prompt_dir),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _refresh_database_schema():
    """Refresh the database schema in the Jinja environment globals."""
    try:
        db_schema = get_db_schema_string()
        _jinja_env.globals["db_schema"] = db_schema
        logger.debug(
            f"Database schema refreshed successfully. Schema length: {len(db_schema)} characters"
        )
        return db_schema
    except Exception as e:
        logger.error(f"Failed to refresh database schema: {e}")
        _jinja_env.globals["db_schema"] = "ERROR: Could not load database schema"
        return None


def load_prompt(template_name: str, **kwargs) -> str:
    """
    Loads and renders a Jinja2 template from the prompts directory.
    Refreshes database schema and logs the rendered prompt for debugging.

    Args:
        template_name: The name of the template file (e.g., 'agents/strategist.j2').
        **kwargs: The context variables to render the template with.

    Returns:
        The rendered prompt as a string.
    """
    try:
        # Refresh database schema to ensure it's current
        _refresh_database_schema()

        # Load and render the template
        template = _jinja_env.get_template(template_name)
        rendered_prompt = template.render(**kwargs)

        # Log the template being loaded and key info
        logger.info(f"Loaded prompt template: {template_name}")
        if kwargs:
            logger.debug(f"Template variables: {list(kwargs.keys())}")

        # Log the rendered prompt for debugging (truncated to avoid spam)
        prompt_preview = (
            rendered_prompt[:500] + "..."
            if len(rendered_prompt) > 500
            else rendered_prompt
        )
        logger.debug(f"Rendered prompt preview (first 500 chars):\n{prompt_preview}")

        # Log full prompt length
        logger.info(f"Full rendered prompt length: {len(rendered_prompt)} characters")

        return rendered_prompt

    except jinja2.TemplateNotFound as e:
        logger.error(f"Template not found: {template_name}")
        raise ValueError(f"Prompt template '{template_name}' not found") from e
    except jinja2.TemplateError as e:
        logger.error(f"Template rendering error for {template_name}: {e}")
        raise ValueError(f"Error rendering template '{template_name}': {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error loading prompt {template_name}: {e}")
        raise


