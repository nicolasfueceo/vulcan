import logging
from jinja2 import Environment, BaseLoader, Template
from typing import Dict, Any

class PromptManager:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("PromptManager")
        self.env = Environment(loader=BaseLoader())
        self.templates = {}

    def add_template(self, name: str, template_str: str):
        self.logger.info(f"Adding template: {name}")
        self.templates[name] = self.env.from_string(template_str)

    def render(self, name: str, context: Dict[str, Any]) -> str:
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found.")
        self.logger.info(f"Rendering template: {name} with context: {context}")
        return self.templates[name].render(**context)
