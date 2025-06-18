import logging
from agentic.utils.prompt_manager import PromptManager

def test_prompt_manager():
    logger = logging.getLogger("test")
    pm = PromptManager(logger=logger)
    template_str = "Hello {{ name }}! You have {{ count }} new messages."
    pm.add_template("greeting", template_str)
    result = pm.render("greeting", {"name": "Alice", "count": 3})
    assert result == "Hello Alice! You have 3 new messages."
