from dotenv import load_dotenv


def pytest_configure(config):
    """
    Load environment variables from .env file before any tests run.
    """
    load_dotenv()

