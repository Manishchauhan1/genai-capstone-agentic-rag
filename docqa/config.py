import os
from dataclasses import dataclass
from typing import Mapping

from dotenv import load_dotenv

from docqa.errors import DocQAError


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Read settings from the environment.

    With env=None, a local .env file is loaded first (without overriding
    variables that are already set) and os.environ is used.
    """
    if env is None:
        load_dotenv()
        env = os.environ
    values = {}
    for name in ("GROQ_API_KEY", "GROQ_MODEL"):
        value = (env.get(name) or "").strip()
        if not value:
            raise DocQAError(
                f"Missing setting {name}. Add it to your .env file "
                "(or to the app's secrets) and try again."
            )
        values[name] = value
    return Settings(groq_api_key=values["GROQ_API_KEY"], groq_model=values["GROQ_MODEL"])
