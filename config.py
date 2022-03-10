
from dynaconf import Dynaconf

settings = Dynaconf(
    load_dotenv=True,
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml", "params.yaml"],
)