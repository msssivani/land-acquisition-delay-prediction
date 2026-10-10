from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
'''
config.py reads .yaml file and converts the configurations to python objects(dictionaries) so that later py file can acess the settings easily

'''
'''
python -c "from src.config import load_config; print(load_config())"
-c means defining command. ""->defines code , ";"->seperates 2 python statements
'''