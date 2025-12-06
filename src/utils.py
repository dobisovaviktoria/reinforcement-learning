import yaml

def load_config(p):
    with open(p) as f: return yaml.safe_load(f)
