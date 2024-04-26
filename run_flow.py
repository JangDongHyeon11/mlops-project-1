import yaml
import json
import argparse
from importlib import import_module

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str,
                    help="path to a config file")
args = parser.parse_args()

with open(args.config, 'r') as f:
    if args.config.endswith(('.yml', '.yaml')):
        config = yaml.safe_load(f)
    else:
        config = json.load(f)
print(config)
# flow_module = import_module(config['flow_module'])

# flow_module.start(config)