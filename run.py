from importlib import import_module

import argparse
import yaml
import json

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str,
                    help="config file 입력")
args = parser.parse_args()

with open(args.config, 'r') as f:
    if args.config.endswith(('.yml', '.yaml')):
        config = yaml.safe_load(f)
    else:
        config = json.load(f)
        
module = import_module(config['flow_module'])

module.start(config)