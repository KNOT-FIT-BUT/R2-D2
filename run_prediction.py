import json
import logging
import os
import sys

from prediction import run_predictions
from utility.utility import setup_logging

if __name__ == "__main__":
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path="configurations/logging.yml")
    infile, outfile = sys.argv[1], sys.argv[2]
    configuration_file = None
    if len(sys.argv) > 3:
        configuration_file = sys.argv[3]
    logging.info(f"Running configuration {configuration_file}")
    with open(configuration_file, "r") as cfg_contents:
        cfg = json.load(cfg_contents)
        logging.debug(json.dumps(cfg, indent=4, sort_keys=True))
    if configuration_file == "":
        raise ValueError("Configuration file needs to be provided")
    run_predictions(infile, outfile, configuration_file)
    
