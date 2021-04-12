import datetime
import logging
import logging.config
import os
import socket
import zipfile
from pathlib import Path
from urllib import request

import progressbar
import torch
import yaml

# Taken from https://stackoverflow.com/a/53643011
from tqdm import tqdm


class DownloadProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def download_item(url, file_path):
    if not os.path.exists(file_path):
        logging.info(f"Downloading {file_path} from {url}")
        Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        try:
            request.urlretrieve(url, file_path, DownloadProgressBar())
        except ValueError:
            # Try prepending http, this is a common mistake
            request.urlretrieve("http://" + url, file_path, DownloadProgressBar())


def unzip(m):
    with zipfile.ZipFile(m, 'r') as zf:
        target_path = os.path.dirname(m)
        for member in tqdm(zf.infolist(), desc=f'Unzipping {m} into {target_path}'):
            zf.extract(member, target_path)


def lazy_unzip(pathToZipFile: str):
    """
    Unzip pathToZipFile file, if pathToZipFile[:-len(".zip")] (unzipped) does not already exists.

    :param pathToZipFile: Path to zip file that should be unziped.
    :type pathToZipFile: str
    """

    if not os.path.isfile(pathToZipFile[:-4]):
        unzip(pathToZipFile)


def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    if sll < 1:
        return results
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))

    return results


def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)


def get_device(t):
    return t.get_device() if t.get_device() > -1 else torch.device("cpu")


def touch(f):
    """
    Create empty file at given location f
    :param f: path to file
    """
    basedir = os.path.dirname(f)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    open(f, 'a').close()


class LevelOnly(object):
    levels = {
        "CRITICAL": 50,
        "ERROR": 40,
        "WARNING": 30,
        "INFO": 20,
        "DEBUG": 10,
        "NOTSET": 0,
    }

    def __init__(self, level):
        self.__level = self.levels[level]

    def filter(self, logRecord):
        return logRecord.levelno <= self.__level


def setup_logging(
        module,
        default_level=logging.INFO,
        env_key='LOG_CFG',
        logpath=os.getcwd(),
        extra_name="",
        config_path=None
):
    """
        Setup logging configuration\n
        Logging configuration should be available in `YAML` file described by `env_key` environment variable

        :param module:     name of the module
        :param logpath:    path to logging folder [default: script's working directory]
        :param config_path: configuration file, has more priority than configuration file obtained via `env_key`
        :param env_key:    evironment variable containing path to configuration file
        :param default_level: default logging level, (in case of no local configuration is found)
    """

    if not os.path.exists(os.path.dirname(logpath)):
        os.makedirs(os.path.dirname(logpath))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    stamp = timestamp + "_" + socket.gethostname() + "_" + extra_name

    path = config_path if config_path is not None else os.getenv(env_key, None)
    if path is not None and os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
            for h in config['handlers'].values():
                if h['class'] == 'logging.FileHandler':
                    h['filename'] = os.path.join(logpath, module, stamp, h['filename'])
                    touch(h['filename'])
            for f in config['filters'].values():
                if '()' in f:
                    f['()'] = globals()[f['()']]
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level, filename=os.path.join(logpath, stamp))
