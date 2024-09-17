import datetime
import gc
import glob
import os
import sys
import json
import multiprocessing as mp
import signal
import shutil
from itertools import product as tensor_product
from datetime import timezone

import torch
import numpy as np


def dir_or_file_exists(d):
    """
    检查指定的目录或文件是否存在。
    
    :param d: 要检查的目录或文件路径
    :return: 如果存在则返回 True，否则返回 False
    """
    return os.path.exists(d)


def tensor_of_dict_of_lists(d: dict):
    """
    将字典的列表值转换为张量产品。
    
    :param d: 包含列表值的字典
    :return: 字典列表，每个字典代表一个可能的组合
    """
    tore_values = list(tensor_product(*d.values()))
    return [{K: V[i] for i, K in enumerate(d.keys())} for V in tore_values]


def colored_background(r: int, g: int, b: int, text):
    """
    r,g,b integers between 0,255
    """
    return f"\033[48;2;{r};{g};{b}m{text}\033[0m"


def batch_indexable(iterable, n=128):
    """
    对可索引的可迭代对象进行简单的批处理。
    
    :param iterable: 可索引的可迭代对象
    :param n: 批次大小，默认为 128
    :yield: 迭代器，每次返回一个批次的数据
    """

    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


class NpEncoder(json.JSONEncoder):
    """
    自定义 JSON 编码器，用于处理 NumPy 和 PyTorch 数据类型。
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def json_valid_dict(obj):
    return json.loads(json.dumps(obj, cls=NpEncoder))


def utc_epoch_now():
    """
    获取当前的 UTC 时间戳。
    
    :return: 当前的 UTC 时间戳
    """
    return datetime.datetime.now().replace(tzinfo=timezone.utc).timestamp()


def makedir(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.
    If a directory is provided, creates that directory. If a file is provided (i.e. isfile == True),
    creates the parent directory for that file.
    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != "":
        os.makedirs(path, exist_ok=True)


def rmdir(path: str):
    """
    Creates a directory given a path to either a directory or file.
    If a directory is provided, creates that directory. If a file is provided (i.e. isfile == True),
    creates the parent directory for that file.
    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    try:
        shutil.rmtree(path)
    except Exception as Ex:
        print("rmdir failure", Ex)


class OnlineEstimator:
    """
    Knuth 在线估计器，用于累积计算均值和方差。
    Simple storage-less Knuth estimator which
    accumulates mean and variance.
    """

    def __init__(self, x_):
        self.n = 1
        self.mean = x_ * 0.0
        self.m2 = x_ * 0.0
        delta = x_ - self.mean
        self.mean += delta / self.n
        delta2 = x_ - self.mean
        self.m2 += delta * delta2

    def __call__(self, x_):
        """
        更新估计器并返回当前的均值和方差。
        
        :param x_: 新的数据点
        :return: 当前的均值和方差
        """
        self.n += 1
        delta = x_ - self.mean
        self.mean += delta / self.n
        delta2 = x_ - self.mean
        self.m2 += delta * delta2
        return self.mean, self.m2 / (self.n - 1)


# useful for debugging Torch memory leaks.
def get_all_allocated_torch_tensors():
    objs = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                objs.append(obj)
        except:
            pass
    return objs


def records_mp(recs, func, args=None, n=None):
    """
    使用多进程对记录应用指定的函数。
    
    :param recs: 要处理的记录列表
    :param func: 要应用的函数
    :param args: 额外的参数
    :param n: 进程数，默认为 CPU 核心数
    :return: 处理后的记录列表
    """
    if n is None:
        n = min([mp.cpu_count(), len(recs)])
    if args is None:
        args = tuple()

    before_ct = len(recs)
    mp_args = [(sub_recs, *args) for sub_recs in batch_indexable(recs, n)]
    with mp.Pool(processes=n) as pool:
        recs = pool.starmap(func, mp_args)

    recs = [rec for sub_recs in recs for rec in sub_recs]
    assert len(recs) == before_ct

    return recs


def execute_with_timeout(method, args, timeout):
    """Execute method with timeout, return None if timeout exceeded"""
    result = None

    def timeout_handler(signum, frame):
        # This function is called when the timeout is reached
        # It raises an exception to stop the execution of the method
        raise TimeoutError("Execution timed out")

    # Set up the timeout signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Start the timeout timer

    try:
        result = method(*args)  # Execute the method
    except TimeoutError:
        pass  # Execution timed out
    finally:
        signal.alarm(0)  # Cancel the timeout timer

    return result


def get_tnet_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def dicts_to_keyval(list_of_dicts, key: str, value: str):
    # convert list of dictionaries with unique keys to key value mapping.
    return {dct[key]: dct[value] for dct in list_of_dicts}


def query_yes_no(question, default=None):
    """https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
