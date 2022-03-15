import os
from .base import BaseIndexer
dir_path = "/content/drive/MyDrive/Hindi Audio/common_voice"
from collections import defaultdict
def f(file,d,have):
    with open(file,"r") as f:
        data = f.read().split("\n")
    for line in range(1,len(data)):
        if len(data[line]) == 0:
            continue
        a = data[line].split()
        fname = a[1].split(".")[0]  + ".wav.npy"
        if fname in have:
            d[a[0]].add(fname)
def get():
    d = defaultdict(set)
    os.chdir(dir_path)
    a = os.listdir()
    os.chdir("features/mel")
    have = set(os.listdir())
    os.chdir("../..")
    for file in a:
        if file.split(".")[-1] != "tsv":
            continue
        f(file,d,have)
    sm = 0
    for v in d:
        sm += len(d[v])
    inv = {}
    for key,arr in d.items():
        for s in arr:
            inv[s] = key
    return d,inv;


class Indexer(BaseIndexer):
    def __init__(self):
        super().__init__()

    def split(self, file_list, split_all, split_train):

        train = defaultdict(list)
        dev = defaultdict(list)
        pwd = os.getcwd()
        _,inv = get()
        os.chdir(pwd)
        for fname in file_list:
            s = fname.split("/")[-1]
            person_name = inv[s]
            if len(train)<split_all.train:
                if len(train[person_name])<split_train.train:
                    train[person_name].append(fname)
            else:
                if len(dev[person_name])<split_train.dev:
                    dev[person_name].append(fname)
        keys = train.keys()
        print(len(train),len(dev))
        for key in keys:
            if len(train[key]) == 0:
                train.pop(key)
        keys= dev.keys()
        for key in keys:
            if len(dev[key]) == 0:
                dev.pop(key)
        train = dict(train)
        dev = dict(dev)
        print(len(train),len(dev))
        indexes = {
                'train': train,
                'dev': dev
                }
        return indexes