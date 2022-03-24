import os
from util.parser import get_parser
from util.config import Config
from pickle import dump
from  glob import glob
from collections import defaultdict

class Indexer():
    def __init__(self):
        self.dir_path = "/Users/sameer/Downloads/cv-corpus-8.0-2022-01-19/hi/"

    def make_indexes(self, input_path, output_path, split_all, split_train):
        print(f'Starting to make indexes from {input_path}.')
        file_list = self.gen_file_list(input_path)
        indexes = self.split(file_list, split_all, split_train)

        assert len(indexes['train'].keys()) <= split_all.train

        os.makedirs(os.path.join(output_path), exist_ok=True)
        out_path = os.path.join(output_path, 'indexes.pkl')
        dump(indexes, open(out_path, 'wb'))
        print(f'The output file is saved to {out_path}')

    def gen_file_list(self, input_path):
        return sorted(glob(os.path.join(input_path, '*')))

    def get(self):
        d = defaultdict(set)
        os.chdir(self.dir_path)
        a = os.listdir()
        os.chdir("features/mel")
        have = set(os.listdir())
        os.chdir("../..")
        for file in a:
            if file.split(".")[-1] != "tsv":
                continue
            self.custom(file,d,have)
        inv = {}
        for key,arr in d.items():
            for s in arr:
                inv[s] = key
        return d,inv;

    def custom(self,file,d,have):
        with open(file,"r") as f:
            data = f.read().split("\n")
        for line in range(1,len(data)):
            if len(data[line]) == 0:
                continue
            a = data[line].split()
            fname = a[1].split(".")[0]  + ".wav.npy"
            if fname in have:
                d[a[0]].add(fname)

    def split(self, file_list, split_all, split_train):
        train,dev = defaultdict(list),defaultdict(list)
        pwd = os.getcwd()
        _ , inv = self.get()
        os.chdir(pwd)
        for fname in file_list:
            s = fname.split("/")[-1]
            person_name = inv[s]
            if len(train)<split_all.train:
                if len(train[person_name])<split_train.train:
                    train[person_name].append(fname)
            else:
                if len(dev[person_name])<split_train.dev and person_name not in train:
                    dev[person_name].append(fname)
        keys = list(train.keys())
        print(len(train),len(dev))
        for key in keys:
            if len(train[key]) == 0:
                train.pop(key)
        keys = list(dev.keys())
        for key in keys:
            if len(dev[key]) == 0:
                dev.pop(key)
        train,dev = dict(train),dict(dev)
        print(len(train),len(dev))
        return {'train': train,'dev': dev}


def get_args():
    parser = get_parser(description='Make indexes.')

    # config
    parser.add_argument('--config', '-c', default='./config/indexes.yaml')

    return parser.parse_args()

if __name__ == '__main__':
    # config
    args = get_args()
    config = Config(args.config)

    # build indexer
    indexer = Indexer()

    # make indexes
    indexer.make_indexes(input_path=config.input_path, output_path=config.output_path, 
        split_all=config.split_all, split_train=config.split_train)