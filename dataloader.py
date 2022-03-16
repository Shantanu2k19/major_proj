import pickle as pk
from functools import partial
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from util.transform import segment 


class BaseDataset(Dataset):
    def __init__(self, dset, indexes_path, feat, feat_path, seglen, njobs, metadata):
        super().__init__()
        if metadata is None:
            data = self.gen_metadata(dset, indexes_path, feat, feat_path, njobs)
        else:
            data = metadata
        self.seglen = seglen
        self.data = BaseDataset.drop_invalid_data(data, verbose=True)
        self.speaker2data = BaseDataset.gen_speaker2data(self.data)

    def gen_metadata(self, dset, indexes_path, feat, feat_path, njobs):
        indexes = pk.load(open(indexes_path, 'rb'))
        sdata = indexes[dset]
        metadata = [(spk, d) for spk, data in sdata.items() for d in data]

        task = partial(self.sub_process, feat=feat, feat_path=feat_path)
        with ThreadPool(njobs) as pool:
            results = list(tqdm(pool.imap(task, metadata), total=len(metadata)))
        print(len(results))
        return results

    @staticmethod
    def gen_speaker2data(data):
        speaker2data = {}
        for i, d in enumerate(data):
            if d['speaker'] in speaker2data.keys():
                speaker2data[d['speaker']].append(i)
            else:
                speaker2data[d['speaker']] = [i]
        return speaker2data

    @staticmethod
    def drop_invalid_data(data, verbose=False):
        return data

    def sub_process(self, each_data, feat, feat_path):
        speaker = each_data[0]
        basename = each_data[1]
        # print(each_data, feat, feat_path)
        ret = {}
        for f in feat:
            path = os.path.join(feat_path, f, basename)
            if os.path.isfile(path):
                ret[f] = np.load(path)
            else:
                print(f'Skip {path} {f}: invalid file.')
                return
        ret['speaker'] = speaker
        return ret
    
    # def __getitem__(self, index):
    #     y = segment(self.data[index], seglen=self.seglen)
    #     return y

    def __getitem__(self, index):
        speaker = self.data[index]['speaker']
        mel = self.data[index]['mel']

        mel = segment(mel, return_r=False, seglen=self.seglen)

        meta = {
            'mel': mel,
        }
        return meta
    def __len__(self):
        return len(self.data)

def get_dataset(dset, dataset_config, njobs, metadata=None):
    return BaseDataset(dset, 
        dataset_config.indexes_path, 
        dataset_config.feat,
        dataset_config.feat_path,
        dataset_config.seglen,
        njobs,
        metadata)

def get_dataloader(dset, dataloader_config, dataset):
    return DataLoader(dataset, **dataloader_config[dset])

