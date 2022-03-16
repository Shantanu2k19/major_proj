from torch.utils.data import DataLoader
import importlib

def get_dataset(dset, dataset_config, njobs, metadata=None):
    Dataset = importlib.import_module(f'.{dataset_config.dataset_name}', package=__package__).Dataset
    return Dataset(dset, 
        dataset_config.indexes_path, 
        dataset_config.feat,
        dataset_config.feat_path,
        dataset_config.seglen,
        njobs,
        metadata)

def get_dataloader(dset, dataloader_config, dataset):
    ret = DataLoader(dataset, **dataloader_config[dset])
    return ret

