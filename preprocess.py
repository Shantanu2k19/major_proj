import os
import copy
import numpy as np
from glob import glob
from tqdm import tqdm
from functools import partial
from multiprocessing.pool import ThreadPool
from util.parser import get_parser
from util.config import Config
from util.dsp import Dsp


def preprocess_one(input_items, module, output_path=''):
    input_path, basename = input_items
    y = module.load_wav(input_path)
    if module.config.dtype == 'wav':
        ret = y
    elif module.config.dtype == 'melspectrogram':
        ret = module.wav2mel(y)
    else:
        print(f'Not implement feature type {module.config.dtype}')
    if output_path == '':
        return ret
    else:
        if type(ret) is np.ndarray:
            np.save(os.path.join(output_path, f'{basename}.npy'), ret)
        else:
            print(f'Feature {module.config.dtype} is not saved: {input_path}.')
        return 1


class Preprocessor():
    def __init__(self, config):
        self.dsp_modules = {feat:Dsp(config.feat[feat]) for feat in config.feat_to_preprocess}

    def preprocess(self, input_path, output_path, feat, njobs):
        file_dict = self.gen_file_dict(input_path)
        print(f'Starting to preprocess from {input_path}.')
        self.preprocess_from_file_dict(
            file_dict=file_dict, output_path=output_path, feat=feat, njobs=njobs)
        print(f'Saving processed file to {output_path}.')

    def preprocess_from_file_dict(self, file_dict, output_path, feat, njobs):
        os.makedirs(os.path.join(output_path, feat), exist_ok=True)
        module = self.dsp_modules[feat]
        task = partial(preprocess_one, module=module,
                       output_path=os.path.join(output_path, feat))
        with ThreadPool(njobs) as pool:
            _ = list(tqdm(pool.imap(task, file_dict.items()),
                     total=len(file_dict), desc=f'Preprocessing '))

    def gen_file_dict(self, input_path):
        lst = glob(os.path.join(input_path, '*.wav'))
        return dict(zip(lst, [os.path.basename(f) for f in lst]))

def get_args():
    parser = get_parser(description='Preprocess')
    # config
    parser.add_argument('--config', '-c', default='./config/preprocess.yaml', help='config yaml file')
    # multi thread
    parser.add_argument('--njobs', '-p', type=int, default=4)
    return parser.parse_args()


if __name__ == '__main__':
    # config
    args = get_args()
    config = Config(args.config)

    # build preprocessor
    preprocessor = Preprocessor(config)
    
    # process
    for feat in config.feat_to_preprocess:
        preprocessor.preprocess(input_path=config.input_path, output_path=config.output_path, feat=feat, njobs=args.njobs)

