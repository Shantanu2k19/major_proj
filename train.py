import os
import torch
from util.parser import get_parser
from util.config import Config
from util.mytorch import same_seeds
from tqdm import tqdm
from model import build_model
from pickle import load,dump
from dataloader import get_dataset, get_dataloader
from util.vocoder import get_vocoder
from util.mytorch import save_checkpoint, load_checkpoint

class Trainer():
    def __init__(self, config, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vocoder = get_vocoder(device=self.device)
        self.mel2wav = self.vocoder.mel2wav
        if args.load != '':
            self.ckpt_dir_flag, self.train_set, self.dev_set, self.train_loader, self.dev_loader = \
                self.load_data(ckpt_path=args.load, 
                    dataset_config=config.dataset, 
                    dataloader_config=config.dataloader,
                    njobs=args.njobs)
            self.model_state, self.step_fn = self.build_model(config.build)
            self.model_state = self.load_model(self.model_state, args.load)
        else:
            self.ckpt_dir_flag, self.train_set, self.dev_set, self.train_loader, self.dev_loader = \
                self.gen_data(ckpt_path=config.ckpt_dir, flag=config.flag,
                    dataset_config=config.dataset, 
                    dataloader_config=config.dataloader,
                    njobs=args.njobs)
            self.model_state, self.step_fn = self.build_model(config.build)

    @staticmethod
    def build_model(build_config, mode, device):
        return build_model(build_config, device=device, mode=mode)

    @staticmethod
    def gen_data(ckpt_path, flag, dataset_config, dataloader_config, njobs):
        ckpt_dir = os.path.join(ckpt_path, '')
        ckpt_dir_flag = os.path.join(ckpt_dir, flag)
        prefix = os.path.basename(os.path.dirname(dataset_config.indexes_path))
        train_pkl = os.path.join(ckpt_dir, f'{prefix}_train.pkl')
        dev_pkl = os.path.join(ckpt_dir, f'{prefix}_dev.pkl')
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(ckpt_dir, 'dirtype'), 'w') as f:
            f.write('ckpt_dir')
        os.makedirs(ckpt_dir_flag, exist_ok=True)
        with open(os.path.join(ckpt_dir_flag, 'dirtype'), 'w') as f:
            f.write('ckpt_dir_flag')
        if os.path.exists(train_pkl):
            train_data = load(open(train_pkl, 'rb'))
            dev_data = load(open(dev_pkl, 'rb'))
            train_set = get_dataset(dset='train', dataset_config=dataset_config, njobs=njobs, metadata=train_data)
            dev_set = get_dataset(dset='dev', dataset_config=dataset_config, njobs=njobs, metadata=dev_data)
        else:
            train_set = get_dataset(dset='train', dataset_config=dataset_config, njobs=njobs)
            dev_set = get_dataset(dset='dev', dataset_config=dataset_config, njobs=njobs)
            dump(train_set.data, open(train_pkl, 'wb'))
            dump(dev_set.data, open(dev_pkl, 'wb'))
        train_loader = get_dataloader(dset='train', dataloader_config=dataloader_config, dataset=train_set)
        dev_loader = get_dataloader(dset='dev', dataloader_config=dataloader_config, dataset=dev_set)
        
        return ckpt_dir_flag, train_set, dev_set, train_loader, dev_loader

    @staticmethod
    def load_data(ckpt_path, dataset_config, dataloader_config, njobs):
        if os.path.isdir(ckpt_path):
            d = os.path.join(ckpt_path, '')
            with open(os.path.join(d, 'dirtype'), 'r') as f:
                dirtype = f.read().strip()
            if dirtype == 'ckpt_dir':
                ckpt_dir = d
                flag = 'default'
                ckpt_dir_flag = os.path.join(ckpt_dir, flag)
                ckpt_path = ckpt_dir_flag
            elif dirtype == 'ckpt_dir_flag':
                ckpt_dir_flag = os.path.dirname(d)
                ckpt_dir = os.path.dirname(ckpt_dir_flag)
                flag = os.path.basename(ckpt_dir_flag)
            else:
                raise NotImplementedError(f'Wrong dirtype: {dirtype} from {d}.')
        else:
            ckpt_dir_flag = os.path.dirname(ckpt_path)
            ckpt_dir = os.path.dirname(ckpt_dir_flag)
            flag = os.path.basename(ckpt_dir_flag)

        prefix = os.path.basename(os.path.dirname(dataset_config.indexes_path))
        train_data = load(open(os.path.join(ckpt_dir, f'{prefix}_train.pkl'), 'rb'))
        dev_data = load(open(os.path.join(ckpt_dir, f'{prefix}_dev.pkl'), 'rb'))

        train_set = get_dataset(dset='train', dataset_config=dataset_config, njobs=njobs, metadata=train_data)
        dev_set = get_dataset(dset='dev', dataset_config=dataset_config, njobs=njobs, metadata=dev_data)
        train_loader = get_dataloader(dset='train', dataloader_config=dataloader_config, dataset=train_set)
        dev_loader = get_dataloader(dset='dev', dataloader_config=dataloader_config, dataset=dev_set)
        
        return ckpt_dir_flag, train_set, dev_set, train_loader, dev_loader


    @staticmethod
    def save_model(model_state, save_path):
        dynamic_state = model_state['_dynamic_state']
        state = {}
        for key in dynamic_state:
            if hasattr(model_state[key], 'state_dict'):
                state[key] = model_state[key].state_dict()
            else:
                state[key] = model_state[key]
        save_checkpoint(state, save_path)

    @staticmethod
    def load_model(model_state, load_path):
        dynamic_state = model_state['_dynamic_state']
        state = load_checkpoint(load_path)
        for key in dynamic_state:
            if hasattr(model_state[key], 'state_dict'):
                try:
                    model_state[key].load_state_dict(state[key])
                except:
                    print(f'Load Model Error: key {key} not found')
            elif key in state.keys():
                try:
                    model_state[key] = state[key]
                except:
                    print('Load Other State Error')
            else:
                print(f'{key} is not in state_dict.')
        return model_state

    def build_model(self, build_config):
        return build_model(build_config, mode='train', device=self.device)

    # ====================================================
    #  train
    # ====================================================
    def train(self, total_steps, verbose_steps, log_steps, save_steps, eval_steps):
        self.model_state['steps'] = 0
        while self.model_state['steps'] <= total_steps:
            train_bar = tqdm(self.train_loader)
            for data in train_bar:
                meta = self.step_fn(self.model_state, data)
                meta['log']['steps'] = self.model_state['steps']
                train_bar.set_postfix(meta['log'])
            self.model_state['steps'] += 1
            if self.model_state['steps'] % save_steps == 0:
                self.save_model(self.model_state, \
                    os.path.join(self.ckpt_dir_flag, f'steps_{self.model_state["steps"]}.pth'))
            if self.model_state['steps'] % eval_steps == 0 and self.model_state['steps'] != 0:
                self.evaluate()

    # ====================================================
    #  evaluate
    # ====================================================
    def evaluate(self):
        try:
            data = next(self.dev_iter)
        except :
            self.dev_iter = iter(self.dev_loader)
            data = next(self.dev_iter)
        with torch.no_grad():
            meta = self.step_fn(self.model_state, data, train=False)
            mels = meta['mels']


def get_args():
    parser = get_parser(description='Train')
    # config
    parser.add_argument('--config', '-c', default='./config/train_again-c4s.yaml', help='config yaml file')
    # seed
    parser.add_argument('--seed', type=int, help='random seed', default=961998)
    parser.add_argument('--load', '-l', type=str, help='Load a checkpoint.', default='')
    parser.add_argument('--njobs', '-p', type=int, help='', default=10)
    parser.add_argument('--total-steps', type=int, help='Total training steps.', default=600)
    parser.add_argument('--verbose-steps', type=int, help='The steps to update tqdm message.', default=1)
    parser.add_argument('--log-steps', type=int, help='The steps to log data for the customed logger (wandb, tensorboard, etc.).', default=1)
    parser.add_argument('--save-steps', type=int, help='The steps to save a checkpoint.', default=5)
    parser.add_argument('--eval-steps', type=int, help='The steps to evaluate.', default=5)
    return parser.parse_args()

if __name__ == '__main__':
    # config
    args = get_args()
    config = Config(args.config)
    same_seeds(args.seed)

    # build trainer
    trainer = Trainer(config, args)

    # train
    trainer.train(total_steps=args.total_steps,
        verbose_steps=args.verbose_steps,
        log_steps=args.log_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps)
    

