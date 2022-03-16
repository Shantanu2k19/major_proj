import os
import torch
from tqdm import tqdm
from .base import BaseAgent

class Trainer(BaseAgent):
    def __init__(self, config, args):
        super().__init__(config, args)
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

    def build_model(self, build_config):
        return super().build_model(build_config, mode='train', device=self.device)

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
