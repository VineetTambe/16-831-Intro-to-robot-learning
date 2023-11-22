from cs16831.hw4_part2.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch

def init_method_1(model):
    if isinstance(model, nn.Linear):
        model.weight.data.uniform_()
        model.bias.data.uniform_()

def init_method_2(model):
    if isinstance(model, nn.Linear):
        model.weight.data.normal_()
        model.bias.data.normal_()

class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        # <TODO>: Create two neural networks:
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f

        self.f = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.output_size,
                n_layers=self.n_layers,
                size=self.size,
            )
        self.f.apply(init_method_1)
        self.f.to(ptu.device)


        self.f_hat = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.output_size,
                n_layers=self.n_layers,
                size=self.size,
            )
        self.f_hat.apply(init_method_2)
        
        self.optimizer = self.optimizer_spec.constructor(self.f_hat.parameters(), **self.optimizer_spec.optim_kwargs)

        self.f_hat.to(ptu.device)


        # print(f'{self.ob_dim=}')
        # print(f'{self.output_size=}')
        # print(f'{self.n_layers=}')
        # print(f'{self.size=}')
        

    def forward(self, ob_no):
        # <TODO>: Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        # pass
        if not torch.is_tensor(ob_no):
            ob_no = ptu.from_numpy(ob_no)
        pred = self.f(ob_no).detach()
        pred_hat = self.f_hat(ob_no)

        # pred_error = torch.norm(pred - pred_hat)
        pred_error = (pred - pred_hat) ** 2
        return pred_error

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        # <TODO>: Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        # pass

        # ob_no = ptu.from_numpy(ob_no)
        # pred = self.f(ob_no).detach()
        # pred_hat = self.f_hat(ob_no)

        # loss = torch.mean((pred - pred_hat)**2/)

        loss = torch.mean(self.forward(ob_no))
        
        self.optimizer.zero_grad()
        
        loss.backward()
        
        self.optimizer.step()
        
        return loss.item()
