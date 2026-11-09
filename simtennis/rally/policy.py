"""Frozen upstream PPO actor for cheap deterministic opponent inference."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FrozenPolicy:
    layers: list
    input_size: int

    @classmethod
    def from_model(cls, model):
        import torch
        layers = []
        for module in list(model.policy.mlp_extractor.policy_net) + [model.policy.action_net]:
            if isinstance(module, torch.nn.Linear):
                layers.append((module.weight.detach().cpu().numpy().copy(),
                               module.bias.detach().cpu().numpy().copy()))
            elif not isinstance(module, torch.nn.Tanh):
                raise ValueError('Only the declared upstream tanh MLP is supported')
        if model.policy.squash_output:
            raise ValueError('Expected unsquashed Gaussian PPO actions')
        return cls(layers, layers[0][0].shape[1])

    @classmethod
    def load(cls, path):
        from stable_baselines3 import PPO
        return cls.from_model(PPO.load(Path(path), device='cpu'))

    def __call__(self, observation):
        values = np.asarray(observation[:self.input_size], dtype=np.float32)
        for index, (weight, bias) in enumerate(self.layers):
            values = weight @ values + bias
            if index+1 < len(self.layers):
                values = np.tanh(values)
        return np.clip(values, -1, 1)

