# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model, build_CNNMLP_model


def build_ACT_model_and_optimizer(args_override):
    class Args:
        def __init__(self, args_dict):
            for k, v in args_dict.items():
                setattr(self, k, v)

    args = Args(args_override)

    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": getattr(args, "lr_backbone", 1e-5),
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=getattr(args, "weight_decay", 1e-4))

    return model, optimizer

def build_CNNMLP_model_and_optimizer(args_override):
    class Args:
        def __init__(self, args_dict):
            for k, v in args_dict.items():
                setattr(self, k, v)

    args = Args(args_override)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": getattr(args, "lr_backbone", 1e-5),
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=getattr(args, "weight_decay", 1e-4))

    return model, optimizer

