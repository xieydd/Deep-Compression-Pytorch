import argparse
import os

import torch
import torch.nn as nn
from net.quantization import apply_weight_sharing
import util
import torchvision.datasets as datasets
from torchvision import datasets, transforms
import dp 

parser = argparse.ArgumentParser(description='This program quantizes weight by using weight sharing')
parser.add_argument('model', type=str, help='path to saved pruned model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--output', default='saves/model_after_weight_sharing.ptmodel', type=str,
                    help='path to model output')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()


# Define the model
model = torch.load(args.model)
criterion = nn.CrossEntropyLoss().cuda()
valdir = os.path.join(args.data, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
#dp.validate(val_loader, model, criterion)

# Weight sharing
model = apply_weight_sharing(model)
print('accuacy after weight sharing')
dp.validate(val_loader, model, criterion)

# Save the new model
os.makedirs('saves', exist_ok=True)
torch.save(model, args.output)
