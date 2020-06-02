import os
import glob
import math
import random
import sys
import time
import numpy as np

from PIL import Image

from scipy.signal import savgol_filter
from six.moves import xrange

import umap
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl


parser = argparse.ArgumentParser(description='VQ-VAE.')
parser.add_argument(
    '--rank',
    default=0,
    type=int,
    help="Rank of the training task"
)
parser.add_argument(
    '--world_size',
    default=1,
    type=int,
    help="World size of training tasks"
)
parser.add_argument(
    '--num_workers',
    default=4,
    type=int,
    help="Number of parallel data loaders"
)
parser.add_argument(
    '--init_method',
    default='tcp://192.168.1.154:23456',
    help="Master host (e.g. 'tcp://192.168.1.154:23456')"
)
parser.add_argument(
    '--epoch_start',
    default=1,
    type=int,
    help="Epoch to start training from (load savepoint)"
)
parser.add_argument(
    '--num_epochs',
    default=15000,
    type=int,
    help="Number of epochs to run"
)
parser.add_argument(
    '--batch_size',
    default=64,
    type=int,
    help="Number of data elements for one pass"
)
parser.add_argument(
    '--image_width',
    default=128,
    type=int,
    help="Horizontal image dimension"
)
parser.add_argument(
    '--image_height',
    default=128,
    type=int,
    help="Vertical image dimension"
)
parser.add_argument(
    '--backend',
    default="nccl",
    help="Distributed backend to use"
)

args = parser.parse_args()
print(args)

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 512
commitment_cost = 0.25
decay = 0.99
learning_rate = 1e-3
saved_models_path = "./saved_models/{epoch:08d}.vq-vae.net"
saved_results_path = "./results/{epoch:08d}.vq-vae.{name}"

if (torch.cuda.is_available()):
    device = "cuda"
    print("CUDA is available.")
else:
    device = "cpu"
    print("This example uses nccl as backend which is only available for (nVidia) GPUs.")
    sys.exit(-1)

pl.seed_everything(12345)

os.makedirs(os.path.dirname(saved_models_path.format(epoch=0)), exist_ok=True)
os.makedirs(os.path.dirname(saved_results_path.format(epoch=0,name='')), exist_ok=True)


class NoisySource_ImageDataset(Dataset):
    def __init__(self, images_pattern, transforms_before=None, transforms_after=None):
        self.images_pattern = images_pattern
        self.transforms_before = transforms_before
        self.transforms_after = transforms_after
        self.image_names = glob.glob(images_pattern)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        while True:
            img_name = self.image_names[idx]
            try:
                image_trg = self._load_image(img_name)
                if (image_trg.size[0] >= args.image_width) and (image_trg.size[1] >= args.image_height):
                    break
                print("Image {} too small ({}).".format(img_name, image_trg.size))
            except Exception as e:
                print("Exception {} - caught.".format(str(e)))
            idx = (idx+1) % len(self)
            
        if self.transforms_before:
            image_trg = self.transforms_before(image_trg)
        image_src = self._pixel_noise(image_trg)
        if self.transforms_after:
            image_trg = self.transforms_after(image_trg)
        if self.transforms_after:
            image_src = self.transforms_after(image_src)
        return {
            "image_src": image_src,
            "image_trg": image_trg,
        }

    def _load_image(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
    def _pixel_noise(self, src):
        factor = random.randint(8,64)
        pix = src.copy()
        nsiz_w = max(int(src.width/factor),1)
        nsiz_h = max(int(src.height/factor),1)
        pix = pix.resize((nsiz_w, nsiz_h))
        pix = pix.resize(src.size, Image.NEAREST)
        w0 = random.randint(0,src.width)
        h0 = random.randint(0,src.height)
        w1 = random.randint(w0, src.width)
        h1 = random.randint(h0, src.height)
        pix = pix.crop((w0,h0,w1,h1))
        res = src.copy()
        res.paste(pix, (w0,h0))
        return res

# Training Data
# Dataset

training_data_dataset = NoisySource_ImageDataset(
    "/data/imagenet/imagenet_images/*/*.jpg",
    transforms_before=transforms.Compose(
        [
            transforms.RandomCrop((args.image_height,args.image_width), padding_mode='reflect')
        ]
    ),
    transforms_after=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
        ]
    ),
)

# training_data_dataset = datasets.ImageFolder(
#     "/data/imagenet",
#     transform=transforms.Compose(
#         [
#             transforms.Resize((args.image_height,args.image_width), interpolation=2),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
#         ]
#     ),
# #    target_transform=None,
# #    loader=<function default_loader>,
# #    is_valid_file=None
# )
# training_data_dataset = datasets.CIFAR10(
#     root="data", train=True, download=True,
#     transform=transforms.Compose(
#         [
#             transforms.Resize(IMAGE_SIZE, interpolation=2),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
#         ]
#     )
# )
# training_data_dataset = datasets.MNIST(
#     root="data",
#     train=True,
#     transform=transforms.Compose(
#         [
#             transforms.Resize(IMAGE_SIZE, interpolation=2),
#             transforms.Grayscale(num_output_channels=3),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
#         ]
#     ),
#     target_transform=None,
#     download=True
# )
# DataSampler
training_data_sampler = DistributedSampler(
    training_data_dataset,
    num_replicas=args.world_size,
    rank=args.rank,
    shuffle=True
)
# DataLoader
training_data_loader = DataLoader(
    training_data_dataset,
    batch_size=args.batch_size, 
    shuffle=(training_data_sampler is None),
    sampler=training_data_sampler,
    pin_memory=True,
    num_workers=args.num_workers,
)

# Validation Data
# Dataset
validation_data_dataset = NoisySource_ImageDataset(
    "/data/imagenet/imagenet_images/*/*.jpg",
    transforms_before=transforms.Compose(
        [
            transforms.RandomCrop((args.image_height,args.image_width), )
        ]
    ),
    transforms_after=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
        ]
    ),
)
# validation_data_dataset = datasets.ImageFolder(
#     "/data/imagenet",
#     transform=transforms.Compose(
#         [
#             transforms.Resize((args.image_height,args.image_width), interpolation=2),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
#         ]
#     ),
# #    target_transform=None,
# #    loader=<function default_loader>,
# #    is_valid_file=None
# )
# validation_data_dataset = datasets.CIFAR10(
#     root="data", train=False, download=True,
#     transform=transforms.Compose(
#         [
#             transforms.Resize(IMAGE_SIZE, interpolation=2),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
#         ]
#     )
# )
# validation_data_dataset = datasets.MNIST(
#     root="data",
#     train=False,
#     transform=transforms.Compose(
#         [
#             transforms.Resize(IMAGE_SIZE, interpolation=2),
#             transforms.Grayscale(num_output_channels=3),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
#         ]
#     ),
#     target_transform=None,
#     download=True
# )
# DataSampler
validation_data_sampler = DistributedSampler(
    validation_data_dataset,
    num_replicas=args.world_size,
    rank=args.rank,
    shuffle=True
)
# DataLoader
validation_data_loader = DataLoader(
    validation_data_dataset,
    batch_size=32,
    shuffle=(validation_data_sampler is None),
    sampler=validation_data_sampler,
    pin_memory=True,
)

training_data_loader_iterator = iter(training_data_loader)
validation_data_loader_iterator = iter(validation_data_loader)


# Vector Quantizer Layer
# 
# This layer takes a tensor to be quantized. The channel dimension will be used as the space in which to quantize. All other dimensions will be flattened and will be seen as different examples to quantize.
# 
# The output tensor will have the same shape as the input.
# 
# As an example for a `BCHW` tensor of shape `[16, 64, 32, 32]`, we will first convert it to an `BHWC` tensor of shape `[16, 32, 32, 64]` and then reshape it into `[16384, 64]` and all `16384` vectors of size `64`  will be quantized independently. In otherwords, the channels are used as the space in which to quantize. All other dimensions will be flattened and be seen as different examples to quantize, `16384` in this case.

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

# We will also implement a slightly modified version  which will use exponential moving averages to update the embedding vectors instead of an auxillary loss. This has the advantage that the embedding updates are independent of the choice of optimizer for the encoder, decoder and other parts of the architecture. For most experiments the EMA version trains faster than the non-EMA version.

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay +                                      (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

# Encoder & Decoder Architecture
# 
# The encoder and decoder architecture is based on a ResNet and is implemented below:

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens//2,
            kernel_size=4,
            stride=2, padding=1
        )
        self._conv_2 = nn.Conv2d(
            in_channels=num_hiddens//2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2, padding=1
        )
        self._conv_3 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1, padding=1
        )
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3, 
            stride=1, padding=1
        )
        
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens
        )
        
        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens, 
            out_channels=num_hiddens//2,
            kernel_size=4, 
            stride=2, padding=1
        )
        
        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens//2, 
            out_channels=3,
            kernel_size=4, 
            stride=2, padding=1
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)


# Model & Optimizer
class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()
        
        self._encoder = Encoder(
            3, num_hiddens,
            num_residual_layers, 
            num_residual_hiddens
        )
        self._pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, 
            out_channels=embedding_dim,
            kernel_size=1, 
            stride=1
        )
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(
                num_embeddings, embedding_dim, 
                commitment_cost, decay
            )
        else:
            self._vq_vae = VectorQuantizer(
                num_embeddings, embedding_dim,
                commitment_cost
            )
        self._decoder = Decoder(
            embedding_dim,
            num_hiddens, 
            num_residual_layers, 
            num_residual_hiddens
        )

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity

model = Model(
    num_hiddens, num_residual_layers, num_residual_hiddens,
    num_embeddings, embedding_dim, 
    commitment_cost, decay
).to(device)

load_model_path = saved_models_path.format(epoch=args.epoch_start)
if os.path.exists(load_model_path):
    print("Loading {}".format(load_model_path))
    load_dict = torch.load(load_model_path)
    model.load_state_dict(load_dict['model_state_dict'])
    print(load_dict.keys())
else:
    print("Could not read {}; no data was loaded.".format(load_model_path))

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

def save_model(model, epoch):
    save_dict = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'args': repr(args),
    }
    torch.save(save_dict, saved_models_path.format(epoch=epoch))

    
# Show reconstructions
def validate():
    model.eval()

    # (valid_originals, _) = next(iter(validation_data_loader))
    while True:
        try:
            data = next(validation_data_loader_iterator)
            break
        except StopIteration:
            validation_data_loader_iterator = iter(validation_data_loader)
    valid_originals = data['image_src']
    valid_originals_targets = data['image_trg']
    valid_originals = valid_originals.to(device)
    vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize)

    save_image(
        valid_reconstructions.cpu().data+0.5,
        fp=saved_results_path.format(epoch=epoch, name='reconstruction.png'),
        nrow=8,
        padding=2,
        normalize=False,
        range=None,
        scale_each=False,
        pad_value=0,
        format="png"
    )
    save_image(
        valid_originals.cpu()+0.5,
        fp=saved_results_path.format(epoch=epoch, name='originals.png'),
        nrow=8,
        padding=2,
        normalize=False,
        range=None,
        scale_each=False,
        pad_value=0,
        format="png"
    )
    save_image(
        valid_originals_targets.cpu()+0.5,
        fp=saved_results_path.format(epoch=epoch, name='targets.png'),
        nrow=8,
        padding=2,
        normalize=False,
        range=None,
        scale_each=False,
        pad_value=0,
        format="png"
    )
    model.train()

    
def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def device_memory_info(device_number=0):
    t = torch.cuda.get_device_properties(device_number).total_memory
    c = torch.cuda.memory_cached(device_number)
    a = torch.cuda.memory_allocated(device_number)
    f = c-a  # free inside cache
    return {
        'total': t,
        'cached': c,
        'allocated': a,
        'free': f,
    }

def print_device_memory_info():
    info = device_memory_info()
    print(
        "Total: {}, Cached: {}, Allocated: {}, Free: {}".format(
            convert_size(info['total']),
            convert_size(info['cached']),
            convert_size(info['allocated']),
            convert_size(info['free']),
        )
    )


class Timer(object):
    def __init__(self):
        self.time_instantiated = time.time()
        self.time_last = None
        self.kwargs_last = None
    
    def start(self, **kwargs):
        self.kwargs_last = kwargs
        self.time_last = time.time()
        return {
            'span': 0,
            'kwargs': kwargs,
        }
        
    def lap(self, **kwargs):
        time_now = time.time()
        span = time_now - self.time_last
        kwargs = self.kwargs_last
        self.time_last = time_now
        self.kwargs_last = kwargs
        result = kwargs
        result['span'] = span
        return result


# Train

print_device_memory_info()

print("Initializing process group...")
dist.init_process_group(
    backend=args.backend,
    init_method=args.init_method,
    rank=args.rank,
    world_size=args.world_size
)

print("Trainig starts!")
model.train()
train_res_recon_error = []
train_res_perplexity = []

timer = Timer()
timer.start(iterations=args.epoch_start)

for epoch in xrange(args.epoch_start, args.epoch_start+args.num_epochs+1):
    try:
        data = next(training_data_loader_iterator)
    except StopIteration:
        training_data_loader_iterator = iter(training_data_loader)
        data = next(training_data_loader_iterator)
    image_src = data['image_src']
    image_trg = data['image_trg']
    image_src = data['image_src'].to(device)
    image_trg = data['image_trg'].to(device)
    optimizer.zero_grad()

    vq_loss, data_recon, perplexity = model(image_src)
    recon_error = F.mse_loss(data_recon, image_trg)
    loss = recon_error + vq_loss
    loss = loss / float(len(data['image_src']))
    loss.backward()
    optimizer.step()
    
    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            param.grad.data /= float(args.world_size)

    if ((epoch % 100 == 0) or (epoch==args.epoch_start+args.num_epochs)) and (epoch != args.epoch_start):
        timer_info = timer.lap(iterations=epoch)
        fract = len(image_trg) * (epoch - timer_info['iterations']) / timer_info['span']
        
        # from IPython.display import clear_output
        # clear_output(wait=True)
        print(
            '{iterations:d} iterations, {fract:f}/s'.format(
                iterations=epoch,
                fract=fract
            )
        )
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
        save_model(model, epoch)
        validate()
        
        print_device_memory_info()
        timer.start(iterations=epoch)

print("Done.")


# # Plot Loss
# train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
# train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)
# 
# f = plt.figure(figsize=(16,8))
# ax = f.add_subplot(1,2,1)
# ax.plot(train_res_recon_error_smooth)
# ax.set_yscale('log')
# ax.set_title('Smoothed NMSE.')
# ax.set_xlabel('iteration')
# 
# ax = f.add_subplot(1,2,2)
# ax.plot(train_res_perplexity_smooth)
# ax.set_title('Smoothed Average codebook usage (perplexity).')
# ax.set_xlabel('iteration')


# def show(img):
#     npimg = img.numpy()
#     fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
#     fig.axes.get_xaxis().set_visible(False)
#     fig.axes.get_yaxis().set_visible(False)

# show(make_grid(valid_reconstructions.cpu().data+0.5, range=(0.0, 1.0), scale_each=True))
# show(make_grid(valid_originals.cpu()+0.5))

# # View Embedding
# proj = umap.UMAP(n_neighbors=3,
#                  min_dist=0.1,
#                  metric='cosine').fit_transform(model._vq_vae._embedding.weight.data.cpu())
# 
# plt.scatter(proj[:,0], proj[:,1], alpha=0.3)
# 

# python vq-vae.py --rank 0 --world_size 2 --epoch_start 0 --num_epochs 15000 --init_method 'tcp://192.168.1.154:23456'
# python vq-vae.py --rank 1 --world_size 2 --epoch_start 0 --num_epochs 15000 --init_method 'tcp://192.168.1.154:23456'
