import os
import warnings
import copy
import random
import warnings
import yaml
import sys

import gc
import torch

import math
from dataclasses import dataclass
from functools import partial
import torch
from torch import nn

import hashlib
import logging
import os
import subprocess
import sys
from os.path import expanduser
from pathlib import Path
from typing import Dict, Any, List


import torch
from PIL import ImageOps
from torchvision.transforms import functional as TF

import torch
from torch.nn import functional as F
from yaml import Loader
from typing import Dict, Union, Optional
from types import SimpleNamespace
from docarray import DocumentArray, Document


import math

import torch
from resize_right import resize
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF


import gc
import random
import clip
import lpips
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from IPython import display
from docarray import DocumentArray, Document
from ipywidgets import Output

from .config import print_args_table
from .helper import parse_prompt, logger
from .nn.losses import spherical_dist_loss, tv_loss, range_loss
from .nn.make_cutouts import MakeCutoutsDango
from .nn.sec_diff import alpha_sigma_to_t





###########################################################################################################################################################################

from . import __resources_path__
with open(f'{__resources_path__}/default.yml') as ymlfile: default_args = yaml.load(ymlfile, Loader=Loader)

from .helper import load_all_models, load_diffusion_model, load_clip_models
from typing import TYPE_CHECKING, overload, List, Optional

if TYPE_CHECKING:
    from docarray import Document

_clip_models_cache = {}

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

__version__ = '0.0.23'
__all__ = ['create']

__resources_path__ = os.path.join(
    os.path.dirname(
        sys.modules.get(__package__).__file__ if __package__ in
        sys.modules else __file__),
    'resources',
)


device = torch.device('cuda:0')
model_config, secondary_model = load_all_models('512x512_diffusion_uncond_finetune_008100',use_secondary_model=True,device=device,)


##################################################################################################################################################################################


@overload
def create(
    text_prompts: Optional[List[str]] = [
        'A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.',
        'yellow color scheme',],
    init_image: Optional[str] = None,
    width_height: Optional[List[int]] = [1280, 768],
    skip_steps: Optional[int] = 10,
    steps: Optional[int] = 250,
    cut_ic_pow: Optional[int] = 1,
    init_scale: Optional[int] = 1000,
    clip_guidance_scale: Optional[int] = 5000,
    tv_scale: Optional[int] = 0,
    range_scale: Optional[int] = 150,
    sat_scale: Optional[int] = 0,
    cutn_batches: Optional[int] = 4,
    diffusion_model: Optional [str] = '512x512_diffusion_uncond_finetune_008100',
    use_secondary_model: Optional[bool] = True,
    diffusion_sampling_mode: Optional[str] = 'ddim',
    perlin_init: Optional[bool] = False,
    perlin_mode: Optional[str] = 'mixed',
    seed: Optional[int] = None,
    eta: Optional[float] = 0.8,
    clamp_grad: Optional[bool] = True,
    clamp_max: Optional[float] = 0.05,
    randomize_class: Optional[bool] = True,
    clip_denoised: Optional[bool] = False,
    fuzzy_prompt: Optional[bool] = False,
    rand_mag: Optional[float] = 0.05,
    cut_overview: Optional[str] = '[12]*400+[4]*600',
    cut_innercut: Optional[str] = '[4]*400+[12]*600',
    cut_icgray_p: Optional[str] = '[0.2]*400+[0]*600',
    display_rate: Optional[int] = 10,
    n_batches: Optional[int] = 4,
    batch_size: Optional[int] = 1,
    batch_name: Optional[str] = '',
    clip_models: Optional[list] = ['ViTB32', 'ViTB16', 'RN50'],)

@overload
def create(init_document: 'Document')

def create(**kwargs)
    from .config import load_config, print_args_table
    from .runner import do_run
    _args = load_config(user_config=kwargs)
    _args = SimpleNamespace(**_args)
    model, diffusion = load_diffusion_model(model_config,_args.diffusion_model,steps=_args.steps,device=device)
    clip_models = load_clip_models(device,enabled=_args.clip_models,clip_models=_clip_models_cache)
    gc.collect()
    torch.cuda.empty_cache()
    try: return do_run(_args, (model, diffusion, clip_models, secondary_model),device)
    except KeyboardInterrupt:
        pass
    finally:
        from IPython import display
        display.clear_output(wait=True)
        _name = _args.name_docarray
        if os.path.exists(f'{_name}.protobuf.lz4'):
            from docarray import DocumentArray
            _da = DocumentArray.load_binary(f'{_name}.protobuf.lz4')
            if _da and _da[0].uri:
                _da.plot_image_sprites(skip_empty=True,show_index=True,keep_aspect_ratio=True)

        print_args_table(vars(_args))


def print_args_table(cfg,console=None,only_non_default: bool = False,console_print: bool = True):
    from rich.table import Table
    from rich import box
    from rich.console import Console
    if console is None:console = Console()
    param_str = Table(title=cfg['name_docarray'],box=box.ROUNDED,highlight=True,title_justify='left',)
    param_str.add_column('Argument', justify='right')
    param_str.add_column('Value', justify='left')

    for k, v in sorted(cfg.items()):
        value = str(v)
        _non_default = False
        if not default_args.get(k, None) == v:
            if not only_non_default:
                k = f'[b]{k}*[/]'
            _non_default = True

        if not only_non_default or _non_default:
            param_str.add_row(k, value)

    if console_print:
        console.print(param_str)
    return param_str


###########################################################################################################################################################################

cache_dir = f'{expanduser("~")}/.cache/{__package__}'

def _get_logger():
    logger = logging.getLogger(__package__)
    _log_level = os.environ.get('DISCOART_LOG_LEVEL', 'INFO')
    logger.setLevel(_log_level)
    ch = logging.StreamHandler()
    ch.setLevel(_log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

logger = _get_logger()

if not os.path.exists(cache_dir):
    logger.info(f'looks like you are running {__package__} for the first time, the first time will take longer time as it will download models. ')
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

logger.debug(f'`.cache` dir is set to: {cache_dir}')

check_model_SHA = False

def _gitclone(url, dest):
    res = subprocess.run(['git', 'clone', '--depth', '1', url, dest],stdout=subprocess.PIPE).stdout.decode('utf-8')
    logger.debug(f'cloned {url} to {dest}: {res}')

def _pip_install(url):
    res = subprocess.run(['pip', 'install', url],stdout=subprocess.PIPE).stdout.decode('utf-8')
    logger.debug(f'pip installed {url}: {res}')

def _clone_repo_install(repo_url, repo_dir):
    if not os.path.exists(repo_dir):_gitclone(repo_url, repo_dir)
    sys.path.append(repo_dir)

def _clone_dependencies():
    try:
        import clip
    except ModuleNotFoundError:
        _pip_install('git+https://github.com/openai/CLIP.git')
    _clone_repo_install('https://github.com/crowsonkb/guided-diffusion',f'{cache_dir}/guided_diffusion')
    _clone_repo_install('https://github.com/assafshocher/ResizeRight',f'{cache_dir}/resize_right')


def _wget(url, outputdir):
    res = subprocess.run(['wget', url, '-q', '-P', f'{outputdir}'],stdout=subprocess.PIPE).stdout.decode('utf-8')
    logger.debug(res)


def load_clip_models(device,enabled: List[str],clip_models: Dict[str, Any] = {}):

    import clip
    # load enabled models
    for k in enabled:
        if k not in clip_models: clip_models[k] = clip.load(k, jit=False)[0].eval().requires_grad_(False).to(device)

    # disable not enabled models to save memory
    for k in clip_models:
        if k not in enabled:
            clip_models.pop(k)

    return list(clip_models.values())

def load_all_models(diffusion_model,use_secondary_model,fallback=False,device=torch.device('cuda:0'),):
    _clone_dependencies()
    model_512_downloaded = False
    model_512_SHA = '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648'
    model_512_link = 'https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt'
    model_512_link_fb = 'https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt'
    model_512_path = f'{cache_dir}/512x512_diffusion_uncond_finetune_008100.pt'

    if fallback:
        model_512_link = model_512_link_fb
    if diffusion_model == '512x512_diffusion_uncond_finetune_008100':
        if os.path.exists(model_512_path) and check_model_SHA:
            logger.debug('Checking 512 Diffusion File')
            with open(model_512_path, "rb") as f:
                bytes = f.read()
                hash = hashlib.sha256(bytes).hexdigest()
            if hash == model_512_SHA:
                logger.debug('512 Model SHA matches')
                if os.path.exists(model_512_path):
                    model_512_downloaded = True
                else:
                    logger.debug('First URL Failed using FallBack')
                    load_all_models(diffusion_model, use_secondary_model, True)
            else:
                logger.debug("512 Model SHA doesn't match, redownloading...")
                _wget(model_512_link, cache_dir)
                if os.path.exists(model_512_path):
                    model_512_downloaded = True
                else:
                    logger.debug('First URL Failed using FallBack')
                    load_all_models(diffusion_model, use_secondary_model, True)
        elif (os.path.exists(model_512_path) and not check_model_SHA
              or model_512_downloaded == True):
            logger.debug('512 Model already downloaded, check check_model_SHA if the file is corrupt')
        else:
            _wget(model_512_link, cache_dir)
            model_512_downloaded = True

    from guided_diffusion.script_util import (model_and_diffusion_defaults, )

    model_config = model_and_diffusion_defaults()

    if diffusion_model == '512x512_diffusion_uncond_finetune_008100':
        model_config.update({
            'attention_resolutions': '32, 16, 8','class_cond': False,
            'diffusion_steps':1000,'rescale_timesteps': True,'timestep_respacing':250,
            'image_size': 512,'learn_sigma': True,'noise_schedule': 'linear',
            'num_channels': 256,'num_head_channels': 64,'num_res_blocks': 2,
            'resblock_updown': True,'use_fp16': device != 'cpu','use_scale_shift_norm': True,})

   #secondary_model = None
   #if use_secondary_model:
   #    from .nn.sec_diff import SecondaryDiffusionImageNet2
   #    secondary_model = SecondaryDiffusionImageNet2()
   #    secondary_model.load_state_dict(
   #        torch.load(f'{cache_dir}/secondary_model_imagenet_2.pth',
   #                   map_location='cpu'))
   #    secondary_model.eval().requires_grad_(False).to(device)
   #
   # return model_config, secondary_model

def load_diffusion_model(model_config, diffusion_model, steps, device):
    from guided_diffusion.script_util import (create_model_and_diffusion, )

    timestep_respacing = f'ddim{steps}'
    diffusion_steps = (1000 // steps) * steps if steps < 1000 else steps
    model_config.update({'timestep_respacing': timestep_respacing,'diffusion_steps': diffusion_steps,})

    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load(f'{cache_dir}/{diffusion_model}.pt', map_location='cpu'))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if 'qkv' in name or 'norm' in name or 'proj' in name:
            param.requires_grad_()
    if model_config['use_fp16']:
        model.convert_to_fp16()

    return model, diffusion

def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])


###########################################################################################################################################################################





def do_run(args, models, device)
    _set_seed(args.seed)
    logger.info('preparing models...')
    model, diffusion, clip_models, secondary_model = models
    normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],std=[0.26862954, 0.26130258, 0.27577711],)
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    side_x = (args.width_height[0] // 64) * 64
    side_y = (args.width_height[1] // 64) * 64
    cut_overview = eval(args.cut_overview)
    cut_innercut = eval(args.cut_innercut)
    cut_icgray_p = eval(args.cut_icgray_p)

    from .nn.perlin_noises import create_perlin_noise, regen_perlin

    skip_steps = args.skip_steps

    loss_values = []

    model_stats = []
    for clip_model in clip_models:
        model_stat = {'clip_model': None,'target_embeds': [],'make_cutouts': None,'weights': [],}
        model_stat['clip_model'] = clip_model
        if isinstance(args.text_prompts, str):
            args.text_prompts = [args.text_prompts]
        for prompt in args.text_prompts:
            txt, weight = parse_prompt(prompt)
            txt = clip_model.encode_text(
                clip.tokenize(prompt).to(device)).float()
            if args.fuzzy_prompt:
                for i in range(25):
                    model_stat['target_embeds'].append(
                        (txt +
                         torch.randn(txt.shape).cuda() * args.rand_mag).clamp(
                             0, 1))
                    model_stat['weights'].append(weight)
            else:
                model_stat['target_embeds'].append(txt)
                model_stat['weights'].append(weight)

        model_stat['target_embeds'] = torch.cat(model_stat['target_embeds'])
        model_stat['weights'] = torch.tensor(model_stat['weights'],
                                             device=device)
        if model_stat['weights'].sum().abs() < 1e-3:
            raise RuntimeError('The weights must not sum to 0.')
        model_stat['weights'] /= model_stat['weights'].sum().abs()
        model_stats.append(model_stat)

    init = None
    if args.init_image:
        d = Document(uri=args.init_image).load_uri_to_image_tensor(
            side_x, side_y)
        init = TF.to_tensor(d.tensor).to(device).unsqueeze(0).mul(2).sub(1)

    if args.perlin_init:
        if args.perlin_mode == 'color':
            init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1,1, False)
            init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4,4, False)
        elif args.perlin_mode == 'gray':
            init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1,1, True)
            init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4,4, True)
        else:
            init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1,1, False)
            init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4,4, True)
        # init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device)
        init = (TF.to_tensor(init).add(
            TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1))
        del init2

    cur_t = None

##################################################################################################################

def interp(t): return 3 * t**2 - 2 * t**3

def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)


def perlin_ms(octaves, width, height, grayscale, device):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)


def create_perlin_noise(octaves, width, height, grayscale, side_y, side_x, device):
    out = perlin_ms(octaves, width, height, grayscale, device)
    if grayscale:
        out = TF.resize(size=(side_y, side_x), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert('RGB')
    else:
        out = out.reshape(-1, 3, out.shape[0] // 3, out.shape[1])
        out = TF.resize(size=(side_y, side_x), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())
    out = ImageOps.autocontrast(out)
    return out


def regen_perlin(perlin_mode, side_y, side_x, device, batch_size):

    if perlin_mode == 'color':
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, False, side_y, side_x, device)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, False, side_y, side_x, device)
    elif perlin_mode == 'gray':
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, True, side_y, side_x, device)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, True, side_y, side_x, device)
    else:
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, False, side_y, side_x, device)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, True, side_y, side_x, device)
    init = (TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1))
    del init2
    return init.expand(batch_size, -1, -1, -1)


##################################################################################################################


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])

def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


##################################################################################################################


skip_augs = False  # @param{type: 'boolean'}

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))

def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose(
            [T.RandomHorizontalFlip(p=0.5),
             T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
             T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
             T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
             T.RandomPerspective(distortion_scale=0.4, p=0.7),
             T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
             T.RandomGrayscale(p=0.15),
             T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),])

    def forward(self, input):
        input = T.Pad(input.shape[2] // 4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn // 4:
                cutout = input.clone()
            else:
                size = int(max_size* torch.zeros(1,).normal_(mean=0.8, std=0.3).clip(float(self.cut_size / max_size), 1.0))
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety: offsety + size, offsetx: offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts


class MakeCutoutsDango(nn.Module):
    def __init__(self, cut_size, Overview=4, InnerCrop=0, IC_Size_Pow=0.5, IC_Grey_P=0.2):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        self.augs = T.Compose(
            [T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(degrees=10,translate=(0.05, 0.05),
                interpolation=T.InterpolationMode.BILINEAR,),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.1),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),])

    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        pad_input = F.pad(input,((sideY - max_size) // 2,(sideY - max_size) // 2,(sideX - max_size) // 2,(sideX - max_size) // 2,),**padargs,)
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview > 0:
            if self.Overview <= 4:
                if self.Overview >= 1:cutouts.append(cutout)
                if self.Overview >= 2:cutouts.append(gray(cutout))
                if self.Overview >= 3:cutouts.append(TF.hflip(cutout))
                if self.Overview == 4:cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

        if self.InnerCrop > 0:
            for i in range(self.InnerCrop):
                size = int(torch.rand([]) ** self.IC_Size_Pow * (max_size - min_size)+ min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety: offsety + size, offsetx: offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
        cutouts = torch.cat(cutouts)
        if skip_augs is not True:
            cutouts = self.augs(cutouts)
        return cutouts


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


padargs = {}

##################################################################################################################


def cond_fn(x, t, y=None):
    with torch.enable_grad():
        x_is_NaN = False
        x = x.detach().requires_grad_()
        n = x.shape[0]
        if secondary_model:
            alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t],device=device,dtype=torch.float32,)
            sigma = torch.tensor(diffusion.sqrt_one_minus_alphas_cumprod[cur_t],device=device,dtype=torch.float32,)
            cosine_t = alpha_sigma_to_t(alpha, sigma)
            out = secondary_model(x, cosine_t[None].repeat([n])).pred
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out * fac + x * (1 - fac)
            x_in_grad = torch.zeros_like(x_in)
        else:
            my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
            out = diffusion.p_mean_variance(model,x,my_t,clip_denoised=False,model_kwargs={'y': y})
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out['pred_xstart'] * fac + x * (1 - fac)
            x_in_grad = torch.zeros_like(x_in)
        for model_stat in model_stats:
            for i in range(args.cutn_batches):
                t_int = (int(t.item()) + 1)  # errors on last step without +1, need to find source
                # when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                try:
                    input_resolution = model_stat['clip_model'].visual.input_resolution
                except:
                    input_resolution = 224
                cuts = MakeCutoutsDango(input_resolution,Overview=cut_overview[1000 - t_int],InnerCrop=cut_innercut[1000 - t_int],IC_Size_Pow=args.cut_ic_pow,IC_Grey_P=cut_icgray_p[1000 - t_int],)
                clip_in = normalize(cuts(x_in.add(1).div(2)))
                image_embeds = (model_stat['clip_model'].encode_image(clip_in).float())
                dists = spherical_dist_loss(image_embeds.unsqueeze(1),model_stat['target_embeds'].unsqueeze(0),)
                dists = dists.view([cut_overview[1000 - t_int] +cut_innercut[1000 - t_int],n,-1,])
                losses = dists.mul(model_stat['weights']).sum(2).mean(0)
                loss_values.append(losses.sum().item())  # log loss, probably shouldn't do per cutn_batch
                x_in_grad += (torch.autograd.grad(losses.sum() * args.clip_guidance_scale, x_in)[0] /args.cutn_batches)
        tv_losses = tv_loss(x_in)
        if secondary_model:
            range_losses = range_loss(out)
        else:
            range_losses = range_loss(out['pred_xstart'])
        sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
        loss = (tv_losses.sum() * args.tv_scale +range_losses.sum() * args.range_scale + sat_losses.sum() * args.sat_scale)
        if init is not None and args.init_scale:
            init_losses = lpips_model(x_in, init)
            loss = loss + init_losses.sum() * args.init_scale
        x_in_grad += torch.autograd.grad(loss, x_in)[0]
        if not torch.isnan(x_in_grad).any():
            grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
        else:
            x_is_NaN = True
            grad = torch.zeros_like(x)
    if args.clamp_grad and not x_is_NaN:
        magnitude = grad.square().mean().sqrt()
        return (grad * magnitude.clamp(max=args.clamp_max) / magnitude)  # min=-0.02, min=-clamp_max,
    return grad

if args.diffusion_sampling_mode == 'ddim':
    sample_fn = diffusion.ddim_sample_loop_progressive
else:
    sample_fn = diffusion.plms_sample_loop_progressive

logger.info('creating artwork...')

image_display = Output()

org_seed = args.seed
for _nb in range(args.n_batches):

    # set seed for each image in the batch
    new_seed = org_seed + _nb
    _set_seed(new_seed)
    args.seed = new_seed

    display.clear_output(wait=True)
    display.display(print_args_table(vars(args),only_non_default=True,console_print=False), image_display)
    gc.collect()
    torch.cuda.empty_cache()

    d = Document(tags=vars(args))
    da_batches.append(d)

    cur_t = diffusion.num_timesteps - skip_steps - 1

    if args.perlin_init:
        init = regen_perlin(args.perlin_mode, args.side_y, side_x, device,args.batch_size)
    if args.diffusion_sampling_mode == 'ddim':
        samples = sample_fn(model,(args.batch_size, 3, side_y, side_x),clip_denoised=args.clip_denoised,model_kwargs={},cond_fn=cond_fn,progress=True,skip_timesteps=skip_steps,init_image=init,randomize_class=args.randomize_class,eta=args.eta,)
    else:
        samples = sample_fn(model,(args.batch_size, 3, side_y, side_x),clip_denoised=args.clip_denoised,model_kwargs={},cond_fn=cond_fn,progress=True,skip_timesteps=skip_steps,init_image=init,randomize_class=args.randomize_class,order=2,)

    threads = []
    for j, sample in enumerate(samples):
        cur_t -= 1
        with image_display:
            if j % args.display_rate == 0 or cur_t == -1:
                for _, image in enumerate(sample['pred_xstart']):
                    image = TF.to_pil_image(
                        image.add(1).div(2).clamp(0, 1))
                    c = Document(tags={'cur_t': cur_t})
                    c.load_pil_image_to_datauri(image)
                    d.chunks.append(c)
                    display.clear_output(wait=True)
                    display.display(image)
                    d.chunks.plot_image_sprites(f'{args.name_docarray}-progress-{_nb}.png',skip_empty=True,show_index=True,keep_aspect_ratio=True,)

                # root doc always update with the latest progress
                d.uri = c.uri
                _start_persist(threads, da_batches, args.name_docarray)

    for t in threads:
        t.join()
display.clear_output(wait=True)
logger.info(f'done! {args.name_docarray}')

return da_batches

def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


    ####################################################################################################################################################################################################################################




