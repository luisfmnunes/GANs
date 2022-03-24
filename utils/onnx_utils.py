import io
import os
import errno
import numpy as np

import torch
from torch import nn
import torch.onnx as pt2onnx
from .networks import GeneratorNet, DiscriminatorNet

def torch_to_onnx(root_dir, discriminator, generator, epoch, logger=None):
    """
    Exports the discriminator and the generator trained models as onnx models for inference with onnxruntime
    """

    discriminator.eval()
    generator.eval()

    d_input = torch.randn(1,784,requires_grad=True)
    g_input = torch.randn(1,100,requires_grad=True)

    d_name = os.path.join(root_dir, f'D_epoch_{epoch}_onnx.onnx')
    g_name = os.path.join(root_dir, f'G_epoch_{epoch}_onnx.onnx')

    d_out = discriminator(d_input)
    g_out = generator(g_input)

    if logger:
      logger.info(f"Exporting Discriminator to {d_name}")
    # Export the Discriminator
    pt2onnx.export( discriminator, d_input, d_name,
                    export_params=True, # Store the trained parameters weights
                    opset_version=10,  # Opset = ONNX version 
                    do_constant_folding=True, # execute constant folding for optimization
                    input_names= ['input'], # name(s) of the onnx model input(s)
                    output_names = ['output'], # name(s) of the onnx model output(s)
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # Sets the batch size as a dynamic input, allowing different mini batches sizes for inference
                  )

    if logger:
      logger.info(f"Exported Discriminator to {d_name}")
      logger.info(f'Exporting Generator to {g_name}')

    # Export the Generator
    pt2onnx.export( generator, g_input, g_name,
                    export_params=True, # Store the trained parameters weights
                    opset_version=10,  # Opset = ONNX version 
                    do_constant_folding=True, # execute constant folding for optimization
                    input_names= ['input'], # name(s) of the onnx model input(s)
                    output_names = ['output'], # name(s) of the onnx model output(s)
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # Sets the batch size as a dynamic input, allowing different mini batches sizes for inference
                  )

    if logger:
      logger.info(f'Exported Generator to {g_name}')

def load_models(root_dir, epoch, logger=None):
    """
    Loads the discriminator and generator pretrained models from .pt files.
    """
    
    d_file = os.path.join(root_dir, f'D_epoch_{epoch}.pt')
    g_file = os.path.join(root_dir, f'G_epoch_{epoch}.pt')

    if not os.path.exists(d_file):
      if logger:
        logger.error(f"File {d_file} not found")
      raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), d_file)

    if not os.path.exists(g_file):
      if logger:
        logger.error(f'File {g_file} not found')
      raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), g_file)

    discriminator = DiscriminatorNet()
    generator = GeneratorNet()

    if logger:
      logger.info(f'Loading pretrained discriminator {d_file}')
    discriminator.load_state_dict(torch.load(d_file))

    if logger:
      logger.info(f'Loading pretrained generator {g_file}')
    generator.load_state_dict(torch.load(g_file))

    return discriminator, generator

