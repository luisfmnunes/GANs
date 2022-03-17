import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch

class Logger:
    """
        A class logger that writes the GAN training summary to display into Tensorboard
    """
    def __init__(self,model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = f"{model_name}_{data_name}"
        self.data_subdir = f"{model_name}/{data_name}"

        # Tensorboard Writer
        self.writer = SummaryWriter(comment = self.comment)

    def log(self, d_error, g_error, epoch, n_batch, num_batches):
        """
        writes data to tensorboard
        """
        d_error = self._autograd_to_numpy(d_error)
        g_error = self._autograd_to_numpy(g_error)

        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(f'{self.comment}/DError', d_error, step)
        self.writer.add_scalar(f'{self.comment}/GError', g_error, step)

    def log_images(self, images, num_images, epoch, n_batch, num_batches, format : str = 'NCHW', normalize=True):
        """
        input images are expected in format NCHW (pytorch default)
        """
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        
        if format=='NCHW':
            images = images.transpose(1,3)

        step = Logger._step(epoch, n_batch, num_batches)
        img_name = f'{self.comment}/images{""}'

        horizontal_grid = vutils.make_grid(images, normalize=normalize, scale_each=True)
        nrows = int(np.sqrt(num_images))
        grid = vutils.make_grid(images, nrow=nrows, normalize=True, scale_each=True)

        self.writer.add_image(img_name, horizontal_grid, step)
        
        self.save_torch_images(horizontal_grid, grid, epoch, n_batch)

    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=True):
        out_dir = f'./data/images/{self.data_subdir}'
        Logger._make_dir(out_dir)

        fig = plt.figure(figsize=(16,16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        if plot_horizontal:
            display.display(plt.gcf())
        self._save_images(fig, epoch, n_batch, 'hori')
        plt.close()

        fig = plt.figure()
        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
        plt.axis('off')
        self._save_images(fig, epoch, n_batch)
        plt.close()
    
    def _save_images(self, fig, epoch, n_batch, comment=''):
        out_dir = f'.data/images/{self.data_subdir}'
        Logger._make_dir(out_dir)
        fig.savefig(f'{out_dir}/{comment}_epoch_{epoch}_batch_{n_batch}.png')

    def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):
        d_error = self._autograd_to_numpy(d_error)
        g_error = self._autograd_to_numpy(g_error)

        d_pred_real = self._autograd_to_numpy(d_pred_real)
        d_pred_fake = self._autograd_to_numpy(d_pred_fake)

        print(f'Epoch: [{epoch}/{num_epochs}], Batch Num: [{n_batch}/{num_batches}]')
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))
       
    def save_models(self, generator, discriminator, epoch):
        out_dir = f'./data/models/{self.data_subdir}'
        Logger._make_dir(out_dir)
        torch.save(generator.state_dict(),
                    f'{out_dir}/G_epoch_{epoch}')
        torch.save(discriminator.state_dict(),
                    f'{out_dir}/D_epoch_{epoch}')

    def close(self):
        self.writer.close()

    @staticmethod
    def _autograd_to_numpy(tensor):
        return tensor.data.cpu().numpy() if isinstance(tensor, torch.autograd.Variable) else tensor

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
