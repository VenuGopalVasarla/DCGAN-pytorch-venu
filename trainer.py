import torch.optim as optim
from torch import nn
import torch
from config import loss, optimizer, lr, beta, latent_vector, \
    batch_size
from saver import save_checkpoint, save_images


class Trainer:
    """
        Scripts to train defined models.
        Inputs: generator, discriminator.
    """

    def __init__(self, gen, dis, device):
        self.gen_net = gen
        self.dis_net = dis
        self.device = device

        if loss == "BCE":
            self.loss_fn = nn.BCELoss()
        else:
            raise Exception('Invalid loss type. Choose between [BCE]')

        if optimizer == "adam":
            self.optim_D = optim.Adam(dis.parameters(),
                                      lr=lr,
                                      betas=(beta, 0.999))
            self.optim_G = optim.Adam(gen.parameters(),
                                      lr=lr,
                                      betas=(beta, 0.999))

    def train(self, num_iters, data, batch_size=batch_size):
        """
            Trains the loaded data by iterating it.
            Args:
                num_iters: number of iterations.
                data: data loaded using dataloader.
        """
        fixed_noise = torch.randn(batch_size, latent_vector,
                                  1, 1, device=self.device)
        real_label = 1
        fake_label = 0
        for epoch in range(num_iters):
            for i, partial_data in enumerate(data, 0):

                # training the real data.
                self.dis_net.zero_grad()
                real_data = partial_data[0].to(self.device)
                batch_size = real_data.size(0)
                label = torch.full((batch_size,), real_label,
                                   dtype=real_data.dtype,
                                   device=self.device)

                output_tensor = self.dis_net(real_data)
                error_dis_real = self.loss_fn(output_tensor, label)
                error_dis_real.backward()
                D_x = output_tensor.mean().item()

                # training with fake data.
                noise = torch.randn(batch_size,
                                    latent_vector, 1, 1, device=self.device)
                fake = self.gen_net(noise)
                label.fill_(fake_label)
                output_tensor = self.dis_net(fake.detach())
                error_dis_fake = self.loss_fn(output_tensor, label)
                error_dis_fake.backward()
                D_G_z1 = output_tensor.mean().item()

                error_dis = error_dis_fake + error_dis_real
                self.optim_D.step()

                # Updating the generator:
                self.gen_net.zero_grad()
                label.fill_(real_label)
                output_tensor = self.dis_net(fake)
                error_gen = self.loss_fn(output_tensor, label)
                error_gen.backward()
                D_G_z2 = output_tensor.mean().item()
                self.optim_G.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f \
                D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, num_iters, i,
                                                    len(data),
                                                    error_dis.item(),
                                                    error_gen.item(),
                                                    D_x, D_G_z1, D_G_z2))
                if i % 100 == 0:
                    save_images(real_data, 'real', epoch)

                    fake = self.gen_net(fixed_noise)
                    save_images(fake.detach(), 'fake', epoch)

            save_checkpoint(self.gen_net.state_dict(), 'generator',
                            epoch)
            save_checkpoint(self.dis_net.state_dict(), 'discriminator',
                            epoch)
