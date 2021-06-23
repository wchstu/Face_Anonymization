import os
import torch
import torchvision
import numpy as np
import torch.utils.data
from torch import nn
from tqdm import tqdm
from pytorch_msssim import SSIM
from utils.Dataset import FNDataset
from models.generator import Generator
from models.discriminator import Discriminator
from models.arcface import resnet_face18

from models.gan_loss import GANLoss, ContrastiveLoss
from utils.logger import setup_logger

class Face_Annoymization:
    def __init__(self, img_size, batch_size, lr, device, checkpoints_path, val_path, weights_path, is_training=True):
        self.img_size = img_size
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.checkpoints_path = checkpoints_path
        self.val_path = val_path
        self.weights_path = weights_path
        if not os.path.exists(self.weights_path):
            os.mkdir(self.weights_path)
        if not os.path.exists(self.val_path):
            os.mkdir(self.val_path)
        if not os.path.exists(self.checkpoints_path):
            os.mkdir(self.checkpoints_path)

        self.generator = Generator(n_input=101, n_emb=512, img_size=self.img_size, device=device).to(device)
        self.generator.train()
        self.discriminator = Discriminator(input_nc=3, img_size=self.img_size).to(device)
        self.discriminator.train()
        self.face_embedding = resnet_face18(use_se=False)
        self.face_embedding = nn.DataParallel(self.face_embedding)
        self.face_embedding.load_state_dict(torch.load(os.path.join(self.weights_path, 'resnet18_100_arc.pth'), map_location=device))
        self.face_embedding.eval()
        self.face_embedding.to(self.device)
        for param in self.face_embedding.parameters():
            param.requires_grad = False

        self.criterion_gan = GANLoss('lsgan').to(self.device)
        self.criterion_identity = ContrastiveLoss(margin=3).to(self.device)
        self.criterion_ssim = SSIM(data_range=1., size_average=True, channel=3).to(self.device)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.schedule_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=2, gamma=0.1)
        self.schedule_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=2, gamma=0.1)

        self.logger = setup_logger('./')

    def training_G(self, G_iters):
        for param in self.generator.parameters():
            param.requires_grad = True
        for param in self.discriminator.parameters():
            param.requires_grad = False

        G_loss = []
        for i in range(G_iters):
            imgs, iden_imgs, msks, lndms = self.loader.next()
            imgs = imgs.to(self.device)
            iden_imgs = iden_imgs.to(self.device)
            msks = msks.to(self.device)
            lndms = lndms.to(self.device)

            self.optimizer_G.zero_grad()

            with torch.no_grad():
                im_embeddings = self.face_embedding(iden_imgs)

            im_background = imgs * msks

            im_gen = self.generator(im_background, lndms, im_embeddings)
            loss_gan = self.criterion_gan(self.discriminator(im_gen), True)     # 判别器损失

            im_gen_embeddings = self.face_embedding(im_gen)
            label_ones = torch.ones(im_gen.shape[0]).to(self.device)
            loss_identity = self.criterion_identity(im_gen_embeddings, im_embeddings, label_ones)   # 身份一致损失

            loss_ssim = 1 - self.criterion_ssim((im_gen + 1) / 2, (imgs + 1) / 2)

            loss = loss_gan + 0.125 * loss_identity + loss_ssim
            # loss = loss_gan + 0.2 * loss_identity + 10 * loss_ssim

            loss.backward()
            self.optimizer_G.step()
            G_loss.append(loss.item())
        return G_loss

    def training_D(self, D_iters):
        for param in self.generator.parameters():
            param.requires_grad = False
        for param in self.discriminator.parameters():
            param.requires_grad = True

        D_loss = []
        for i in range(D_iters):
            self.optimizer_D.zero_grad()
            imgs, iden_imgs, msks, lndms = self.loader.next()
            imgs = imgs.to(self.device)
            iden_imgs = iden_imgs.to(self.device)
            msks = msks.to(self.device)
            lndms = lndms.to(self.device)

            with torch.no_grad():
                im_embeddings = self.face_embedding(iden_imgs)
                im_background = imgs * msks
                im_gen = self.generator(im_background, lndms, im_embeddings)
            loss_fake = self.criterion_gan(self.discriminator(im_gen), False)
            loss_real = self.criterion_gan(self.discriminator(imgs), True)
            loss = loss_real + loss_fake
            loss.backward()
            self.optimizer_D.step()
            D_loss.append(loss.item())
        return D_loss

    def training(self, loader, epochs, save_epoch, G_iters=2, D_iters=1, G_path=None, D_path=None):
        if G_path is not None:
            self.generator.load_state_dict(torch.load(os.path.join(self.checkpoints_path, G_path), map_location=self.device))
            self.logger.info('Loaded generator checkpoint: {}'.format(G_path))
        if D_path is not None:
            self.discriminator.load_state_dict(torch.load(os.path.join(self.checkpoints_path, D_path), map_location=self.device))
            self.logger.info('Loaded discriminator checkpoint: {}'.format(D_path))

        steps = len(loader) // (G_iters + D_iters)

        self.logger.info('Training begining!')
        self.logger.info('Total epochs: {}, {} steps every epoch.'.format(epochs, steps))

        for epoch in tqdm(range(1, epochs+1)):
            self.loader = iter(loader)
            G_losses = []
            D_losses = []

            for step in tqdm(range(1, steps + 1)):
                D_loss = self.training_D(D_iters)
                G_loss = self.training_G(G_iters)
                G_losses += G_loss
                D_losses += D_loss

                if step % 50 == 0:
                    self.logger.info('Epoch [{}/{}], Step [{}/{}] G_loss: {:.4f}, D_loss: {:.4f}'.format(epoch, epochs, step, steps,
                                                                                        np.sum(G_losses)/len(G_losses),
                                                                                        np.sum(D_losses)/len(D_losses)))
                if step % 200 == 0:
                    self.validate(epoch, step)
                if step % 2000 == 0:
                    self.save_model(epoch)
            if epoch % save_epoch == 0:
                self.save_model(epoch)
            self.validate(epoch)
            self.schedule_G.step()
            self.schedule_D.step()

    def validate(self, epoch, step=None):
        transfer_fun = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                                        std=(0.5, 0.5, 0.5))])
        dataset = FNDataset('./dataset', img_size=img_size, transform=transfer_fun, is_training=False)
        loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=n_workers, batch_size=64,
                                             shuffle=False, drop_last=False)
        imgs, msks, lndms = iter(loader).next()
        imgs = imgs.to(self.device)
        msks = msks.to(self.device)
        lndms = lndms.to(self.device)

        self.generator.eval()
        with torch.no_grad():
            im_embeddings = self.face_embedding(imgs)

        im_background = imgs * msks
        im_gen = self.generator(im_background, lndms, im_embeddings)
        if step is None:
            torchvision.utils.save_image(im_gen / 2 + 0.5, './val/{}.jpg'.format(epoch))
        else:
            torchvision.utils.save_image(im_gen / 2 + 0.5, './val/{}_{}.jpg'.format(epoch, step))
        self.generator.train()

    def save_model(self, epoch):
        torch.save(self.generator.state_dict(), os.path.join(self.checkpoints_path, 'G_{}.pth'.format(epoch)))
        torch.save(self.discriminator.state_dict(), os.path.join(self.checkpoints_path, 'D_{}.pth'.format(epoch)))
        self.logger.info('Model saved at G_{}.pth and D_{}.pth'.format(epoch, epoch))


if __name__ == '__main__':
    img_size = 128
    batch_size = 64
    epochs = 9
    lr = 0.0001
    save_epoch = 1
    n_workers = 8
    checkpoints_path = './checkpoints'
    data_path = './dataset'
    val_path = './val'
    weights_path = './weights'
    is_training = True

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True
    transfer_fun = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    dataset = FNDataset(data_path, img_size=img_size, transform=transfer_fun, is_training=is_training)
    loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=n_workers, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True)

    model = Face_Annoymization(img_size=img_size, batch_size=batch_size, lr=lr, device=device,
                               checkpoints_path=checkpoints_path, val_path=val_path,
                               weights_path=weights_path, is_training=is_training)
    model.training(loader=loader, epochs=epochs, save_epoch=save_epoch, G_iters=3)