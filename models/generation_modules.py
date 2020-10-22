
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import torch
import numpy as np
from models.blocks import *
import models



def reconstruction_loss(x, x_recon, distribution='gaussian'):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        #x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def dynamic_selective_kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds_mu = -0.5 * (1 - mu.pow(2))
    klds_logvar = -0.5 * (1 + logvar - logvar.exp())

    median_var, i = torch.median(logvar, 1)
    median_var_expanded = median_var.unsqueeze(1).expand(median_var.size(0), klds_logvar.size(1))
    mask = torch.ge(logvar, median_var_expanded)
    after_mask_logvar = klds_logvar*mask
    new_klds = klds_mu + after_mask_logvar.view(-1, 64)
    total_kld = new_klds.sum(1).mean(0, True)

    return total_kld


class Generator_for_vae(nn.Module):
    def __init__(self, params):
        super().__init__()

        #self.fc = nn.Linear(params['nz'], params['nz'])

        self.tconv1 = nn.ConvTranspose2d(params['nz'], params['ngf']*8,
            kernel_size=5, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(params['ngf']*8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(params['ngf']*8, params['ngf']*4,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ngf']*4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(params['ngf']*4, params['ngf']*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ngf']*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(params['ngf']*2, params['ngf'],
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ngf'])

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(params['ngf'], params['nc'],
            4, 2, 1, bias=False)
        #Output Dimension: (nc) x 64 x 64

    def forward(self, x):

        x = F.relu(self.bn1(self.tconv1(x)))
        #print("222: ", x.shape)
        x = F.relu(self.bn2(self.tconv2(x)))
        #print("333: ", x.shape)
        x = F.relu(self.bn3(self.tconv3(x)))
        #print("444: ", x.shape)
        x = F.relu(self.bn4(self.tconv4(x)))
        #print("555: ", x.shape)

        x = F.tanh(self.tconv5(x))

        return x


class Encoder_for_cem_module(nn.Module):
    def __init__(self, z_size):
        super(Encoder_for_cem_module, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        self._enc_mu = torch.nn.Linear(32*4*4, z_size)
        self._enc_log_sigma = torch.nn.Linear(32*4*4, z_size)

    def _sample_latent(self, mu, log_sigma):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        #mu = self._enc_mu(h_enc)
        #log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False).cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(self.batch_norm4(x))

        mu = self._enc_mu(x.view(-1, 32*4*4))
        log_sigma = self._enc_log_sigma(x.view(-1, 32*4*4))

        #z = self._sample_latent(mu, log_sigma)

        return mu

class Encoder_module(nn.Module):
    def __init__(self, z_size):
        super(Encoder_module, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        self._enc_mu = torch.nn.Linear(32*4*4, z_size)
        self._enc_log_sigma = torch.nn.Linear(32*4*4, z_size)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(self.batch_norm4(x))

        mu = self._enc_mu(x.view(-1, 32*4*4))
        log_sigma = self._enc_log_sigma(x.view(-1, 32*4*4))

        return mu, log_sigma



class cem_emb_to_gen(torch.nn.Module):


    def __init__(self, with_noise):
        super(cem_emb_to_gen, self).__init__()

        self.with_noise = with_noise
        if with_noise:
            self._enc_log_sigma = nn.Sequential(nn.Linear(128*3, 128, bias=False),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(128, 128, bias=False),
                                                     nn.ReLU(inplace=True),
                                                     nn.Linear(128, 64, bias=False))


        self._enc_mu = nn.Sequential(nn.Linear(128 * 3, 128, bias=False),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(128, 128, bias=False),
                                                 nn.ReLU(inplace=True),
                                                 nn.Linear(128, 64, bias=False))





        self.high_dim = 64
        self.mid_dim = 128
        self.low_dim = 256
        self.mlp_dim_low = 128
        self.latent_dim = 64



        self.res1_high = ResBlock(self.high_dim, 2 * self.high_dim, stride=2,
                                  downsample=nn.Sequential(conv1x1(self.high_dim, 2 * self.high_dim, stride=2),
                                                           nn.BatchNorm2d(2 * self.high_dim)
                                                           )
                                  )

        self.res2_high = ResBlock(2 * self.high_dim, 128, stride=2,
                                  downsample=nn.Sequential(conv1x1(2 * self.high_dim, 128, stride=2),
                                                           nn.BatchNorm2d(128)
                                                           )
                                  )

        self.res1_mid = ResBlock(self.mid_dim, 2 * self.mid_dim, stride=2,
                                 downsample=nn.Sequential(conv1x1(self.mid_dim, 2 * self.mid_dim, stride=2),
                                                          nn.BatchNorm2d(2 * self.mid_dim)
                                                          )
                                 )

        self.res2_mid = ResBlock(2 * self.mid_dim, 128, stride=2,
                                 downsample=nn.Sequential(conv1x1(2 * self.mid_dim, 128, stride=2),
                                                          nn.BatchNorm2d(128)
                                                          )
                                 )


        self.res1_low = nn.Sequential(conv1x1(self.low_dim, self.mlp_dim_low),
                                           nn.BatchNorm2d(self.mlp_dim_low),
                                           nn.ReLU(inplace=True))
        self.res2_low = ResBlock1x1(self.mlp_dim_low, self.mlp_dim_low)



        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mu = 0
        self.logvar = 0

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps



    def forward(self, x_high, x_mid, x_low):
        N, _, _, _ = x_high.shape
        res1_out_high = self.res1_high(x_high.view(N, self.high_dim, 20, 20))
        res2_in_high = res1_out_high.view(N, 1, 2 * self.high_dim, 10, 10)
        out_high = self.res2_high(res2_in_high.view(N, 2 * self.high_dim, 10, 10))
        final_high = self.avgpool(out_high)
        final_high = final_high.view(-1, 128)

        res1_out_mid = self.res1_mid(x_mid.view(N, self.mid_dim, 5, 5))
        res2_in_mid = res1_out_mid.view(N, 1, 2 * self.mid_dim, 3, 3)
        out_mid = self.res2_mid(res2_in_mid.view(N, 2 * self.mid_dim, 3, 3))
        final_mid = self.avgpool(out_mid)
        final_mid = final_mid.view(-1, 128)

        res1_out_low = self.res1_low(x_low.view(N, self.low_dim, 1, 1))
        out_low = self.res2_low(res1_out_low.view(N, self.mlp_dim_low, 1, 1))
        final_low = self.avgpool(out_low)
        final_low = final_low.view(-1, 128)


        all_final = torch.cat([final_low, final_mid, final_high], dim=1)

        self.mu = self._enc_mu(all_final)
        if self.with_noise:
            self.logvar = self._enc_log_sigma(all_final)
            to_return = self.reparametrize(self.mu, self.logvar)
        else:
            to_return = self.mu



        return to_return.view(-1, self.latent_dim, 1, 1)



class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(params['nc'], params['ndf'],
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(params['ndf'], params['ndf']*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ndf']*2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(params['ndf']*2, params['ndf']*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ndf']*4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(params['ndf']*4, params['ndf']*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ndf']*8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(params['ndf']*8, 1, 4, 1, 0, bias=False)

        self.bn5 = nn.BatchNorm2d(1)

        # added
        self.conv6 = nn.Conv2d(1, 1, 2, 1, 0, bias=False)

        self.dropout = nn.Dropout(0.5)



    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        x = F.leaky_relu(self.conv5(x))
        x = F.sigmoid(self.conv6(x))
        return x.squeeze(1).squeeze(1).squeeze(1)



class VAE(torch.nn.Module):


    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = 64


    def reparametrize(self, mu, logvar):
        self.z_mean = mu
        self.z_sigma = logvar
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps


    def _sample_latent(self, mu, log_sigma):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False).cuda()  # Reparameterization trick

    def no_random(self, state):
        mu, log_sigma = self.encoder(state)
        self.z_mean = mu
        self.z_sigma = log_sigma
        return self.decoder(mu.view(-1, self.latent_dim, 1, 1))

    def forward(self, state):
        mu, sigma = self.encoder(state)
        z = self.reparametrize(mu, sigma)
        return self.decoder(z.view(-1, self.latent_dim, 1, 1))



class generation_cycle_res3(torch.nn.Module):


    def __init__(self, cem, vae, with_discriminator=True, with_dynamic_embeddings_loss_coef=False):
        super(generation_cycle_res3, self).__init__()
        self.latent_dim = 64
        self.high_dim = 64
        self.mid_dim = 128
        self.low_dim = 256
        self.b_for_kld_from_cem = 0.1
        self.cem = cem.cuda()
        self.vae = vae
        self.b_for_vae = 4
        self.optimizer_vae = optim.Adam(self.vae.parameters(), lr=0.0003)
        self.vae_loss_coef = 0.1
        self.generator = (self.vae.decoder).cuda()
        self.encoder = (self.vae.encoder).cuda()
        self.with_noise = True
        self.cem_emb_to_gen = cem_emb_to_gen(self.with_noise).cuda()
        self.with_discriminator = with_discriminator
        self.with_ds_kld = True
        self.with_meta = True

        self.optimizer_cen = optim.Adam(self.cem.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)


        if with_discriminator:
            params_for_discriminator = {
                'nc': 1,  # Number of channles in the training images. For coloured images this is 3.
                'nz': 128,  # Size of the Z latent vector (the input to the generator).
                'ndf': 64,  # Size of features maps in the discriminator. The depth will be multiples of this.
                'lr': 0.0002,  # Learning rate for optimizers
                'beta1': 0.5  # Beta1 hyperparam for Adam optimizer
            }
            self.discriminator = Discriminator(params_for_discriminator)
            self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=params_for_discriminator['lr'],
                                         betas=(params_for_discriminator['beta1'], 0.999), weight_decay=0.05)
            self.discriminator.cuda()
            self.discriminator.train()
        else:
            self.discriminator = None
            self.optimizerD = None

        self.optimizer = optim.Adam(self.cem_emb_to_gen.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)

        self.with_dynamic_embeddings_loss_coef = with_dynamic_embeddings_loss_coef
        if self.with_dynamic_embeddings_loss_coef:
            self.contrastiveLoss_func = models.ContrastiveLoss(with_updating_sum_dist=True)
            self.avg_negative_dist = 0
            self.avg_positive_dist = 0
        else:
            self.contrastiveLoss_func = models.ContrastiveLoss()
        self.contrastiveLoss_coef = 0.0001
        self.disc_coef = 1
        self.criterion_BCE = nn.BCELoss()
        self.cem.train()
        self.cem_emb_to_gen.train()
        self.generator.train()
        self.encoder.eval()
        self.counter = 0
        self.positive_embeddings_loss_coef = 30
        self.classification_loss_coef = 10




    def _triples(self, input_features):
        N,  _, C, H, W = input_features.shape
        choices_features = input_features[:, 8:, :, :, :].unsqueeze(2)  # N, 8, 64, 20, 20 -> N, 8, 1, 64, 20, 20

        row3_pre = input_features[:, 6:8, :, :, :].unsqueeze(1).expand(N, 9, 2, C, H, W)  # N, 2, 64, 20, 20 -> N, 1, 2, 64, 20, 20 -> N, 8, 2, 64, 20, 20
        row3_features = torch.cat((row3_pre, choices_features), dim=2).view(N*9, 3, C, H, W)

        col3_pre = input_features[:, 2:8:3, :, :, :].unsqueeze(1).expand(N, 9, 2, C, H, W)  # N, 2, 64, 20, 20 -> N, 1, 2, 64, 20, 20 -> N, 8, 2, 64, 20, 20
        col3_features = torch.cat((col3_pre, choices_features), dim=2).view(N * 9, 3, C, H, W)
        return row3_features, col3_features

    def _get_relation_vectors(self, x):

        x = x.view(-1, 17, 80, 80)
        N, _, H, W = x.shape

        # Perception Branch
        input_features_high = self.cem.perception_net_high(x.view(-1, 80, 80).unsqueeze(1))
        input_features_mid = self.cem.perception_net_mid(input_features_high)
        input_features_low = self.cem.perception_net_low(input_features_mid)

        # High res
        row3_features_high, col3_features_high = self._triples(input_features_high.view(N, 17, self.high_dim, 20, 20))

        row_feats_high = self.cem.g_function_high(row3_features_high)
        row_feats_high = self.cem.bn_row_high(self.cem.conv_row_high(row_feats_high))
        col_feats_high = self.cem.g_function_high(col3_features_high)
        col_feats_high = self.cem.bn_col_high(self.cem.conv_col_high(col_feats_high))


        _, C, H, W = row_feats_high.shape
        row_feats_high = row_feats_high.view(N, 9, C, H, W)
        _, C, H, W = col_feats_high.shape
        col_feats_high = col_feats_high.view(N, 9, C, H, W)
        reduced_feats_high = row_feats_high + col_feats_high


        row3_features_mid, col3_features_mid = self._triples(input_features_mid.view(N, 17, self.mid_dim, 5, 5))

        row_feats_mid = self.cem.g_function_mid(row3_features_mid)
        row_feats_mid = self.cem.bn_row_mid(self.cem.conv_row_mid(row_feats_mid))
        col_feats_mid = self.cem.g_function_mid(col3_features_mid)
        col_feats_mid = self.cem.bn_col_mid(self.cem.conv_col_mid(col_feats_mid))

        _, C, H, W = row_feats_mid.shape
        row_feats_mid = row_feats_mid.view(N, 9, C, H, W)
        _, C, H, W = col_feats_mid.shape
        col_feats_mid = col_feats_mid.view(N, 9, C, H, W)
        reduced_feats_mid = row_feats_mid + col_feats_mid


        row3_features_low, col3_features_low =  self._triples(input_features_low.view(N, 17, self.low_dim, 1, 1))

        row_feats_low = self.cem.g_function_low(row3_features_low)
        row_feats_low = self.cem.bn_row_low(self.cem.conv_row_low(row_feats_low))
        col_feats_low = self.cem.g_function_low(col3_features_low)
        col_feats_low = self.cem.bn_col_low(self.cem.conv_col_low(col_feats_low))

        _, C, H, W = row_feats_low.shape
        row_feats_low = row_feats_low.view(N, 9, C, H, W)
        _, C, H, W = col_feats_low.shape
        col_feats_low = col_feats_low.view(N, 9, C, H, W)
        reduced_feats_low = row_feats_low + col_feats_low

        return reduced_feats_high, reduced_feats_mid, reduced_feats_low

    def loss_D_target(self, image, target):
        b_size = image.size(0)
        label_target = torch.full((b_size,), 1).type(torch.LongTensor)
        label_target = label_target.cuda()

        # -------loss D target----------
        imgs_just_target = image[torch.arange(image.shape[0]), target].unsqueeze(1)
        d_out = self.discriminator(imgs_just_target)
        d_out = d_out.cuda()
        loss_d_target = self.criterion_BCE(d_out, label_target.type(torch.FloatTensor).cuda())
        loss_d_target = loss_d_target * self.disc_coef
        loss_d_target.backward()
        pred = np.round(d_out.cpu().data).type(torch.LongTensor).cuda()
        correct = pred.eq(label_target.data).cpu().sum().numpy()
        accuracy_discriminator_target = correct * 100. / label_target.size()[0]
        return loss_d_target, accuracy_discriminator_target

    def loss_D_generated(self, generated_images):
        # ------- loss D generated ----------
        b_size = generated_images.size(0)
        label_wrong = torch.full((b_size,), 0).type(torch.LongTensor)
        label_wrong = label_wrong.cuda()

        d_out = self.discriminator(generated_images.detach())  # ********
        d_out = d_out.cuda()
        loss_d_generated = self.criterion_BCE(d_out, label_wrong.type(torch.FloatTensor).cuda())
        loss_d_generated = loss_d_generated * self.disc_coef
        loss_d_generated.backward()
        pred = np.round(d_out.cpu().data).type(torch.LongTensor).cuda()
        correct = pred.eq(label_wrong.data).cpu().sum().numpy()
        accuracy_discriminator_generated = correct * 100. / label_wrong.size()[0]
        return loss_d_generated, accuracy_discriminator_generated

    def loss_from_D(self, generated_images_for_loss_from_d):
        # ------- discriminator loss  on  generated ----------
        b_size = generated_images_for_loss_from_d.size(0)
        label_target = torch.full((b_size,), 1).type(torch.LongTensor)
        label_target = label_target.cuda()

        generated_images_for_loss_from_d.retain_grad()
        d_out = self.discriminator(generated_images_for_loss_from_d)  # ********
        d_out = d_out.cuda()
        loss_g_generated = self.criterion_BCE(d_out, label_target.type(torch.FloatTensor).cuda())
        loss_g_generated = loss_g_generated * self.disc_coef
        return loss_g_generated

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps


    def loss_embed_D_target(self, image, target):

        b_size = image.size(0)
        label_target = torch.full((b_size,), 1).type(torch.LongTensor)
        label_target = label_target.cuda()

        # -------loss D target----------
        imgs_just_target = image[torch.arange(image.shape[0]), target].unsqueeze(1)
        mu, log_sigma = self.encoder(imgs_just_target)
        target_embed = self.reparametrize(mu, log_sigma)
        target_embed = target_embed.view(-1, self.latent_dim)
        d_out = self.discriminator(target_embed)
        d_out = d_out.cuda()
        loss_d_target = self.criterion_BCE(d_out, label_target.type(torch.FloatTensor).cuda())
        loss_d_target = loss_d_target * self.disc_coef
        loss_d_target.backward()
        pred = np.round(d_out.cpu().data).type(torch.LongTensor).cuda()
        correct = pred.eq(label_target.data).cpu().sum().numpy()
        accuracy_discriminator_target = correct * 100. / label_target.size()[0]
        return loss_d_target, accuracy_discriminator_target

    def loss_embed_D_generated(self, generated_embed):
        # ------- loss D generated ----------
        b_size = generated_embed.size(0)
        label_wrong = torch.full((b_size,), 0).type(torch.LongTensor)
        label_wrong = label_wrong.cuda()

        generated_embed = generated_embed.view(-1, self.latent_dim)
        d_out = self.discriminator(generated_embed.detach())  # ********

        d_out = d_out.cuda()
        loss_d_generated = self.criterion_BCE(d_out, label_wrong.type(torch.FloatTensor).cuda())
        loss_d_generated = loss_d_generated * self.disc_coef
        loss_d_generated.backward()
        # self.optimizerD.step()
        pred = np.round(d_out.cpu().data).type(torch.LongTensor).cuda()

        correct = pred.eq(label_wrong.data).cpu().sum().numpy()
        accuracy_discriminator_generated = correct * 100. / label_wrong.size()[0]

        # self.optimizerD.zero_grad()
        return loss_d_generated, accuracy_discriminator_generated

    def loss_from_embed_generated(self, generated_embed):
        # ------- discriminator loss  on  generated ----------
        b_size = generated_embed.size(0)
        label_target = torch.full((b_size,), 1).type(torch.LongTensor)
        label_target = label_target.cuda()

        generated_embed = generated_embed.view(-1, self.latent_dim)
        generated_embed.retain_grad()
        d_out = self.discriminator(generated_embed)  # ********
        d_out = d_out.cuda()
        loss_g_generated = self.criterion_BCE(d_out, label_target.type(torch.FloatTensor).cuda())
        loss_g_generated = loss_g_generated * self.disc_coef
        loss_g_generated.backward(retain_graph=True)
        d_grads = torch.norm(generated_embed.grad)
        return loss_g_generated, d_grads

    def calc_classification_loss(self, output, output_meta, Model_CEM, target, meta_target):
        loss_target = F.cross_entropy(output, target)
        all_loss = None
        if self.with_meta:
            loss_meta_target = Model_CEM.compute_meta_loss(output_meta, meta_target)
            all_loss = loss_target + loss_meta_target*10
        else:
            all_loss = loss_target

        return all_loss


    def get_relation_array_not_target(self, relation_array, target):
        to_return = []

        for b in range(0, target.shape[0]):
            for_each_b = torch.cat([relation_array[b, i + (target[b] <= i) ,:].unsqueeze(0) for i in range(7)], dim=0)
            to_return.append(for_each_b)

        to_return = torch.cat(to_return, dim=0)

        return to_return


    def get_not_target(self, target):
        # print("choice_embeddings.shape[0]: ", target.shape[0])


        not_target = target.clone()
        for i in range(0, not_target.shape[0]):
            if not_target[i] == 0:
                not_target[i] = 1
            else:
                not_target[i] = not_target[i] - 1
        return not_target

    def update_embeddings_loss_coef(self):
        avg_negative_dist = self.contrastiveLoss_func.sum_negative_dist/float(self.counter)
        self.avg_negative_dist = avg_negative_dist
        avg_positive_dist = self.contrastiveLoss_func.sum_positive_dist/float(self.counter)
        self.avg_positive_dist = avg_positive_dist

        middle_dist = (self.avg_negative_dist + self.avg_positive_dist)/2
        self.contrastiveLoss_func = models.ContrastiveLoss(margin=middle_dist, with_updating_sum_dist=True)
        self.counter = 0

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train_cem_generation(self, input, target, meta_target):
        self.counter += 1

        losses_and_acc = {}

        generated_imgs, generated_embed, output, meta_output = self(input)


        if self.with_discriminator:
            self.optimizerD.zero_grad()
            loss_d_target, accuracy_discriminator_target = self.loss_D_target(input, target)
            losses_and_acc["loss_d_target"] = loss_d_target.item()
            losses_and_acc["accuracy_discriminator_target"] = accuracy_discriminator_target
            loss_d_generated, accuracy_discriminator_generated = self.loss_D_generated(generated_imgs)
            losses_and_acc["loss_d_generated"] = loss_d_generated.item()
            losses_and_acc["accuracy_discriminator_generated"] = accuracy_discriminator_generated
            self.optimizerD.step()
            self.optimizerD.zero_grad()

            self.optimizer.zero_grad()
            self.optimizer_G.zero_grad()
            loss_from_D_to_generator = self.loss_from_D(generated_imgs)
            losses_and_acc["loss_g_generated"] = loss_from_D_to_generator.item()

        else:
            self.optimizer.zero_grad()


        # ============ kl_divergence on cem generated ==================
        if self.with_noise:
            if self.with_ds_kld:
                total_kld = dynamic_selective_kl_divergence(self.cem_emb_to_gen.mu,
                                                                        self.cem_emb_to_gen.logvar)
                total_kld = total_kld * self.b_for_kld_from_cem
                losses_and_acc["total_kld_of_cem_generated"] = total_kld.item()
            else:
                total_kld, dimension_wise_kld, mean_kld = kl_divergence(self.cem_emb_to_gen.mu, self.cem_emb_to_gen.logvar)
                total_kld = total_kld*self.b_for_kld_from_cem
                losses_and_acc["total_kld_of_cem_generated"] = total_kld.item()
        else:
            losses_and_acc["total_kld_of_cem_generated"] = 0
        #=====================================================



        input_with_generated = torch.cat((input, generated_imgs.view(-1, 1, 80, 80)), dim=1)

        self.set_requires_grad([self.cem], False)
        reduced_feats_high, reduced_feats_mid, reduced_feats_low = self._get_relation_vectors(input_with_generated)
        self.set_requires_grad([self.cem], True)

        reduced_feats_high_choice, reduced_feats_mid_choice, reduced_feats_low_choice = reduced_feats_high[:, :8, :, :], reduced_feats_mid[:, :8, :, :], reduced_feats_low[:, :8,
                                                                                               :, :]
        reduced_feats_high_generated, reduced_feats_mid_generated, reduced_feats_low_generated = reduced_feats_high[:, 8:9, :, :], reduced_feats_mid[:, 8:9, :, :], reduced_feats_low[
                                                                                                      :, 8:9, :, :]


        label_target = torch.full((reduced_feats_high_choice.shape[0],), 1).type(torch.LongTensor)
        label_target = label_target.cuda()



        reduced_feats_high_choice_just_target = reduced_feats_high_choice[
            torch.arange(reduced_feats_high_choice.shape[0]), target]
        reduced_feats_mid_choice_just_target = reduced_feats_mid_choice[
            torch.arange(reduced_feats_mid_choice.shape[0]), target]
        reduced_feats_low_choice_just_target = reduced_feats_low_choice[
            torch.arange(reduced_feats_low_choice.shape[0]), target]

        reduced_feats_high_generated = reduced_feats_high_generated.squeeze(1).view(-1, 64 * 20 * 20)
        reduced_feats_high_choice_just_target = reduced_feats_high_choice_just_target.view(-1, 64 * 20 * 20)
        reduced_feats_mid_choice_just_target = reduced_feats_mid_choice_just_target.view(-1, 128 * 5 * 5)
        reduced_feats_mid_generated = reduced_feats_mid_generated.squeeze(1).view(-1, 128 * 5 * 5)
        reduced_feats_low_choice_just_target = reduced_feats_low_choice_just_target.view(-1, 256 * 1 * 1)
        reduced_feats_low_generated = reduced_feats_low_generated.squeeze(1).view(-1, 256 * 1* 1)




        embeddings_loss_high = self.contrastiveLoss_func(reduced_feats_high_choice_just_target,
                                                         reduced_feats_high_generated,
                                                         label_target.type(torch.FloatTensor).cuda())
        embeddings_loss_mid = self.contrastiveLoss_func(reduced_feats_mid_choice_just_target,
                                                        reduced_feats_mid_generated,
                                                        label_target.type(torch.FloatTensor).cuda())
        embeddings_loss_low = self.contrastiveLoss_func(reduced_feats_low_choice_just_target,
                                                        reduced_feats_low_generated,
                                                        label_target.type(torch.FloatTensor).cuda())


        embeddings_loss = (embeddings_loss_high + embeddings_loss_mid + embeddings_loss_low) * self.contrastiveLoss_coef * self.positive_embeddings_loss_coef

        losses_and_acc["embeddings_loss"] = embeddings_loss.item()



        # all not target ------------
        reduced_feats_high_choice_just_not_target = self.get_relation_array_not_target(
            reduced_feats_high_choice.view(-1, 8, 64 * 20 * 20), target)
        reduced_feats_mid_choice_just_not_target = self.get_relation_array_not_target(
            reduced_feats_mid_choice.view(-1, 8, 128 * 5 * 5), target)
        reduced_feats_low_choice_just_not_target = self.get_relation_array_not_target(
            reduced_feats_low_choice.view(-1, 8, 256 * 1 * 1), target)
        label_wrong = torch.full((reduced_feats_high_choice.shape[0] * 7,), 0).type(torch.LongTensor)
        label_wrong = label_wrong.cuda()

        new_reduced_feats_high_generated = reduced_feats_high_generated.unsqueeze(1).expand(-1, 7, -1)
        new_reduced_feats_mid_generated = reduced_feats_mid_generated.unsqueeze(1).expand(-1, 7, -1)
        new_reduced_feats_low_generated = reduced_feats_low_generated.unsqueeze(1).expand(-1, 7, -1)

        embeddings_loss_high_not_target = self.contrastiveLoss_func(reduced_feats_high_choice_just_not_target,
                                                                    new_reduced_feats_high_generated.contiguous().view(
                                                                        -1, 64 * 20 * 20),
                                                                    label_wrong.type(torch.FloatTensor).cuda())

        embeddings_loss_mid_not_target = self.contrastiveLoss_func(reduced_feats_mid_choice_just_not_target,
                                                                   new_reduced_feats_mid_generated.contiguous().view(-1,
                                                                                                                     128 * 5 * 5),
                                                                   label_wrong.type(torch.FloatTensor).cuda())

        embeddings_loss_low_not_target = self.contrastiveLoss_func(reduced_feats_low_choice_just_not_target,
                                                                   new_reduced_feats_low_generated.contiguous().view(-1,
                                                                                                                     256 * 1 * 1),
                                                                   label_wrong.type(torch.FloatTensor).cuda())
        embeddings_loss_not_target = (embeddings_loss_high_not_target + embeddings_loss_mid_not_target + embeddings_loss_low_not_target) * self.contrastiveLoss_coef

        classification_loss = self.calc_classification_loss(output, meta_output, self.cem, target, meta_target) * self.classification_loss_coef

        if self.with_noise:
            all_loss = embeddings_loss_not_target + embeddings_loss + total_kld + loss_from_D_to_generator + classification_loss
        else:
            all_loss = embeddings_loss_not_target + embeddings_loss + loss_from_D_to_generator + classification_loss
        all_loss.backward(retain_graph=True)


        losses_and_acc["embeddings_loss_on_negative_choice"] = embeddings_loss_not_target.item()


        if self.with_dynamic_embeddings_loss_coef:
            losses_and_acc["avg_negative_dist"] = self.avg_negative_dist
            losses_and_acc["avg_positive_dist"] = self.avg_positive_dist

        self.optimizer_G.step()
        self.optimizer.step()
        self.optimizer_cen.step()

        return losses_and_acc, generated_imgs

    def train_(self, input, target, meta_target):
        losses_and_acc, generated_imgs = self.train_cem_generation(input, target, meta_target)
        losses_and_acc_vae = self.train_vae(input, target)

        losses_and_acc = losses_and_acc.copy()  # start with x's keys and values
        losses_and_acc.update(losses_and_acc_vae)
        return losses_and_acc, generated_imgs



    def train_vae(self, input, target):
        losses_and_acc_vae = {}
        input = input[0:2, 8:, :, :]
        inputs = input.contiguous().view(-1, 1, 80, 80)
        self.optimizer_vae.zero_grad()
        dec = self._forward_vae(inputs)
        total_kld, dimension_wise_kld, mean_kld = kl_divergence(self.vae.z_mean, self.vae.z_sigma)
        total_kld = total_kld*self.vae_loss_coef
        losses_and_acc_vae["total_kld"] = total_kld.item()

        loss_pic = reconstruction_loss(inputs, dec, distribution='gaussian')*self.vae_loss_coef

        losses_and_acc_vae["loss_pic"] = loss_pic.item()
        all_loss = (loss_pic + total_kld * self.b_for_vae)
        all_loss.backward()
        self.optimizer_vae.step()

        return losses_and_acc_vae

    def _forward_vae(self, imgs):
        dec = self.vae(imgs)
        return dec


    def forward(self, imgs):
        output, meta_output, x_high, x_mid, x_low = self.cem(imgs)
        embed = self.cem_emb_to_gen(x_high, x_mid, x_low)
        generated_imgs = self.generator(embed)
        return generated_imgs, embed, output, meta_output





