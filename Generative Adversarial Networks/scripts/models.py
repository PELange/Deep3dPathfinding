import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, lr_policy, niter, niter_decay, lr_decay_iters):
    epoch_count = 1

    if lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + epoch_count - niter) / float(niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler

def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02):
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, num_filters, num_dim=3, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             use_ce=False, n_blocks=9, activation='LeakyReLU'):
    norm_layer = get_norm_layer(norm_type=norm)
    if activation == 'ReLU':
        activation = nn.ReLU(True)
    elif activation == 'tanh':
        activation = nn.Tanh()
    else:
        activation = nn.LeakyReLU(True)
    
    net = ResnetGenerator(input_nc, output_nc, activation, num_dim, num_filters, norm_layer=norm_layer,
        use_dropout=use_dropout, n_blocks=n_blocks, use_ce=use_ce, padding_type='zero')
    return init_net(net, init_type, init_gain)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, activation, num_dim, num_filters=21, norm_layer=nn.BatchNorm3d, use_dropout=False,
                 n_blocks=9, use_ce=False, padding_type='zero'):

        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.num_filters = num_filters
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.inc = Inconv(input_nc, num_filters, num_dim, norm_layer, use_bias, activation)
        self.down1 = Down(num_filters, num_filters * 2, num_dim, norm_layer, use_bias, activation)
        self.down2 = Down(num_filters * 2, num_filters * 4, num_dim, norm_layer, use_bias, activation)

        model = []
        for i in range(n_blocks):
            model += [ResBlock(num_filters * 4, num_dim, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)]
        self.resblocks = nn.Sequential(*model)

        self.up1 = Up(num_filters * 4, num_filters * 2, num_dim, norm_layer, use_bias, activation)
        self.up2 = Up(num_filters * 2, num_filters, num_dim, norm_layer, use_bias, activation)

        self.outc = Outconv(num_filters, output_nc, num_dim, activation, use_ce)

    def forward(self, input):
        out = {}

        out['in'] = self.inc(input)
        out['d1'] = self.down1(out['in'])
        out['d2'] = self.down2(out['d1'])
        out['bottle'] = self.resblocks(out['d2'])
        out['u1'] = self.up1(out['bottle'])
        out['u2'] = self.up2(out['u1'])

        return self.outc(out['u2'])


class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch, num_dim, norm_layer, use_bias, activation):
        super(Inconv, self).__init__()
        
        self.inconv = nn.Sequential(
                        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1,
                                    bias=use_bias),
                        norm_layer(out_ch),
                        activation
                    )

    def forward(self, x):
        x = self.inconv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, num_dim, norm_layer, use_bias, activation):
        super(Down, self).__init__()
        
        if num_dim == 3:
            kernel_size = 4
        else:
            kernel_size = (3, 4, 4)
            
        self.down = nn.Sequential(
                        nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size,
                                    stride=1, padding=1, bias=use_bias),
                        norm_layer(out_ch),
                        activation
                    )

    def forward(self, x):
        x = self.down(x)
        return x


# Define a Resnet block
class ResBlock(nn.Module):
    def __init__(self, dim, num_dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()
        self.num_dim = num_dim
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=1, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return nn.ReLU(True)(out)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, num_dim, norm_layer, use_bias, activation):
        super(Up, self).__init__()
        
        if num_dim == 3:
            kernel_size = 4
        else:
            kernel_size = (3, 4, 4)
        
        self.up = nn.Sequential(
                        nn.ConvTranspose3d(in_ch, out_ch,
                                            kernel_size=kernel_size, stride=1,
                                            padding=1, output_padding=0,
                                            bias=use_bias),
                        norm_layer(out_ch),
                        activation
                    )

    def forward(self, x):
        x = self.up(x)
        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch, num_dim, activation, use_ce=False):
        super(Outconv, self).__init__()
            
        if use_ce:
            self.outconv = nn.Sequential(
                                         nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                                        )
        else:
            self.outconv = nn.Sequential(
                                         nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                                         activation
                                        )

    def forward(self, x):
        x = self.outconv(x)
        return x


def define_D(input_nc, ndf, norm='batch', use_sigmoid=False,
             init_type='normal', init_gain=0.02):
    norm_layer = get_norm_layer(norm_type=norm)
    net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)

    return init_net(net, init_type, init_gain)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 3
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                    kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            ]
            sequence += [
                norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                    kernel_size=kw, stride=1, padding=padw, bias=use_bias)]

        sequence += [norm_layer(ndf * nf_mult),
                     nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)