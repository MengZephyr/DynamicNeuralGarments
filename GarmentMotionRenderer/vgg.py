import torch.nn as nn
import torchvision.models as models
import torch

vgg_mean = [0.485, 0.456, 0.406]
vgg_std = [0.229, 0.224, 0.225]


class vgg_Normalization(nn.Module):
    def __init__(self, mean, std, device):
        super(vgg_Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).to(device).view(-1, 1, 1)
        self.std = torch.tensor(std).to(device).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# define the VGG
class VGG19(nn.Module):
    def __init__(self, device, ifNorm=False):
        super(VGG19, self).__init__()

        # load the vgg model's features
        if ifNorm:
            self.normalization = vgg_Normalization(vgg_mean, vgg_std, device).to(device)
        self.vgg = models.vgg19(pretrained=True).features
        self.device = device
        self.ifNorm = ifNorm

    def get_content_layer(self, lay=7):
        return self.vgg[:lay]

    def get_content_actList(self, x, layerList):
        y = x
        if self.ifNorm:
            y = self.normalization(x)
        bID = 0
        outF = []
        for Lid in layerList:
            if len(outF) == 0:
                outF.append(self.vgg[:Lid](y))
            else:
                ty = outF[-1]
                outF.append(self.vgg[bID:Lid](ty))
            bID = Lid

        return outF

    def get_style_layers(self):
        return [self.vgg[:4]] + [self.vgg[:7]] + [self.vgg[:12]] + [self.vgg[:21]] + [self.vgg[:30]]

    def get_content_activations(self, x: torch.Tensor, lay=7) \
            -> torch.Tensor:
        """
            Extracts the features for the content loss from the block4_conv2 of VGG19
            Args:
                x: torch.Tensor - input image we want to extract the features of
            Returns:
                features: torch.Tensor - the activation maps of the block2_conv1 layer
        """
        y = x
        if self.ifNorm:
            y = self.normalization(x)
        features = self.vgg[:lay](y)
        return features

    def get_style_activations(self, x):
        """
            Extracts the features for the style loss from the block1_conv1,
                block2_conv1, block3_conv1, block4_conv1, block5_conv1 of VGG19
            Args:
                x: torch.Tensor - input image we want to extract the features of
            Returns:
                features: list - the list of activation maps of the block1_conv1,
                    block2_conv1, block3_conv1, block4_conv1, block5_conv1 layers
        """
        y = x
        if self.ifNorm:
            y = self.normalization(x)
        features = [self.vgg[:4](y)] + [self.vgg[:7](y)] + [self.vgg[:12](y)] + [self.vgg[:21](y)] + [self.vgg[:30](y)]
        return features

    def forward(self, x):
        y = x
        if self.ifNorm:
            y = self.normalization(x)
        return self.vgg(y)
