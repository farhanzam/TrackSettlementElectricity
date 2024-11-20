import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn.functional import relu, pad

class DoubleConvHelper(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Module that implements 
            - a convolution
            - a batch norm
            - relu
            - another convolution
            - another batch norm
        """
        super().__init__()
        self.kernel_size = 2

        # if no mid_channels are specified, set mid_channels as out_channels
        if mid_channels == None:
            mid_channels = out_channels

        # create a convolution from in_channels to mid_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        
        # create a batch_norm2d of size mid_channels
        self.mid_batch = nn.BatchNorm2d(mid_channels)
        
        # create a relu
        self.relu = nn.ReLU(inplace=True)
        
        # create a convolution from mid_channels to out_channels
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=self.kernel_size, padding=1)
        
        # create a batch_norm2d of size out_channels
        self.out_batch = nn.BatchNorm2d(out_channels)
        
        return


    def forward(self, x):
        """Forward pass through the layers of the helper block"""
        # conv1
        x = self.conv1(x)
        
        # batch_norm1
        x = self.mid_batch(x)
        
        # relu
        x = self.relu(x)
        
        # conv2
        x = self.conv2(x)
        
        # batch_norm2
        x = self.out_batch(x)
        
        # relu
        x = self.relu(x)        

        return x


class Encoder(nn.Module):
    """ Downscale using the maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # create a maxpool2d of kernel_size 2 and padding = 0
        self.kernel_size = 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # create a doubleconvhelper
        self.dconv = DoubleConvHelper(in_channels, out_channels)

        return
        

    def forward(self, x):
        # maxpool2d
        x = self.pool1(x)
        
        # doubleconv
        x = self.dconv(x)
        
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.kernel_size = 2

        # create up convolution using convtranspose2d from in_channels to in_channels//2
        self.upconv1 = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)

        # use a doubleconvhelper from in_channels to out_channels
        self.dconv = DoubleConvHelper(in_channels, out_channels)

        return

    def forward(self, x1, x2):
        # step 1 x1 is passed through the convtranspose2d
        x1 = self.upconv1(x1)

        # step 2 The difference between x1 and x2 is calculated to account for differences in padding
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]

        # step 3 x1 is padded (or not padded) accordingly
        x1 = pad(x1, (diffX // 2, diffX - diffX//2,
                      diffY // 2, diffY - diffY//2))

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # step 4 & 5
        # x2 represents the skip connection
        # Concatenate x1 and x2 together with torch.cat
        x = torch.cat([x1, x2], dim=1)

        # step 6 Pass the concatenated tensor through a doubleconvhelper
        x = self.dconv(x)

        # step 7 Return output 
        return x
    
class OutConv(nn.Module):
    """ OutConv is the replacement of the final layer to ensure
    that the dimensionality of the output matches the correct number of
    classes for the classification task.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # create a convolution with in_channels = in_channels and out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        return

    def forward(self, x):
        # evaluate x with the convolution
        x = self.conv1(x)
        
        return x
    
class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_encoders: int = 2,
                 embedding_size: int = 64, scale_factor: int = 50, **kwargs):
        """
        Implements a unet, a network where the input is downscaled
        down to a lower resolution with a higher amount of channels,
        but the residual images between encoders are saved
        to be concatednated to later stages, creatin the
        nominal "U" shape.

        In order to do this, we will need n_encoders-1 encoders. 
        The first layer will be a doubleconvhelper that
        projects the in_channels image to an embedding_size
        image of the same size.

        After that, n_encoders-1 encoders are used which halve
        the size of the image, but double the amount of channels
        available to them (i.e, the first layer is 
        embedding_size -> 2*embedding size, the second layer is
        2*embedding_size -> 4*embedding_size, etc)

        The decoders then upscale the image and halve the amount of
        embedding layers, i.e., they go from 4*embedding_size->2*embedding_size.

        We then have a maxpool2d that scales down the output to by scale_factor,
        as the input for this architecture must be the same size as the output,
        but our input images are 800x800 and our output images are 16x16.
        """
        super(UNet, self).__init__()

        # save in_channels, out_channels, n_encoders, embedding_size to self
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_encoders = n_encoders
        self.embedding_size = embedding_size
        self.encoders = []

        # create a doubleconvhelper
        self.dconv = DoubleConvHelper(in_channels, embedding_size)

        # for each encoder (there's n_encoders encoders)
        for encoder in range(n_encoders):
        
            # append a new encoder with embedding_size as input and 2*embedding_size as output
            self.encoders.append(Encoder(embedding_size, 2 * embedding_size))
            
            # double the size of embedding_size
            embedding_size = 2 * embedding_size
            
        
        # store it in self.encoders as an nn.ModuleList
        self.encoders = nn.ModuleList(self.encoders)
        
        self.decoders = []
        # for each decoder (there's n_encoders decoders)
        for decoder in range(n_encoders):        
            curr_decoder = None
            # if it's the last decoder
            if decoder == n_encoders-1:
            
                # create a decoder of embedding_size input and out_channels output
                curr_decoder = Decoder(embedding_size, out_channels)

                # create a decoder of embeding_size input and embedding_size//2 output
                # this is some crad code vvv
            else:
                curr_decoder = Decoder(embedding_size, embedding_size//2)

            self.decoders.append(curr_decoder)
                
            # halve the embedding size
            embedding_size = embedding_size//2
            
        
        # save the decoder list as an nn.ModuleList to self.decoders
        self.decoders = nn.ModuleList(self.decoders)

        # create a MaxPool2d of kernel size scale_factor as the final pooling layer
        self.pool_scale = nn.MaxPool2d(kernel_size=(scale_factor, scale_factor))

        return

    def forward(self, x):
        """
            The image is passed through the encoder layers,
            making sure to save the residuals in a list.

            Following this, the residuals are passed to the
            decoder in reverse, excluding the last residual
            (as this is used as the input to the first decoder).

            The ith decoder should have an input of shape
            (batch, some_embedding_size, some_width, some_height)
            as the input image and
            (batch, some_embedding_size//2, 2*some_width, 2*some_height)
            as the residual.
        """
        # evaluate x with self.inc
        x = self.dconv(x)
        
        # create a list of the residuals, with its only element being x
        residuals = []
        residuals.append(x)
        
        # for each encoder
        for e in self.encoders:
            # run the residual through the encoder, append the output to the residual
            x = e(x)
            residuals.append(x)

        # set x to be the last value from the residuals
        x = residuals.pop()
        
        # for each residual except the last one
        for d in self.decoders:
            # evaluate it with the decoder
            x = d(x, residuals.pop())
        
        # evaluate the final pooling layer
        x = self.pool_scale(x)

        # return x
        return x