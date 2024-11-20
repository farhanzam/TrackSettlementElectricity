from torchvision.models.segmentation import fcn_resnet101
import torch
from torch import nn
from torchvision import transforms

class DeepLabV3Transfer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=50, **kwargs):
        """
        Loads the fcn_resnet101 model from torch hub,
        then replaces the first and last layer of the network
        in order to adapt it to our current problem, 
        the first convolution of the fcn_resnet must be changed
        to an input_channels -> 64 Conv2d with (7,7) kernel size,
        (2,2) stride, (3,3) padding and no bias.

        The last layer must be changed to be a 512 -> output_channels
        conv2d layer, with (1,1) kernel size and (1,1) stride. 

        A final pooling layer must then be added to pool each 50x50
        patch down to a 1x1 image, as the original FCN resnet is trained to
        have the segmentation be the same resolution as the input.
        
        Input:
            input_channels: number of input channels of the image of shape (batch, input_channels, width, height)
            output_channels: number of output channels of prediction; prediction is shape (batch, output_channels, width//scale_factor, height//scale_factor)
            scale_factor: number of input pixels that map to 1 output pixel. For example, if the input is 800x800 and the output is 16x6, then the scale factor is 800/16 = 50.
        """
        super().__init__()

        # save in_channels and out_channels to self
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        # use torch.hub to load 'pytorch/vision', 'fcn_resnet101', make sure to use pretrained=True
        # save it to self.model
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        # change self.model.backbone.conv1 to use in_channels as input
        self.model.backbone.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        # change self.model.classifier[-1] to use out_channels as output
        self.model.classifier[4] = nn.Conv2d(256, self.out_channels, kernel_size=(1,1), stride=(1,1))
        # create a final pooling layer that's a maxpool2d, of kernel size scale_factor
        self.final_pooling_layer = nn.MaxPool2d(kernel_size = scale_factor)

        self.preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        
    def forward(self, x):
        """
        Runs predictions on the modified FCN resnet
        followed by pooling

        Input:
            x: image to run a prediction of, of shape
            (batch, self.input_channels, width, height)
            with width and height divisible by
            self.scale_factor
        Output:
            pred_y: predicted labels of size
            (batch, self.output_channels, width//self.scale_factor, height//self.scale_factor)
        """
        x = x.to(torch.float32)
        #x = self.preprocess(x)
        # run x through self.model
        new_x = self.model(x)
        # pool the model_output's "out" value
        pred_y = self.final_pooling_layer(new_x['out'])
        return pred_y