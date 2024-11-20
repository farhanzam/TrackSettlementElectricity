import torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import torchmetrics

from src.models.supervised.segmentation_cnn import SegmentationCNN
from src.models.supervised.unet import UNet
from src.models.supervised.resnet_transfer import FCNResnetTransfer
from src.models.supervised.deeplabv3_transfer import DeepLabV3Transfer

# import wandb

class ESDSegmentation(pl.LightningModule):
    def __init__(self, model_type, in_channels, out_channels, 
                 learning_rate=1e-3, model_params: dict = {}):
        '''
        Constructor for ESDSegmentation class.
        '''
        # call the constructor of the parent class
        super().__init__()
        
        # use self.save_hyperparameters to ensure that the module will load
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        # store in_channels and out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # if the model type is segmentation_cnn, initalize a unet as self.model
        if model_type == 'SegmentationCNN':
            self.model = SegmentationCNN(in_channels, out_channels, **model_params)
        
        # if the model type is unet, initialize a unet as self.model
        elif model_type == 'UNet':
            self.model = UNet(in_channels, out_channels, **model_params)
        
        # if the model type is fcn_resnet_transfer, initialize a fcn_resnet_transfer as self.model
        elif model_type == 'FCNResnetTransfer':
            self.model = FCNResnetTransfer(in_channels, out_channels, **model_params)
        
        elif model_type == 'DeepLabV3Transfer':
            self.model = DeepLabV3Transfer(in_channels, out_channels, **model_params)
        
        # initialize the accuracy metrics for the semantic segmentation task
        self.train_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=out_channels)
        self.val_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=out_channels)
        self.train_iou = torchmetrics.classification.MulticlassJaccardIndex(num_classes=out_channels)
        self.val_iou = torchmetrics.classification.MulticlassJaccardIndex(num_classes=out_channels)
    
    def forward(self, X):
        # evaluate self.model
        X = torch.nan_to_num(X)
        X = X.to(torch.float32)
        return self.model(X)
    
    def training_step(self, batch, batch_idx):
        # get sat_img and mask from batch
        sat_img, mask = batch
        mask = mask.long()
        
        # evaluate batch
        batch_eval = self.forward(sat_img)
        
        # calculate cross entropy loss
        cross_entropy = nn.CrossEntropyLoss()
        loss = cross_entropy(batch_eval, mask)

        # return loss
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        print('error in satellite validation_step')
        # get sat_img and mask from batch
        sat_img, mask = batch
        mask = mask.long()

        # evaluate batch for validation
        val_eval = self.forward(sat_img)

        # get the class with the highest probability
        prediction = torch.argmax(val_eval, dim=1)

        # evaluate each accuracy metric and log it in wandb
        acc = self.val_accuracy(prediction, mask)
        # wandb.log('validation accuracy', self._val_accuracy, prog_bar=True)

        # return validation loss
        cross_entropy = nn.CrossEntropyLoss() 
        return cross_entropy(val_eval, mask)
    
    def configure_optimizers(self):
        # initialize optimizer
        #ADD WEIGHT DECAY TO HELP WITH OVERFITTING
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        # return optimizer
        return optimizer