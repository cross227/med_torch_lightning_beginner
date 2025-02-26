import pytorch_lightning
import pytorch_lightning.tuner
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import lightning as L
from lightning.pytorch.tuner import Tuner
from pytorch_lightning.tuner.lr_finder import _lr_find
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter

logger = TensorBoardLogger('tb_logs', name='my_model_MNIST')

class CTDataset(Dataset):
    def __init__(self, filepath):
        #using this "self.x, self.y = torch.load(filepath)"
        self.x, self.y = torch.load(filepath)
        #is the same as using this (see below)
        # x, y = torch.load('C:/Users/csross/PycharmProjects/pythonProject/training.pt')

        ##now the pixel data go btw 0-255 and we want it to go btw 0-1
        self.x = self.x / 255.

        ## we use the y data to process the x input data (same as 1.2)
        self.y = F.one_hot(self.y, num_classes=10).to(float) # will convert probabilities to float values

    def __len__(self):
        return self.x.shape[0] #pulls the first value in the shape of [60000, 784] which is the 60000

    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]
        #this will pull the item from the index in self.x and its resp. label from self.y
        # this will output a tensor in One-hot form which indicates the possible label


class myLightning_NN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(28**2, 100)
        self.Matrix2 = nn.Linear(100, 50)
        self.Matrix3 = nn.Linear(50, 10)
        self.R = nn.ReLU()
        self.learning_rate=0.01
        self.loss = nn.CrossEntropyLoss()
        self.training_losses=[]
        # self.encoder=encoder


    #forward param req an arg to be used when calling out the model
    def forward(self,x):
        x=x.view(-1,28**2) #this takes the [batchsize, 784] images and then feeds into NN
        x1=self.R(self.Matrix1(x)) #applies the first Linear layer "self.Matrix1" to the input 'x' followed by ReLU act.
        x2=self.R(self.Matrix2(x1))
        x3=self.Matrix3(x2)
        # return x3.squeeze() #note: .squeeze() removes only the specified index if a shape is size 1 or -1
        return x3
    def configure_optimizers(self):
        _optimizer=SGD(self.parameters(), lr=self.learning_rate)
        return _optimizer


    def training_step(self, batch):

        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        self.training_losses.append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)

        return val_loss

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.encoder(x)
    #     test_loss = F.cross_entropy(y_hat, y)
    #     self.log("test_loss", test_loss)
    #
    # def predict_step(self, batch, batch_idx):
    #     x, y = batch
    #     pred = self.encoder(x)
    #     return pred

train_ds = CTDataset('C:/Users/csross/PycharmProjects/pythonProject/training.pt')
test_ds = CTDataset('C:/Users/csross/PycharmProjects/pythonProject/test.pt')

# def main():
#     model = myLightning_NN()
#     train_dataLoad = DataLoader(train_ds, batch_size=5)
#     trainer = L.Trainer(max_epochs=5, logger=logger)
#     trainer.fit(model, train_dataloaders=train_dataLoad)


# if __name__ == "__main__":
#     main()
model = myLightning_NN()
train_dataLoad = DataLoader(train_ds, batch_size=5)
trainer = L.Trainer(max_epochs=7, logger=logger)

#FIND an Optimal Learning Rate
# tuner=L.pytorch.tuner.Tuner(trainer)
# lr_find_results = tuner.lr_find(model, train_dataloaders=train_dataLoad,min_lr=0.001,max_lr=1.0, early_stop_threshold=None)
# suggested_lr=lr_find_results.suggestion()
# print(f"lr_find() suggests {suggested_lr:.5f} for the learning rate.")

trainer.fit(model, train_dataloaders=train_dataLoad)
losses = model.training_losses
print(len(losses))
epochs=range(1,len(losses)+1)

#Plot all the loss Values on graph
# plt.plot(epochs,losses)
# plt.xlabel("Epoch")
# plt.ylabel("Cross Entropy Loss")
# plt.show()

#NOTE: .reshape ONLY applies to NUMPY ARRAYS so the lists "losses" and "epochs" must be converted
losses =np.array(losses)
epochs=np.array(epochs)

epoch_data_avg = epochs.reshape(20,-1).mean(axis=1) #mean(axis=1) -> takes mean of 12000 since its axis =1
loss_data_avg = losses.reshape(20,-1).mean(axis=1)
#
plt.plot(epoch_data_avg, loss_data_avg, 'o--')
plt.xlabel('Epoch Number')
plt.ylabel('Cross Entropy')
plt.title('Cross Entropy (avg per epoch)')
plt.show()

#obtain the logged loss values and plot on graph
# train_loss_values = trainer.logged_metrics["train_loss"]
# print(train_loss_values.shape)
# epochs = range(1, len(train_loss_values)+1)
#
# plt.plot(epochs, train_loss_values)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Average Loss per Epoch")
# plt.show()

"""Allows users to validate results and the user could go on paint and create a drawing of a number and the tensor
output would correctly validate the number"""
### VALIDATE RESULTS ###
# 1. Input a Single image
# 2. Input multiple images
# 3. Input Images the machine hasn't seen yet for comparison

## 1. Input single image

# Observing a Single Image from the Trained dataset
# y_pred = train_ds[4][1]
# print(y_pred)
# x_pred= train_ds[4][0]
# print(x_pred)
#
# y_hat=torch.argmax(model(x_pred))
# print(y_hat)
#
# plt.imshow(x_pred)
# plt.show()


#observing multiple images
# xs, ys = train_ds[0:2000]
# y_hats=model(xs).argmax(axis=1)
# fig, ax = plt.subplots(10,4,figsize=(10,15))
# for image in range(40):
#     plt.subplot(10,4,image+1)
#     plt.imshow(xs[image])
#     plt.title(f'Predicted Digit: {y_hats[image]}')
#
# fig.tight_layout()
# plt.show()

#Test out a New Image that the machine has not yet seen
# xs, ys = test_ds[:2000]
# y_hats = model(xs).argmax(axis=1)
# fig, ax = plt.subplots(10,4,figsize=(10,15))
# for image in range(40):
#     plt.subplot(10,4,image+1)
#     #NOTE: ".imshow" reqs an 'X' input as its argument
#     plt.imshow(xs[image])
#     plt.title(f'Predicted Digit: {y_hats[image]}')
#
# fig.tight_layout()
# plt.show()
