import pytorch_lightning
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


x, y = torch.load('C:/Users/csross/PycharmProjects/pythonProject/training.pt')
# print(x.shape) #(60000, 28, 28)
# print(x[3]) #will give a tensor value with a [28x28] matrix and its respective RGB inputs (0-255)
# print(y) #will output the following labels (which represent the image inputs via pixels)
x.view(-1, 28**2).shape
# print(x.view(-1, 28**2).shape)

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


train_ds = CTDataset('C:/Users/csross/PycharmProjects/pythonProject/training.pt')
test_ds = CTDataset('C:/Users/csross/PycharmProjects/pythonProject/test.pt')



# #apply the DataLoader for Lightning
# train_dl = DataLoader(train_ds, batch_size=5)

#set loss fcn
# class_loss = nn.CrossEntropyLoss()

class myLightning_NN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(28**2, 100)
        self.Matrix2 = nn.Linear(100, 50)
        self.Matrix3 = nn.Linear(50, 10)
        self.R = nn.ReLU()
        self.learning_rate=0.01
        self.loss = None


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
        self.loss = nn.CrossEntropyLoss()
        return _optimizer

    # def calculate_average_loss(losses):
    #     """
    #     Calculates the average loss per epoch.
    #
    #     Args:
    #         losses (list): A list of losses from each batch.
    #
    #     Returns:
    #         The average loss per epoch.
    #     """
    #
    #     avg_loss = torch.stack(losses).mean()
    #     return avg_loss


    def training_step(self, batch):

        x, y = batch
        output = self.forward(x)
        loss = self.loss(output, y)
        avg_loss = torch.stack([loss]).mean()
        # epochs.append(self.current_epoch)
        accuracy = (output.argmax(dim=1) == y).sum().item() / len(y)
        # logger.experiment.add_histogram("Loss/Train",
        #                              accuracy,
        #                              self.current_epoch)
        logger.experiment.add_scalar("Loss/Train",
                                        loss,
                                        self.current_epoch)

        epoch_dictionary ={'loss': loss, 'accuracy': accuracy}
        return epoch_dictionary



    # def on_train_epoch_end(self):
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     self.logger.log_metrics({'Loss/Train': avg_loss}, step=self.current_epoch)



# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


inputs, labels = train_ds[:]
dataset = TensorDataset(inputs,labels)
train_dl = DataLoader(dataset, batch_size=10)
model=myLightning_NN()
trainer = L.Trainer(max_epochs=20, logger=logger)
trainer.fit(model, train_dataloaders=train_dl)
# def show_dataset(dataset):
#     fig, axs = plt.subplots(3, 3, figsize=(10, 15))
#     fig.tight_layout()
#
#     for i, ax in enumerate(axs.flat):
#         img, label = dataset[i]
#         ax.imshow(img.squeeze(), cmap='gray')
#         ax.set_title(f'Label: {label}')
#         ax.axis('off')
#
#     plt.show()
#
# # Assuming you have initialized the 'myLightning_NN' class and have access to its training data
# train_dataset = model.train_dataloader().dataset
# show_dataset(train_dataset)


# epoch_data, loss_data = [], []

# for epoch in range(10):
#     for i, batch in enumerate(train_dl):
#         x,y=batch
#         loss=model.training_step(batch,i)
#         epoch_data.append(epoch + i / len(train_dl))
#
#         loss_data.append(loss.item())
#         print(f'Epoch {epoch}, Loss value: {loss}')









        # epoch_data, loss_data = train_model(train_dl, model)
        #
        # # NOTE: the shape of epoch_data and loss_data are both tuples of (240000, ) b/c 60000/5 batchs = 12000 *20 epoch = 240k
        #
        # # plot the data
        # plt.plot(epoch_data, loss_data)
        # plt.xlabel('Epoch Number')
        # plt.ylabel('Cross Entropy')
        # plt.show()  # will show a graph and while the learning rate is decrementing the loss data and epoch data can't be seen
        #
        # # so you want to avg the loss across all the data per epoch to get the loss for all 60000 images
        # # this produces a single point which shows a cleaner entropy loss line
        # # done by reshaping and getting 20 epochs for 12000
        # # loss_data.reshape(20,-1).shape # (20, 12000)
        # epoch_data_avg = epoch_data.reshape(20, -1).mean(
        #     axis=1)  # mean(axis=1) -> takes mean of 12000 since its axis =1
        # loss_data_avg = loss_data.reshape(20, -1).mean(axis=1)
        #
        # plt.plot(epoch_data_avg, loss_data_avg, 'o--')
        # plt.xlabel('Epoch Number')
        # plt.ylabel('Cross Entropy')
        # plt.title('Cross Entropy (avg per epoch)')
        # plt.show()



# plt.plot(epoch_data, loss_data)
# plt.xlabel('Epoch Number')
# plt.ylabel('Train Loss')
# plt.show()


# output_values = model(inputs)
# sns.set(style='whitegrid')
# sns.barplot(x=inputs, y=output_values.detach(), color='green')
# plt.ylabel('Cross Entropy')
# plt.xlabel('Epoch Number')
# plt.show()






#     def train_model(dl, model, n_epochs=20):  # goes thru all 60000 items 20 times
#
#
#         # Train model for the training step
#         losses = []  # CrossEntropy of vectors at each iteration
#         epochs = []  # corresponding epochs
#         for epoch in range(n_epochs):
#
#             # N = 12000
#             N = len(dl)
#             # loop thru the Dataloader b/c theres 12000 each with batch size =5
#             # 'i' reps the Index for each datapoint (x,y)
#             for i, batch in enumerate(dl):
#                 x, y = batch
#                 # Update the weights of the network
#                 _optimizer.zero_grad()  # each epoch gets a fresh grad
#                 # apply loss fcn
#                 loss_value = L(model(x), y)
#                 # apply backprop
#                 loss_value.backward()
#                 # apply next Step Size
#                 _optimizer.step()
#
#                 # Store training data
#                 # "epoch + i/N" lets you know how far into the loop you are
#                 # 'i' is iterating and keeping count via 'enumerate(dl)' and N = 12000
#                 epochs.append(epoch + i / N)
#                 # will then store the loss_value in the "losses" array
#                 losses.append(loss_value.item())
#             print(f'Epoch {epoch}, Loss value: {loss_value}')
#         return np.array(epochs), np.array(losses)
#
#  self.optimize.zero_grad()
#         data = T.tensor(data).to(self.device)
#         labels = T.tensor(labels).to(self.device)
#
#         predictions = self.forward(data)
#
#         cost = self.loss(predictions, labels)
#
#         cost.backward()
#         self.optimize.step()
#
#
#
# class BasicLightning_train(L.LightningModule):
#     def __init__(self):
#         super().__init__()
#
#         #1.1 we call out the weights and biases
#         #weights and biases row 1
#         self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
#         self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
#         self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
#
#         # weights and biases row 2
#         self.w10 = nn.Parameter(torch.tensor(8.4), requires_grad=True)
#         self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
#         self.w11 = nn.Parameter(torch.tensor(1.5), requires_grad=True)
#
#         # final bias as rows 1 and 2 converge to the single output
#         self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad=True)
#
#         # we will set an initial Learning Rate but this is just a placeholder
#         ## b/c we are going to do something that finds the ideal learning rate for us
#         self.learning_rate=0.01
#
#     # 1.2 We def the forward() prop fcn
#     def forward(self,input):
#         # we want to connect the input to the activation fcn on row 1
#         input_to_row1_relu = input * self.w00 + self.b00
#         row1_relu_output = F.relu(input_to_row1_relu)  # at this point we are at the activation fcn on row 1
#
#         # now we go from the activation fcn to the weight afterwards in row 1 by scaling
#         scaled_row1_relu_output = row1_relu_output * self.w01
#
#         # we want to connect the input to the activation fcn on row 2
#         input_to_row2_relu = input * self.w10 + self.b10
#         row2_relu_output = F.relu(input_to_row2_relu)  # at this point we are at the activation fcn on row 2
#
#         # now we go from the activation fcn to the weight afterward in row 2 by scaling
#         scaled_row2_relu_output = row2_relu_output * self.w11
#
#         # ### now we add rows 1 and 2 values to the Final Bias
#         inputs_to_final_relu = scaled_row1_relu_output + scaled_row2_relu_output + self.final_bias
#         #
#         # ## use sum as input to the Final ReLU to get the output value
#         output = F.relu(inputs_to_final_relu)
#         return output
#
#     ## in Lightning we can combine different ops including Optimizers and epoch training steps
#
#     def configure_optimizers(self):
#         #note: this learning rate will continue to improve itself in a bit (from the initial "self.learning_rate=0.01")
#         return SGD(self.parameters(), lr=self.learning_rate)
#
#
#     # transformed from epoch training step in Torch_1.py
#     def training_step(self, batch, batch_idx):
#
#         input_i, label_i = batch
#         output_i = self.forward(input_i)
#         loss= (output_i - label_i)**2
#
#         return loss
