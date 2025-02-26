## we want to use a tool that could help generate the LR for us

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import random
import numpy as np

import lightning as L
from lightning.pytorch.tuner import Tuner
from pytorch_lightning.tuner.lr_finder import _lr_find
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

## 1.0 we will use a Lightning Module instead of an nn.Module
# class BasicLightning(L.LightningModule):
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
#         self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
#         self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
#         self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)
#
#         # final bias as rows 1 and 2 converge to the single output
#         self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)
#
#     # 1.2 We def the forward() prop fcn
#     def forward(self,input, **kwargs):
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
#         ### now we add rows 1 and 2 values to the Final Bias
#         inputs_to_final_relu = scaled_row1_relu_output + scaled_row2_relu_output + self.final_bias
#
#         ## use sum as input to the Final ReLU to get the output value
#         output = F.relu(inputs_to_final_relu)
#
#         return output


input_doses = torch.linspace(start=0, end=1, dtype=torch.float64, steps=11)

# model = BasicLightning()
# output_values = model(input_doses)
#
# sns.set(style="whitegrid")
# sns.lineplot(x=input_doses, y=output_values, color='green', linewidth=2.5)
# plt.xlabel('Dose')
# plt.ylabel('Effectiveness')
# plt.show()


## 3.0 now we will use LIGHTNING to training

class BasicLightning_train(L.LightningModule):
    def __init__(self):
        super().__init__()

        #1.1 we call out the weights and biases
        #weights and biases row 1
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        # weights and biases row 2
        self.w10 = nn.Parameter(torch.tensor(8.4), requires_grad=True)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(1.5), requires_grad=True)

        # final bias as rows 1 and 2 converge to the single output
        self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad=True)

        # we will set an initial Learning Rate but this is just a placeholder
        ## b/c we are going to do something that finds the ideal learning rate for us
        self.learning_rate=0.01

    # 1.2 We def the forward() prop fcn
    def forward(self,input):
        # we want to connect the input to the activation fcn on row 1
        input_to_row1_relu = input * self.w00 + self.b00
        row1_relu_output = F.relu(input_to_row1_relu)  # at this point we are at the activation fcn on row 1

        # now we go from the activation fcn to the weight afterwards in row 1 by scaling
        scaled_row1_relu_output = row1_relu_output * self.w01

        # we want to connect the input to the activation fcn on row 2
        input_to_row2_relu = input * self.w10 + self.b10
        row2_relu_output = F.relu(input_to_row2_relu)  # at this point we are at the activation fcn on row 2

        # now we go from the activation fcn to the weight afterward in row 2 by scaling
        scaled_row2_relu_output = row2_relu_output * self.w11

        # ### now we add rows 1 and 2 values to the Final Bias
        inputs_to_final_relu = scaled_row1_relu_output + scaled_row2_relu_output + self.final_bias
        #
        # ## use sum as input to the Final ReLU to get the output value
        output = F.relu(inputs_to_final_relu)
        return output

    ## in Lightning we can combine different ops including Optimizers and epoch training steps

    def configure_optimizers(self):
        #note: this learning rate will continue to improve itself in a bit (from the initial "self.learning_rate=0.01")
        return SGD(self.parameters(), lr=self.learning_rate)


    # transformed from epoch training step in Torch_1.py
    def training_step(self, batch, batch_idx):

        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss= (output_i - label_i)**2

        return loss

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BasicLightning_train()

# sns.set(style="whitegrid")
# sns.lineplot(x=input_doses, y=output_values.detach(), color='green', linewidth=2.5)
# plt.xlabel('Dose')
# plt.ylabel('Effectiveness')
# plt.show()

inputs=torch.tensor([0.,0.5,1.])
labels=torch.tensor([0.,1.,0.])

# print(type(inputs))
# print(type(labels))

##3.1 when using Lightning wrap the data into a DataLoader
dataset = TensorDataset(inputs,labels)
#DataLoader - make it easy to access data in batches, easy to shuffle data each epoch and ...
## uses a small fraction of data to help with training and then debugging
dataloader = DataLoader(dataset, shuffle=True)

#### what Lightning does is it allows you to combine ALL the operations within the same "def __init__(self): " op

trainer=L.Trainer(max_epochs=1800)
# lr_find_results = L.pytorch.tuner.lr_finder._lr_find()

tuner=L.pytorch.tuner.Tuner(trainer)
# # # #we can use the "trainer.tuner.lr_find()" fcn to create 100 candidate Learning Rates
# # # ## and by setting "early_stop_threshold=None" we can test all of them
lr_find_results = tuner.lr_find(model, train_dataloaders=dataloader,min_lr=0.001,max_lr=1.0, early_stop_threshold=None)


# fig = lr_find_results.plot(suggest=True)
# fig.show()

# # ####################################################################################
# # #can access the improved learning rate by calling on "suggestion()"
suggested_lr=lr_find_results.suggestion()
print(f"lr_find() suggests {suggested_lr:.5f} for the learning rate.")

# NOTE: this "model.learning_rate" supercededs the initial 0.0=001 learning rate set in the original model
model.learning_rate=suggested_lr

#NOTE: in Lightning's ".fit" there is also a .ckpt path that can be inserted
#the trainer will then call the model's "configure_optimizers()" fcn
# then configure SGD using the new lr -> "model.learning_rate=suggested_lr"
#then calculate loss
trainer.fit(model, train_dataloaders=dataloader)
#### the "training_step" in the model calls the following below that we had to code out in Torch_1.py
##NOTE2: trainer calls "optimizer.zero_grad()" - each epoch starts with fresh gradient
##NOTE3: trainer calls "loss.backward()" to calculate new gradient
##NOTE4: trainer calls "optimizer.step()" (SSP) to take a step toward optimal values for params

print(model.final_bias.data)

output_values = model(input_doses)
sns.set(style='whitegrid')
sns.lineplot(x=input_doses, y=output_values.detach(), color='green', linewidth=2.5)
plt.ylabel('Effectiveness')
plt.xlabel('Doses')
plt.show()