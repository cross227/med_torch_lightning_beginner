## intro to Torch ##

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

## 1.0 create NN (simple form)

class BasicNN(nn.Module):
    def __init__(self):
        super().__init__()
        ######## 1.1 first input weight (w0#) and biases (b0#) on row 1 w/o the ACTIVATION FCNS####
        ##NOTE: using ".Parameter" gives us option to optimize it using Gradient via "requires_grad ="
        ##NOTE2: b/c "requires_grad = False" then the .tensor(1.7) weight is optimal
        self.w00=nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00=nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01=nn.Parameter(torch.tensor(-40.8),requires_grad=False)

        #weights and biases row 2
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        #final bias as rows 1 and 2 converge to the single output
        self.final_bias = nn.Parameter(torch.tensor(-16.),requires_grad=False)

        ### 1.2 now we need to make a Forward pass thru the NN that uses weights and biases we initialized
        ######### in rows 1 and 2

    ## in "forward" we define the Dense layers and their operations
    ## we then specify the Activation function
    ## finally we calculate the output value for each row/Dense layer
    ## we then add these output values to the other rows and add it to one final Bias value
    ## that sum then uses a Final Activation algo before the final output
    ## what we can also do is use a Softmax Algo after the output and specify a 0-1 value reps prob of action happening
    def forward(self, input):
        # we want to connect the input to the activation fcn on row 1
        input_to_row1_relu = input*self.w00+self.b00
        row1_relu_output =F.relu(input_to_row1_relu) #at this point we are at the activation fcn on row 1

        #now we go from the activation fcn to the weight afterwards in row 1 by scaling
        scaled_row1_relu_output = row1_relu_output * self.w01

        # we want to connect the input to the activation fcn on row 2
        input_to_row2_relu = input * self.w10 + self.b10
        row2_relu_output = F.relu(input_to_row2_relu)  # at this point we are at the activation fcn on row 2

        # now we go from the activation fcn to the weight afterward in row 2 by scaling
        scaled_row2_relu_output = row2_relu_output * self.w11

        ### now we add rows 1 and 2 values to the Final Bias
        inputs_to_final_relu = scaled_row1_relu_output + scaled_row2_relu_output +self.final_bias

        ## use sum as input to the Final ReLU to get the output value
        output = F.relu(inputs_to_final_relu)

        return output


#2.0 now we need to create some input values and call out the NN created above
# ##### NOTE: (multiple ways seen below but will go with value not commented out)
input_doses=torch.linspace(start=0, end=1, steps=11, dtype=torch.float64) #torch.Tensor type

# values = np.random.randn(11)
# # print(type(values)) #numpy.ndarray
#
# #to convert to tensor with specific data type
# values = np.random.randn(11)
# tensor_val = torch.tensor(values, dtype=torch.float32)
# print(type(tensor_val))#torch.Tensor type

print(input_doses)

## 2.1 now we run these values through the NN model from above

# model=BasicNN()
# output_values = model(input_doses)
#
# ## 2.2 generate a visual representation
#
# sns.set(style="whitegrid")
# sns.lineplot(x=input_doses, y=output_values, color='green', linewidth=2.5)
# plt.xlabel('Dose')
# plt.ylabel('Effectiveness')
# plt.show()

        ### just want to see the difference as the params are used
        # class LinearClassifier(nn.Module):
        #     def __init__(self, lr, n_classes, input_dims):
        #         super(LinearClassifier, self).__init__()
        #         # 'fc' fully connected layer
        #         self.fc1 = nn.Linear(*input_dims, 128)
        #         self.fc2 = nn.Linear(128, 256)
        #         self.fc3 = nn.Linear(256, n_classes)
        # .parameters comes from nn.Module
        # self.optimize = optim.Adam(self.parameters(), lr=lr)
        # self.loss = nn.CrossEntropyLoss()  # nn.MSELoss
        #
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #
        # self.to(self.device)



## 3.0 now we will train the model without an optimal Final Bias

class BasicNN_train(nn.Module):
    def __init__(self):
        super().__init__()
        ######## 1.1 first input weight (w0#) and biases (b0#) on row 1 w/o the ACTIVATION FCNS####
        ##NOTE: using ".Parameter" gives us option to optimize it using Gradient via "requires_grad ="
        ##NOTE2: b/c "requires_grad = False" then the .tensor(1.7) weight is optimal
        self.w00=nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00=nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01=nn.Parameter(torch.tensor(-40.8),requires_grad=False)

        #weights and biases row 2
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        #final bias is not optimal and requires GD to train to become optimal
        self.final_bias = nn.Parameter(torch.tensor(0.),requires_grad=True)

        ### 1.2 now we need to make a Forward pass thru the NN that uses weights and biases we initialized
        ######### in rows 1 and 2

    def forward(self, input):
        # we want to connect the input to the activation fcn on row 1
        input_to_row1_relu = input*self.w00+self.b00
        row1_relu_output =F.relu(input_to_row1_relu) #at this point we are at the activation fcn on row 1

        #now we go from the activation fcn to the weight afterwards in row 1 by scaling
        scaled_row1_relu_output = row1_relu_output * self.w01

        # we want to connect the input to the activation fcn on row 2
        input_to_row2_relu = input * self.w10 + self.b10
        row2_relu_output = F.relu(input_to_row2_relu)  # at this point we are at the activation fcn on row 2

        # now we go from the activation fcn to the weight afterward in row 2 by scaling
        scaled_row2_relu_output = row2_relu_output * self.w11

        ### now we add rows 1 and 2 values to the Final Bias
        inputs_to_final_relu = scaled_row1_relu_output + scaled_row2_relu_output +self.final_bias

        ## use sum as input to the Final ReLU to get the output value
        output = F.relu(inputs_to_final_relu)

        return output


## 3.1 now we run these values through the NN model from above and train

# model=BasicNN_train()
# output_values_ = model(input_doses)

## 2.2 generate a visual representation

# sns.set(style="whitegrid")
# #b/c "final_bias" has a gradient..detach is called on the output values to create a new tensor that only has the values
# ## b/c seaborn doesn't know what to do with the gradient we strip it off via detach
# sns.lineplot(x=input_doses, y=output_values_.detach(), color='green', linewidth=2.5)
# plt.xlabel('Dose')
# plt.ylabel('Effectiveness')
# plt.show()

### this second model shows the data is trained but it does not match the initial optimal graph we have
#### therefore we need to input data using an optimizer

model=BasicNN_train()
#the "input_doses" arg is done via the "def forward(self, input)"


inputs=torch.tensor([0.,0.5,1.])
labels=torch.tensor([0.,1.,0.])


#the SGD optimizer will optimize every Parameter in our model that has "requires_grad=True"
optimizer = SGD(model.parameters(), lr=0.1)

#this allows us to see values as final bias becomes optimized
print("Final bias, before optimization: " + str(model.final_bias.data) + "\n")

for epoch in range(100):
    #initialize the total loss and then keep track to see how well it fits data
    total_loss =0

    for iteration in range(len(inputs)):
        input_i = inputs[iteration]
        label_i = labels[iteration]

        output_i = model(input_i)

        #SSR (diff btw output value and the know label value
        loss=(output_i - label_i)**2

        ## using backprop
        loss.backward() #d(SR_1)/d(b_finalbias)
        #now this accumulates the derivatives each time we go thru nested loop

        #keeps tabs on how well data fits
        total_loss += float(loss)

    ##if model fits really well then we can stop the training
    if(total_loss <0.0001):
        print("Num steps: " +str(epoch))
        break

    optimizer.step()
    optimizer.zero_grad()

    print("Step: " +str(epoch) + " Final bias: " + str(model.final_bias.data) +"\n")

output_values = model(input_doses)
sns.set(style="whitegrid")
sns.lineplot(x=input_doses, y=output_values.detach(), color='green', linewidth=2.5)
plt.xlabel('Dose')
plt.ylabel('Effectiveness')
plt.show()

