# IMAGE CLASSIFICATION
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt




# x- images, y - labels
x, y = torch.load('C:/Users/csross/PycharmProjects/pythonProject/training.pt')

### 0.0 Load the Data  ############
# print(x.shape) #(60000, 28, 28)
# print(x[3]) #will give a tensor value with a [28x28] matrix
# print(y) #will output the following labels (which represent the image inputs via pixels)
## output - > ([5,0,4,...,5,6,8]) NOTE: *** THERE IS NO "dtype = "
### b/c the "LABELS" are not values that have ANY MAGNITUDE ...almost like a drawing
### but when you associate with a numpy array now it has a magnitude
# print(y[2].numpy()) #array(4, dtype=int64)

#because this is a tensor for the image we want to see the image
# plt.imshow(x[2000].numpy())
# plt.title(f'Number is {y[2000].numpy()}')
# plt.colorbar()
# plt.show()

########### The ONE-HOT ENCODER ###########################
# 1.1
# y_orig = torch.tensor([2,4,3,0,1,1]) #right now these LABELS have no magnitude
# y_new = F.one_hot(y_orig)
# print(y_new)
# output -> is ...
# tensor([[0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 1],
#         [0, 0, 0, 1, 0],
#         [1, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0]
#         [0, 1, 0, 0, 0]])
### this matrix says each value has its own unit vector in its resp. 1x5 matrix which combine to make a 6x5
### also there are up to <insert hightest value> possible values ..so if 8 were inserted this would be a 6x9
### and within each '1' pos within each 1D vector there is a possible value in its resp. pos.

#### another one hot method is to specify # of classes
#1.2
# this says that within each 1D vector there are 10 possible values so a 1x10 vector will be created with
# unit values of '1' appearing where possible values are located with respect to the 'y' values (which in this case
# are the "inputs")
# y_new =F.one_hot(y, num_classes=10)
# print(y_new[:,5]) #this accesses all 60000 vectors and will print the element at pos. 5 for each (creates a 1x60000)
# print(y_new[1:5,4]) #this accesses vectors 1,2,3,4 and prints the element at pos. 4 for each (creates a 1x4)
# print(y_new[1:5,4].to(float)) #this converts value to float
# print(y_new[1:5+1,3]) #this accesses vectors 1 thru 5 and prints the element at pos. 3 for each (creates a 1x5)

#### now these tensor values that have a 1 will actually become probability float values (ex. 0.989234234)
### these tell us the probability of a value being located at that particular position within the tensor

### now we want to convert these images into a vector length 28^2 for training (or into a single vector)
## there will be 784 FREE parameters in a 28x28 image
x.view(-1, 28**2).shape
# print(x.view(-1, 28**2).shape) #so it has been reshaped to [60000,784]

### now we are going to load the data using Pytorch Dataset Object
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

    def __getitem__(self, ix): # dataset[ix] - indexing syntax that retrieves an item for the index
        return self.x[ix], self.y[ix] #this returns as a tuple
        #this will pull the item from the index in self.x and its resp. label from self.y
        # this will output a tensor in One-hot form which indicates the possible label




train_ds = CTDataset('C:/Users/csross/PycharmProjects/pythonProject/training.pt')
test_ds = CTDataset('C:/Users/csross/PycharmProjects/pythonProject/test.pt')

# print(len(train_ds)) #outputs 60000
# print(train_ds[0])  ### in this case the label is PROBABLY a "5" (via the one-hot encoder below)
## the output shows the 28 x28 pixels and where the frames are getting filled in with "pixel data or COLOR" (via self.x)
# to generate an image that contributes to the output LABEL
## and that LABEL is specified as a "one-hot encoder" (via self.y)
#(tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0118, 0.0706, 0.0706, 0.0706, 0.4941, 0.5333,
        #  0.6863, 0.1020, 0.6510, 1.0000, 0.9686, 0.4980, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1176,
        #  0.1412, 0.3686, 0.6039, 0.6667, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922,
        #  0.8824, 0.6745, 0.9922, 0.9490, 0.7647, 0.2510, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1922, 0.9333,
        #  0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9843,
        #  0.3647, 0.3216, 0.3216, 0.2196, 0.1529, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0706, 0.8588,
        #  0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.7765, 0.7137, 0.9686, 0.9451,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3137,
        #  0.6118, 0.4196, 0.9922, 0.9922, 0.8039, 0.0431, 0.0000, 0.1686, 0.6039,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0549, 0.0039, 0.6039, 0.9922, 0.3529, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.5451, 0.9922, 0.7451, 0.0078, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0431, 0.7451, 0.9922, 0.2745, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.1373, 0.9451, 0.8824, 0.6275, 0.4235, 0.0039,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.3176, 0.9412, 0.9922, 0.9922, 0.4667,
        #  0.0980, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1765, 0.7294, 0.9922, 0.9922,
        #  0.5882, 0.1059, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0627, 0.3647, 0.9882,
        #  0.9922, 0.7333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.9765,
        #  0.9922, 0.9765, 0.2510, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1804, 0.5098, 0.7176, 0.9922,
        #  0.9922, 0.8118, 0.0078, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.1529, 0.5804, 0.8980, 0.9922, 0.9922, 0.9922,
        #  0.9804, 0.7137, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0941, 0.4471, 0.8667, 0.9922, 0.9922, 0.9922, 0.9922, 0.7882,
        #  0.3059, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0902,
        #  0.2588, 0.8353, 0.9922, 0.9922, 0.9922, 0.9922, 0.7765, 0.3176, 0.0078,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0706, 0.6706, 0.8588,
        #  0.9922, 0.9922, 0.9922, 0.9922, 0.7647, 0.3137, 0.0353, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.2157, 0.6745, 0.8863, 0.9922, 0.9922,
        #  0.9922, 0.9922, 0.9569, 0.5216, 0.0431, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.5333, 0.9922, 0.9922, 0.9922, 0.8314,
        #  0.5294, 0.5176, 0.0627, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #  0.0000]]), tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], dtype=torch.float64))


# or if you want to use slicing
xs, ys = train_ds[0:6]
# print(xs.shape) # this shows vector 1 out of 60000 with a 28x28 height and width (3D vector)
# print(ys.shape) #this shows a 1x10 matrix because there are 10 possibilities (via num_classes=10) for the
## first matrix to have a label of 0-9 or 1-10 (i know its NOT 11-20 b/c I pre-processed data above to see an image)

####NOTE: 'xs' and 'ys' CAN CHANGE SHAPE

###2.0 Pytorch DataLoader Object ####
train_dl = DataLoader(train_ds, batch_size=5) #each iteration thru Dataloader will yield a tensor w/1st dim = batch size
##NOTE: the 'batch size' is NOT considered as a dim in any Neural Network
for x, y in train_dl:
    print(x.shape)
    print(y.shape) # (batch size, num_classes) - num_class
    break

### 3.0 PRE-PROCESS DATA and utilize the NN ##
L = nn.CrossEntropyLoss()

# 3.1 Create the NN (note: this is not Lightning)
class myNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(28**2, 100)
        self.Matrix2 = nn.Linear(100, 50)
        self.Matrix3 = nn.Linear(50, 10)
        self.R = nn.ReLU()

    #forward param req an arg to be used when calling out the model
    def forward(self,x):
        x=x.view(-1,28**2) #this takes the [batchsize, 784] images and then feeds into NN - origial input
        x1=self.R(self.Matrix1(x)) #applies the first Linear layer "self.Matrix1" to the input 'x' followed by ReLU act.
        x2=self.R(self.Matrix2(x1))
        x3=self.Matrix3(x2)
        return x3.squeeze() #note: .squeeze() removes only the specified index if a shape is size 1 or -1

#### NN Architecture #####
    # Input(28
    # x28)
    # ↓
    # Linear(784, 100)
    # ↓
    # ReLU
    # ↓
    # Linear(100, 50)
    # ↓
    # ReLU
    # ↓
    # Linear(50, 10)
    # ↓
    # Output(10
    # classes)

#Define the NN with a variable
model = myNeuralNet()

#Look at network predictions (before optimization)
### use earlier example from "slicing
# xs, ys = train_ds[0:6]
print(model(xs)) #will get a lot of values BEFORE they're optimized with a Gradient and applied loss fcn

#NOTE: model(xs) is the same as saying 'ys' since model is the fcn of xs
#########  (or say f(xs) with f() = model) so shapes is same
# print(model(xs).shape) # in this case [6,10]
# print(ys.shape) # [6,10]
# print(xs.shape) # [6,28, 28]


#3.1 Compute the Loss fcn
loss=L(model(xs),ys) #crossEntropy loss function
print(loss)

# in this loss function we want the "model(xs)" predictions to match 'ys' (actual labels)
# so the loss function must be as small as possible

## 4.0 TRAIN THE OBJECT(S)
def train_model(dl,model,n_epochs=5): #goes thru all 60000 items 20 times
    #configure the optimizer
    #the gradient optimizer takes in all parameters from 'model'
    _optimizer = torch.optim.sgd.SGD(model.parameters(), lr=0.01)
    # _optimizer = torch.optim.adam.Adam(model.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()


    #Train model for the training step
    losses=[] #CrossEntropy of vectors at each iteration
    epochs =[] #corresponding epochs
    for epoch in range(n_epochs):

        #N = 12000
        N = len(dl)
        #loop thru the Dataloader b/c theres 12000 each with batch size =5
        # 'i' reps the Index for each datapoint (x,y)
        for i, batch in enumerate(dl):
            x,y=batch
            # Update the weights of the network
            _optimizer.zero_grad() #each epoch gets a fresh grad
            #apply loss fcn
            loss_value = L(model(x),y)
            #apply backprop
            loss_value.backward()
            #apply next Step Size
            _optimizer.step()

            #Store training data
            # "epoch + i/N" lets you know how far into the loop you are
            # 'i' is iterating and keeping count via 'enumerate(dl)' and N = 12000
            epochs.append(epoch+i/N)
            #will then store the loss_value in the "losses" array
            losses.append(loss_value.item())
        print(f'Epoch {epoch}, Loss value: {loss_value}')
    return np.array(epochs), np.array(losses)



#Call out and train the data
epoch_data, loss_data = train_model(train_dl, model)

#NOTE: the shape of epoch_data and loss_data are both tuples of (240000, ) b/c 60000/5 batchs = 12000 *20 epoch = 240k

#plot the data
plt.plot(epoch_data, loss_data)
plt.xlabel('Epoch Number')
plt.ylabel('Cross Entropy')
plt.show() # will show a graph and while the learning rate is decrementing the loss data and epoch data can't be seen

#so you want to avg the loss across all the data per epoch to get the loss for all 60000 images
#this produces a single point which shows a cleaner entropy loss line
#done by reshaping and getting 20 epochs for 12000
# loss_data.reshape(20,-1).shape # (20, 12000)
epoch_data_avg = epoch_data.reshape(20,-1).mean(axis=1) #mean(axis=1) -> takes mean of 12000 since its axis =1
loss_data_avg = loss_data.reshape(20,-1).mean(axis=1)

# plt.plot(epoch_data_avg, loss_data_avg, 'o--')
# plt.xlabel('Epoch Number')
# plt.ylabel('Cross Entropy')
# plt.title('Cross Entropy (avg per epoch)')
# plt.show()

#Observing a Single Image from the Trained dataset
y_pred = train_ds[4][1]
print(y_pred)
x_pred= train_ds[4][0]
print(x_pred)

y_hat=torch.argmax(model(x_pred))
print(f'value is: {y_hat}')

plt.imshow(x_pred)
plt.show()

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
