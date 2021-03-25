# Method and its Implementation

## Proposed methods of gesture recognition:

### Constructing a convolutional neural network (CNN)

The initial and originally proposed solution is building a convolutional neural networks (CNNs) by myself. 

#### The implementation of CNN: 

First and foremost, it is essential to prepare sufficient datasets for classes 0-9 to train and test the model. As mentioned in the introduction, datasets are the backbone of CNNs. It is necessary to prepare a enough training data to train the model and testing data to test the model. If the model is well-trained, it can output the floor number precisely by feeding an image of gesture. Currently, each class will contain around 1300 images for training and 200 images for testing. 

After the preparation of the datasets, the next step will be building the infrastructure of CNNs by using Pytorch. The documentation of Pytorch is well organized and user-friendly for beginners and its syntax is similar to Python. 
```python
# import libraries
import torch
import torch.nn as nn
import torch.optim as optim  
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```
In the beginning, multiple libraries including Pytorch will be imported (Fig.6). Besides, Numpy will be imported for supporting N-dimensional arrays and offering comprehensive mathematical functions. 

<br>
```python
train_dir = './data/numbers_images/training_set'   # The datas will be prepared by the openpose program later.
test_dir = './data/numbers_images/test_set'


# dataset augmentation + define a transform to convert images to tensor and normalize
transforms = transforms.Compose([
                                transforms.Resize((128, 128)), 
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load and transform the data
trainset = datasets.ImageFolder(train_dir, transform=transforms)
testset = datasets.ImageFolder(test_dir, transform=transforms)

... More detail on the source code ...

# data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,sampler=train_sampler)
validloader = torch.utils.data.DataLoader(trainset, batch_size=64,sampler=valid_sampler)
testloader = torch.utils.data.DataLoader(testset, batch_size=64)
```

Here, the training dataset and the testing dataset will be loaded and transformed by the dataset augmentation. In order to do an early stopping and avoid overfitting, the training dataset will be split around 20% to validation dataset. 
<br>

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()    

        self.conv1 = nn.Conv2d(3, 16, 4, stride=2, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        #linear transformation, y=wx+b, with 2048 inputs and 8 outputs 
        #self.fc = nn.Linear(32*8*8, 8)        
        self.fc = nn.Linear(2048, 10)        

    def forward(self, x):        
        x = torch.relu(self.conv1(x))        
        x = self.pool(x)        
        x = torch.relu(self.conv2(x))        
        x = self.pool(x)        
        x = self.dropout(x)        
        # Fatten the tensor
        # change input shapes to our batch size
        x = x.view(x.shape[0], -1) 
        x = torch.relu(self.fc(x))         
        x = torch.softmax(x,dim=1) 
        return x

model = Net() # create the model
```
Then, the model will be created by multiple layers. Besides the last layer using the SoftMax function, the other layers will use the ReLu function as a non-linear activation function. Also, the dropout technique will be imposed for preventing overfitting. For easy understanding, the Fig.8 shows the graphical representation of the designed CNN.

<center>
<img  src="https://user-images.githubusercontent.com/56287097/112486704-17a2aa00-8db7-11eb-906c-4d7e165449f5.PNG"  /><br>
Fig.8 Graphical Representation of CNN</center>
<br>
<br>
```python
# define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1) 
```
After that, the next crucial step will be setting the loss function and optimizing algorithm. The cross-entropy loss function will be adopted here since it performs well when working on classification tasks. Also, stochastic gradient descent (SGD) optimizer with learning rate 0.1 will be set. Inside the optimizer, L2 regularization which is also known as weight decay will applied. It can reduce overfitting to quite an extent.  


```python
# train the network
epochs = 10
train_losses = np.zeros(epochs)
valid_losses = np.zeros(epochs)

for epoch in range(epochs):  # loop over the dataset multiple times
    model.train()
    train_loss = []     
    for data in trainloader:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward propagation
        outputs = model(inputs)
        
        # loss calculation
        loss = criterion(outputs, labels)
        
        # backward propagation
        loss.backward()
        
        # weight optimization
        optimizer.step()

        train_loss.append(loss.item())    
    model.eval()
```
Finally, the network will be trained with 30 epochs. In each epoch, the network will do the forward calculation to calculate the output. Then, the loss calculation will be done to obtain the difference between the ground-truth output and the calculated output. Based on the difference, the backward propagation can be done to optimize the weight and one epoch finishes. 

When the training finishes, the CNN would be created and ready for making prediction. This is the implementation of the CNN solution. 


#### Results and Performance of CNN: 
<center>
<img  src="https://github.com/TomKwanyingkin/hiuraunt/blob/main/11.png?raw=true"  /><br>
Fig.9 Accuracy of the network</center>
<br>
As shown in the fig.12, the accuracy of my trained CNN is 80% 
<br>
<br>
<center>
<img  src="https://github.com/TomKwanyingkin/hiuraunt/blob/main/Picture11.png?raw=true" width="200"  />

<img  src="https://github.com/TomKwanyingkin/hiuraunt/blob/main/Picture13.png?raw=true" width="221"  />
<br>
Fig.10 Misclassified results   Fig.11 Correctly classified results</center>
<br>
The fig.10 and fig.11 respectively show five misclassified results and correctly classified results. It can be seen that the predicted result is quite different with the true label. For instance, the true label should be 6, but the model predicts it as 0. 
<br>
<br>
<center>
<img  src="https://github.com/TomKwanyingkin/hiuraunt/blob/main/Picture14.png?raw=true" />
<br>
Fig.12 Confusion matrix of the model</center>


#### Obstacles of using CNN: 
The accuracy of my CNN model is 80% which cannot surpass 90% accuracy. The poor precision implies that the model cannot predict and recognize the gesture well. There will be numerous misclassified examples. Also, remember that there are 0 to 9 classes only. If we try to train the model for 0 to 99 classes for recognizing the hand gestures that represent 0th to 99th floor, the result might be worse. The elevator would keep bring the users to the wrong floor. There are two possible reasons that are responsible for this bad performance and high loss. 

First and foremost, the number of the dataset is insufficient. The sum of the number of prepared training data and testing data is around 10000 only. It is inadequate to train the model well and robustly when comparing with other CNN model. In light of this, it is necessary to add much more data. 

The second reason will be no annotation of the datasets. After seeking help to professors and do some research, it is realized that the dataset should be annotated. The data annotation is crucial since it can label the interested objects in the image by providing the x, y position and the size of it. One image should have one annotated document (Fig.13). 

<center>
<img  src="https://github.com/TomKwanyingkin/hiuraunt/blob/main/Picture16.png?raw=true" />
<br>
Fig.13 The document of annotation</center>

However, this reason implies that I need to annotate over one hundred thousand of data. It is very time-consuming and impossible to finish it alone. With this in mind, it is appropriate to find another solution. 

### Transfer learning of MediaPipe

Transfer learning of MediaPipe will be the alternative approach. It offers cross-platform, customizable machine learning solutions for live and streaming media. MediaPipe – Hands which is a Machine Learning (ML) solution utilizes the pipeline that is consisting of two successfully trained models working together.  


The first model is a palm detection model. It operates on the full image and returns an oriented hand bounding box like fig.14. 

<center>
<img  src="https://github.com/TomKwanyingkin/hiuraunt/blob/main/Picture17.png?raw=true" />
<br>
Fig.13 The document of annotation</center>

The second model is a hand landmark model. It operates on the cropped image region defined by the palm detector model and returns 3D hand landmarks.

<center>
<img  src="https://github.com/TomKwanyingkin/hiuraunt/blob/main/Picture18.png?raw=true" />
<br>
Fig.14 The document of annotation</center>

#### The implementation of MediaPipe (Singer user’s hand gesture recognition mode): 

```python
# Construction
mp_hands = mp.solutions.hands

# Initialize MediaPipe Hands.
hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.9)

mp_drawing = mp.solutions.drawing_utils 
```
Initially, in order to utilize the MediaPipe, the construction and initialization needs to be done. The first parameter which is static_image_mode will set to False, so the MediaPipe will treat the input images as a video stream. The max_num_hands will set to two since this mode tries to detect the hands of one singer user only. The min_detection_confidence is the value for the detection to be considered successful. 0.9 will be the most appropriate value based on the current hardware setting. It can ensure all the landmarks precisely landed on the hands of users. Also, it reveals that this proposed approach will be more accurate comparing to building CNN from zero. 

Then, the remaining codes will be running in the while loop, so that the system will keep recognizing until the system shuts down. Inside the while loop, each captured frame will be converted from BGR to RGB by using cv2.cvtColor() function. Then, the rgb frame will be ready for further processing. 

```python
if (count == 4):
    results = hands.process(cv2.flip(rgb, 1))
    count = 0
else:
    continue
```

Since the speed of capturing a frame is much faster than that of processing of a frame, the hands.process() function would process every fourth frame to get the recognized result (Fig. 20). By so doing, the latency can be reduced, and the results would be more updated and catch up the current frame. 

The attribute “results” will store the output from the hands.process() function. It is composed by multi_handedness and multi_hand_landmarks. The multi_handedness is a collection of handedness of the detected hands. It contains a score and a label for each hand (Fig.15). The label will indicate whether the hand is left or right while the score is the estimated probability of the handedness. The multi_hand_landmarks is a collection of detected/tracked hands, where each hand is represented as a list of 21 hand landmarks and each landmark is composed of x, y ,and z (Fig.16).

<center>
<img  src="https://github.com/TomKwanyingkin/hiuraunt/blob/main/Picture19.png?raw=true" width="200"  />

<img  src="https://github.com/TomKwanyingkin/hiuraunt/blob/main/Picture20.png?raw=true" width="150"  />
<br>
<p>Fig.15 multi_handedness    Fig.16 multi_hand_landmarks</p></center>


With a list of twenty-one hand landmarks of each hand, it is easy to know which finger is extending or touching the palm by using multiple if-statements to do the comparison of x and y position of each landmark. 

<center>
<img  src="https://github.com/TomKwanyingkin/hiuraunt/blob/main/Picture21.png?raw=true" />
<br>
Fig.17 The document of annotation</center>

For instance, when a user raises up his/her index finger to represent the first floor. The y-position of landmark[7] would be smaller than that of landmark[8]. As the middle finger is still touching the palm, the y-position of landmark[12] would be larger than that of landmark[9]. By knowing each status of each finger, the hand gesture made by the user can be distinguished and the corresponding floor number will be obtained.

However, this is still not the end of the program. In order to ensure the recognized result is accurate and prevent the users making wrong hand gestures, it is necessary for the user to hold up his/her hand gesture for around 6 seconds. This can be achieved by the following codes.
```python

resultsList.append(int(result))

if (wholeLoop == 25):
    bothHands = False
    tenDresultsList = []

    for result in resultsList: 
        if (result >= 10):
            tenDresultsList.append(result)

    if(bothHands == False):
        finalResult = statistics.mode(resultsList)
    else:
        finalResult = statistics.mode(tenDresultsList)

    print(finalResult, "th floor")
    print("Deletion Complete!!")

    wholeLoop = 0
    resultsList = []
```

A resultList will be created for storing the recognized result for 25 times. If one of the result exist over 9 times, that result will be confirmed and printed out in the terminal to simulate sending signal to the elevator and then one loop finishes. The coding of this mode ends here. 

In the real world, the user in the lift just need to make and hold the hand gesture for 6 seconds in front of the camera in the position shown in fig.18. The above program will recognize the hand gesture and bring the user to that floor.

<center>
<img  src="https://github.com/TomKwanyingkin/hiuraunt/blob/main/Picture22.png?raw=true" />

<img  src="https://github.com/TomKwanyingkin/hiuraunt/blob/main/Picture23.png?raw=true" />

<p>Fig.18 The document of annotation   Fig.19 Real-world application</p></center>

#### The implementation of MediaPipe (Multi-users’ hand gestures recognition mode): 

The basic infrastructure of this mode is similar to the last one. However, the multi-users’ hand gestures recognition would be much more challenging. 

Imagine that there are two or more users are using the elevator. Although the previous mode can deal with this situation by recognizing their hands one-by-one like fig.19, it is quite inconvenient since the MediaPipe model itself and the program cannot identify which two hands belong to the same person when numerous hands exists and cannot control the sequence of detection and recognition.


<center>
<img  src="https://github.com/TomKwanyingkin/hiuraunt/blob/main/Picture24.png?raw=true" />
<br>
Fig.20 The mirror image of Multiple hands of two users</center>
<br>

In other words, it is not capable to gather the recognized results of Tom’s each hand to output 55th floor when there are multiple hands in one frame as shown in fig.27. The recognized result of Tom’s right hand may be combined with that of Ernest’s left hand. As a result, the elevator may go to  65th undesired location. Based on the above, multi-users’ hand gestures recognition mode will be designed for recognizing all the hands at once. To achieve this, various methods has been considered.

The first approach that comes out my mind is k-means clustering. The k-means clustering aims to partition n data into k clusters in which each data belongs to the cluster with the nearest mean.

<center>
<img  src="https://github.com/TomKwanyingkin/hiuraunt/blob/main/Picture25.png?raw=true" />
<br>
Fig.21 K-means clustering</center>
<br>
Inputting the landmark positions of the multiple hands into the k-means clustering algorithm, the output would be like the Fig.29. 

<center>
<img  src="https://github.com/TomKwanyingkin/hiuraunt/blob/main/Picture26.png?raw=true" />
<br>
Fig.22 K-means clustering</center>
<br>
It seems that k-means clustering algorithm is able to tackle the obstacles and help the program to group the results. However, if Tom’s left hand is closer to Mike’s right hand, they will form an  cluster(Fig.23). The cluster would combine the result of Tom’s left hand and Mike’s right hand leading to incorrect result. As a consequence, k-means clustering is not the one we need.

<center>
<img  src="https://github.com/TomKwanyingkin/hiuraunt/blob/main/Picture27.png?raw=true" />
<br>
Fig.23 Expected result of k-means clustering</center>
<br>
The second approach is training an extra CNN model to identify and recognize whose hand this is. If the CNN model is well-trained, it is expected that the output would like the fig.31. 
<center>
<img  src="https://github.com/TomKwanyingkin/hiuraunt/blob/main/Picture28.png?raw=true" />
<br>
Fig.24 Expected output of Fast R-CNN </center>
<br>
It can be seen that the predicted joints will link up user’s body and arms, so that it would help the program to indicate which hands belong to which user. Also, one crucial fact is that the CNN model can solve the obstacle faced by the k-means clustering algorithm (Fig.30). No matter how close the hands of the other users, the recognized result will not be affected if the model is robust. Based on the above, it is guaranteed that the performance and the accuracy will be the best among most of the solutions. Unfortunately, it is expected the model will require an enormous amount of data and annotation. The preparation time would consume lots of time, so this approach will be denied. 
<br>
<br>
The last suggested method will be the quick-sort algorithm. Quick sort is a divide and conquer algorithm. It picks an element as pivot and partitions the given array around the picked pivot. In the multi-users’ hand gestures recognition mode, it will be used to sort multiple hands of users starting from y-coordinate to x-coordinate based on the coordinates of their thumbs. For easy understanding, an example will be given.
<center>
<img  src="https://github.com/TomKwanyingkin/hiuraunt/blob/main/Picture29.png?raw=true" />
<br>
Fig.25 Expected result of k-means clustering</center>
<br>
In the fig.25, there are different coordinates of different landmarks on various users’ hands. Remember that the outputs of MediaPipe model which are multi_handedness list (fig.21) and multi_hand_landmarks list (fig.22)  are separated. That means it is hard to know whether the landmark (x=1, y=2) belongs to left hand or right hand. Thus, a dictionary will be created first in this mode to connect the list of landmarks and its corresponding handedness. 
```python
# Create a dictionary to store two keys for combining the output of MediaPipe  
for i in range(0, len(results.multi_handedness)):
    dictList.append({"handedness":results.multi_handedness[i], "landmarks":results.multi_hand_landmarks[i]})
```
<br>

Then, the dictionary would be sorted by quick-sort function. In the sortXY(), (1, 2), which is the coordinate of the landmark [4], would represent the all the landmarks of Tom’ left hand, (4, 3) would represent the all the landmarks of Tom’ right hand, (9, 9) would represent the all the landmarks of Ernest’s left hand and so on. Those representatives would be sorted based on their x, y coordinates. 
```python
def sortXY(item1, item2):
    if item1["landmarks"].landmark[4].y < item2["landmarks"].landmark[4].y:
        return -1
    elif item1["landmarks"].landmark[4].y > item2["landmarks"].landmark[4].y:
        return 1
    elif item1["landmarks"].landmark[4].y == item2["landmarks"].landmark[4].y and item1["landmarks"].landmark[4].x < item2["landmarks"].landmark[4].x:
        return -1
    elif item1["landmarks"].landmark[4].y == item2["landmarks"].landmark[4].y and item1["landmarks"].landmark[4].x > item2["landmarks"].landmark[4].x:
        return 1
    elif item1["landmarks"].landmark[4].y == item2["landmarks"].landmark[4].y and item1["landmarks"].landmark[4].x == item2["landmarks"].landmark[4].x:
        return 0
```
<br>
After that, a sorted_list would be returned by sortXY. It would be like 

[{“handedness”: left, “landmarks”: all the landmarks of Tom’s left hand},
 {“handedness”: right, “landmarks”: all the landmarks of Tom’s right hand},
 {“handedness”: left, “landmarks”: all the landmarks of Ernest’s left hand},
 {“handedness”: right, “landmarks”: all the landmarks of Ernest’s right hand},]. 

Remember that this approach will not know the landmarks belongs to whom, the order of the list just depends on  x, y coordinate. The above list is just for easy illustration. 

Since the floor number can be obtained by knowing the landmarks of each hand, detectedRlist will be created and like:

[{“handedness”: left, “gesture”: 5},
{“handedness”: right, “gesture”: 5},
{“handedness”: left, “gesture”: 6},
{“handedness”: right, “gesture”: 0},].

The final step would be combining the gesture in the detectedRlist like the following code. Inside the while loop, there are two if-statements. The first if-statement is to check the length of detectedRlist is larger than 1 not. If not, the detectedRlist[0][ “gesture”] will be the result directly. The second if-statement is checking whether the detectedRlist[0] [ “handedness”] equals to “left” and the detectedRlist[1] [ “handedness”] equals to “right”. If yes, they would combine as a result of floor and pop out. Otherwise, the detectedRlist[0][“ gesture”] would be the result alone. 
```python
# printing result to simulate the outputting process
while(len(detectedRlist) > 0):
    if(len(detectedRlist) > 1):
        if(detectedRlist[0]["handedness"] == "Left" and detectedRlist[1]["handedness"] == "Right"):    
            result =  str(detectedRlist[0]["gesture"]) + str(detectedRlist[1]["gesture"])
            resultsList.append(int(result))

            detectedRlist.pop(0)
            detectedRlist.pop(0)
        else:
            result =  str(detectedRlist[0]["gesture"])
            resultsList.append(int(result))

            detectedRlist.pop(0)
    else:
        result =  str(detectedRlist[0]["gesture"])
        resultsList.append(int(result))

        detectedRlist.pop(0)
```
Based on the above, the results of floors in the example will be 55th floor and the 60th floor that matches to our observation. 

To ensure the quick-sort approach is workable, an extra example (fig.35) will be tested and goes through briefly. 

<center>
<img  src="https://github.com/TomKwanyingkin/hiuraunt/blob/main/Picture31.png?raw=true" />
<br>
Fig.26 Expected result of k-means clustering</center>

In the beginning, the MediaPipe model will offer the coordinates of the landmarks after processing the input image. Then, a dictionary will store the result of multi_handedness and multi_hand_landmarks. After the sorting, a sorted_list would be produced as follows:

                [{“handedness”: left, “landmarks”: all the landmarks of that hand},
                 {“handedness”: right, “landmarks”: all the landmarks of that hand},
                 {“handedness”: left, “landmarks”: all the landmarks of that hand}]

Next, floor number represented by the gesture would be obtained based on the landmarks’ position. 

                            [{“handedness”: left, “gesture”: 3},
                             {“handedness”: right, “gesture”: 2},
                             {“handedness”: left, “gesture”: 5},]

Finally, the gesture in the detectedRlist would be combining by the coding in Fig.34. Since the initial length of the list is three, it would enter the second if-statement. The detectedRlist[0][ “gesture”] would combine detectedRlist[1][ “gesture”] to produce a floor number which is 32nd floor as the detectedRlist[0] [ “handedness”] equals to “left” and the detectedRlist[1] [ “handedness”] equals to “right”. The list will be shortened to length 1 after the popping out. The last remaining element would be result which is 5th floor directly. In the end, the output of the floor number would be right.  

In conclusion, it is proven that the implementation of quick-sort method can deal with most cases. No matter how close the hands of different users, the result will not be interrupted. Therefore, it will be adopted in the multi-users’ hand gestures recognition mode for gathering the output from MediaPipe to produce correct floor numbers. 

#### The implementation of MediaPipe (GUI version): 

To consummate the application, it is necessary to design the GUI for the gesture recognition. To make the GUI operatable, the whole programming structure of the previous two modes needs to be restructured since they are not well-organized, and the processes are not running concurrently but sequentially. During the reconstruction, coroutines and asyncio queue are the key components of the concurrency. However, what are they?

- Coroutines can be entered, exited, and resumed at many different points. They can be implemented with the async def statement. 
<br>

- Asyncio queue, which is a FIFO queue, is designed to be used in async/await code with different useful functions. The put(item) function allows the asynchronous functions to put an item into the queue. If the queue is full, the function will wait until a free slot is available before putting a new item into it. Also, the get() function permit the asynchronous functions to remove and return an item from the queue. If the queue is empty, that function will wait until an item is available in the list. In short, the asyncio queue provide a channel for the asynchronous functions to exchange the items. 


With the basic understanding of the coroutine and queue, it is time to discuss the design blue print of the program with GUI. The idea is basically dividing the program into five asynchronous functions which are main(), capture(), keepget(), processing() and prediction().

<br>

```python
async def main():
    ... (more details on the source codes)

    capimg = asyncio.Queue()
    samples = asyncio.Queue()
    processed = asyncio.Queue()
    rgbImg = asyncio.Queue()

    await asyncio.gather(
        capture(capimg, root, photocanvas),
        keepget(capimg, samples),
        processing(samples, processed),
        prediction(processed, rgbImg, root, textcanvas, textArea),
    )
```
In the main() function, there are four created asyncio queues used for transferring items. They will be parameters passed to the other asynchronous functions. Besides, the asyncio.gather() will run capture(), keepget(), processing() and prediction() concurrently inside the main(). To conclude, main() acts as a kick starter of the program.

<br>
```python
async def capture(q1, win, canvas):
    cap = cv2.VideoCapture(0)

    while True:
        # Frame capturing  
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # GUI design
        photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(rgb))
        canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
        win.update()

        # Putting the rgb into the queue and waiting for the keepget() function to get it. 
        await q1.put(rgb)
        await asyncio.sleep(0.00001)
```
The main duty of capture() is keep capturing and transferring the converted rgb to keepget() function through the queue q1. Meanwhile, the rgb will be projected in GUI for the users to see whether their hands are inside the camera. 
<br>

```python
async def keepget(q1, q2):
    while True:
        rgb = await q1.get()
        # print("keepget",len(rgb))
        await q2.put(rgb)
        await asyncio.sleep(0.00001)
```
The asynchronous function keepget() will non-stop get the rgb from the shared queue q1 
whenever the capture() puts the rgb into the q1. Then, it will pass the rgb to the processing(). In short, keepget() is like a transfer station. 
<br>

```python
async def processing(q2, q3):

    mp_drawing = mp.solutions.drawing_utils 

    ...
    
    while True:

        ...

        rgb = await q2.get()

        results = hands.process(cv2.flip(rgb,1))

        unsortedList.append({'image': rgb,'hands':[{'classification':{'label':h.classification[0].label, 'index':h.classification[0].index}, 'floor':getGesture(h.classification[0].label, l.landmark), 'landmark':l.landmark[4]} for h, l in zip(results.multi_handedness, results.multi_hand_landmarks)]})

        await q3.put(unsortedList)

        await asyncio.sleep(0.00001)
```
In the processing() function, it will immediately pass the rgb gotten from the queue to the MediaPipe model to get the multi_handedness and multi_hand_landmarks. Also, in order to make the life easier, a unsortedList will be created. It stores the rgb frame, left or right hand, floor and the x, y, z coordinate of landmark[4]. The floor is the return value of getGesture() function. The logic of getGesture() is similar to the coding we mentioned before. It utilizes the multi_hand_landmarks to know the status of each fingers and then predict the floor number. 

The processing() will send the unsortedList to prediction() for further prediction. 

For instance, 









