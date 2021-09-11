import torch
import torchvision

from torch.utils.data import Dataset, random_split
# from config import device

import student

# AL additions
import time
# import gsheets
import datetime
import pandas as pd
import os
import shutil
import logging
import sys
import progressbar
import PIL
from torch.autograd import Variable

MODEL_DICT_PATH = "checkModel_2021-08-03_1502_epoch_2260.pth"
NUM_WORKERS = 2
# LABELS = {0:"grandpa",
#           1:"apu",
#           2:"bart",
#           3:"burns",
#           4:"wiggum",
#           5:"homer",
#           6:"krusty",
#           7:"lisa",
#           8:"marge",
#           9:"milhouse",
#           10:"moe",
#           11:"flanders",
#           12:"skinner",
#           13:"bob"}

# not sure why the internal mapping doesn't match the folder labels
LABELS = {0:"grandpa",
          1:"apu",
          2:"moe",
          3:"flanders",
          4:"skinner",
          5:"bob",
          6:"bart",
          7:"burns",
          8:"wiggum",
          9:"homer",
          10:"krusty",
          11:"lisa",
          12:"marge",
          13:"milhouse"}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info("using device: " + str(device))

##################################################
# COMMAND LINE INPUTS
##################################################
# defaults (not command line args)
FOLDER_TO_SCORE = "test_new"
test_mode = False

if len(sys.argv) == 2:  
    if sys.argv[1] == "test" or sys.argv[1] == "test_subset":
        FOLDER_TO_SCORE = sys.argv[1]
        test_mode = True

##################################################
# LOGGING
##################################################
# logging.basicConfig(level=logging.DEBUG)
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)  # need to set root logger to the lowest level

# Test network on validation set, if it exists.
def test_network(net, testloader):
    net.eval()
    total_images = 0
    total_correct = 0
    with torch.no_grad():
        for data in progressbar.progressbar(testloader):
            images, labels = data
#             logging.info("images " + str(images))
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            logging.debug("predicted " + str(predicted))
            logging.debug("labels " + str(labels))
            logging.debug("predicted.shape " + str(predicted.shape))
            
    model_accuracy = total_correct / total_images * 100
    print('      Accuracy on {0} test images: {1:.2f}%'.format(
                                total_images, model_accuracy))
    net.train()
    
    return model_accuracy

# initialise model
logging.info("instantiating net")
net = student.net.to(device)
# net.load_state_dict(torch.load(PATH))
logging.info("loading state dict using device " + str(device))
net.load_state_dict(torch.load(MODEL_DICT_PATH, map_location=torch.device(device)))
logging.info("setting model to eval mode")
net.eval()

criterion = student.lossFunc
optimiser = student.optimiser

logging.info("defining dataset")

if test_mode:
    logging.info("test mode using test data")
    data = torchvision.datasets.ImageFolder(root=FOLDER_TO_SCORE,
                                            transform=student.transform('test'))

    logging.info("defining test dataloader")
    testloader = torch.utils.data.DataLoader(data,
                                              batch_size=student.batch_size,
                                             shuffle=False,
                                              num_workers = NUM_WORKERS,
                                              pin_memory=True);

    logging.info("testing network")
    test_network(net, testloader)
# score on a new smaller set of data, far less efficient than the batch processing in "test mode", but easier
# for the user (they don't have to move images into a separate folder for each target label).
# used https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864 as a reference
else:
    logging.info("scoring on new data")
    transform=student.transform('test')    
    filenames = []
    predictions = []
    
    with torch.no_grad():
        for filename in os.listdir(FOLDER_TO_SCORE):
            if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg") or filename.lower().endswith(".png"):                
                data = PIL.Image.open(os.path.join(FOLDER_TO_SCORE, filename))    
                data = transform(data).float()

                # Add an extra batch dimension since pytorch treats all images as batches
                image_tensor = data.unsqueeze_(0).to(device)

                # Turn the input into a Variable
                input = Variable(image_tensor)

                # Predict the class of the image
                output = net(input)
                _, predicted = torch.max(output.data, 1)

                print(os.path.join(FOLDER_TO_SCORE, filename), "is predicted to be", LABELS[predicted.item()], "label", predicted)
        
                filenames.append(os.path.join(FOLDER_TO_SCORE, filename))
                predictions.append(LABELS[predicted.item()])

    net.train()
    
    logging.info("output scores to csv")
    pd.DataFrame({"file":filenames, "prediction":predictions}).to_csv(os.path.join(FOLDER_TO_SCORE, "results.csv"))

