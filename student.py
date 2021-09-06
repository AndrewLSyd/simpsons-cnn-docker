#!/usr/bin/env python3
"""
student.py

From my major assignment for UNSW COMP9444 Neural Networks and Deep Learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
********************************************************************************
* choice of architecture, algorithms, and enhancements
********************************************************************************
CNNs, with their automatic image feature generation and efficient parameter sharing [1] were the natural choice for this task. I briefly experimented with regular fully connected ANNs of varying depths and numbers of hidden layers, but they did not perform that well, achieving accuracies of only 30-40%. It was found that CNNs easily outperformed ANNs.

Given all the possible architectures (width, depth, max pooling, batch normalisation, skip connections, …), activation functions and optimisers and hyper parameters, clearly, the size of the search space make a brute force, purely random search unfeasible.

Thus, being methodological was paramount; the transformations, architecture and training configurations of all models were programmatically saved to a google spreadsheet (see this public gsheets https://docs.google.com/spreadsheets/d/1u9UyFmrOLXEPozd-dPBhQxp_Lt-UcxKKHXtPyDW9nX4/edit?usp=sharing) to keep track of experiments performed. Given the search space, it’s also very important to tune and experiment in order of importance (e.g., tuning the learning rate before finding a decent architecture or set of image transforms is putting the cart before the horse). My order of experimentation was as follows: find a baseline model, experiment with data transforms, experiment with layers (depth, width, max pooling, batch normalisation), experiment with activation functions and then experiment with optimisers and their parameters. Finally, after locking everything in I retrained on 95% of the training data (using the 5% to check if the model had converged). Ultimately, what the deciding factor for model architecture was its performance on the validation (and test) data. Looking at this google sheet, you can see that over time my validation score improves, but I quickly hit a point of diminishing returns and struggle to get validation accuracy much greater than my final submission score of 96-7%. To an extent, this method of stepwise experimentation falsely assumes independence of each step I am tuning, but given the vast size of the search space, it is a reasonable concession.

I saved every 10 epochs of all models and generally trained them for 3000 epochs towards the end of my experimentation. This allowed me to train a model past convergence into overfit territory and subsequently retrieve an epoch that had just converged (according to the validation data). I had access to a non-preemptible virtual machine on Google Cloud Platform with a P100 GPU running 24/7 that could fit two models in parallel comfortably. I also set up a clean repo/bash setup script that allowed me to easily spin up an arbitrary number of GCP virtual machines to increase parallel search capacity, but around this stage I was already achieving a final submission score of 90-91% so was deemed unnecessary.

Start with a reasonable, baseline model
I started with a baseline CNN model to build on a tweak by taking inspiration from AlexNet as well as model structures used in image recognition tasks like the MNIST, Fashion MNIST as well as CIFAR-10. I also looked to recommendations from [1, 2, 4] to guide my “default” baseline model choice.

Depth (numbers of convolutional layers) - I tested various depths, from 1 to 6. 5 Seemed to work best for me. A kernel size and stride of 5x5 and 1 were used for all convolutional layers (same as AlexNet’s first CNN layer).

Width – taking inspiration from AlexNet and a comment Professor Blair made in one of the lectures, initially, a “funnel” shape was used with the first layer having a width of 64 before expanding to up to 512 for deeper layers. Later on, I discovered success by keeping the number of filters mostly constant and testing different numbers of filters.

Fully connected network - I experimented with 1-3 hidden layers. 2 layers seemed to give similar performance, so I proceeded with just 1 hidden layer.

Max pooling - my initial experimentation had max pooling layers for each layer but cause too much information to be lost from subsampling. Taking inspiration from AlexNet, my final network had a max pooling layer at the beginning and somewhere towards the end of the network. I found that max pooling layers helped somewhat in overfitting.

Batch normalisation - was recommended by [1], [2]. At a cost of only a few parameters per layer, batch normalisation can help ameliorate vanishing/exploding gradients. Batch normalisation - was used at every layer to speed up gradient descent to discourage vanishing and exploding gradients by keeping inputs and outputs to each layer inside a healthy range and speed up the training process.
Had I more time, I would have investigated skip connections/Resnet as well as taking inspiration from other well-known CNNs (e.g., ResNet, VCG).

********************************************************************************
* choice of loss function and optimiser
********************************************************************************
Loss function – this is a multiclass classification problem. The two obvious candidates are to either output nn.LogSoftmax and use nn.NLLLoss or to output logits and use nn.CrossEntropyLoss. According to [4, pp187] they are equivalent so we have gone with outputting logits and using nn.CrossEntropyLoss.
Optimiser - based on the recommendation of Geron [1] and He (of He initialisation fame) [3] I initially used Nesterov accelerated gradient descent. After my model architecture was starting to finalise, I experimented with the popular Adam and it seemed to perform a bit better (perhaps it’s less sensitive to hyperparameters) so I proceeded with Adam. At this stage, my model was performing at 96-97% accuracy on the final submission, so I only bothered testing another learning rate for Adam (which performed poorly).

********************************************************************************
* choice of image transformations
********************************************************************************
One of the best ways to improve model generalisability and performance is to feed it more data [3]. Unfortunately, getting more clean labelled data is often difficult. This is where data augmentation comes in – by randomly transforming the data in such a way that a human would still recognise it as the labelled object we are generating additional pseudo-data for our model [4, pp346-7].

Greyscale transform – given the data is inherently greyscale, removing the 2 redundant channels was a no-brainer.

Random horizontal flip – flipping images along the y-axis creates a realistic pseudo-image that in most cases, humans would not realise was flipped. Also, a no-brainer. A flip probability of 50% was used, as proposed by [3]

Colour jitter – was used to randomly alter the brightness and hue of the image as proposed by [3]
Auto augment policy – it was found that in addition to the above transforms, using an auto augment policy that randomly transforms images according to a well-known policy helped model performance. The incredibly well-known Image Net transforms as well as CIFAR-10’s transforms were tested. Image net seemed to perform marginally better and was thus used. It was observed that using the image net auto augment policy was so effective at creating pseudo data that even in later epochs (2500+), the network struggled to “overfit” and achieve even training accuracies more than 3-4% above validation data. 

********************************************************************************
* (d) tuning of meta parameters
********************************************************************************
Optimisers and hyper parameters
I experimented with batch sizes of 256, 512 and 1024 (which didn’t fit into my GPU’s RAM). 256 rather than 512 would possibly help speed up training with quicker auto grad calculations, and some additional randomness to push it out of local optima. In the end, it didn’t appear to make a material difference, so I stuck with 256. Powers of two were experimented with, as X. 
Learning rate and momentum – were kept at default values. Had I more time I would have run a quick grid search across these values.

Activation function – whilst the ever-popular ReLU provides us with non-linearity, saturation for only small values (and hence fewer vanishing gradients) and quick computation [1], it suffers from the possibility of “dead neurons” – neurons with a negative input and gradients stuck at zero. For much of my experimentation, ELU was used an alternative that also provides non-linearity and saturation in one direction but gives the opportunity for “dead neurons” to revive and is recommended over ReLU by Geron [1] and Clevert et al. [5]. Interestingly, experimentation towards the end showed ReLU outperforming ELU slightly, so my final model uses ReLU rather than ELU.

********************************************************************************
* use of validation set, and any other steps taken to improve generalization and avoid overfitting
********************************************************************************
Validation data – was used with a typical training split of 80% to estimate final test set error. The ratio of training to validation error was also used as a measure of level of overfit (e.g., a model with 95% training accuracy and 75% validation accuracy shows signs of overfitting) and helped guide training decisions like whether to implement more regularisation techniques and how many more epochs to train for. Models were allowed to fit pasted optimal convergence into overfit territory, before retrieving an earlier model that had “just converged” – like early stopping, except I allowed the model to far overfit, in case early stopping occurred at a local optimum.
Max pooling – reduces the risk of overfitting by subsampling and reducing the number of parameters [1, page 371].
Hinton’s dropout layers – were applied at each convolutional layer to randomly “turn off” neurons during training, to reduce co-dependence between neurons and making the network less sensitive to reliance on a few input neurons [1, pp 312].
Heavy use of transforms – the auto transforms based on image net provided the model with huge increase in amount of training data. It was observed that with the random transforms, the network struggled to overfit over, event with 3000 epochs (e.g., the network’s training accuracy struggled to climb much higher than 4% over the validation accuracy) as the network was fed a “different” data set at each epoch, making it difficult to overfit to any particular set of images.
References
[1]	A. Geron, Hands-On Machine Learning with Scikit-Learn and TensorFlow. 2018.
[2]	F. Chollet, Deep Learning with Python. 2020.
[3]	He et al., Bag of Tricks for Image Classification with Convolutional Neural Networks. 2019.
[4]	E Stevens, L Antiga, T Viehmann, Deep Learning with PyTorch. 2020
[5]	D Clevert, T Unterthiner, S Hochreiter, Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':        
        my_transform = transforms.Compose([
            # convert to greyscale, remove redundant channels
            transforms.Grayscale(num_output_channels=1),
            # apply imagenet auto augment
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            # randomly change brightness and hue
            transforms.ColorJitter(brightness=.5, hue=.3),
            # flip across the y-axis 50% of the time
            transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor()
                                          ])

    elif mode == 'test':
        # for testing there is no need to apply random augmentations
        my_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64,64)),
            transforms.ToTensor()
        ])
    return my_transform

############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class CNN_30(nn.Module):
    """
    Like CNN 29, but moved max pooling layer to the end and removed a max pooling layer
    """
    def __init__(self):
        super(CNN_30, self).__init__()
        self.model = nn.Sequential(
            # conv layer 1            
            nn.Conv2d(1, 64, kernel_size = 5, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.5),
            
            # conv layer 2
            nn.Conv2d(64, 128, kernel_size = 5, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            # conv layer 3
            nn.Conv2d(128, 128, kernel_size = 5, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),     
            nn.Dropout(p=0.5),
            
             # conv layer 4
            nn.Conv2d(128, 128, kernel_size = 5, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.5),
            
             # conv layer 5
            nn.Conv2d(128, 128, kernel_size = 5, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),            
            
            # fully connected layer
            nn.Flatten(),
            nn.Linear(12800, 64),
            nn.ReLU(),
            nn.Linear(64, 14),
        )        
    def forward(self, x):
        output = self.model(x)
        return output    
    
net = CNN_30()

# logits and cross entropy loss used for multi-class classification
lossFunc = torch.nn.CrossEntropyLoss()  # pp187 of deep learning with pytorchsays use logits as outputs and CrossEntropyLoss()

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = .8
batch_size = 256
# let it run past the point of convergence to overfit territory
epochs = 3000
optimiser = optim.Adam(net.parameters(), lr=0.001)
# optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)

"""
LIST OF PACKAGES USED TO REPLICATE RESULTS
! conda list
# packages in environment at /home/jovyan/my-conda-envs/pytorch-gpu:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                 conda_forge    conda-forge
_openmp_mutex             4.5                       1_gnu    conda-forge
abseil-cpp                20210324.1           h9c3ff4c_0    conda-forge
aiohttp                   3.7.4.post0      py39h3811e60_0    conda-forge
arrow-cpp                 4.0.0           py39h6ac3dd5_3_cpu    conda-forge
async-timeout             3.0.1                   py_1000    conda-forge
attrs                     21.2.0             pyhd8ed1ab_0    conda-forge
aws-c-cal                 0.5.11               h95a6274_0    conda-forge
aws-c-common              0.6.2                h7f98852_0    conda-forge
aws-c-event-stream        0.2.7               h3541f99_13    conda-forge
aws-c-io                  0.10.5               hfb6a706_0    conda-forge
aws-checksums             0.1.11               ha31a3da_7    conda-forge
aws-sdk-cpp               1.8.186              hb4091e7_3    conda-forge
backcall                  0.2.0              pyh9f0ad1d_0    conda-forge
backports                 1.0                        py_2    conda-forge
backports.functools_lru_cache 1.6.4              pyhd8ed1ab_0    conda-forge
blas                      1.0                         mkl    defaults
blessings                 1.7                      pypi_0    pypi
blinker                   1.4                        py_1    conda-forge
brotli                    1.0.9                h7f98852_5    conda-forge
brotli-bin                1.0.9                h7f98852_5    conda-forge
brotlipy                  0.7.0           py39h3811e60_1001    conda-forge
bzip2                     1.0.8                h7f98852_4    conda-forge
c-ares                    1.17.1               h7f98852_1    conda-forge
ca-certificates           2021.5.30            ha878542_0    conda-forge
cachetools                4.1.1                    pypi_0    pypi
catalogue                 2.0.4            py39hf3d152e_0    conda-forge
certifi                   2021.5.30        py39hf3d152e_0    conda-forge
cffi                      1.14.6           py39he32792d_0    conda-forge
chardet                   3.0.4                    pypi_0    pypi
charset-normalizer        2.0.0              pyhd8ed1ab_0    conda-forge
click                     7.1.2              pyh9f0ad1d_0    conda-forge
colorama                  0.4.4              pyh9f0ad1d_0    conda-forge
cryptography              3.4.7            py39hbca0aa6_0    conda-forge
cudatoolkit               10.2.89              h8f6ccaa_8    conda-forge
cycler                    0.10.0                     py_2    conda-forge
cymem                     2.0.5            py39he80948d_2    conda-forge
cython-blis               0.7.4            py39hce5d2b2_0    conda-forge
dataclasses               0.7                      pypi_0    pypi
dbus                      1.13.6               h48d8840_2    conda-forge
debugpy                   1.3.0            py39he80948d_0    conda-forge
decorator                 5.0.9              pyhd8ed1ab_0    conda-forge
en-core-web-sm            3.1.0                    pypi_0    pypi
expat                     2.4.1                h9c3ff4c_0    conda-forge
ffmpeg                    4.3                  hf484d3e_0    pytorch
fontconfig                2.13.1            hba837de_1005    conda-forge
freetype                  2.10.4               h0708190_1    conda-forge
fsspec                    2021.7.0           pyhd8ed1ab_0    conda-forge
gcsfs                     2021.7.0           pyhd8ed1ab_0    conda-forge
gettext                   0.19.8.1          h0b5b191_1005    conda-forge
gflags                    2.2.2             he1b5a44_1004    conda-forge
glib                      2.68.3               h9c3ff4c_0    conda-forge
glib-tools                2.68.3               h9c3ff4c_0    conda-forge
glog                      0.5.0                h48cff8f_0    conda-forge
gmp                       6.2.1                h58526e2_0    conda-forge
gnutls                    3.6.13               h85f3911_1    conda-forge
google-api-core           1.22.4                   pypi_0    pypi
google-api-core-grpc      1.31.0               hd8ed1ab_0    conda-forge
google-api-python-client  2.14.0             pyhd8ed1ab_0    conda-forge
google-auth               1.22.1                   pypi_0    pypi
google-auth-httplib2      0.1.0              pyhd8ed1ab_0    conda-forge
google-auth-oauthlib      0.4.4              pyhd8ed1ab_0    conda-forge
google-cloud-bigquery     2.21.0             pyheb06c22_0    conda-forge
google-cloud-bigquery-core 2.21.0             pyheb06c22_0    conda-forge
google-cloud-bigquery-storage 2.0.0                    pypi_0    pypi
google-cloud-bigquery-storage-core 2.0.0              pyh9f0ad1d_1    conda-forge
google-cloud-core         1.7.1              pyh6c4a22f_0    conda-forge
google-crc32c             1.1.2            py39hb81f231_0    conda-forge
google-resumable-media    1.3.1              pyh6c4a22f_0    conda-forge
googleapis-common-protos  1.52.0                   pypi_0    pypi
gpustat                   0.6.0                    pypi_0    pypi
grpc-cpp                  1.37.1               h2519f57_2    conda-forge
grpcio                    1.32.0                   pypi_0    pypi
gst-plugins-base          1.14.0               hbbd80ab_1    defaults
gstreamer                 1.14.0               h28cd5cc_2    defaults
httplib2                  0.19.1             pyhd8ed1ab_0    conda-forge
icu                       58.2              hf484d3e_1000    conda-forge
idna                      2.10                     pypi_0    pypi
intel-openmp              2021.2.0           h06a4308_610    defaults
ipykernel                 6.0.1            py39hef51801_0    conda-forge
ipython                   7.25.0           py39hef51801_1    conda-forge
ipython_genutils          0.2.0                      py_1    conda-forge
jedi                      0.18.0           py39hf3d152e_2    conda-forge
jinja2                    3.0.1              pyhd8ed1ab_0    conda-forge
joblib                    1.0.1              pyhd8ed1ab_0    conda-forge
jpeg                      9b                   h024ee3a_2    defaults
jupyter_client            6.1.12             pyhd8ed1ab_0    conda-forge
jupyter_core              4.7.1            py39hf3d152e_0    conda-forge
kiwisolver                1.3.1            py39h1a9c180_1    conda-forge
krb5                      1.19.1               hcc1bbae_0    conda-forge
lame                      3.100             h7f98852_1001    conda-forge
lcms2                     2.12                 h3be6417_0    defaults
ld_impl_linux-64          2.36.1               hea4e1c9_1    conda-forge
libblas                   3.9.0                     9_mkl    conda-forge
libbrotlicommon           1.0.9                h7f98852_5    conda-forge
libbrotlidec              1.0.9                h7f98852_5    conda-forge
libbrotlienc              1.0.9                h7f98852_5    conda-forge
libcblas                  3.9.0                     9_mkl    conda-forge
libcrc32c                 1.1.1                h9c3ff4c_2    conda-forge
libcst                    0.3.13                   pypi_0    pypi
libcurl                   7.77.0               h2574ce0_0    conda-forge
libedit                   3.1.20191231         he28a2e2_2    conda-forge
libev                     4.33                 h516909a_1    conda-forge
libevent                  2.1.10               hcdb4288_3    conda-forge
libffi                    3.3                  h58526e2_2    conda-forge
libgcc-ng                 9.3.0               h2828fa1_19    conda-forge
libgfortran-ng            9.3.0               hff62375_19    conda-forge
libgfortran5              9.3.0               hff62375_19    conda-forge
libglib                   2.68.3               h3e27bee_0    conda-forge
libgomp                   9.3.0               h2828fa1_19    conda-forge
libiconv                  1.16                 h516909a_0    conda-forge
liblapack                 3.9.0                     9_mkl    conda-forge
libnghttp2                1.43.0               h812cca2_0    conda-forge
libpng                    1.6.37               h21135ba_2    conda-forge
libprotobuf               3.16.0               h780b84a_0    conda-forge
libsodium                 1.0.18               h36c2ea0_1    conda-forge
libssh2                   1.9.0                ha56f1ee_6    conda-forge
libstdcxx-ng              9.3.0               h6de172a_19    conda-forge
libthrift                 0.14.1               he6d91bd_2    conda-forge
libtiff                   4.2.0                h85742a9_0    defaults
libutf8proc               2.6.1                h7f98852_0    conda-forge
libuuid                   2.32.1            h7f98852_1000    conda-forge
libuv                     1.41.1               h7f98852_0    conda-forge
libwebp-base              1.2.0                h7f98852_2    conda-forge
libxcb                    1.13              h7f98852_1003    conda-forge
libxml2                   2.9.12               h03d6c58_0    defaults
lz4-c                     1.9.3                h9c3ff4c_0    conda-forge
markupsafe                2.0.1            py39h3811e60_0    conda-forge
matplotlib                3.4.2            py39hf3d152e_0    conda-forge
matplotlib-base           3.4.2            py39h2fa2bec_0    conda-forge
matplotlib-inline         0.1.2              pyhd8ed1ab_2    conda-forge
mkl                       2021.2.0           h06a4308_296    defaults
mkl-service               2.4.0            py39h3811e60_0    conda-forge
mkl_fft                   1.3.0            py39h42c9631_2    defaults
mkl_random                1.2.2            py39hde0f152_0    conda-forge
multidict                 5.1.0            py39h3811e60_1    conda-forge
murmurhash                1.0.5            py39he80948d_0    conda-forge
mypy_extensions           0.4.3            py39hf3d152e_3    conda-forge
ncurses                   6.2                  h58526e2_4    conda-forge
nettle                    3.6                  he412f7d_0    conda-forge
ninja                     1.10.2               h4bd325d_0    conda-forge
nltk                      3.6.2              pyhd8ed1ab_0    conda-forge
numpy                     1.20.2           py39h2d18471_0    defaults
numpy-base                1.20.2           py39hfae3a4d_0    defaults
nvidia-ml-py3             7.352.0                  pypi_0    pypi
oauthlib                  3.1.1              pyhd8ed1ab_0    conda-forge
olefile                   0.46               pyh9f0ad1d_1    conda-forge
openh264                  2.1.1                h780b84a_0    conda-forge
openjpeg                  2.4.0                hb52868f_1    conda-forge
openssl                   1.1.1k               h7f98852_0    conda-forge
orc                       1.6.7                h89a63ab_2    conda-forge
packaging                 21.0               pyhd8ed1ab_0    conda-forge
pandas                    1.3.0            py39hde0f152_0    conda-forge
pandas-gbq                0.15.0             pyh44b312d_0    conda-forge
parquet-cpp               1.5.1                         2    conda-forge
parso                     0.8.2              pyhd8ed1ab_0    conda-forge
pathy                     0.6.0              pyhd8ed1ab_0    conda-forge
patsy                     0.5.1                      py_0    conda-forge
pcre                      8.45                 h9c3ff4c_0    conda-forge
pexpect                   4.8.0              pyh9f0ad1d_2    conda-forge
pickleshare               0.7.5                   py_1003    conda-forge
pillow                    8.3.1            py39h2c7a002_0    defaults
pip                       21.1.3             pyhd8ed1ab_0    conda-forge
preshed                   3.0.5            py39he80948d_1    conda-forge
prompt-toolkit            3.0.19             pyha770c72_0    conda-forge
proto-plus                1.10.1                   pypi_0    pypi
protobuf                  3.13.0                   pypi_0    pypi
psutil                    5.8.0                    pypi_0    pypi
pthread-stubs             0.4               h36c2ea0_1001    conda-forge
ptyprocess                0.7.0              pyhd3deb0d_0    conda-forge
pyarrow                   4.0.0           py39h3ebc44c_3_cpu    conda-forge
pyasn1                    0.4.8                      py_0    conda-forge
pyasn1-modules            0.2.8                    pypi_0    pypi
pycparser                 2.20               pyh9f0ad1d_2    conda-forge
pydantic                  1.8.2            py39h3811e60_0    conda-forge
pydata-google-auth        1.2.0              pyhd8ed1ab_0    conda-forge
pygments                  2.9.0              pyhd8ed1ab_0    conda-forge
pyjwt                     2.1.0              pyhd8ed1ab_0    conda-forge
pyopenssl                 20.0.1             pyhd8ed1ab_0    conda-forge
pyparsing                 2.4.7              pyh9f0ad1d_0    conda-forge
pyqt                      5.9.2            py39h2531618_6    defaults
pysocks                   1.7.1            py39hf3d152e_3    conda-forge
python                    3.9.6           h49503c6_0_cpython    conda-forge
python-dateutil           2.8.1                      py_0    conda-forge
python_abi                3.9                      2_cp39    conda-forge
pytorch                   1.9.0           py3.9_cuda10.2_cudnn7.6.5_0    pytorch
pytz                      2020.1                   pypi_0    pypi
pyu2f                     0.1.5              pyhd8ed1ab_0    conda-forge
pyyaml                    5.3.1                    pypi_0    pypi
pyzmq                     22.1.0           py39h37b5a0c_0    conda-forge
qt                        5.9.7                h5867ecd_1    defaults
re2                       2021.04.01           h9c3ff4c_0    conda-forge
readline                  8.1                  h46c0cb4_0    conda-forge
regex                     2021.7.6         py39h3811e60_0    conda-forge
requests                  2.24.0                   pypi_0    pypi
requests-oauthlib         1.3.0              pyh9f0ad1d_0    conda-forge
rsa                       4.7.2              pyh44b312d_0    conda-forge
s2n                       1.0.10               h9b69904_0    conda-forge
scikit-learn              0.24.2           py39h4dfa638_0    conda-forge
scipy                     1.7.0            py39hee8e79c_0    conda-forge
seaborn                   0.11.1               hd8ed1ab_1    conda-forge
seaborn-base              0.11.1             pyhd8ed1ab_1    conda-forge
setuptools                49.6.0           py39hf3d152e_3    conda-forge
shellingham               1.4.0              pyh44b312d_0    conda-forge
simplejson                3.17.3           py39h3811e60_0    conda-forge
sip                       4.19.13          py39h2531618_0    defaults
six                       1.15.0                   pypi_0    pypi
smart_open                5.1.0              pyhd8ed1ab_1    conda-forge
snappy                    1.1.8                he1b5a44_3    conda-forge
spacy                     3.1.0            py39h5472131_0    conda-forge
spacy-legacy              3.0.8              pyhd8ed1ab_0    conda-forge
sqlite                    3.36.0               h9cd32fc_0    conda-forge
srsly                     2.4.1            py39he80948d_0    conda-forge
statsmodels               0.12.2           py39hce5d2b2_0    conda-forge
thinc                     8.0.7            py39h5472131_0    conda-forge
threadpoolctl             2.1.0              pyh5ca1d4c_0    conda-forge
tk                        8.6.10               h21135ba_1    conda-forge
torchaudio                0.9.0                      py39    pytorch
torchvision               0.10.0               py39_cu102    pytorch
tornado                   6.1              py39h3811e60_1    conda-forge
tqdm                      4.61.2             pyhd8ed1ab_1    conda-forge
traitlets                 5.0.5                      py_0    conda-forge
typer                     0.3.2              pyhd8ed1ab_0    conda-forge
typing-extensions         3.7.4.3                  pypi_0    pypi
typing-inspect            0.6.0                    pypi_0    pypi
typing_extensions         3.10.0.0           pyha770c72_0    conda-forge
typing_inspect            0.7.1              pyh6c4a22f_0    conda-forge
tzdata                    2021a                he74cb21_1    conda-forge
uritemplate               3.0.1                      py_0    conda-forge
urllib3                   1.25.10                  pypi_0    pypi
wasabi                    0.8.2              pyh44b312d_0    conda-forge
wcwidth                   0.2.5              pyh9f0ad1d_2    conda-forge
wheel                     0.36.2             pyhd3deb0d_0    conda-forge
xorg-libxau               1.0.9                h7f98852_0    conda-forge
xorg-libxdmcp             1.1.3                h7f98852_0    conda-forge
xz                        5.2.5                h516909a_1    conda-forge
yaml                      0.2.5                h516909a_0    conda-forge
yarl                      1.6.3            py39h3811e60_2    conda-forge
zeromq                    4.3.4                h9c3ff4c_0    conda-forge
zlib                      1.2.11            h516909a_1010    conda-forge
zstd                      1.4.9                ha95c52a_0    conda-forge

"""
