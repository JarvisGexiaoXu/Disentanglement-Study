from xml.etree.ElementTree import tostring
import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import logging

logging.basicConfig(filename='dci1.log', level=logging.DEBUG)
with open('dci1.log','w'):
    pass

# hyperparameters -------------------------------------------------------------

dataset_path = '~/datasets'
cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu") 

batch_size = 100 # 训练一个batch有100个samples

x_dim  = 784 # 图片样本dimension （输入 encoder （输出 decoder
hidden_dim = 400 # 隐藏层向量dimension （隐藏层 encoder （隐藏层 decoder
latent_dim = 200 # 潜在向量dimension （输出 encoder （输入 decoder

lr = 1e-3 # 学习率

epochs = 1

beta = 4

# Define model -----------------------------------------------------------------

# Encoder ----------------------------------------------------------------------
class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var

# Decoder ----------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat

# Model ------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var

# Combine Encoder and Decoder into the model -----------------------------------         
encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

# Save and Load ----------------------------------------------------------------

model = torch.load("VAE")
encoder = torch.load("encoder")
decoder = torch.load("decoder")
print("loading complete...")
logging.info("loading complete...")

# image generation (from noise) -------------------------------------------------
with torch.no_grad():
    noise = torch.randn(batch_size, latent_dim).to(DEVICE)
    generated_images = decoder(noise)

save_image(generated_images.view(batch_size, 1, 28, 28), 'VAE model/generated_sample.png')
print("Image generation complete...")
logging.info("Image generation complete...")





''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''                             DCI Evaluation                                         '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import scipy
import scipy.stats
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import utils
from _base import DisentDataset
print(type(DisentDataset))
mnist_transform = transforms.Compose([
        transforms.ToTensor(),
])

def metric_dci(
    dataset: DisentDataset,
    representation_function: callable,
    num_train: int = 10000,
    num_test: int = 5000,
    batch_size: int = 16,
    boost_mode = 'sklearn',
    show_progress = False, 
    ):
    logging.info("Generating training set. -----For DCI")
    print(123)
    print("!",type(dataset))
    mus_train, ys_train = utils.generate_batch_factor_code(dataset, representation_function, num_train, batch_size, show_progress=False)
    assert mus_train.shape[1] == num_train
    assert ys_train.shape[1] == num_train
    mus_test, ys_test = utils.generate_batch_factor_code(dataset, representation_function, num_test, batch_size, show_progress=False)


    logging.info("Computing DCI score.")
    scores = compute_dci(mus_train, ys_train, mus_test, ys_test, boost_mode=boost_mode, show_progress=show_progress)
    

    return scores["dci.disentanglement"]

def compute_dci(mus_train, ys_train, mus_test, ys_test, boost_mode='sklearn', show_progress=False):
    """Computes score based on both training and testing codes and factors."""
    importance_matrix, train_err, test_err = compute_importance_gbt(mus_train, ys_train, mus_test, ys_test, boost_mode=boost_mode, show_progress=show_progress)
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    return {
        "dci.informativeness_train": train_err,                      # "dci.explicitness" -- Measuring Disentanglement: A Review of Metrics
        "dci.informativeness_test": test_err,                        # "dci.explicitness" -- Measuring Disentanglement: A Review of Metrics
        "dci.disentanglement": disentanglement(importance_matrix),  # "dci.modularity"   -- Measuring Disentanglement: A Review of Metrics
        "dci.completeness": completeness(importance_matrix),        # "dci.compactness"  -- Measuring Disentanglement: A Review of Metrics
    }

def compute_importance_gbt(x_train, y_train, x_test, y_test, boost_mode='sklearn', show_progress=False):
    # Compute importance based on gradient boosted trees.
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zerps(shape=[num_codes, num_factors], dtype=np.float64)
    train_loss = []
    test_loss = []
    for i in tqdm(range(num_factors),disable=(not show_progress)):
        if boost_mode == 'sklearn':
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier
        model.fix(x_train.T, y_train[i, :])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
        test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))

    return importance_matrix, np.mean(train_loss), np.mean(test_loss)

def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11, base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()
    return np.sum(per_code * code_importance)


def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix + 1e-11, base=importance_matrix.shape[0])


def completeness(importance_matrix):
    """"Compute completeness of the representation."""
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor * factor_importance)

print(metric_dci(dataset=dataset_path, representation_function=callable))
