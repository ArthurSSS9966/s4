import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from dPCA import dPCA
from numpy import random
from sklearn.metrics import (
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from regularizers import compute_regularizer_term


# Define LSTM model
class LSTMClassifier(nn.Module):
    def __init__(
        self, input_dim, output_dim, lr=0.001, hidden_dim=50, weight_decay=1e-5
    ):
        super(LSTMClassifier, self).__init__()
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, bidirectional=True
        ).to("cuda:0")
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.hidden2out = nn.Linear(hidden_dim * 2, output_dim).to("cuda:0")
        self.softmax = nn.Softmax(dim=1).to("cuda:0")

    def forward(self, x):
        x = x.to("cuda:0")
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last output for classification
        output = self.hidden2out(lstm_out)

        output = self.softmax(output)
        return output

    def random_init(self):
        # Randomly initialize weights and biases
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.uniform_(
                    param.data, -0.1, 0.1
                )  # Uniform initialization between -0.1 and 0.1
            elif "bias" in name:
                nn.init.zeros_(param.data)  # Zero initialization for biases

    def load_data(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Convert to PyTorch tensors
        # self.X_train = torch.from_numpy(self.X_train).float().to('cuda:0')
        # self.X_test = torch.from_numpy(self.X_test).float().to('cuda:0')
        # self.y_train = torch.from_numpy(self.y_train).long().to('cuda:0')
        # self.y_test = torch.from_numpy(self.y_test).long()

        # Convert to PyTorch tensors
        self.X_train = torch.tensor(self.X_train.tolist()).float().to("cuda:0")
        self.X_test = torch.tensor(self.X_test.tolist()).float().to("cuda:0")
        self.y_train = torch.tensor(self.y_train.tolist()).long().to("cuda:0")
        self.y_test = torch.tensor(self.y_test.tolist()).long()

    def train_test_using_optimizer(self, epoches=50):
        epoch_accuracies = []
        epoch_loss = []
        epoch_test_loss = []
        for epoch in range(epoches):
            self.train()
            self.optimizer.zero_grad()

            output = self(self.X_train)
            loss = self.criteria(output, self.y_train)

            loss.backward()

            self.optimizer.step()

            # Evaluate on the test set
            test_output = self(self.X_test).cpu().detach()
            _, predicted = torch.max(test_output, 1)
            correct = (predicted == self.y_test).sum().item()
            accuracy = correct / len(self.y_test)
            epoch_accuracies.append(accuracy)

            loss_cpu = loss.cpu().detach()
            epoch_loss.append(loss_cpu.item())

            test_loss = self.criteria(test_output, self.y_test).cpu().detach()
            epoch_test_loss.append(test_loss.item())

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epoches}], Loss: {loss.item()}, Test"
                    f" Accuracy: {accuracy * 100}%"
                )

        return epoch_accuracies, epoch_loss, epoch_test_loss


# Define GRU+Transformer model
class PositionalEncoding(nn.Module):
    def __init__(self, hiddendim, lens, device):
        super(PositionalEncoding, self).__init__()
        self.hiddendim = hiddendim
        self.positional_encoding = self.generate_positional_encoding(
            hiddendim, lens
        ).to(device, non_blocking=True)

    def generate_positional_encoding(self, hiddendim, lens):
        pe = torch.zeros(lens, hiddendim)
        position = torch.arange(0, lens).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hiddendim, 2) * -(math.log(10000.0) / hiddendim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.positional_encoding[:, : x.size(1)]
        return x


class GTModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.loss_fn = torch.nn.MSELoss()

    def load_data(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.timestep = X_train.shape[1]
        # Convert to PyTorch tensors
        self.X_train = (
            torch.from_numpy(self.X_train)
            .float()
            .to(self.device, non_blocking=True)
        )
        self.X_test = (
            torch.from_numpy(self.X_test)
            .float()
            .to(self.device, non_blocking=True)
        )
        self.y_train = (
            torch.from_numpy(self.y_train)
            .float()
            .to(self.device, non_blocking=True)
        )
        self.y_test = (
            torch.from_numpy(self.y_test)
            .float()
            .to(self.device, non_blocking=True)
        )

    def Build(
        self,
        hiddendim,
        middle_dim,
        nhead,
        num_layers,
        learningRate=0.001,
        weight_decay=0.0001,
        hidden_prior="Uniform",
        hidden_prior2="False",
        hyperatio=1.0,
        lambda_=0.001,
        c=1,
    ):
        """
        Transformer parameters:
        nhead =                             (default)
        num_encoder_layer =                 (default)
        dropout = 0.1                       (default)
        activation = relu                   (default)
        norm_first                          (default)
        """

        inputdim = self.X_train.shape[2]
        outputsize = self.y_train.shape[2]
        self.input_dim = inputdim
        self.hiddendim = hiddendim
        self.nhead = nhead
        self.num_layers = num_layers
        self.output_dim = outputsize
        # self.embedding = nn.Linear(self.input_dim,self.hiddendim).to(self.device,non_blocking=True)
        self.middle_dim = middle_dim
        self.gru = nn.GRU(
            self.input_dim,
            self.hiddendim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        ).to(self.device, non_blocking=True)
        self.gru_fc = nn.Linear(self.hiddendim * 2, self.hiddendim).to(
            self.device, non_blocking=True
        )
        self.position_encode = PositionalEncoding(
            self.hiddendim, self.timestep, self.device
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hiddendim, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.num_layers
        ).to(self.device, non_blocking=True)

        self.fc_out = nn.Linear(self.hiddendim, self.middle_dim).to(
            self.device, non_blocking=True
        )
        self.fc_out1 = nn.Linear(self.middle_dim, self.middle_dim).to(
            self.device, non_blocking=True
        )
        self.fc_out2 = nn.Linear(self.middle_dim, self.middle_dim).to(
            self.device, non_blocking=True
        )
        self.fc_out3 = nn.Linear(self.middle_dim, self.middle_dim).to(
            self.device, non_blocking=True
        )
        self.fc_out4 = nn.Linear(self.middle_dim, self.output_dim).to(
            self.device, non_blocking=True
        )

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learningRate,
            weight_decay=weight_decay,
            eps=1e-3,
        )
        # Set up the cosine learning rate scheduler
        lr_min = learningRate * 0.5
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=lr_min
        )

        # regularizer
        self.lambda_ = lambda_
        self.hidden_prior = hidden_prior
        self.hidden_prior2 = hidden_prior2
        self.hyperatio = hyperatio
        self.c = c

    def forward(self, xt):
        xt = xt.to(self.device, non_blocking=True)
        # xt = self.embedding(xt)

        gru_out, h = self.gru(xt)
        gru_out = self.gru_fc(gru_out)
        trans_out = self.position_encode(gru_out)

        trans_out = self.transformer_encoder(trans_out)
        trans_out = self.fc_out(trans_out)
        trans_out = self.fc_out1(trans_out)
        trans_out = self.fc_out2(trans_out)
        trans_out = self.fc_out3(trans_out)
        velocity = self.fc_out4(trans_out)

        return velocity

    def train_fit(self, N_epoches):
        """

        :param N_epoches:
        :param x_latent: train X
        :param V_train: train Y
        :param x_latent_cv: test X
        :param V_cv: test Y
        :return:
        """
        x_latent = self.X_train
        V_train = self.y_train
        x_latent_cv = self.X_test
        V_cv = self.y_test

        MSE_cv_linear_epoch = torch.empty([N_epoches])
        MSE_train_linear_epoch = torch.empty([N_epoches])

        for i in tqdm(range(N_epoches), "epochs"):
            v_predict_cv = self(x_latent_cv)
            MSE_cv_linear_epoch[i] = self.loss_fn(v_predict_cv, V_cv)

            self.train()
            self.optimizer.zero_grad()
            output = self(x_latent)
            loss = self.loss_fn(output, V_train)
            reg_term = self.hidden_layer_regularizer()

            loss = loss + reg_term
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            MSE_train_linear_epoch[i] = loss

        return MSE_cv_linear_epoch, MSE_train_linear_epoch

    def predict_velocity(self, x_latent):
        self.eval()
        with torch.no_grad():
            v_predict = self(x_latent).detach().cpu().numpy()
        return v_predict

    def hidden_layer_regularizer(self):
        """
        Compute the regularization loss of the hidden layers.
        """

        assert self.hidden_prior in [
            "Uniform",
            "Cauchy",
            "Gaussian",
            "Laplace",
            "Sinc_squared",
            "negcos",
            "SinFouthPower",
        ], (
            "Change the data name to 'uniform', 'Cauchy', 'Gaussian',"
            " 'Laplace', or 'Sinc_squared','Sinc_squared', 'negcos',"
            " 'SinFouthPower'."
        )
        reg_loss = torch.tensor([0.0], device=self.device)

        if self.hidden_prior != "Uniform":
            for name, param in self.named_parameters():
                if (
                    (name[0:9] not in ["embedding"])
                    & (name[-4:] != "bias")
                    & (name[0:3] != "enc")
                    & (name[0:8] not in ["gru.bias"])
                ):
                    # if (name[0:3] == 'fc_') &(name[-4:]!='bias'):
                    reg_loss = reg_loss + compute_regularizer_term(
                        wgts=param,
                        lambda_=self.lambda_,
                        hidden_prior=self.hidden_prior,
                        hidden_prior2=self.hidden_prior2,
                        hyperatio=self.hyperatio,
                        c=self.c,
                    )
        return reg_loss


# Define Attention Feedforward Network
class Affn(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.loss_fn = torch.nn.MSELoss()

    def load_data(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.timestep = X_train.shape[1]
        # Convert to PyTorch tensors
        self.X_train = (
            torch.from_numpy(self.X_train)
            .float()
            .to(self.device, non_blocking=True)
        )
        self.X_test = (
            torch.from_numpy(self.X_test)
            .float()
            .to(self.device, non_blocking=True)
        )
        self.y_train = (
            torch.from_numpy(self.y_train)
            .float()
            .to(self.device, non_blocking=True)
        )
        self.y_test = (
            torch.from_numpy(self.y_test)
            .float()
            .to(self.device, non_blocking=True)
        )

    def Build(
        self,
        hiddendim,
        middle_dim,
        nhead,
        num_layers,
        learningRate=0.001,
        weight_decay=0.0001,
        hidden_prior="Uniform",
        hidden_prior2="False",
        hyperatio=1.0,
        lambda_=0.001,
        c=1,
    ):
        """
        Transformer parameters:
        nhead =                             (default)
        num_encoder_layer =                 (default)
        dropout = 0.1                       (default)
        activation = relu                   (default)
        norm_first                          (default)
        """

        inputdim = self.X_train.shape[2]
        outputsize = self.y_train.shape[2]
        self.input_dim = inputdim
        self.hiddendim = hiddendim
        self.nhead = nhead
        self.num_layers = num_layers
        self.output_dim = outputsize
        # self.embedding = nn.Linear(self.input_dim,self.hiddendim).to(self.device,non_blocking=True)
        self.middle_dim = middle_dim
        # self.embeddingv = nn.Linear(self.output_dim,self.hiddendim).to(self.device,non_blocking=True)
        self.position_encode = PositionalEncoding(
            self.hiddendim, self.timestep, self.device
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hiddendim, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.num_layers
        ).to(self.device, non_blocking=True)

        self.fc_out = nn.Linear(self.hiddendim, self.middle_dim).to(
            self.device, non_blocking=True
        )
        self.fc_out1 = nn.Linear(self.middle_dim, self.middle_dim).to(
            self.device, non_blocking=True
        )
        self.fc_out2 = nn.Linear(self.middle_dim, self.middle_dim).to(
            self.device, non_blocking=True
        )
        self.fc_out3 = nn.Linear(self.middle_dim, self.middle_dim).to(
            self.device, non_blocking=True
        )
        self.fc_out4 = nn.Linear(self.middle_dim, self.output_dim).to(
            self.device, non_blocking=True
        )

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learningRate,
            weight_decay=weight_decay,
            eps=1e-3,
        )
        # Set up the cosine learning rate scheduler
        lr_min = learningRate * 0.5
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=lr_min
        )

        # regularizer
        self.lambda_ = lambda_
        self.hidden_prior = hidden_prior
        self.hidden_prior2 = hidden_prior2
        self.hyperatio = hyperatio
        self.c = c

    def forward(self, xt):
        xt = xt.to(self.device, non_blocking=True)
        # xt = self.embedding(xt)
        trans_out = self.position_encode(xt)

        trans_out = self.transformer_encoder(trans_out)
        trans_out = self.fc_out(trans_out)
        trans_out = self.fc_out1(trans_out)
        trans_out = self.fc_out2(trans_out)
        trans_out = self.fc_out3(trans_out)
        velocity = self.fc_out4(trans_out)

        return velocity

    def train_fit(self, N_epoches):
        """

        :param N_epoches:
        :param x_latent: train X
        :param V_train: train Y
        :param x_latent_cv: test X
        :param V_cv: test Y
        :return:
        """
        x_latent = self.X_train
        V_train = self.y_train
        x_latent_cv = self.X_test
        V_cv = self.y_test

        MSE_cv_linear_epoch = torch.empty([N_epoches])
        MSE_train_linear_epoch = torch.empty([N_epoches])

        for i in tqdm(range(N_epoches), "epochs"):
            v_predict_cv = self(x_latent_cv)
            MSE_cv_linear_epoch[i] = self.loss_fn(v_predict_cv, V_cv)

            self.train()
            self.optimizer.zero_grad()
            output = self(x_latent)
            loss = self.loss_fn(output, V_train)
            reg_term = self.hidden_layer_regularizer()

            loss = loss + reg_term
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            MSE_train_linear_epoch[i] = loss

        return MSE_cv_linear_epoch, MSE_train_linear_epoch

    def predict_velocity(self, x_latent):
        self.eval()
        with torch.no_grad():
            v_predict = self(x_latent).detach().cpu().numpy()
        return v_predict

    def hidden_layer_regularizer(self):
        """
        Compute the regularization loss of the hidden layers.
        """

        assert self.hidden_prior in [
            "Uniform",
            "Cauchy",
            "Gaussian",
            "Laplace",
            "Sinc_squared",
            "negcos",
            "SinFouthPower",
        ], (
            "Change the data name to 'uniform', 'Cauchy', 'Gaussian',"
            " 'Laplace', or 'Sinc_squared','Sinc_squared', 'negcos',"
            " 'SinFouthPower'."
        )
        reg_loss = torch.tensor([0.0], device=self.device)

        if self.hidden_prior != "Uniform":
            for name, param in self.named_parameters():
                if (
                    (name[0:9] not in ["embedding"])
                    & (name[-4:] != "bias")
                    & (name[0:3] != "enc")
                ):
                    # if (name[0:3] == 'fc_') &(name[-4:]!='bias'):
                    reg_loss = reg_loss + compute_regularizer_term(
                        wgts=param,
                        lambda_=self.lambda_,
                        hidden_prior=self.hidden_prior,
                        hidden_prior2=self.hidden_prior2,
                        hyperatio=self.hyperatio,
                        c=self.c,
                    )
        return reg_loss


# Define Bidirectional GRU model
class BiGRU(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.loss_fn = torch.nn.MSELoss()

    def load_data(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Convert to PyTorch tensors
        self.X_train = (
            torch.from_numpy(self.X_train)
            .float()
            .to(self.device, non_blocking=True)
        )
        self.X_test = (
            torch.from_numpy(self.X_test)
            .float()
            .to(self.device, non_blocking=True)
        )
        self.y_train = (
            torch.from_numpy(self.y_train)
            .float()
            .to(self.device, non_blocking=True)
        )
        self.y_test = (
            torch.from_numpy(self.y_test)
            .float()
            .to(self.device, non_blocking=True)
        )

    def Build(self, hiddendim, learningRate=0.001, weight_decay=0.0001):
        inputdim = self.X_train.shape[2]
        outputsize = self.y_train.shape[2]

        self.input_dim = inputdim
        self.hidden_dim = hiddendim
        self.out_dim = outputsize
        self.gru = nn.GRU(
            self.input_dim,
            self.hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        ).to(self.device, non_blocking=True)
        self.fc_out = nn.Linear(self.hidden_dim * 2, self.out_dim).to(
            self.device, non_blocking=True
        )
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learningRate, weight_decay=weight_decay
        )

        # Set up the cosine learning rate scheduler
        lr_min = learningRate * 0.1
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=lr_min
        )

    def forward(self, xt):
        xt = xt.to(self.device, non_blocking=True)
        gruoutput, h = self.gru(xt)
        velocity = self.fc_out(gruoutput)
        return velocity

    def train_fit(self, N_epoches):
        """

        :param N_epoches:
        :param x_latent: train X
        :param V_train: train Y
        :param x_latent_cv: test X
        :param V_cv: test Y
        :return:
        """
        x_latent = self.X_train
        V_train = self.y_train
        x_latent_cv = self.X_test
        V_cv = self.y_test

        MSE_cv_linear_epoch = torch.empty([N_epoches])
        MSE_train_linear_epoch = torch.empty([N_epoches])

        for i in tqdm(range(N_epoches), "epochs"):
            v_predict_cv = self(x_latent_cv)
            MSE_cv_linear_epoch[i] = self.loss_fn(v_predict_cv, V_cv)

            self.train()

            output = self(x_latent)
            loss = self.loss_fn(output, V_train)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            MSE_train_linear_epoch[i] = loss

        return MSE_cv_linear_epoch, MSE_train_linear_epoch

    def predict_velocity(self, x_latent):
        self.eval()
        with torch.no_grad():
            v_predict, hidden_states = self.gru(x_latent)
            v_predict = self.fc_out(v_predict)

        v_predict = v_predict.cpu().detach().numpy()
        hidden_states = hidden_states.cpu().detach().numpy()
        return v_predict, hidden_states


# Define Kalman Filter + GRU Model
class InitGRU:
    def __init__(self, F, H, K0, m, n, T):
        """
        yt和xtt1目前这里都是仅考虑一个sample, 后续需要扩展为多个sample
        """
        self.F = F
        self.H = H
        self.K0 = K0
        self.m = m
        self.n = n
        self.T = T


class KalmanNetNN(nn.Module):
    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    def Build(self, ssModel):
        self.InitSystemDynamics(ssModel.F, ssModel.H, ssModel.K0)

        # Number of neurons in the 1st hidden layer
        H1_KNet = (
            (ssModel.m + ssModel.n) * 10 * 8
        )  # 第一层的neuron 个数 (GRU前) 原(m + n) * (10) * 8

        # Number of neurons in the 2nd hidden layer
        H2_KNet = (
            ssModel.m * ssModel.n
        ) * 4  # 第二层的neuron 个数 (GRU后) 原 (m * n) * 1 * (4)

        self.InitKGainNet(
            H1_KNet, H2_KNet
        )  # Init网络结构, 两个hidden layer neuro个数需要自己定义

    def InitSystemDynamics(self, F, H, K0):
        # Set State Evolution Matrix
        self.F = F.to(self.device, non_blocking=True)
        self.F_T = torch.transpose(F, 0, 1)  # 有啥用?
        self.m = self.F.size()[0]

        # Set Observation Matrix
        self.H = H.to(self.device, non_blocking=True)
        self.H_T = torch.transpose(H, 0, 1)  # 有啥用?
        self.n = self.H.size()[0]

        # Set KalmanGain at t0
        self.KGain = K0.to(self.device, non_blocking=True)

    def InitKGainNet(self, H1, H2):
        # Input Dimensions
        D_in = (
            self.m + self.n
        )  # x(t-1), y(t)   #在这里增加KG的维度(mn) self.m + self.n + self.m*self.n

        # Output Dimensions
        D_out = self.m * self.n  # Kalman Gain

        ###################
        ### Input Layer ###
        ###################
        # Linear Layer
        self.KG_l1 = torch.nn.Linear(D_in, H1, bias=True).to(
            self.device, non_blocking=True
        )

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu1 = torch.nn.ReLU().to(self.device, non_blocking=True)

        ###########
        ### GRU ###
        ###########
        # Input Dimension
        self.input_dim = H1
        # Hidden Dimension
        self.hidden_dim = (
            1000  # GRU 的hidden dimension; 原来是(m^2+n^2)*10, out of memory!
        )
        # Number of Layers
        self.n_layers = 1
        # Batch Size
        self.batch_size = 1
        # Input Sequence Length
        self.seq_len_input = 1
        # Hidden Sequence Length
        self.seq_len_hidden = self.n_layers

        # batch_first = False
        # dropout = 0.1 ;

        # Initialize a Tensor for GRU Input
        # self.GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim)

        # Initialize a Tensor for Hidden State
        self.hn = torch.randn(
            self.seq_len_hidden, self.batch_size, self.hidden_dim
        ).to(self.device, non_blocking=True)

        # Iniatialize GRU Layer
        self.rnn_GRU = nn.GRU(
            self.input_dim, self.hidden_dim, self.n_layers
        ).to(self.device, non_blocking=True)

        ####################
        ### Hidden Layer ###
        ####################
        self.KG_l2 = torch.nn.Linear(self.hidden_dim, H2, bias=True).to(
            self.device, non_blocking=True
        )

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu2 = torch.nn.ReLU().to(self.device, non_blocking=True)

        ####################
        ### Output Layer ###
        ####################
        self.KG_l3 = torch.nn.Linear(H2, D_out, bias=True).to(
            self.device, non_blocking=True
        )

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0):  # 传入 xtt1_0这个在pipeline里trian的时候用到
        self.m1x_prior = M1_0.to(self.device, non_blocking=True)

        self.m1x_posterior = M1_0.to(self.device, non_blocking=True)

        self.state_process_posterior_0 = M1_0.to(self.device, non_blocking=True)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):
        # Compute the 1-st moment of x based on model knowledge and without process noise
        self.state_process_prior_0 = torch.matmul(
            self.F, self.state_process_posterior_0
        )

        # Compute the 1-st moment of y based on model knowledge and without noise
        self.obs_process_0 = torch.matmul(self.H, self.state_process_prior_0)

        # Predict the 1-st moment of x
        self.m1x_prev_prior = self.m1x_prior
        self.m1x_prior = torch.matmul(self.F, self.m1x_posterior)

        # Predict the 1-st moment of y
        self.m1y = torch.matmul(self.H, self.m1x_prior)

        ##############################

    ### Kalman Gain Estimation ###      这步第二步调用, 被knetstep调用
    ##############################
    def step_KGain_est(self, y):  # 传入KG
        # Reshape and Normalize the difference in X prior
        # Featture 4: x_t|t - x_t|t-1
        # dm1x = self.m1x_prior - self.state_process_prior_0
        dm1x = self.m1x_posterior - self.m1x_prev_prior
        dm1x_reshape = torch.squeeze(dm1x)
        dm1x_norm = func.normalize(
            dm1x_reshape, p=2, dim=0, eps=1e-12, out=None
        )  # 这个方法查下, 这里传入gru做了norm

        # Feature 2: yt - y_t+1|t
        dm1y = y - torch.squeeze(self.m1y)
        dm1y_norm = func.normalize(
            dm1y, p=2, dim=0, eps=1e-12, out=None
        )  # 这个方法查下, 这里传入gru做了norm

        """
        # Feature 3: Kalman Gain t-1
        !上一步的kg
        ! 这里用 self.KGain, 需要reshape为一维, 底下拼起来, 可能还需要normalize
        !
        """

        # KGain Net Input
        KGainNet_in = torch.cat([dm1y_norm, dm1x_norm], dim=0)  # 这里可能需要拼KGain

        # Kalman Gain Network Step
        KG = self.KGain_step(KGainNet_in)

        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.m, self.n))

    #######################
    ### Kalman Net Step ###   这步最先被forward函数调用,
    #######################
    def KNet_step(self, y):  # 传入真y (spikes)
        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Innovation
        y_obs = y
        dy = y_obs - self.m1y

        # Compute the 1-st posterior moment
        INOV = torch.matmul(self.KGain, dy)
        self.m1x_posterior = self.m1x_prior + INOV

        # return
        return torch.squeeze(self.m1x_posterior)

    ########################
    ### Kalman Gain Step ###    这步最后被kgain est步调用
    ########################
    def KGain_step(self, KGainNet_in):  # FC+GRU+FC模块, 算出来ht
        ###################
        ### Input Layer ###
        ###################
        L1_out = self.KG_l1(KGainNet_in)
        La1_out = self.KG_relu1(L1_out)

        ###########
        ### GRU ###
        ###########
        GRU_in = torch.empty(
            self.seq_len_input, self.batch_size, self.input_dim
        ).to(self.device, non_blocking=True)
        GRU_in[0, 0, :] = La1_out
        GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn)
        GRU_out_reshape = torch.reshape(GRU_out, (1, self.hidden_dim))

        ####################
        ### Hidden Layer ###
        ####################
        L2_out = self.KG_l2(GRU_out_reshape)
        La2_out = self.KG_relu2(L2_out)

        ####################
        ### Output Layer ###
        ####################
        L3_out = self.KG_l3(La2_out)
        return L3_out

    ###############
    ### Forward ###
    ###############
    def forward(self, yt):
        yt = yt.to(self.device, non_blocking=True)
        return self.KNet_step(yt)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(
            self.n_layers, self.batch_size, self.hidden_dim
        ).zero_()
        self.hn = hidden.data


if __name__ == "__main__":
    #############################TEST dPCA CODE##################################
    # Dummy data for illustration purposes
    # Replace this with your actual data
    # number of neurons, time-points and stimuli
    N, T, S = 120, 2000, 3

    # noise-level and number of trials in each condition
    noise, n_samples = 3, 30

    # build two latent factors
    zt = np.arange(T) / float(T)
    zs = np.arange(S) / float(S)

    # build trial-by trial data
    trialR = noise * random.randn(n_samples, N, S, T)
    time_component = (
        random.randn(N)[None, :, None, None] * zt[None, None, None, :]
    )
    stimulus_component = (
        random.randn(N)[None, :, None, None] * zs[None, None, :, None]
    )
    trialR += time_component
    trialR += stimulus_component

    # trial-average data
    R = np.mean(trialR, 0)

    # center data
    R -= np.mean(R.reshape((N, -1)), 1)[:, None, None]

    dpca = dPCA.dPCA(labels="st", regularizer="auto")
    dpca.protect = ["t"]

    Z = dpca.fit_transform(R, trialR)

    time = np.arange(T)

    plt.figure(figsize=(16, 7))
    plt.subplot(131)

    for s in range(S):
        plt.plot(time, Z["t"][0, s])

    plt.title("1st time component", fontsize=20)

    plt.subplot(132)

    for s in range(S):
        plt.plot(time, Z["s"][0, s])

    plt.ylim([np.amin(Z["s"]) - 1, np.amax(Z["s"]) + 1])

    plt.title("1st stimulus component", fontsize=20)

    plt.subplot(133)

    for s in range(S):
        plt.plot(time, Z["st"][0, s])

    dZ = np.amax(Z["st"]) - np.amin(Z["st"])

    plt.ylim([np.amin(Z["st"]) - dZ / 10.0, np.amax(Z["st"]) + dZ / 10.0])

    plt.title("1st mixing component", fontsize=20)
    plt.legend(["Left", "Right", "Nothing"], fontsize=15)
    plt.xlabel("Time (ms)", fontsize=15)
    plt.ylabel("dPCA axis", fontsize=15)
    plt.show()

    # reshape to [n_samples * n_timepoints * n_neurons]
    X = trialR.transpose(0, 2, 3, 1).reshape(-1, T, N)
    # create labels as [0,1,2,0,1,2,...]
    Y = np.tile([0, 1, 2], 30)

    X_transformed = dpca.transform(X.transpose(2, 0, 1))["s"]
    # Plot the first component of the transformed data
    X_transformed_to_plot = X_transformed[0, :, :]
    original_data_to_plot = X[0, :, :]

    # Plot the transformed data and first three trials of the transformed data in a separate subplot
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.imshow(X_transformed_to_plot, aspect="auto")
    ax1.set_title("Transformed data")
    ax2.plot(X_transformed_to_plot[0, :])
    ax2.plot(X_transformed_to_plot[1, :])
    ax2.plot(X_transformed_to_plot[2, :])
    ax2.set_title("First three trials of the transformed data")
    plt.show()

    # Plot the original data and first three trials of the original data in a separate subplot
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.imshow(original_data_to_plot, aspect="auto")
    ax1.set_title("Original data")
    ax2.plot(original_data_to_plot[0, :])
    ax2.plot(original_data_to_plot[1, :])
    ax2.plot(original_data_to_plot[2, :])
    ax2.set_title("First three trials of the original data")
    plt.show()

    # Transpose the last two dimensions to match the PyTorch convention
    X_transformed = X_transformed.transpose(1, 2, 0)

    #############################TEST LSTM CODE##################################
    # Shuffle and split data and label
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2023)
    for train_index, test_index in sss.split(X_transformed, Y):
        X_transformed_train, X_transformed_test = (
            X_transformed[train_index],
            X_transformed[test_index],
        )
        y_train, y_test = Y[train_index], Y[test_index]
        X_train, X_test = X[train_index], X[test_index]

    # Hyperparameters
    input_dim = N
    output_dim = S

    model = LSTMClassifier(input_dim, output_dim)
    model.load_data(X_train, X_test, y_train, y_test)
    (
        epoch_accuracies,
        epoch_precision,
        epoch_recall,
        epoch_f1,
    ) = model.train_test_using_optimizer(epoches=100)

    model = LSTMClassifier(10, output_dim)
    model.load_data(X_transformed_train, X_transformed_test, y_train, y_test)
    (
        trans_epoch_accuracies,
        trans_epoch_precision,
        trans_epoch_recall,
        trans_epoch_f1,
    ) = model.train_test_using_optimizer(epoches=100)

    # Plot accuracy over epochs
    plt.plot(epoch_accuracies, "b", label="Original data")
    plt.plot(trans_epoch_accuracies, "r", label="Transformed data")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy for transformed data Over Epochs")
    plt.legend()
    plt.show()

    # Plot precision over epochs
    plt.plot(epoch_precision, "b", label="Original data")
    plt.plot(trans_epoch_precision, "r", label="Transformed data")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Precision for transformed data Over Epochs")
    plt.legend()
    plt.show()

    # Plot recall over epochs
    plt.plot(epoch_recall, "b", label="Original data")
    plt.plot(trans_epoch_recall, "r", label="Transformed data")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Recall for transformed data Over Epochs")
    plt.legend()
    plt.show()

    # Plot f1 over epochs
    plt.plot(epoch_f1, "b", label="Original data")
    plt.plot(trans_epoch_f1, "r", label="Transformed data")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("F1 for transformed data Over Epochs")
    plt.legend()
    plt.show()
