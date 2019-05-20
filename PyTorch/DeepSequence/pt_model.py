from __future__ import print_function

import numpy as np
from scipy.special import erfinv, gammaln
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_tensor_environment(float_size, random_seed=False):
    """
    used to ensure GPU & float-size compatibility throughout model.
    :param float_size: integer 16, 32, or 64 to set byte size
    :param random_seed: optional setting of random seed
    """
    if torch.cuda.is_available(): # if there's a GPU, make cuda tensors
        if float_size == 16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        elif float_size == 32:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        elif float_size == 64:
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else: # otherwise make normal cpu tensors
        if float_size == 16:
            torch.set_default_tensor_type(torch.HalfTensor)
        elif float_size == 32:
            torch.set_default_tensor_type(torch.FloatTensor)
        elif float_size == 64:
            torch.set_default_tensor_type(torch.DoubleTensor)
    if random_seed:
        torch.manual_seed(random_seed)


class Encoder(nn.Module):
    """
    Encoder class shared by SVI and MLE VAEs, inherits from nn.Module so as to share access
    with VAE class. initialization defines architecture, and the forward pass through this
    architecture is specified in forward().

    N.B. capabilities beyond two hidden layers to be added.
    """
    def __init__(self, hidden_architecture, z_dim, convolve_encoder, conv_encoder_size,
                 alph_size, seq_len, nonlinearity):
        """
        initialize key parameters and architecture
        :param hidden_architecture: length-2 list detailing sizes of the two hidden layers
        :param z_dim: size of latent dimension
        :param convolve_encoder: whether to perform 1d convolution on input
        :param conv_encoder_size: specify size of above convolution
        :param alph_size: alphabet size of sequence alignment
        :param seq_len: sequence length of sequence alignment
        :param nonlinearity: what non-linear function to use
        """
        super(Encoder, self).__init__()
        # parameters used in forward()
        self.seq_len = seq_len
        self.convolve = convolve_encoder
        self.alph_size = alph_size

        # create architecture
        if convolve_encoder:
            self.conv_layer = nn.Conv1d(alph_size,conv_encoder_size,kernel_size=1,stride=1,bias=False)
            self.channel_size = conv_encoder_size
        else:
            self.channel_size = alph_size # setting channel size allows subsequent architecture consistency
        self.hidden1 = nn.Linear((self.channel_size*seq_len),hidden_architecture[0])
        self.hidden2 = nn.Linear(hidden_architecture[0],hidden_architecture[1])
        self.final1 = nn.Linear(hidden_architecture[1],z_dim, bias=True)
        self.final2 = nn.Linear(hidden_architecture[1],z_dim, bias=True)

        # perform glorot normal initialisation of weights
        # initialise biases to 0.1 as per original model.
        if convolve_encoder:
            nn.init.xavier_normal_(self.conv_layer.weight)
        for layer in (self.hidden1,self.hidden2):
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.1)
        for layer in (self.final1,self.final2):
            nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(self.final1.bias, 0.1)
        nn.init.constant_(self.final2.bias, -5) # -5 as per logsig_init of full model

        # set up non-linearity
        if nonlinearity == 'relu':
            self.nonlinear = nn.ReLU()
        elif nonlinearity == 'tanh':
            self.nonlinear = nn.Tanh()
        elif nonlinearity == 'sigmoid':
            self.nonlinear = nn.Sigmoid()
        elif nonlinearity == 'elu':
            self.nonlinear = nn.ELU()

    def forward(self, x):
        """
        forward pass of encoder network
        :param x: input data
        :return: z_loc - mu variable for latent distribution, z_logsig - log-sigma variable for latent dis.
        """
        if self.convolve:
            x = x.permute(0,2,1) # arrange x to have alphabet dimension convolved
            x_conv = self.conv_layer(x)
            x_reshaped = x_conv.view(-1,self.seq_len*self.channel_size) # rearrange for model architecture
        else:
            x_reshaped = x.view(-1,self.seq_len*self.channel_size) # no conv; just rearrange
        hidden1 = self.nonlinear(self.hidden1(x_reshaped))
        hidden2 = self.nonlinear(self.hidden2(hidden1))
        z_loc = self.final1(hidden2)
        z_logsig = self.final2(hidden2)
        return z_loc, z_logsig


class DecoderMLE(nn.Module):
    """
    Decoder class for the maximum likelihood estimation version of the VAE model. Initialization
    defines architecture, and the forward pass through this architecture is specified in forward().

    N.B. capabilities beyond two hidden layers to be added.
    """
    def __init__(self, hidden_architecture, z_dim, convolve_patterns, conv_size,
                 alph_size, seq_len, nonlinearity, final_nonlinearity, sparsity,
                 final_pwm_scale, dropout, n_patterns, output_bias):
        """
        :param hidden_architecture: 2-length list of hidden layer sizes
        :param z_dim: latent dimension size
        :param convolve_patterns: whether to perform 1D convolution on output layer
        :param conv_size: the size of above convolution
        :param alph_size: size of alphabet of sequence alignment
        :param seq_len: length of sequence in sequence alignment
        :param nonlinearity: the first hidden layer non-linearity to use
        :param final_nonlinearity: the second hidden layer non-linearity to use
        :param sparsity: the type of structured sparsity argument to use.
        :param final_pwm_scale: whether to include temperature parameter
        :param dropout: whether to include dropout
        :param n_patterns: number of times to tile sparsity weights over output
        :param output_bias: whether to include a bias in the output layer
        """
        super(DecoderMLE, self).__init__()
        # parameters used in forward()
        self.seq_len = seq_len
        self.alph_size = alph_size
        self.convolve = convolve_patterns
        self.sparsity = sparsity
        self.final_output_size = hidden_architecture[-1]
        self.n_patterns = n_patterns

        # create hidden architecture
        self.hidden1 = nn.Linear(z_dim, hidden_architecture[0])
        self.hidden2 = nn.Linear(hidden_architecture[0], hidden_architecture[1])
        # if true, final layer convolves from conv_size to alph_size for output
        if convolve_patterns:
            self.final = nn.Linear(hidden_architecture[1], conv_size*seq_len, bias=output_bias)
            self.conv_layer = nn.Conv1d(conv_size, alph_size, kernel_size=1, stride=1)
            self.channel_size = conv_size
        # otherwise, final layer goes directly to alph_size
        else:
            self.final = nn.Linear(hidden_architecture[1], (alph_size*seq_len), bias=output_bias)
            self.channel_size = alph_size
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        # create sparsity parameter which will tile over output weights.
        if sparsity: # create parameter sized such that it can tile over final layer
            self.scale_weight = nn.Parameter(torch.randn(int(hidden_architecture[-1]/n_patterns),seq_len))
        # inverse temperature parameter
        if final_pwm_scale:
            self.final_pwm_scale = nn.Parameter(torch.ones(1))
        else:
            self.final_pwm_scale = False

        # with architecture specified, perform glorot & bias=0.1 initialisation, as per original model.
        if convolve_patterns:
            nn.init.xavier_normal_(self.conv_layer.weight)
            nn.init.constant_(self.conv_layer.bias, 0.1)
        for layer in (self.hidden1, self.hidden2):
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.1)
        nn.init.xavier_normal_(self.final.weight)
        if output_bias:
            nn.init.constant_(self.final.bias, 0.1)
        if sparsity:
            nn.init.xavier_normal_(self.scale_weight)

        # set up non-linearity functions
        if nonlinearity == 'relu':
            self.nonlinear = nn.ReLU()
        elif nonlinearity == 'tanh':
            self.nonlinear = nn.Tanh()
        elif nonlinearity == 'sigmoid':
            self.nonlinear = nn.Sigmoid()
        elif nonlinearity == 'elu':
            self.nonlinear = nn.ELU()
        # including the final layer
        if final_nonlinearity == 'relu':
            self.final_nonlinear = nn.ReLU()
        elif final_nonlinearity == 'tanh':
            self.final_nonlinear = nn.Tanh()
        elif final_nonlinearity == 'sigmoid':
            self.final_nonlinear = nn.Sigmoid()
        elif final_nonlinearity == 'elu':
            self.final_nonlinear = nn.ELU()

    def forward(self, x, z):
        """
        :param x: original x input, used to calculate logpxz
        :param z: latent variable
        :return: x_recon - reconstruction of x, logpxz - value for log(P(x|z)), out - model output patterns
        """
        hidden1 = self.nonlinear(self.hidden1(z))
        hidden2 = self.final_nonlinear(self.hidden2(hidden1))
        if self.sparsity:
            scale_tiled = self.scale_weight.repeat(self.n_patterns,1)
            scale_unsqueezed = torch.unsqueeze(scale_tiled,2) # add third dimension for dim compatability
            if self.sparsity == 'logit':
                weight_permute = self.final.weight.permute(1,0)
                sparse_weight = weight_permute.view(self.final_output_size, self.seq_len, self.channel_size) \
                                * torch.sigmoid(scale_unsqueezed)
                self.final.weight = nn.Parameter(sparse_weight) # return this as the original parameter
            else:
                weight_permute = self.final.weight.permute(1, 0)
                sparse_weight = weight_permute.view(self.final_output_size, self.seq_len,self.channel_size) \
                                * torch.exp(scale_unsqueezed)
                self.final.weight = nn.Parameter(sparse_weight)
            final_sparse_weight = self.final.weight.view(self.final_output_size, self.seq_len \
                                                         * self.channel_size)
            self.final.weight = nn.Parameter(final_sparse_weight.permute(1,0))
        if self.convolve:
            final = self.final(hidden2)
            final = final.view(-1,self.channel_size,self.seq_len)
            out = self.conv_layer(final)
        else:
            final = self.final(hidden2)
            out = final.view(-1,self.channel_size,self.seq_len)

        x_recon_unnorm = out.permute(0,2,1) # return to original dimensions

        if self.final_pwm_scale: # apply temperative as strictly positive parameter.
            x_recon_unnorm = x_recon_unnorm * torch.log(1.0+torch.exp(self.final_pwm_scale[0]))

        # softmax over amino acids
        e_x = torch.exp(x_recon_unnorm - x_recon_unnorm.max(dim=2, keepdim=True)[0])
        x_recon = e_x / e_x.sum(dim=2, keepdim=True)

        xdev = x_recon_unnorm - x_recon_unnorm.max(dim=2, keepdim=True)[0]
        log_softmax = xdev - torch.log(torch.sum(torch.exp(xdev), dim=2, keepdim=True)[0])
        logpxz = torch.sum(torch.sum((x*log_softmax), dim=-1), dim=-1)
        return x_recon, logpxz, out

class DecoderSVI(nn.Module):
    """
    Decoder that allows stochastic variational inference with variational approximation of parameters.

    N.B. capabilities beyond two hidden layers to be added."""
    def __init__(self, hidden_architecture, z_dim, convolve_patterns, conv_size,
                 alph_size, seq_len, nonlinearity, final_nonlinearity, sparsity,
                 final_pwm_scale, n_patterns, output_bias, logsig_init, pattern_sigma, rng):
        """
        :param hidden_architecture: list of hidden layer sizes
        :param z_dim: latent dimension size
        :param convolve_patterns: whether to perform output convolution
        :param conv_size: size of above convolution
        :param alph_size: alphabet size of data
        :param seq_len: sequence length of data
        :param nonlinearity: first hidden layer non linear function
        :param final_nonlinearity: final hidden layer non linear function
        :param sparsity: the type of sparsity to be used
        :param final_pwm_scale: whether to apply temperature parameter
        :param n_patterns: how many times to tile sparsity parameters over output
        :param output_bias: whether to include bias in output
        :param logsig_init: the initial value used for logsigma initialization
        :param pattern_sigma: the initial value to set for prior sigma for final patterns
        :param rng: the random number generator inherited from VAE for sampler function
        """
        super(DecoderSVI, self).__init__()
        self.seq_len = seq_len
        self.alph_size = alph_size
        self.convolve = convolve_patterns
        self.sparsity = sparsity
        self.final_output_size = hidden_architecture[-1]
        self.n_patterns = n_patterns
        self.output_bias = output_bias
        self.final_pwm_scale = final_pwm_scale
        self.srng = rng
        self.variational_param_identifiers = []
        self.variational_param_name_to_sigma = {}

        # create hidden architecture
        """NB: These architectures serve only as containers for weights, biases and their gradients"""
        self.mu_h1 = nn.Linear(z_dim, hidden_architecture[0])
        self.ls_h1 = nn.Linear(z_dim, hidden_architecture[0]) # ls = logsig
        self.variational_param_identifiers.extend(['h1.weight','h1.bias'])
        self.variational_param_name_to_sigma['h1'] = 1.0
        self.mu_h2 = nn.Linear(hidden_architecture[0],hidden_architecture[1])
        self.ls_h2 = nn.Linear(hidden_architecture[0],hidden_architecture[1])
        self.variational_param_identifiers.extend(['h2.weight','h2.bias'])
        self.variational_param_name_to_sigma['h2'] = 1.0
        if convolve_patterns:
            self.mu_fn = nn.Linear(hidden_architecture[1], conv_size*seq_len, bias=output_bias) #fn=final layer
            self.ls_fn = nn.Linear(hidden_architecture[1], conv_size*seq_len, bias=output_bias)
            self.mu_cv = nn.Conv1d(conv_size, alph_size, kernel_size=1, stride=1, bias=False)
            self.ls_cv = nn.Conv1d(conv_size, alph_size, kernel_size=1, stride=1, bias=False) # cv = conv layer
            self.variational_param_identifiers.extend(['cv.weight'])
            self.variational_param_name_to_sigma['cv'] = 1.0
            self.channel_size = conv_size
        else:
            self.mu_fn = nn.Linear(hidden_architecture[1], alph_size*seq_len, bias=output_bias)
            self.ls_fn = nn.Linear(hidden_architecture[1], alph_size*seq_len, bias=output_bias)
            self.channel_size = alph_size
        self.variational_param_identifiers.append('fn.weight')
        if output_bias:
            self.variational_param_identifiers.append('fn.bias')
        self.variational_param_name_to_sigma['fn'] = pattern_sigma
        if sparsity:
            self.scale_mu = nn.Parameter(torch.zeros(int(hidden_architecture[-1]/n_patterns),seq_len))
            self.scale_ls = nn.Parameter(logsig_init*
                                         torch.ones(int(hidden_architecture[-1]/n_patterns),seq_len))
        if final_pwm_scale:
            self.mu_pw = nn.Parameter(torch.ones(1))
            self.ls_pw = nn.Parameter(-5*torch.ones(1))
            self.variational_param_identifiers.append('pw')
            self.variational_param_name_to_sigma['pw'] = 1.0

        # with architecture specified, perform glorot & bias=0.1 initialisation, as per original model.
        if convolve_patterns:
            nn.init.xavier_normal_(self.mu_cv.weight)
            nn.init.constant_(self.ls_cv.weight, logsig_init)
        for layer in (self.mu_h1, self. mu_h2):
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.1)
        for layer in (self.ls_h1, self.ls_h2):
            nn.init.constant_(layer.weight, logsig_init)
            nn.init.constant_(layer.bias, logsig_init)
        nn.init.xavier_normal_(self.mu_fn.weight)
        nn.init.constant_(self.ls_fn.weight, logsig_init)
        if output_bias:
            nn.init.constant_(self.mu_fn.bias, 0.1)
            nn.init.constant_(self.ls_fn.bias, logsig_init)

        # set up non-linearity functions
        if nonlinearity == 'relu':
            self.nonlinear = nn.ReLU()
        elif nonlinearity == 'tanh':
            self.nonlinear = nn.Tanh()
        elif nonlinearity == 'sigmoid':
            self.nonlinear = nn.Sigmoid()
        elif nonlinearity == 'elu':
            self.nonlinear = nn.ELU()
        # including the final layer
        if final_nonlinearity == 'relu':
            self.final_nonlinear = nn.ReLU()
        elif final_nonlinearity == 'tanh':
            self.final_nonlinear = nn.Tanh()
        elif final_nonlinearity == 'sigmoid':
            self.final_nonlinear = nn.Sigmoid()
        elif final_nonlinearity == 'elu':
            self.final_nonlinear = nn.ELU()

    def sampler(self, mu, logsig):
        """
        reparameterised sampling, redefined from full VAE, inheriting the VAE's rng.
        :param mu: mu value
        :param logsig: logsigma value
        :return: gaussian sample defined by above arguments
        """
        eps = self.srng(mu.shape)
        z = mu + torch.exp(logsig) * eps
        return z

    def forward(self, x, z):
        """
        forward pass; all weights and parameters are sampled from the actual model weight containers
        :param x: original x input, used to calculate logpxz
        :param z: latent variable
        :return: x_recon - reconstruction of x, logpxz - value for log(P(x|z)), out - model output patterns
        """
        h1_W = self.sampler(self.mu_h1.weight, self.ls_h1.weight)
        h1_b = self.sampler(self.mu_h1.bias, self.ls_h1.bias)
        hidden1 = self.nonlinear(F.linear(z, weight=h1_W, bias=h1_b))

        h2_W = self.sampler(self.mu_h2.weight, self.ls_h2.weight)
        h2_b = self.sampler(self.mu_h2.bias, self.ls_h2.bias)
        hidden2 = self.final_nonlinear(F.linear(hidden1, weight=h2_W, bias=h2_b))

        fn_W = self.sampler(self.mu_fn.weight, self.ls_fn.weight)
        if self.output_bias:
            fn_b = self.sampler(self.mu_fn.bias, self.ls_fn.bias)

        if self.sparsity:
            scale_weights = self.sampler(self.scale_mu,self.scale_ls)
            scale_tiled = scale_weights.repeat(self.n_patterns, 1)
            scale_unsqueezed = torch.unsqueeze(scale_tiled,2) # add third dimension
            if self.sparsity == 'logit':
                weight_permute = fn_W.permute(1,0)
                fn_W = weight_permute.view(self.final_output_size, self.seq_len, self.channel_size) \
                    * torch.sigmoid(scale_unsqueezed)
            else:
                weight_permute = fn_W.permute(1,0)
                fn_W = weight_permute.view(self.final_output_size, self.seq_len, self.channel_size) \
                    * torch.exp(scale_unsqueezed)
            fn_W = fn_W.view(self.final_output_size, self.seq_len * self.channel_size).permute(1, 0)
        if self.convolve:
            cv_W = self.sampler(self.mu_cv.weight, self.ls_cv.weight)
            if self.output_bias:
                final = F.linear(hidden2, weight=fn_W, bias=fn_b)
                final = final.view(-1, self.channel_size, self.seq_len)
                out = F.conv1d(final, cv_W)
            else:
                final = F.linear(hidden2, weight=fn_W)
                final = final.view(-1, self.channel_size, self.seq_len)
                out = F.conv1d(final, cv_W)
        else:
            if self.output_bias:
                final = F.linear(hidden2, weight=fn_W, bias=fn_b)
            else:
                final = F.linear(hidden2, weight=fn_W)
            out = final.view(-1,self.channel_size,self.seq_len) # ensure output in correct dimension

        x_recon_unnorm = out.permute(0, 2, 1)
        if self.final_pwm_scale:
            pwm_scale = self.sampler(self.mu_pw, self.ls_pw)[0]
            x_recon_unnorm = x_recon_unnorm * torch.log(1 + torch.exp(pwm_scale))

        # softmax over amino acids
        e_x = torch.exp(x_recon_unnorm - x_recon_unnorm.max(dim=2, keepdim=True)[0])
        x_recon = e_x / e_x.sum(dim=2, keepdim=True)

        # Numerically stable softmax using logsumexp trick
        xdev = x_recon_unnorm - x_recon_unnorm.max(dim=2, keepdim=True)[0]
        log_softmax = xdev - torch.log(torch.sum(torch.exp(xdev), dim=2, keepdim=True)[0])
        logpxz = torch.sum(torch.sum((x*log_softmax), dim=-1), dim=-1)
        return x_recon, logpxz, out


class VAE_MLE(nn.Module):
    """
    Maximum likehood estimation variational autoencoder. Constructs full model and forward pass,
    serves as a callable object for model training.
    """
    def __init__(self,
            data,
            encoder_architecture=[1500,1500],
            decoder_architecture=[100,500],
            n_latent=2,
            n_patterns=4,
            batch_size=100,
            encode_nonlinearity_type="relu",
            decode_nonlinearity_type="relu",
            final_decode_nonlinearity="sigmoid",
            global_scale=1.0,
            convolve_encoder=False,
            convolve_patterns=True,
            conv_decoder_size=10,
            conv_encoder_size=10,
            warm_up=0.0,
            output_bias=True,
            final_pwm_scale=False,
            working_dir=".",
            learning_rate=0.001,
            random_seed=42,
            sparsity_lambda=0.0,
            l2_lambda=0.0,
            sparsity='logit',
            kl_scale=1.0,
            logit_p=0.01,
            logit_sigma=4.0,
            dropout=False,
            float_size = 32):
        """
        :param data: instance of DataHelper class with loaded data
        :param encoder_architecture: list detailing size of encoder hidden layers
        :param decoder_architecture: list detailing size of decoder hidden layers
        :param n_latent: size of latent dimension
        :param n_patterns: how many times to tile sparsity parameters over output
        :param batch_size: size of mini-batch to use
        :param encode_nonlinearity_type: non-linearity used in encoder layers
        :param decode_nonlinearity_type: non-linearity used in decoder's first layers
        :param final_decode_nonlinearity: non-linearity used in decoder's final layer
        :param global_scale: global scale parameter used for sparsity, default 1. Used in
                analytic, laplacian, horseshoe, ARD sparsity priors only.
        :param convolve_encoder: whether to perform width-1 1D convolution on input.
        :param convolve_patterns: whether to perform width-1 1D convolution on output,
                also known as a dictionary
        :param conv_decoder_size: size of decoder convolution/dictionary
        :param conv_encoder_size: size of encoder convolution
        :param warm_up: number of updates for which an annealing procedure should be applied to
                the KL divergence loss; this lessens the importance of KL divergence in early updates for
                greater stability.
        :param output_bias: whether to include a bias on the output weights
        :param final_pwm_scale: whether to use a temperature parameter
        :param working_dir: directory for saving and loading parameters
        :param learning_rate: Adam learning rate
        :param random_seed: initialisation seed for random number generators
        :param sparsity_lambda: Regularization strength of sparsity parameters
        :param l2_lambda: Regularization strength of decoder parameters
        :param sparsity: Sparsity type, using a noncentered reparameterization.
                Options include: logit, analytic, laplacian, horseshoe, ard
                See Ingraham and Marks, 2016 (https://arxiv.org/abs/1602.03807)
        :param kl_scale: Scale of KL of latent variables, default 1.0
                Scale < 1.0 approaches a normal autoencoder
                Scale > 1.0 turns into beta-autoencoder (Higgins et al, 2016)
        :param logit_p: Global scale prior for logit sparsity
        :param logit_sigma: Prior sigma for scale prior for logit sparsity
        :param dropout: Include dropout on the decoder (probability set to 0.5)
        :param float_size: 16, 32 or 64 for Torch Tensor size.
        """
        super(VAE_MLE, self).__init__()
        # SET UP TENSOR ENVIRONMENT
        set_tensor_environment(float_size, random_seed)
        if float_size == 16:
            self.epsilon = 1e-6
            self.dtype = torch.float16
        elif float_size == 32:
            self.epsilon = 1e-8
            self.dtype = torch.float32
        elif float_size == 64:
            self.epsilon = 1e-8
            self.dtype = torch.float64
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # key parameters:
        self.type = "MLE"
        self.working_dir = working_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seq_len = data.seq_len
        self.alphabet_size = data.alphabet_size
        self.convolve_patterns = convolve_patterns
        self.convolve_encoder = convolve_encoder
        self.final_pwm_scale = final_pwm_scale
        self.warm_up = torch.tensor(warm_up,device=self.device,dtype=self.dtype)
        self.srng = lambda *shape: torch.randn(*shape, device = self.device, dtype = self.dtype)
        self.n_latent = n_latent * n_patterns
        self.kl_scale = torch.tensor(kl_scale,device=self.device,dtype=self.dtype)
        self.global_scale = torch.tensor(global_scale, device=self.device, dtype=self.dtype)
        self.inv_global_scale = torch.tensor((1.0 / global_scale), device=self.device, dtype=self.dtype)
        self.sparsity_lambda = sparsity_lambda
        self.sparsity = sparsity
        self.l2_lambda = l2_lambda
        self.logit_p = logit_p
        self.logit_mu = torch.tensor(np.sqrt(2.0) * logit_sigma * erfinv(2.0 * logit_p - 1.0), device=self.device,
                                     dtype=self.dtype)
        self.logit_sigma = torch.tensor(logit_sigma, device=self.device, dtype=self.dtype)

        # define nn modules with given arguments:
        self.encoder = Encoder(encoder_architecture, self.n_latent, convolve_encoder, conv_encoder_size,
                               self.alphabet_size, self.seq_len, encode_nonlinearity_type)
        self.decoder = DecoderMLE(decoder_architecture, self.n_latent, convolve_patterns, conv_decoder_size,
                                  self.alphabet_size, self.seq_len, decode_nonlinearity_type,
                                  final_decode_nonlinearity,sparsity,final_pwm_scale,dropout,n_patterns, output_bias)

    def sampler(self, mu, logsig):
        """
        Samples from a diagonal Gaussian for stochastic variables. Applies reparameterization trick
        :param mu: sample mean
        :param logsig: sample log sigma
        :return: sampled variable z
        """
        eps = self.srng(mu.shape)
        z = mu + torch.exp(logsig) * eps
        return z

    def _anneal(self, update_num):
        """
        Anneal the KL if using annealing; reduces the KLD loss effect in early updates
        :param update_num: current update number
        :return: scaling factor for KLD loss
        """
        # If true, return first, else return second
        condition = update_num < self.warm_up
        KL_scale = torch.where(condition, update_num/self.warm_up,
                               torch.tensor(1.0, device=self.device,dtype=self.dtype))
        return KL_scale

    def update(self, logpxz, mu, logsig, update_num, Neff):
        """
        Calculate updated loss values with regularization and sparsity, returning various loss components.
        :param logpxz: Log(P(x|z)) value determined through forward pass of decoder.
        :param mu: mu of z determined through forward pass of encoder.
        :param logsig: log sigma of z determined through forward pass of encoder
        :param update_num: current update number used for annealing if required.
        :param Neff: effective sample N, used in loss calculations.
        :return: logpx_update - final ELBO loss value,
        torch.mean(logpxz) - value for reconstruction loss,
        regularization_loss,
        torch.mean(KLD_latent) - KL divergence loss of latent variables.
        """
        # calculate KLD of latent variables
        KLD_latent = 0.5 * torch.sum(1.0 + 2.0 * logsig - mu**2.0 \
            - torch.exp(2.0 * logsig), dim=1)

        # if sparsity used, calculate the sparsity loss depending on which sparsity prior specified.
        # include l2 regularization loss in the calculation.
        if self.sparsity:
            l2_loss = 0.0
            if self.l2_lambda > 0.0:
                l2_loss += 0.5*torch.sum(self.decoder.hidden1.weight*self.decoder.hidden1.weight)
                l2_loss += 0.5*torch.sum(self.decoder.hidden2.weight*self.decoder.hidden2.weight)
                if self.convolve_patterns:
                    l2_loss += 0.5*torch.sum(self.decoder.conv_layer.weight*self.decoder.conv_layer.weight)
                l2_loss += 0.5*torch.sum(self.decoder.final.weight*self.decoder.final.weight)
                l2_loss += 0.5*torch.sum(self.decoder.final_pwm_scale*self.decoder.final_pwm_scale)

            if self.sparsity == "logit":
                # Use a continuous relaxation of a spike and slab prior
                #    with a logit normit scale distribution
                group_sparsity_loss = - torch.sum((-0.5*torch.log(2.0*np.pi*self.logit_sigma**2.0))\
                                                  - ((self.decoder.scale_weight-self.logit_mu)**2.0\
                                                  / (2.0*(self.logit_sigma**2.))))
            elif self.sparsity == "analytic":
                # Use a moment-matched Gaussian approximation to the
                #   log-space Hyperbolic Secant hyperprior of the Horseshoe
                analytic_mu = torch.log(self.global_scale)
                analytic_sigma = np.pi / 2.0
                group_sparsity_loss = - torch.sum((-0.5*torch.log(2.*np.pi\
                    *(analytic_sigma**2.0)))
                    - ((self.decoder.scale_weight - analytic_mu)**2.0\
                        /(2.*(analytic_sigma**2.0))))
            else:
                # Estimate KL divergence for the sparsity
                # scale parameters (Fadeout) by sampling

                out_scale_exp = torch.exp(self.decoder.scale_weight)
                if self.sparsity == "horseshoe":
                    # Horsehoe sparsity has Half-Cauchy hyperprior
                    group_sparsity_loss = -torch.sum(torch.log(2.0) + torch.log(self.global_scale) \
                        - torch.log(np.pi) + torch.log(out_scale_exp) \
                        - torch.log(self.global_scale*self.global_scale \
                        + out_scale_exp * out_scale_exp))
                elif self.sparsity == "laplacian":
                    # Laplace sparsity has exponential hyperprior
                    group_sparsity_loss = -torch.sum(torch.log(2.0)\
                        + torch.log(self.inverse_global_scale)\
                        - self.inverse_global_scale * out_scale_exp\
                        * out_scale_exp + 2.0 * self.decoder.scale_weight)

                elif self.sparsity == "ard":
                    # Automatic Relevance Determination sparsity
                    #  has Inverse-Gamma hyperprior
                    group_sparsity_loss = -torch.sum(torch.log(2.0) + (self.global_scale \
                        * torch.log(self.global_scale)) \
                        - gammaln(self.global_scale) - (self.global_scale \
                        / ((out_scale_exp * out_scale_exp) + self.epsilon)) \
                        - (2.0 * self.global_scale * torch.log(out_scale_exp)))
        else:
            # without sparsity, just calculate the l2 regularisation loss
            l2_loss = 0.0
            if self.l2_lambda > 0.0:
                l2_loss += torch.sum(self.decoder.hidden1.weight*self.decoder.hidden1.weight)
                l2_loss += torch.sum(self.decoder.hidden2.weight*self.decoder.hidden2.weight)
                if self.convolve_patterns:
                    l2_loss += torch.sum(self.decoder.conv_layer.weight*self.decoder.conv_layer.weight)
                l2_loss += torch.sum(self.decoder.final.weight*self.decoder.final.weight)
                if self.decoder.final_pwm_scale:
                    l2_loss += torch.sum(self.decoder.final_pwm_scale*self.decoder.final_pwm_scale)

            group_sparsity_loss = 0.0
            if self.sparsity_lambda > 0.0:
                out_lasso = self.decoder.final.weight.view(self.seq_len, self.decoder.channel_size,
                                                           self.decoder.final_output_size).permute(2,0,1)
                group_sparsity_loss = torch.sum(torch.sqrt(torch.sum(out_lasso*out_lasso, dim=2)+self.epsilon))

        regularization_loss = (-(self.sparsity_lambda * group_sparsity_loss)\
            - (self.l2_lambda * l2_loss)) / Neff

        warm_up_scale = self._anneal(update_num)
        KLD_latent_update = KLD_latent * self.kl_scale
        # get final loss; negative because we want to maximize ELBO and Adam() minimizes
        logpx_update = -(torch.mean(logpxz + warm_up_scale * KLD_latent_update) \
                       + (warm_up_scale * regularization_loss))
        return logpx_update, torch.mean(logpxz), regularization_loss, torch.mean(KLD_latent)

    def likelihoods(self, x):
        """
        calculates ELBO of Log(P(x)) for specific x.
        :param x: sequence
        :return: logpx_i likelihood value for that x.
        """
        mu, logsig = self.encoder.forward(x)
        z = self.sampler(mu, logsig)
        reconstructed_x, logpxz, pattern_activations = self.decoder.forward(x, z)
        KLD_latent = 0.5 * torch.sum(1.0 + 2.0 * logsig - mu**2.0 - torch.exp(2.0 * logsig), dim=1)
        logpx_i = logpxz + KLD_latent
        return logpx_i

    def all_likelihood_components(self, x):
        """
        As above, but returning all likelihood components individually
        :param x: x
        :return: likelihood components, KLD of latent space, reconstruction loss, and their sum.
        """
        mu, logsig = self.encoder.forward(x)
        z = self.sampler(mu, logsig)
        reconstructed_x, logpxz, pattern_activations = self.decoder.forward(x, z)
        KLD_latent = 0.5 * torch.sum(1.0 + 2.0 * logsig - mu**2.0 - torch.exp(2.0 * logsig), dim=1)
        logpx_i = logpxz + KLD_latent
        return logpx_i, KLD_latent, logpxz

    def recognize(self, x):
        """
        given a sequence x, return its latent mu and log-sigma.
        """
        mu, logsig = self.encoder.forward(x)
        return mu, logsig

    def get_pattern_activations(self, x):
        """
        return output patterns for a specific input sequence.
        """
        mu, logsig = self.encoder.forward(x)
        z = self.sampler(mu, logsig)
        _, _, pattern_activations = self.decoder.forward(x, z)
        return pattern_activations


class VAE_SVI(nn.Module):
    """
    Variational Autoencoder class that implements stochastic variational inference of decoder parameters.
    Constructs full model and forward pass, serves as a callable object for model training.
    """
    def __init__(self,
        data,
        encoder_architecture=[1500,1500],
        decoder_architecture=[100,500],
        n_latent=2,
        n_patterns=4,
        batch_size=100,
        encode_nonlinearity_type="relu",
        decode_nonlinearity_type="relu",
        final_decode_nonlinearity="sigmoid",
        sparsity="logit",
        global_scale=1.0,
        logit_p=0.01,
        logit_sigma=4.0,
        pattern_sigma=1.0,
        warm_up=0.0,
        convolve_encoder=False,
        convolve_patterns=True,
        conv_decoder_size=10,
        conv_encoder_size=10,
        output_bias=True,
        final_pwm_scale=False,
        working_dir=".",
        learning_rate=0.001,
        kl_scale=1.0,
        random_seed=42,
        float_size = 32):
        """
        :param data: instance of DataHelper class with loaded data
        :param encoder_architecture: list detailing size of encoder hidden layers
        :param decoder_architecture: list detailing size of decoder hidden layers
        :param n_latent: size of latent dimension
        :param n_patterns: how many times to tile sparsity parameters over output
        :param batch_size: size of mini-batch to use
        :param encode_nonlinearity_type: non-linearity used in encoder layers
        :param decode_nonlinearity_type: non-linearity used in decoder's first layers
        :param final_decode_nonlinearity: non-linearity used in decoder's final layer
        :param sparsity: Sparsity type, using a noncentered reparameterization.
                Options include: logit, analytic, laplacian, horseshoe, ard
                See Ingraham and Marks, 2016 (https://arxiv.org/abs/1602.03807)
        :param global_scale: Global scale prior for sparsity: analytic, laplacian, horseshoe, ard
        :param logit_p: Global scale prior for logit sparsity
        :param logit_sigma: Prior sigma for scale prior for logit sparsity
        :param pattern_sigma: Prior sigma for variational weights on the final layer
        :param warm_up: number of updates for which annealing should be performed, lessening the effect
                of KLD loss in early updates.
        :param convolve_encoder:  whether to include 1D convolution on the input sequences
        :param convolve_patterns: whether to include 1D convolution on output, also known as dictionary
        :param conv_decoder_size: size of decoder convolution
        :param conv_encoder_size: size of encoder convolution
        :param output_bias: whether to include bias on output layer
        :param final_pwm_scale: use a temperature parameter on final layer
        :param working_dir: directory for saving and loading parameters
        :param learning_rate: Adam learning rate
        :param kl_scale: Scale of KL of latent variables, default 1.0
                Scale < 1.0 approaches a normal autoencoder
                Scale > 1.0 turns into beta-autoencoder (Higgins et al, 2016)
        :param random_seed: seed for random number generator
        :param float_size: 16, 32, 64 - specifies the torch tensor float size.
        """
        super(VAE_SVI, self).__init__()
        # SET UP TENSOR ENVIRONMENT
        set_tensor_environment(float_size, random_seed)
        if float_size == 16:
            self.epsilon = 1e-6
            self.dtype = torch.float16
        elif float_size == 32:
            self.epsilon = 1e-8
            self.dtype = torch.float32
        elif float_size == 64:
            self.epsilon = 1e-8
            self.dtype = torch.float64
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # key parameters:
        self.type = 'SVI'
        self.convolve_patterns = convolve_patterns
        self.convolve_encoder = convolve_encoder
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seq_len = data.seq_len
        self.alphabet_size = data.alphabet_size
        self.warm_up = torch.tensor(warm_up, device=self.device, dtype=self.dtype)
        self.sigma_init = 0.01
        self.logsig_init = -5
        self.srng = lambda *shape: torch.randn(*shape, device=self.device, dtype=self.dtype)
        self.working_dir = working_dir
        self.sparsity = sparsity
        self.final_pwm_scale = final_pwm_scale
        self.logit_mu = torch.tensor(np.sqrt(2.0)*logit_sigma*erfinv(2.0*logit_p-1.0),
                                     device=self.device, dtype=self.dtype)
        self.logit_sigma = torch.tensor(logit_sigma, device=self.device, dtype=self.dtype)
        self.global_scale = torch.tensor(global_scale, device=self.device, dtype=self.dtype)
        self.kl_scale = torch.tensor(kl_scale, device=self.device, dtype=self.dtype)
        self.global_scale = torch.tensor(global_scale, device=self.device, dtype=self.dtype)
        self.inv_global_scale = torch.tensor((1.0/global_scale), device=self.device, dtype=self.dtype)
        self.logit_p = logit_p
        self.sigma_init = 0.01
        self.logsig_init = -5

        # define nn modules with given arguments:
        self.encoder = Encoder(encoder_architecture, n_latent, convolve_encoder, conv_encoder_size,
                 self.alphabet_size, self.seq_len, encode_nonlinearity_type)

        self.decoder = DecoderSVI(decoder_architecture, n_latent, convolve_patterns, conv_decoder_size,
                                  self.alphabet_size, self.seq_len, decode_nonlinearity_type,
                                  final_decode_nonlinearity, sparsity, final_pwm_scale, n_patterns,
                                  output_bias, self.logsig_init, pattern_sigma, self.srng)

    def KLD_diag_gaussians(self, mu, logsig, p_mu, p_logsig):
        """
        KLD divergence between diagonal gaussian with prior diagonal gaussian
        :param mu: mean value of gaussian
        :param logsig: log sigma of gaussian
        :param p_mu: mu of prior gaussian
        :param p_logsig: log sigma of prior gaussian
        :return: KL divergence value
        """
        KLD = p_logsig - logsig + 0.5*(torch.exp(2.0*logsig)+torch.pow(mu-p_mu,2))*torch.exp(-2.0*p_logsig)-0.5
        return KLD

    def sampler(self, mu, logsig):
        """
        Samples from a diagonal Gaussian for stochastic variables. Applies reparameterization trick
        :param mu: sample mean
        :param logsig: sample log sigma
        :return: sampled variable z
        """
        eps = self.srng(mu.shape)
        z = mu + torch.exp(logsig) * eps
        return z

    def _anneal(self, update_num):
        """
        Anneal the KL if using annealing; reduces the KLD loss effect in early updates
        :param update_num: current update number
        :return: scaling factor for KLD loss
        """
        # If true, return first, else return second
        condition = update_num < self.warm_up
        KL_scale = torch.where(condition, update_num/self.warm_up,
                               torch.tensor(1.0, device=self.device, dtype=self.dtype))
        return KL_scale

    def gen_kld_params(self):
        """
        Iterates through all variational parameters using their identifiers and the model's state_dict(),
        calculated KL divergence value, summing over all parameters
        :return: sum of KLD losses
        """
        KLD_params = 0.0
        for identifier in self.decoder.variational_param_identifiers: # iterate over every variational parameter
            mu_id = "mu_"+identifier # construct IDs for mu and logsigma parameters
            ls_id = "ls_"+identifier
            pl_id = identifier[0:2] # get ID for prior logsig list
            mu = self.decoder.state_dict(keep_vars=True)[mu_id].flatten() #extract weights with gradients
            ls = self.decoder.state_dict(keep_vars=True)[ls_id].flatten()
            prior_sg = torch.tensor(self.decoder.variational_param_name_to_sigma[pl_id], # extract prior sigma
                                    device=self.device, dtype=self.dtype)
            prior_ls = torch.log(prior_sg) # convert to log-sigma
            KLD_params += 4*torch.sum(-self.KLD_diag_gaussians(mu,ls,0.0,prior_ls)) # calculate KLD loss, summing
        return KLD_params # return sum of losses

    def gen_kld_sparsity(self, sparsity):
        """
        Calculates KL divergence loss for sparsity parameters depending on what sparsity is used
        :param sparsity: specified sparsity to use
        :return: KL divergence loss
        """
        if sparsity == "logit":
            # Use a continuous relaxation of a spike and slab prior
            # with a logit normit scale distribution
            KLD_fadeout = -self.KLD_diag_gaussians(
                self.decoder.state_dict(keep_vars=True)['scale_mu'],
                self.decoder.state_dict(keep_vars=True)['scale_ls'],
                self.logit_mu,
                torch.log(self.logit_sigma)
            )
        if sparsity == "analytic":
            # Use a moment-matched Gaussian approximation to the
            #   log-space Hyperbolic Secant hyperprior of the Horseshoe
            KLD_fadeout = -self.KLD_diag_gaussians(
                self.decoder.state_dict(keep_vars=True)['scale_mu'],
                self.decoder.state_dict(keep_vars=True)['scale_ls'],
                torch.log(self.global_scale),
                torch.log(np.pi/2.0)
            )
        else:
            # Estimate KL divergence for the sparsity
            #   scale parameters (Fadeout) by sampling
            W_scale = torch.exp(self.sampler(
                self.decoder.state_dict(keep_vars=True)['scale_mu'],
                self.decoder.state_dict(keep_vars=True)['scale_ls']
            ))
            if sparsity == "horseshoe":
                KLD_fadeout = (torch.log(2.0) + torch.log(self.global_scale) - torch.log(np.pi) + torch.log(W_scale) \
                    - torch.log(self.global_scale*self.global_scale + W_scale*W_scale)) \
                    + (self.decoder.state_dict(keep_vars=True)['scale_ls'] + 0.5*torch.log(2.0*np.pi*np.e))
            elif sparsity == 'laplacian':
                # Laplace sparsity has exponential hyperprior
                KLD_fadeout = (torch.log(2.0) + torch.log(self.inv_global_scale) \
                    - self.inv_global_scale * W_scale * W_scale + 2.0 \
                    * torch.log(W_scale)) + (self.decoder.state_dict(keep_vars=True)['scale_ls'] \
                    + 0.5 * torch.log(2.0 * np.pi * np.e))
            elif sparsity == "ard":
                # Automatic Relevance Determination sparsity
                #  has Inverse-Gamma hyperprior
                KLD_fadeout = (torch.log(2.0) + (self.global_scale \
                    * torch.log(self.global_scale)) \
                    - gammaln(self.global_scale) - (self.global_scale \
                    / ((W_scale * W_scale) + self.epsilon)) \
                    - (2.0 * self.global_scale * torch.log(W_scale))) \
                    + (self.decoder.state_dict(keep_vars=True)['scale_ls']  \
                    + 0.5 * torch.log(2.0 * np.pi * np.e))
        return torch.sum(KLD_fadeout)

    def update(self, logpxz, mu, logsig, update_num, Neff):
        """
        Calculate updated loss values with KLD loss and sparsity, returning various loss components.
        :param logpxz: Log(P(x|z)) value determined through forward pass of decoder.
        :param mu: mu of z determined through forward pass of encoder.
        :param logsig: log sigma of z determined through forward pass of encoder
        :param update_num: current update number used for annealing if required.
        :param Neff: effective sample N, used in loss calculations.
        :return: logpx_update - final ELBO loss value,
        torch.mean(logpxz) - value for reconstruction loss,
        regularization_loss,
        torch.mean(KLD_latent) - KL divergence loss of latent variables.
        """
        # latent dimension KL divergence loss
        KLD_latent = 0.5*torch.sum(1.0 + 2.0*logsig - torch.pow(mu,2) - torch.exp(2.0*logsig), dim=1)
        # decoder parameter KL divergence loss
        KLD_params_all = self.gen_kld_params()
        # sparsity parameter KL divergence loss
        if self.sparsity:
            KLD_params_all += self.gen_kld_sparsity(self.sparsity)
        # get warm up scale for latent space if using annealing
        warm_up_scale = self._anneal(update_num)
        # apply KL scale if using
        KLD_latent_update = KLD_latent * self.kl_scale
        # get final loss; negative because we want to maximize ELBO and Adam() minimizes
        logpx_update = -(torch.mean(logpxz + (warm_up_scale * KLD_latent_update)) \
            + (warm_up_scale * (KLD_params_all / Neff)))
        # return all loss components
        return logpx_update, torch.mean(logpxz), (KLD_params_all/Neff), torch.mean(KLD_latent)

    def likelihoods(self, x):
        """
        calculates ELBO of Log(P(x)) for specific x.
        :param x: sequence
        :return: logpx_i likelihood value for that x.
        """
        mu, logsig = self.encoder.forward(x)
        z = self.sampler(mu, logsig)
        reconstructed_x, logpxz, pattern_activations = self.decoder.forward(x, z)
        KLD_latent = 0.5*torch.sum(1.0 + 2.0*logsig - mu**2.0 - torch.exp(2.0*logsig), dim=1)
        logpx_i = logpxz + KLD_latent
        return logpx_i

    def all_likelihood_components(self, x):
        """
        As above, but returning all likelihood components individually. Used in effect prediction
        :param x: x
        :return: likelihood components, KLD of latent space, reconstruction loss, and their sum.
        """
        mu, logsig = self.encoder.forward(x)
        z = self.sampler(mu, logsig)
        reconstructed_x, logpxz, pattern_activations = self.decoder.forward(x, z)
        KLD_latent = 0.5 * torch.sum(1.0 + 2.0 * logsig - mu**2.0 - torch.exp(2.0 * logsig), dim=1)
        logpx_i = logpxz + KLD_latent
        return logpx_i, KLD_latent, logpxz

    def recognize(self, x):
        """
        given a sequence x, return its latent mu and log-sigma.
        """
        mu, logsig = self.encoder.forward(x)
        return mu, logsig

    def get_pattern_activations(self, x):
        """
        return output patterns for a specific input sequence.
        """
        mu, logsig = self.encoder.forward(x)
        z = self.sampler(mu, logsig)
        _, _, pattern_activations = self.decoder.forward(x, z)
        return pattern_activations





