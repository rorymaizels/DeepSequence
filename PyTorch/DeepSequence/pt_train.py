from __future__ import print_function
import numpy as np
import time
import torch
import torch.optim as optim


def save(model, file_prefix='unnamed_model', working_dir=".",):
    """ save the given model with given name in given directory"""
    file_path = working_dir + '/model_params/' + file_prefix
    torch.save(model.state_dict(), file_path)


def load(model, path, eval=True, cuda=True):
    """
    load model weights
    :param model: model with weights to be updated
    :param path: path to weights
    :param eval: set model to eval() mode
    :param cuda: enable GPU/cuda
    :return: Nothing
    """
    if cuda:
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path, map_location='cpu'))
    if eval:
        model.eval()



def train(data,
    model,
    save_progress=False,
    save_parameters=False,
    num_updates=300000,
    verbose=True,
    job_string="",
    embeddings=False,
    update_offset=0,
    print_neff=True,
    print_iter=1,
    use_cuda=False):
    """
    Main function to train DeepSequence models
    :param data: DataHelper class instance
    :param model: Model class instance
    :param save_progress: save log files of losses during training
    :param save_parameters: number of iterations on which to save parameters
    :param num_updates: number of training iterations / epochs
    :param verbose: Print losses and details
    :param job_string: string for saving model etc.
    :param embeddings:  save latent variables every k iterations (int)
            or "log": save latent variables during training on log scale iterations
            or False (bool)
    :param update_offset:  Offset use for Adam in training
            Change this to keep training parameters from an old model
    :param print_neff: Print the effective sample size of the alignment
    :param print_iter: how many iterations to print information if verbose
    :param use_cuda: GPU/cuda capability
    :return: Nothing
    """
    torch_dtype = model.dtype
    torch_device = model.device
    batch_size = model.batch_size
    batch_order = np.arange(data.x_train.shape[0])
    seq_sample_probs = data.weights / np.sum(data.weights)
    update_num = 0
    LB_list = []
    reg_list = []
    KLD_latent_list = []
    recon_list = []

    if save_progress:
        err_filename = data.working_dir+"/logs/"+job_string+"_err.csv"
        OUTPUT = open(err_filename, "w+")
        if print_neff:
            OUTPUT.write("Neff:\t"+str(data.Neff)+"\n")
        OUTPUT.close()

    start = time.time()

    if embeddings == "log":
        start_embeddings = 10
        log_embedding_interpolants = sorted(list(set(np.floor(np.exp(\
            np.linspace(np.log(start_embeddings),np.log(50000),1250))).tolist())))
        log_embedding_interpolants = [int(val) for val in log_embedding_interpolants]

    solver = optim.Adam(model.parameters(), lr=model.learning_rate)

    while (update_num + update_offset) < num_updates:
        # iterate
        update_num += 1
        # prepare data
        batch_index = np.random.choice(batch_order, batch_size, \
            p=seq_sample_probs).tolist()
        batch = torch.tensor(data.x_train[batch_index], dtype=torch_dtype, device=torch_device, requires_grad=False)

        if use_cuda:
            batch = batch.cuda()
            model = model.cuda()
        neff = data.Neff

        solver.zero_grad() # torch accumulates gradients, so this should be called to reset

        # forward step
        batch_mu, batch_logsig = model.encoder.forward(batch)
        batch_z = model.sampler(batch_mu,batch_logsig)
        batch_recon, logpxz, output = model.decoder.forward(batch, batch_z)

        # find loss
        LB_loss, recon_entropy, reg_loss, KLD_latent = model.update(logpxz, batch_mu, batch_logsig, update_num, neff)

        # store results of different loss components.
        LB_list.append(LB_loss.cpu().detach().numpy())
        reg_list.append(reg_loss.cpu().detach().numpy())
        KLD_latent_list.append(KLD_latent.cpu().detach().numpy())
        recon_list.append(recon_entropy.cpu().detach().numpy())

        # backward step
        LB_loss.backward()

        # update
        solver.step()

        #housekeeping; stop gradients accumulating
        for p in model.parameters():
            p.grad.data.zero_()

        # saving functions
        if save_parameters != False and update_num % save_parameters == 0:
            if verbose:
                print("Saving Parameters")
            name = job_string+"_epoch"+str(update_num+update_offset)
            save(model, name)

        # Make embeddings in roughly log-time
        if embeddings:
            if embeddings == "log":
                if update_num + update_offset in log_embedding_interpolants:
                    data.get_embeddings(model, update_num + update_offset, filename_prefix=job_string)
            else:
                if update_num % embeddings == 0:
                    data.get_embeddings(model, update_num + update_offset, filename_prefix=job_string)

        if update_num % print_iter == 0: # printing if verbose, saving loss to log files if saving progress
            mean_index = np.arange(update_num-print_iter,update_num)

            LB = np.mean(np.asarray(LB_list)[mean_index])
            KLDP = np.mean(np.asarray(reg_list)[mean_index])
            KLDL = np.mean(np.asarray(KLD_latent_list)[mean_index])
            reconstruct = np.mean(np.asarray(recon_list)[mean_index])

            template = "Update {0}. LB : {1:.2f}, Params: {2:.2f}, Latent: {3:.2f}, Reconstruct: {4:.2f}, Time: {5:.2f}"
            progress = template.format(update_num+update_offset, LB, KLDP, KLDL, reconstruct, time.time() - start)

            if verbose:
                print(progress)

            if save_progress:
                OUTPUT = open(err_filename, "a")
                OUTPUT.write(progress+"\n")
                OUTPUT.close()
