import numpy as np
import pandas as pd
import scipy.stats as ss
# use following command if model code in different directory
import sys
sys.path.insert(0, "../DeepSequence/")
import pt_model as model
import pt_helper as helper
import pt_train as train

# specify path to model parameters
model_name = ""

# specify dataset
data_params = {"dataset":"BLAT_ECOLX"}

# specify model params
model_params = {
    "bs"                :   100,
    "encode_dim_zero"   :   1500,
    "encode_dim_one"    :   1500,
    "decode_dim_zero"   :   500,
    "decode_dim_one"    :   1500,
    "n_latent"          :   30,
    "logit_p"           :   0.001,
    "sparsity"          :   "logit",
    "final_decode_nonlin":  "sigmoid",
    "final_pwm_scale"   :   True,
    "n_pat"             :   4,
    "r_seed"            :   12345,
    "conv_pat"          :   True,
    "convolve_encoder"  :   False,
    "d_c_size"          :   40
    }

def generate_spearmanr(mutant_name_list, delta_elbo_list, mutation_filename, phenotype_name):
    """
    function that takes in mutant data from model and experimental data and prints a Spearman R correlation coefficient.
    """
    measurement_df = pd.read_csv(mutation_filename, sep=',')

    mutant_list = measurement_df.mutant.tolist()
    expr_values_ref_list = measurement_df[phenotype_name].tolist()

    mutant_name_to_pred = {mutant_name_list[i]: delta_elbo_list[i] for i in range(len(delta_elbo_list))}

    # If there are measurements
    wt_list = []
    preds_for_spearmanr = []
    measurements_for_spearmanr = []

    for i, mutant_name in enumerate(mutant_list):
        expr_val = expr_values_ref_list[i]

        # Make sure we have made a prediction for that mutant
        if mutant_name in mutant_name_to_pred:
            multi_mut_name_list = mutant_name.split(':')

            # If there is no measurement for that mutant, pass over it
            if np.isnan(expr_val):
                pass

            # If it was a codon change, add it to the wt vals to average
            elif mutant_name[0] == mutant_name[-1] and len(multi_mut_name_list) == 1:
                wt_list.append(expr_values_ref_list[i])

            # If it is labeled as the wt sequence, add it to the average list
            elif mutant_name == 'wt' or mutant_name == 'WT':
                wt_list.append(expr_values_ref_list[i])

            else:
                measurements_for_spearmanr.append(expr_val)
                preds_for_spearmanr.append(mutant_name_to_pred[mutant_name])

    if wt_list != []:
        measurements_for_spearmanr.append(np.mean(wt_list))
        preds_for_spearmanr.append(0.0)

    num_data = len(measurements_for_spearmanr)
    spearman_r, spearman_pval = ss.spearmanr(measurements_for_spearmanr, preds_for_spearmanr)
    print("N: " + str(num_data) + ", Spearmanr: " + str(spearman_r) + ", p-val: " + str(spearman_pval))


# make datahelper and model
data_helper = helper.DataHelper(dataset=data_params["dataset"], working_dir=".", calc_weights=False)

vae_model = model.VAE_SVI(data_helper,
                          batch_size=model_params["bs"],
                          encoder_architecture=[model_params["encode_dim_zero"],
                                                model_params["encode_dim_one"]],
                          decoder_architecture=[model_params["decode_dim_zero"],
                                                model_params["decode_dim_one"]],
                          n_latent=model_params["n_latent"],
                          logit_p=model_params["logit_p"],
                          sparsity=model_params["sparsity"],
                          encode_nonlinearity_type="relu",
                          decode_nonlinearity_type="relu",
                          final_decode_nonlinearity=model_params["final_decode_nonlin"],
                          final_pwm_scale=model_params["final_pwm_scale"],
                          conv_decoder_size=model_params["d_c_size"],
                          convolve_patterns=model_params["conv_pat"],
                          convolve_encoder=model_params["convolve_encoder"],
                          n_patterns=model_params["n_pat"],
                          random_seed=model_params["r_seed"],
                          )

print ("Model built")

path = "model_params/"+model_name
# load weights
train.load(vae_model, path)
print ("Parameters loaded\n\n")
mutation = "mutations/BLAT_ECOLX_Ranganathan2015.csv"

# construct model's mutation predictions
custom_mutant_name_list, custom_delta_elbos = data_helper.custom_mutant_matrix(mutation,
                                                                               vae_model,
                                                                               N_pred_iterations=500)
# find Spearman R value
generate_spearmanr(custom_mutant_name_list, custom_delta_elbos, mutation, "2500")

