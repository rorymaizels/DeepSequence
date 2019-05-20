import sys
import torch
# to store and access model code in different folders, use this example command:
sys.path.insert(0, "../DeepSequence/")
import pt_model
import pt_helper
import pt_train

# specify the data set to use and any other DataHelper parameters
data_params = {
    "dataset"           :   "BLAT_ECOLX"
    }

#specify model details
model_params = {
    "bs"                :   100,
    "encode_dim_zero"   :   1500,
    "encode_dim_one"    :   1500,
    "decode_dim_zero"   :   100,
    "decode_dim_one"    :   500,
    "n_latent"          :   30,
    "logit_p"           :   0.001,
    "sparsity"          :   "logit",
    "final_decode_nonlin":  "sigmoid",
    "final_pwm_scale"   :   True,
    "n_pat"             :   4,
    "r_seed"            :   12345,
    "conv_pat"          :   True,
    "convolve_encoder"  :   False,
    'warm_up'           :   0.0,
    "d_c_size"          :   40
    }

# specify training details
train_params = {
    "num_updates"       :   3,
    "save_progress"     :   False,
    "verbose"           :   True,
    "save_parameters"   :   False,
    "unique_ID"         :   "test_run",
    "cuda"              :   False
    }

# speed up run time with GPU and algorithmic benchmarking
cuda = torch.cuda.is_available()
torch.backends.cudnn.benchmark = True

# construct data, model and train!
if __name__ == "__main__":

    data_helper = pt_helper.DataHelper(dataset=data_params["dataset"],
                                        calc_weights=True)
    vae_model = pt_model.VAE_SVI(data_helper,
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
                                   warm_up=model_params['warm_up']
                                   )
    job_string = pt_helper.gen_simple_job_string(vae_model, data_params, train_params["unique_ID"])

    print("MODEL PARAMS:")
    print(model_params)
    print("TRAINING PARAMS:")
    print(train_params)
    print("Data:")
    print(data_params)

    pt_train.train(data_helper, vae_model,
        num_updates             =   train_params["num_updates"],
        save_progress           =   train_params["save_progress"],
        save_parameters         =   train_params["save_parameters"],
        verbose                 =   train_params["verbose"],
        job_string              =   job_string,
        use_cuda                =   train_params["cuda"])

    # pt_train.save(vae_model, job_string)