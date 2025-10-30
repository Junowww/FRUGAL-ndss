
# parameters setting
BATCH_SIZE = 32
LEARNING_RATE = 0.001
ACTOR_LR = 3e-4
CRITIC_LR = 3e-3
ALPHA_LR = 3e-4
TAU = 0.005
GAMMA = 0.99
num_features = 1000
trace_len =5000
reward_scale = 0.9
state_dim=1000
hidden_dim=125
action_dim=5

# model_path
# q_policy_model_path = './saved_trained_models/DF_1026-web95-sampleweights_sac_ongs20.pth'
q_policy_model_path = './saved_trained_models/sac_models/df_0716-bwo0.3weights_sac_ongs20.pth'
# MI_model_path = './saved_trained_models/MI_DF_0902-btach-bwo0.3weights_sac_ongs20.pth' #trained MI estimator
MI_model_path = './saved_trained_models/df_cw.pth'
conv_model_path = './saved_trained_models/state_encoder.pth' # for 95 websites
DF_model_path = './saved_trained_models/df_cw.pth'
VarCNN_model_path = './saved_trained_models/varcnn_cw.pth'
TF_model_path = './saved_trained_models/tf_cw.pth'
NetCLR_model_path = './saved_trained_models/net_cw.pth'
AWF_model_path = './saved_trained_models/awf_cw.pth'
# RF_model_path = './saved_trained_models/rf_cw_3d_new.pth'

