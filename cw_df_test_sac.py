import torch
from sac_defense import SACAgent
from network_SAC import TrafficFeatureExtractor
from models import DF, VarCNN, TF, AWF, NetCLR
from enviroment import Environment
import csv
import argparse
from utility import *
from config import *

def dqn_test_in_cw(agent, env, traffic, traffic_idx, i, mode, bwo_para):
    t = 0
    done = False
    ori_max_index = env.get_ori_label(traffic_idx)
    ori_len = np.count_nonzero(traffic.cpu())
    reward_sum = 0
    done_num = 0
    bwo = 0
    insert_num = bwo_para * (ori_len // 5)
    max_step = int(insert_num)
    while t < max_step:
        t += 1
        modified_traffic, action = agent.single_trace_mutate(traffic)
        done = env.check_done(modified_traffic, traffic_idx)
        still_in_prob, feedback_new = env.get_single_trace_environment_feedback(modified_traffic, traffic_idx, num=82)
        reward = 1 - still_in_prob.item()
        total_reward = feedback_new
        reward_sum += reward
        reward_avg = reward_sum / t
        if t == max_step:
            inser_num = t
            bwo = inser_num / ori_len
        if done:
            done_num += 1
            done = False
        traffic = modified_traffic.clone()
    print(f"Episode {i}: Total Reward = {reward_sum}, avg={reward_avg}")
    return traffic, bwo

def load_testdata():
    X_test, y_test = LoadDataNoDefCW()
    X_test = X_test[:, np.newaxis, :]
    print("X: Testing data's shape: ", X_test.shape)
    print("y: Testing data's shape: ", y_test.shape)
    test_dataset = MyDataset(X_test, y_test)
    test_data = WholeDatasetIterator(test_dataset)
    total_num = y_test.shape[0]
    return test_data, total_num

def load_model(modelname, modelpath, nb_classes, device):
    if modelname == 'conv_net':
        model = TrafficFeatureExtractor(nb_classes).to(device)
    elif modelname == 'DF':
        model = DF.DF(trace_len, nb_classes).to(device)
    elif modelname == 'VarCNN':
        model = VarCNN.VarCNN(nb_classes).to(device)
    elif modelname == 'TF':
        model = TF.TF(nb_classes).to(device)
    elif modelname == 'AWF':
        model = AWF.AWF(nb_classes).to(device)
    elif modelname == 'NetCLR':
        model = NetCLR.NetCLR(nb_classes).to(device)
    else:
        raise ValueError(f"Unknown model name: {modelname}")

    model.load_state_dict(torch.load(modelpath, map_location=device, weights_only=True))
    model.eval()
    return model

def get_pred_label(trace, model, acc_count, ori_max_index):
    output = model(trace)
    probs = torch.softmax(output[1], dim=-1)
    pred_label = torch.max(probs, 0)[1].data
    pre_label = pred_label.tolist()
    if pre_label == ori_max_index:
        acc_count += 1
    return acc_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU device')
    parser.add_argument('--subdir', type=str, required=True, help='subdir')
    parser.add_argument('--attack_model', type=str, required=True, choices='[DF, VarCNN, TF, AWF, NetCLR]', help='attack model')
    parser.add_argument('--bwo_para', type=float, required=True, help='bwo contro parameter')
    parser.add_argument('--nb_classes', type=int, default=95, help='number of classes')
    args = parser.parse_args()
    
    test_data, total_num = load_testdata()
    conv_net = load_model('conv_net', conv_model_path, args.nb_classes, args.device)
    df_net = load_model('DF', DF_model_path, args.nb_classes, args.device)
    varcnn_net = load_model('VarCNN', VarCNN_model_path, args.nb_classes, args.device)
    tf_net = load_model('TF', TF_model_path, args.nb_classes, args.device)
    awf_net = load_model('AWF', AWF_model_path, args.nb_classes, args.device)
    netCLR = load_model('NetCLR', NetCLR_model_path, args.nb_classes, args.device)

    attack_net = load_model(args.attack_model, MI_model_path, args.nb_classes, args.device)
    agent = SACAgent(state_dim, hidden_dim, action_dim, conv_net, ACTOR_LR, CRITIC_LR, ALPHA_LR, BATCH_SIZE, TAU, GAMMA, args.device)
    agent.load_q_model(q_policy_model_path)
    env = Environment(attack_net)

    df_acc_count = 0
    varcnn_acc_count = 0
    tf_acc = 0
    awf_acc = 0
    netCLR_acc = 0
    bwo_total = 0
    bwo_avg = 0
    for i,(b_x, b_y) in enumerate(test_data):
        b_x = b_x.to(args.device)
        b_y = b_y.to(args.device)
        _, ori_max_index = torch.max(b_y.data, dim=-1)
        traffic, bwo = dqn_test_in_cw(agent, env, b_x, b_y, i, args.test_mode, args.bwo_para)
        bwo_total += bwo
        ori_max_index = ori_max_index.tolist()
        b_x_1 = b_x.repeat(2, 1, 1)
        b_xnew = traffic.repeat(2, 1, 1)

        df_acc_count = get_pred_label(b_xnew, df_net, df_acc_count, ori_max_index)
        varcnn_acc_count = get_pred_label(b_xnew, varcnn_net, varcnn_acc_count, ori_max_index)
        tf_acc = get_pred_label(b_xnew, tf_net, tf_acc, ori_max_index)
        awf_acc = get_pred_label(b_xnew, awf_net, awf_acc, ori_max_index)
        netCLR_acc = get_pred_label(b_xnew, netCLR, netCLR_acc, ori_max_index)
    
    df_accuracy = df_acc_count / total_num
    varcnn_accuracy = varcnn_acc_count / total_num
    tf_accuracy = tf_acc / total_num
    awf_accuracy = awf_acc / total_num
    netCLR_accuracy = netCLR_acc / total_num
    bwo_avg = bwo_total / total_num
    print(f"DF-Accuracy is {df_accuracy}.Var-CNN-Accuracy is {varcnn_accuracy}.TF-Accuracy  is {tf_accuracy}.AWF-Accuracy is {awf_accuracy}.NetCLR-Accuracy is {netCLR_accuracy}. BWO is {bwo_avg}")
    