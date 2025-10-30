from config import *
from utility import *
from network_SAC import *
from enviroment import Environment
from models import DF,VarCNN, TF, AWF, NetCLR
from sac_defense import SACAgent
import argparse


def single_trace_train_agent(agent, env, ori_traffic, ori_label, i, reward_scale, bwo_para, nb_classes):

    state = agent.get_state(ori_traffic)
    reward_sum = 0.0
    t = 0
    done = False
    reward = 0
    _, ori_index = torch.max(ori_label.data, dim=-1)
    ori_len = np.count_nonzero(ori_traffic.cpu())
    ori_max_index = env.get_ori_label(ori_label)
    insert_num = ori_len // 5
    max_step = int(bwo_para * insert_num) 
    while t < max_step:
        t += 1
        modified_traffic, action = agent.single_trace_mutate(ori_traffic)
        next_state = agent.get_state(modified_traffic)
        # 应基于最新的 modified_traffic 检查是否完成
        done = env.check_done(modified_traffic, ori_label)
        max_prob, still_in_prob, reward2 = env.get_single_trace_environment_feedback(modified_traffic, ori_label, nb_classes)
        reward = 1 - still_in_prob
        reward_sum += reward
        reward_avg = reward_sum / t
        mask = 1.0 if not done else 0.0
        # agent.memo.add_memo((reward.detach().cpu(), mask, state.detach().cpu(), action.detach().cpu(), next_state.detach().cpu()))
        agent.memo.add_memo((reward, mask, state.detach().cpu(), action.detach().cpu(), next_state.detach().cpu()))
        agent.memo.init_after_add_memo()
        if agent.memo.now_len >= agent.batch_size:
            agent.learn()
            env.MI_update(modified_traffic, ori_label, i) #update MI estimator every 5 iterations
            if t % 5 ==0:
                agent.soft_update()
        state = next_state.clone()
        ori_traffic = modified_traffic.clone()
    print(f"Episode {i}: Total Reward = {reward_sum}, avg={reward_avg}")
    return modified_traffic

def load_goodsample(limits):
    # each website 20 traces for training
    X_goodSample, y_goodSample = LoadGoodSampleCW(limits)
    gs_dataset = MyDataset(X_goodSample, y_goodSample)
    gs_data = WholeDatasetIterator(gs_dataset)
    return gs_data

def load_train_data():
    X_train, y_train = LoadDataNoDefCW()
    X_train = X_train[:, np.newaxis, :]
    train_dataset = MyDataset(X_train, y_train)
    train_data = WholeDatasetIterator(train_dataset)
    return train_data

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU device')
    parser.add_argument('--subdir', type=str, required=True, help='subdir')
    parser.add_argument('--attack_model', type=str, required=True, choices='[DF, VarCNN, TF, AWF, NetCLR]', help='attack model')
    parser.add_argument('--limits', type=int, default=20, help='number of goodsamples')
    parser.add_argument('--bwo_para', type=float, required=True, help='bwo contro parameter')
    parser.add_argument('--nb_classes', type=int, default=95, help='number of classes')
    args = parser.parse_args()

    # gs_data = load_goodsample(args.limits)
    gs_data = load_train_data()
    print("conv_model_path:", conv_model_path)  
    conv_net = load_model('conv_net', conv_model_path, args.nb_classes, args.device)

    attack_model_path = {
        'DF': DF_model_path,
        'VarCNN': VarCNN_model_path,
        'TF': TF_model_path,
        'AWF': AWF_model_path,
        'NetCLR': NetCLR_model_path
    }[args.attack_model]
    attack_net = load_model(args.attack_model, attack_model_path, args.nb_classes, args.device)
    agent = SACAgent(state_dim, hidden_dim, action_dim, conv_net, ACTOR_LR, CRITIC_LR, ALPHA_LR, BATCH_SIZE, TAU, GAMMA, args.device)
    env = Environment(attack_net)

    for i, (b_x, b_y) in enumerate(gs_data):
        b_x = b_x.to(args.device, non_blocking=True)
        b_y = b_y.to(args.device, non_blocking=True)
        modified_traffic = single_trace_train_agent(agent, env, b_x, b_y, i, reward_scale, args.bwo_para, args.nb_classes)
            # if i % 500 == 0 and i != 0:
            #     save_checkpoint(agent.actor, agent.attack_model, args.subdir, i)

    agent.actor.to('cpu')
    print("actor is saving.")
    torch.save(agent.actor.state_dict(), './saved_trained_models/sac_models/{}_{}weights_sac_ongs{}.pth'.format(args.attack_model, args.subdir, args.limits))
    env.attack_net.to('cpu')
    # torch.save(env.attack_net.state_dict(), './saved_trained_models/MIestimator_{}_{}weights_sac_ongs{}.pth'.format(args.attack_model, args.subdir, args.limits))
    print("Training progress ends.")
