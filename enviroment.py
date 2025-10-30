import torch
import numpy as np
import torch.optim as optim


class Environment():
    def __init__(self, attack_net):
        super(Environment, self).__init__()
        self.attack_net = attack_net
        self.mi_optimizer = torch.optim.Adam(self.attack_net.parameters(), lr=1e-3)
        self.mi_loss_fn = torch.nn.CrossEntropyLoss()

    def get_single_trace_attackmodel_result(self, x_sample, y_sample):
        self.attack_net.eval()
        _, ori_label_tensor = torch.max(y_sample.data, dim=-1)
        ori_label = ori_label_tensor.item()
        b_xnew = x_sample.reshape((1, 1, x_sample.shape[1]))
        b_xnew = x_sample.repeat(2, 1, 1)
        output = self.attack_net(b_xnew.float())
        probs = torch.softmax(output[0], dim=-1)
        max_value, max_index = torch.max(probs.data, dim=0)
        prob_list = probs.tolist()
        still_in_prob = prob_list[ori_label]

        return max_value, still_in_prob, probs

    def attack_reward(self, x_sample, y_sample, num):
        x_sample = x_sample.reshape((1, 1, x_sample.shape[1]))
        x_sample = x_sample.repeat(2, 1, 1)
        prob = self.attack_net(x_sample)[0]
        pred = torch.softmax(prob, dim=-1)
        _, ori_max_index = torch.max(y_sample.data, dim=-1)
        means = torch.tensor(0.0, device=pred.device)
        for i in range(num):
            if i == ori_max_index:
                continue
            else:
                means = means + torch.log(1-pred[i])
        mean = means / (num-1)
        reward = torch.log(pred[ori_max_index]) - mean.detach()
        positive_reward = torch.log(pred[ori_max_index])
        negative_reward = (mean.detach())
        return reward, positive_reward, negative_reward

    def get_single_trace_environment_feedback(self, x_sample, y_sample, num):
        reward, positive_reward, negative_reward = self.attack_reward(x_sample, y_sample, num)
        max_prob, still_in_prob, probs = self.get_single_trace_attackmodel_result(x_sample, y_sample)
        return max_prob, still_in_prob, reward

    def get_pred_label(self, traffic):
        b_xnew = traffic.reshape((1, 1, traffic.shape[1]))
        b_xnew = traffic.repeat(2, 1, 1)
        output = self.attack_net(b_xnew)
        probs = torch.softmax(output[1], dim=-1)
        # _, pred_label = torch.argmax(output.data, dim=-1)
        pred_label = torch.max(probs, 0)[1].data
        pre_label = pred_label.tolist()
        # print("pre_label:", pre_label)
        return pre_label

    def get_ori_label(self, traffic_idx):
        ori_max_value, ori_max_index = torch.max(traffic_idx.data, dim=-1)
        ori_max_index = ori_max_index.tolist()
        return ori_max_index

    def check_done(self, traffic, traffic_idx):
        pred_label = self.get_pred_label(traffic)
        ori_max_index = self.get_ori_label(traffic_idx)
        if pred_label != ori_max_index: 
            return True
        else:
            return False 
    
    def MI_update(self, traffic, traffic_idx, i, max_features=5):
        predictor = self.attack_net
        predictor.train()
        predictor.zero_grad()

        # estimator update
        traffic = traffic.reshape((1, 1, traffic.shape[1]))
        traffic = traffic.repeat(2, 1, 1)
        pred = predictor(traffic)
        loss = loss_fn(pred[0], torch.argmax(traffic_idx, dim=0))
        (loss / max_features).backward()

        if i % max_features == 0:
            opt.step()
            opt.zero_grad()





