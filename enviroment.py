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
        with torch.no_grad():
            _, ori_label_tensor = torch.max(y_sample, dim=-1)
            ori_label = int(ori_label_tensor.item())
            b_xnew = x_sample.reshape((1, 1, x_sample.shape[1]))
            b_xnew = x_sample.repeat(2, 1, 1)
            output = self.attack_net(b_xnew.float())
            probs = torch.softmax(output[0], dim=-1)
            max_value, max_index = torch.max(probs, dim=0)
            still_in_prob = probs[ori_label].item()

        return max_value, still_in_prob, probs

    def attack_reward(self, x_sample, y_sample, num):
        with torch.no_grad():
            x_sample = x_sample.reshape((1, 1, x_sample.shape[1]))
            x_sample = x_sample.repeat(2, 1, 1)
            prob = self.attack_net(x_sample)[0]
            pred = torch.softmax(prob, dim=-1)
            _, ori_max_index = torch.max(y_sample, dim=-1)
            means = torch.tensor(0.0, device=pred.device)
            for i in range(num):
                if i == int(ori_max_index.item()):
                    continue
                else:
                    means = means + torch.log(1 - pred[i])
            mean = means / (num - 1)
            reward = torch.log(pred[ori_max_index]) - mean
            positive_reward = torch.log(pred[ori_max_index])
            negative_reward = mean
        return reward, positive_reward, negative_reward

    def get_single_trace_environment_feedback(self, x_sample, y_sample, num):
        reward, positive_reward, negative_reward = self.attack_reward(x_sample, y_sample, num)
        max_prob, still_in_prob, probs = self.get_single_trace_attackmodel_result(x_sample, y_sample)
        return max_prob, still_in_prob, reward

    def get_pred_label(self, traffic):
        with torch.no_grad():
            b_xnew = traffic.reshape((1, 1, traffic.shape[1]))
            b_xnew = traffic.repeat(2, 1, 1)
            output = self.attack_net(b_xnew)
            probs = torch.softmax(output[1], dim=-1)
            pred_label = torch.max(probs, 0)[1]
            pre_label = int(pred_label.item())
            return pre_label

    def get_ori_label(self, traffic_idx):
        with torch.no_grad():
            _, ori_max_index = torch.max(traffic_idx, dim=-1)
            return int(ori_max_index.item())

    def check_done(self, traffic, traffic_idx):
        pred_label = self.get_pred_label(traffic)
        ori_max_index = self.get_ori_label(traffic_idx)
        if pred_label != ori_max_index: 
            return True
        else:
            return False 
    
    def MI_update(self, traffic, traffic_idx, i, update_every=5):
        # 使用 attack_net 进行交叉熵更新，降低更新频率
        self.attack_net.train()
        self.attack_net.zero_grad()
        b_x = traffic.reshape((1, 1, traffic.shape[1]))
        b_x = traffic.repeat(2, 1, 1)
        logits = self.attack_net(b_x)[0]
        target = torch.argmax(traffic_idx, dim=0)
        loss = self.mi_loss_fn(logits, target)
        (loss/update_every).backward()
        if i % update_every == 0:
            self.mi_optimizer.step()
            self.mi_optimizer.zero_grad()




