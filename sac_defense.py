from network_SAC import *
import random 
from reply_buffer import BufferArray



class SACAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, conv_net, actor_lr, critic_lr, alpha_lr, batch_size, tau, gamma, device):
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.conv_net = conv_net
        self.memo = BufferArray(memo_max_len=4000000, state_dim=state_dim, action_dim=action_dim)

        self.actor = PolicyNet(state_dim, hidden_dim, state_dim).to(self.device)
        self.critic_1 = QValueNet(state_dim, hidden_dim, state_dim).to(self.device)
        self.critic_2 = QValueNet(state_dim, hidden_dim, state_dim).to(self.device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim, state_dim).to(self.device) 
        self.target_critic_2 = QValueNet(state_dim, hidden_dim, state_dim).to(self.device)  

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),lr=self.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),lr=self.critic_lr)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float) # alpha hyperparameter
        self.log_alpha.requires_grad = True  
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=self.alpha_lr)
        self.target_entropy = -5  # target_entropy size

    def load_q_model(self, q_policy_model_path):
        self.actor.load_state_dict(torch.load(q_policy_model_path, map_location=self.device, weights_only=True))
        self.actor.eval()

    def get_state(self, traffic):
        state = self.conv_net.return_fullyconnect(traffic)
        return state[0]

    def take_action(self, traffic):
        state = self.get_state(traffic)
        probs = self.actor(state)
        return probs

    def single_trace_mutate(self, traffic, n=5):
        probs = self.take_action(traffic)
        ori_tra_len = int(torch.count_nonzero(traffic[0, :]).item())
        # print("probs: ", probs.shape)
        # dist = torch.distributions.Categorical(probs)
        # max_index = dist.sample((n,))
        max_index = torch.topk(probs, n, largest=True, sorted=True)[1]
        # print("max_index: ", max_index)
        tra_len = ori_tra_len // 5

        while (True):
            if torch.sum(max_index >= tra_len) == 0:
                 break
            else:
                for j in range(n):
                    if max_index[j] >= tra_len:
                        probs[max_index[j]] = float('-inf')
                        max_index = torch.topk(probs, n, largest=True, sorted=True)[1]
        action = max_index

        for j in range(n):
            insert_idx = action[j] * 5
            insert_idx = random.randint(insert_idx, insert_idx+5)
            insert_num = torch.tensor(1, device=self.device) 
            before_insert = traffic[:, :insert_idx] 
            after_insert = traffic[:, insert_idx:]  
            insert_num_expanded = insert_num.unsqueeze(0).expand(before_insert.size(0), -1)
            modified_traffic = torch.cat((before_insert, insert_num_expanded, after_insert), dim=1)[:, :-1]
            traffic = modified_traffic.clone()

        return traffic, action

    def single_trace_mutate_test(self, traffic, n=5):
        probs = self.take_action(traffic)
        ori_tra_len = int(torch.count_nonzero(traffic[0, :]).item())
        tra_len = ori_tra_len // 5

        _, max_index = torch.topk(probs, n, largest=True, sorted=True)
        while (True):
            if torch.sum(max_index >= tra_len) == 0:
                 break
            else:
                for j in range(n):
                    if max_index[j] >= tra_len:
                        probs[max_index[j]] = float('-inf')
                        _, max_index = torch.topk(probs, n, largest=True, sorted=True)
        action = max_index

        for j in range(n):
            insert_idx = action[j] * 5
            insert_idx = random.randint(insert_idx, insert_idx+5)
            insert_num = torch.tensor(1, device=self.device) 
            before_insert = traffic[:, :insert_idx] 
            after_insert = traffic[:, insert_idx:]  
            insert_num_expanded = insert_num.unsqueeze(0).expand(before_insert.size(0), -1)
            modified_traffic = torch.cat((before_insert, insert_num_expanded, after_insert), dim=1)[:, :-1]
            traffic = modified_traffic.clone()

        return traffic, action

    def calc_target(self, rewards, next_states, dones): 
        next_probs  = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def learn(self, n=5):
        experiences = self.memo.random_sample(self.batch_size, self.device)
        rewards = experiences[0]
        masks = experiences[1]
        states = experiences[2]
        actions = experiences[3]
        next_states = experiences[4]

        td_target = self.calc_target(rewards, next_states, masks)
        q1_all = self.critic_1(states)  # [B, A]
        q2_all = self.critic_2(states)  # [B, A]
        gather_indices = actions.long().to(self.device)  # [B, n]
        critic_1_q_values = q1_all.gather(1, gather_indices).sum(dim=1, keepdim=True)
        critic_2_q_values = q2_all.gather(1, gather_indices).sum(dim=1, keepdim=True)
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),dim=1,keepdim=True) 
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def soft_update(self):
        for param_target, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        for param_target, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)



        

