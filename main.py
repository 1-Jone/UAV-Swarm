import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from make_env import make_env
import argparse
from replay_buffer import ReplayBuffer
# from maddpg import MADDPG
from matd3 import MATD3
import copy
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(columns=['t','r','ee'])
df.to_csv("train_data.csv",index=False)
df1 = pd.DataFrame(columns=['t','x','y'])
df1.to_csv("train_data1.csv",index=False)
df2 = pd.DataFrame(columns=['t','x','y'])
df2.to_csv("train_data2.csv",index=False)
df3 = pd.DataFrame(columns=['t','x','y'])
df3.to_csv("train_data3.csv",index=False)
df4 = pd.DataFrame(columns=['t','x','y'])
df4.to_csv("train_data4.csv",index=False)
df5 = pd.DataFrame(columns=['t','1','2','3','4','5','6'])
df5.to_csv("train_data5.csv",index=False)

df6 = pd.DataFrame(columns=['t','1','2','3','4','5','6','7'])
df6.to_csv("train_data6.csv",index=False)

df7 = pd.DataFrame(columns=['t','1','2','3','4','5','6','7','8'])
df7.to_csv("train_data7.csv",index=False)
class Runner:
    def __init__(self, args, env_name,number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        self.env = make_env()  # Continuous action space
        self.env_evaluate = make_env()
        self.args.N = self.env.n_uav  # The number of agents
        self.args.obs_dim_n = [10,10,10,10]  # obs dimensions of N agents
        self.args.action_dim_n = [3,3,3,3]
        #print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        #print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create N agents
        if self.args.algorithm == "MADDPG":
            print("Algorithm: MADDPG")
            self.agent_n = [MADDPG(args, agent_id) for agent_id in range(args.N)]
        elif self.args.algorithm == "MATD3":
            print("Algorithm: MATD3")
            self.agent_n = [MATD3(args, agent_id) for agent_id in range(args.N)]
        else:
            print("Wrong!!!")

        self.replay_buffer = ReplayBuffer(self.args)

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0

        self.noise_std = self.args.noise_std_init  # Initialize noise_std
    def run(self, ):
        # self.evaluate_policy()
        reward = []
        uav = []

        while self.total_steps < self.args.max_train_steps:
            obs_n = self.env.reset()
            # print(obs_n)

            episodereward = 0
            epsiode = 0
            RR = 0
            if self.total_steps == self.args.max_train_steps - 1:
                for _ in range(200):
                    a_n = [agent.test_choose_action(obs, noise_std=self.noise_std) for agent, obs in zip(self.agent_n, obs_n)]
                    obs_next_n, r_n, done_n, R,uavp,rupsinr,rdownsinr,SINR1,SINR2,SINR3,down_sinr1,down_sinr2,down_sinr3,r11,r21,r31,r41,r12,r22,r32,r42,f= self.env.step(copy.deepcopy(a_n))
                    episodereward = episodereward + r_n[0]+r_n[1]+r_n[2]+r_n[3]
                    epsiode = epsiode + 1
                    RR = RR + R
                    uav.append(uavp)
                    list1 = [epsiode,uavp[0][0],uavp[0][1]]
                    data1 = pd.DataFrame([list1])
                    data1.to_csv("train_data1.csv", mode='a', header=False, index=False)
                    
                    list2 = [epsiode,uavp[1][0],uavp[1][1]]
                    data2 = pd.DataFrame([list2])
                    data2.to_csv("train_data2.csv", mode='a', header=False, index=False)
                    
                    list3 = [epsiode,uavp[2][0],uavp[2][1]]
                    data3 = pd.DataFrame([list3])
                    data3.to_csv("train_data3.csv", mode='a', header=False, index=False)
                    
                    list4 = [epsiode,uavp[3][0],uavp[3][1]]
                    data4 = pd.DataFrame([list4])
                    data4.to_csv("train_data4.csv", mode='a', header=False, index=False)

                    list5 = [epsiode,SINR1,SINR2,SINR3,down_sinr1,down_sinr2,down_sinr3]
                    data5 = pd.DataFrame([list5])
                    data5.to_csv("train_data5.csv", mode='a', header=False, index=False)

                    obs_n = obs_next_n
                for agent_id in range(self.args.N):
                    self.agent_n[agent_id].save_model(self.args.algorithm, self.total_steps, agent_id)
            else:
                for _ in range(200):
                    a_n = [agent.choose_action(obs, noise_std=self.noise_std) for agent, obs in zip(self.agent_n, obs_n)]
                    obs_next_n, r_n, done_n,R,uavp,rupsinr,rdownsinr,SINR1,SINR2,SINR3,down_sinr1,down_sinr2,down_sinr3,r11,r21,r31,r41,r12,r22,r32,r42,f = self.env.step(copy.deepcopy(a_n))

                    episodereward = episodereward + r_n[0]+r_n[1]+r_n[2]+r_n[3]
                    epsiode = epsiode + 1
                    RR = RR + R
                    # print(R)
                    uav.append(uavp)
                    list1 = [epsiode,uavp[0][0],uavp[0][1]]
                    data1 = pd.DataFrame([list1])
                    data1.to_csv("train_data1.csv", mode='a', header=False, index=False)
                    
                    list2 = [epsiode,uavp[1][0],uavp[1][1]]
                    data2 = pd.DataFrame([list2])
                    data2.to_csv("train_data2.csv", mode='a', header=False, index=False)
                    
                    list3 = [epsiode,uavp[2][0],uavp[2][1]]
                    data3 = pd.DataFrame([list3])
                    data3.to_csv("train_data3.csv", mode='a', header=False, index=False)
                    
                    list4 = [epsiode,uavp[3][0],uavp[3][1]]
                    data4 = pd.DataFrame([list4])
                    data4.to_csv("train_data4.csv", mode='a', header=False, index=False)

                    list5 = [epsiode,SINR1,SINR2,SINR3,down_sinr1,down_sinr2,down_sinr3]
                    data5 = pd.DataFrame([list5])
                    data5.to_csv("train_data5.csv", mode='a', header=False, index=False)
                    
                    list6 = [epsiode,SINR1,SINR2,SINR3,down_sinr1,down_sinr2,down_sinr3,f]
                    data6 = pd.DataFrame([list6])
                    data6.to_csv("train_data6.csv", mode='a', header=False, index=False)
                    
                    list7 = [epsiode,r11,r21,r31,r41,r12,r22,r32,r42]
                    data7 = pd.DataFrame([list7])
                    data7.to_csv("train_data7.csv", mode='a', header=False, index=False)
                    
                    self.replay_buffer.store_transition(obs_n, a_n, r_n, obs_next_n, done_n)

                    if self.args.use_noise_decay:
                        self.noise_std = self.noise_std - self.args.noise_std_decay if self.noise_std - self.args.noise_std_decay > self.args.noise_std_min else self.args.noise_std_min

                    if self.replay_buffer.current_size > self.args.batch_size:
                        # Train each agent individually
                        for agent_id in range(self.args.N):
                            self.agent_n[agent_id].train(self.replay_buffer, self.agent_n)
                    obs_n = obs_next_n
            self.total_steps += 1
            reward.append(episodereward/epsiode)
            print("total_steps:{} \t episodereward:{} \t r:{} \t episode:{} ".format(self.total_steps, episodereward/epsiode,RR,epsiode))
            list = [self.total_steps, episodereward/epsiode,RR]
            data = pd.DataFrame([list])
            data.to_csv("train_data.csv", mode='a', header=False, index=False)
        return reward,uav,epsiode



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG and MATD3 in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3000), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")
    parser.add_argument("--max_action", type=float, default=1.0, help="Max action")

    parser.add_argument("--algorithm", type=str, default="MATD3", help="MADDPG or MATD3")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.2, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.05, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=3e5, help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-3, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Softly update the target network")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    # --------------------------------------MATD3--------------------------------------------------------------------
    parser.add_argument("--policy_noise", type=float, default=0.2, help="Target policy smoothing")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="Clip noise")
    parser.add_argument("--policy_update_freq", type=int, default=2, help="The frequency of policy updates")

    args = parser.parse_args()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

    env_names = ["simple_speaker_listener", "simple_spread"]
    env_index = 0
    runner = Runner(args,env_names[env_index], number=1, seed=0)
    reward,uav,epsiode = runner.run()
    # print(uav)
    x= [[],[],[],[]]
    y = [[], [], [], []]
    z = [[], [], [], []]
    plt.figure(1, dpi=200)
    for i in range(epsiode):
        for j in range(4):
            x[j].append(uav[i][j][0])
            y[j].append(uav[i][j][1])
            z[j].append(uav[i][j][2])
    ax = plt.axes(projection='3d')
    ax.set_xlim(0,600)
    ax.set_ylim(-400,400)
    ax.set_zlim(0,250)

    ax.scatter(100,-300,150, color='r')
    ax.scatter(50,-350,140, color='b')
    ax.scatter(150,-350,140 , color='g')
    ax.scatter(50,-250,140 , color='y')
    ax.scatter(200,-50 ,0, color='r')
    ax.scatter(350,50,0, color='r')
    
    ax.plot(x[0], y[0], z[0],color='r', linestyle='-', linewidth=2)
    ax.plot(x[1], y[1],z[1], color='b', linestyle='-', linewidth=2)
    ax.plot(x[2], y[2],z[2], color='g', linestyle='-', linewidth=2)
    ax.plot(x[3], y[3], z[3],color='y', linestyle='-', linewidth=2)

    plt.savefig('tf-logs/fig1.png')

    plt.figure(2,dpi=200)
    plt.plot(reward,color='b', linewidth=1, linestyle='-')
    plt.xlabel('Episodes')
    plt.ylabel('Total reward')
    plt.savefig('tf-logs/fig2.png')

    

    x1= [[],[],[],[]]
    y1 = [[], [], [], []]
    for i in range(epsiode):
        for j in range(4):
            x[j].append(uav[i][j][0])
            y[j].append(uav[i][j][1])
    plt.figure(3, dpi=200)
    plt.xlim(0,600)
    plt.ylim(-400,400)
    plt.plot(x[0],y[0],color='r')
    plt.scatter(100,-300, color='r')
    plt.scatter(200,-50, color='r')
    plt.scatter(350,50, color='r')
    plt.savefig('tf-logs/fig3.png')


    plt.figure(4,dpi=200)
    plt.xlim(0,600)
    plt.ylim(-400,400)
    plt.scatter(50,-350, color='b')
    plt.scatter(150,-350, color='g')
    plt.scatter(50,-250, color='y')
    plt.scatter(200,-50, color='r')
    plt.scatter(350,50, color='r')
    plt.plot(x[1],y[1],color='b')
    plt.plot(x[2],y[2],color='g')
    plt.plot(x[3],y[3],color='y')
    plt.savefig('tf-logs/fig4.png')

    plt.show()