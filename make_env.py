import numpy as np
import math
from systemmodel import SystemModel
from numpy import random


class make_env:
    def __init__(self):
        self.n_uav = 4
        self.n_jammer = 2
        self.sys = SystemModel()
        self.t = 0.5
        self.vmax = 60.0
        self.vmin = 10.0
        self.pmax = 0.1
        self.pmin = 0.01
        self.bmax = 4*5*10**4
        self.bmin = 5*10**4
        self.sinr_th = 3
        self.dismin = 40
        self.dismax = 100
        self.xrange = [0,600]
        self.yrange = [-400,400]
        self.zrange = [0,250]
        self.sinrth = 3
        # self.start_location = [[100,-300,150],[50,-350,140],[150,-350,140],[50,-250,140]]
        # self.jammer_location = [[200,-50],[350,50]]
        # self.end_location = [[500,300,150],[450,250,140],[550,250,140],[450,350,140]]
        # self.start_v = [10,10,10,10]
        # self.start_p = [0.01,0.01,0.01,0.01]

        

#观察空间   uav 坐标xyz  uav与终点的距离 速度 ；；；；；  随机生成初始无人机的位置，终点位置，干扰源位置 速度 发射功率
    def reset(self):
        self.start_location = [[100,-300,150],[50,-350,140],[150,-350,140],[50,-250,140]]
        self.jammer_location = [[200,-50],[350,50]]
        self.end_location = [[500,300,150],[450,250,140],[550,250,140],[450,350,140]]
        self.start_v = [10,10,10,10]
        self.start_p = [0.01,0.01,0.01,0.01]

        R, SINR, comm_rate_up = self.sys.ofdma_t_up_(self.jammer_location, self.start_location, self.start_p[1:4])

        xyz = []
        for i in range(self.n_uav):
            #终点
            d1 = math.sqrt((self.start_location[i][0]-self.end_location[i][0])**2+(self.start_location[i][1]-self.end_location[i][1])**2+(self.start_location[i][2]-self.end_location[i][2])**2)/1000
            #uav位置坐标
            d2 = self.start_location[i][0]/600
            d3 = self.start_location[i][1]/800
            d4 = self.start_location[i][2]/150
            #到干扰器距离
            d5 = math.sqrt((self.start_location[i][0]-self.jammer_location[0][0])**2+(self.start_location[i][1]-self.jammer_location[0][1])**2+(self.start_location[i][2])**2)/1000
            d6 = math.sqrt((self.start_location[i][0]-self.jammer_location[1][0])**2+(self.start_location[i][1]-self.jammer_location[1][1])**2+(self.start_location[i][2])**2)/1000
            #速度
            d7 = self.start_v[i]/50
            #功率
            d8 = self.start_p[i]/0.1
            #干扰功率
            d9 = self.sys.ch_r_jp(self.jammer_location, self.start_location,i)/0.2
            #上行总速率
            d10 = R/400
            b = []
            b.append(d1)
            b.append(d2)
            b.append(d3)
            b.append(d4)
            b.append(d5)
            b.append(d6)
            b.append(d7)
            b.append(d8)
            b.append(d9)
            b.append(d10)
            xyz.append(b)
        self.state = np.vstack([per_uav for (_, per_uav) in enumerate(xyz)])
        return self.state




#[seita fan v] [seita fan v p b]
    def step(self, action):
        Flag = 0
        done = [False, False, False, False]
        state = self.state
        self.action = action
#==========================================================转出环境动作=================================
        env_action = []
        for a in self.action:
            act = []
            fan = (a[0]+1) * math.pi/2
            vv = (self.vmax-self.vmin)/2.0*a[1]+(self.vmax+self.vmin)/2.0
            pp = (self.pmax-self.pmin)/2.0*a[2]+(self.pmax+self.pmin)/2.0
            dz = 0
            dx = vv*self.t*math.cos(fan)
            dy = vv*self.t*math.sin(fan)
            act.append(dx)
            act.append(dy)
            act.append(dz)
            act.append(vv)
            act.append(pp)
            env_action.append(act)

#--------------------------------------------------------------------------------------
        #更新无人机速度
        v= [env_action[0][3],env_action[1][3],env_action[2][3],env_action[3][3]]
        #更新无人机功率
        p = [env_action[0][4],env_action[1][4],env_action[2][4],env_action[3][4]]
        #更新uav位置
        dp_uav = []
        mk = []
        for a in env_action:
            b = a[0:3]  # 列表
            dp_uav.append(b)
        for i, j in zip(self.start_location, dp_uav):
            result = [a + b for a, b in zip(i, j)]
            mk.append(result)
        self.start_location = mk  # 更新uav位置
#--------------------------------------------------------------------------------------------
        #计算无人机到目标的位置
        ddd = []
        for i in range(self.n_uav):
            d = math.sqrt((self.start_location[i][0]-self.end_location[i][0])**2+(self.start_location[i][1]-self.end_location[i][1])**2+(self.start_location[i][2]-self.end_location[i][2])**2)
            ddd.append(d)
#-----------------------------------------------------------------------------------------------
        #得到下一个状态
        ########################################
        R, SINR, comm_rate_up = self.sys.ofdma_t_up_(self.jammer_location, self.start_location, p[1:4])
        down_sinr1 = self.sys.down_sinr(self.jammer_location, self.start_location, p[0], 1)
        down_sinr2 = self.sys.down_sinr(self.jammer_location, self.start_location, p[0], 2)
        down_sinr3 = self.sys.down_sinr(self.jammer_location, self.start_location, p[0], 3)
        down_sinr = [down_sinr1,down_sinr2,down_sinr3]

        xyz = []
        for i in range(self.n_uav):
            #终点
            d1 = math.sqrt((self.start_location[i][0]-self.end_location[i][0])**2+(self.start_location[i][1]-self.end_location[i][1])**2+(self.start_location[i][2]-self.end_location[i][2])**2)/1000
            #uav位置坐标
            d2 = self.start_location[i][0]/600
            d3 = self.start_location[i][1]/800
            d4 = self.start_location[i][2]/150
            #到干扰器距离
            d5 = math.sqrt((self.start_location[i][0]-self.jammer_location[0][0])**2+(self.start_location[i][1]-self.jammer_location[0][1])**2+(self.start_location[i][2])**2)/1000
            d6 = math.sqrt((self.start_location[i][0]-self.jammer_location[1][0])**2+(self.start_location[i][1]-self.jammer_location[1][1])**2+(self.start_location[i][2])**2)/1000
            #速度
            d7 = v[i]/50
            #功率
            d8 = p[i]/0.1
            #干扰功率
            d9 = self.sys.ch_r_jp(self.jammer_location, self.start_location,i)/0.2
            #上行总速率
            d10 = R/400
            b = []
            b.append(d1)
            b.append(d2)
            b.append(d3)
            b.append(d4)
            b.append(d5)
            b.append(d6)
            b.append(d7)
            b.append(d8)
            b.append(d9)
            b.append(d10)
            xyz.append(b)
        self.state = np.vstack([per_uav for (_, per_uav) in enumerate(xyz)])
#--------------------------------------------------------------------------------------------------
    #计算奖励
        Reward = []
        r11 = ddd[0]
        r21 = ddd[1]
        r31 = ddd[2]
        r41 = ddd[3]
        r12 = self.sys.safe_dis(self.start_location,self.dismin)
        r22 = self.sys.safe_dis(self.start_location,self.dismin)
        r32 = self.sys.safe_dis(self.start_location,self.dismin)
        r42 = self.sys.safe_dis(self.start_location,self.dismin)
#         rupsinr = 0
#         for i in range(self.n_uav-1):
#             if (self.sinrth-SINR[i][0])<0:
#                 rupsinr = rupsinr+0
#             else:
#                 rupsinr = rupsinr + (self.sinrth-SINR[i][0])

#         rdownsinr = 0
#         for i in range(self.n_uav-1):
#             if (self.sinrth-down_sinr[i])<0:
#                 rdownsinr = rdownsinr+0
#             else:
#                 rdownsinr = rdownsinr + (self.sinrth-down_sinr[i])
                
        
        rupsinr = 0
        for i in range(self.n_uav-1):
            if SINR[i][0]<self.sinrth:
                rupsinr = rupsinr+1

        rdownsinr = 0
        for i in range(self.n_uav-1):
            if down_sinr[i]<self.sinrth:
                rdownsinr = rdownsinr+1

#         reward1 = R-r11-100*rupsinr-100*rdownsinr-1000*r12
#         reward2 = R-r21-100*rupsinr-100*rdownsinr-1000*r22
#         reward3 = R-r31-100*rupsinr-100*rdownsinr-1000*r32
#         reward4 = R-r41-100*rupsinr-100*rdownsinr-1000*r42


        # reward1 = R-0.1*r11-50*rupsinr-50*rdownsinr
        # reward2 = R-0.1*r21-50*rupsinr-50*rdownsinr
        # reward3 = R-0.1*r31-50*rupsinr-50*rdownsinr
        # reward4 = R-0.1*r41-50*rupsinr-50*rdownsinr
        
        
        reward1 = R-0.01*r11
        reward2 = R-0.01*r21
        reward3 = R-0.01*r31
        reward4 = R-0.01*r41

        Reward = [reward1,reward2,reward3,reward4]
        if r12 > 0:
            Reward[0] = Reward[0] - 1000
            Reward[1] = Reward[1] - 1000
            Reward[2] = Reward[2] - 1000
            Reward[3] = Reward[3] - 1000
        
        if rupsinr > 0:
            Reward[0] = Reward[0] - 1000
            Reward[1] = Reward[1] - 1000
            Reward[2] = Reward[2] - 1000
            Reward[3] = Reward[3] - 1000
            
        if rdownsinr > 0:
            Reward[0] = Reward[0] - 1000
            Reward[1] = Reward[1] - 1000
            Reward[2] = Reward[2] - 1000
            Reward[3] = Reward[3] - 1000
        for i in range(4):
            if ddd[i]<100:
                Reward[i] = Reward[i] + 1000
                done[i] = True
            if self.start_location[i][0] < 0 or self.start_location[i][0] > 600 or self.start_location[i][1]<-400 or self.start_location[i][1]>400 or self.start_location[i][2] < 50 or self.start_location[i][2] >250:
                Reward[i] = Reward[i] - 1000
                Flag = Flag + 1
        for i in range(4):
            Reward[i] = Reward[i]/100
##################################################################################3
        return self.state, Reward, done, R,self.start_location,rupsinr,rdownsinr,SINR[0][0],SINR[1][0],SINR[2][0],down_sinr1,down_sinr2,down_sinr3,r11,r21,r31,r41,r12,r22,r32,r42,Flag