import math
import numpy as np

np.seterr(divide='ignore', invalid='ignore')

# np.random.seed(1)

#if dataset Dk is fixed or not
# f_uav_num=5
# a=np.random.randint(800,1000,size=(f_uav_num,1))
class SystemModel(object):
    def __init__(self,f_uav_num=3):

        self.f_uav_num = f_uav_num              # 底层无人机数，K
        self.n0 = -174 # dBm/hz 噪声功率谱密度
        self.beita = 1.42*10**-4  #  -60db
        self.power_j = 0.1
        self.subbandwidth = 5*10**4
        self.N_B = (10**(-3))*self.subbandwidth*(10**(self.n0/10))
        # self.N_B = 10**(-9)
        self.num_j = 2


        self.p0=80
        self.pi=89
        self.U_tip=120
        self.v0=4.03
        self.zeta=0.6
        self.s=0.05
        self.rou=1.225
        self.A=0.5


        self.k = 0.05
        self.a = 9.53
        self.b = 0.41
        self.Qm = 40
    def p_fly(self,v):
        sum = 0
        e = []
        for i in range(4):
            P=self.p0*(1+3*v[i]**2/(self.U_tip**2))+self.pi*self.v0/v[i]+0.5*self.zeta*self.s*self.rou*self.A*v[i]**3
            sum = sum+P
            e.append(P)
        return sum,e


    #model based on location


    def ch_r_jp(self,jammer_p,uav_p,id):
        dis_jl = []
        G_fl = []
        G_jl = []
        sum_J = 0
        #计算干扰器与无人机的距离
        for i in range(self.num_j):
            dis_i_jl = (jammer_p[i][0]-uav_p[id][0])**2+(jammer_p[i][1]-uav_p[id][1])**2+(uav_p[id][2])**2
            dis_i_jl = math.sqrt(dis_i_jl)
            dis_jl.append(dis_i_jl-self.Qm)

        Plos = []
        for i in range(self.num_j):
            s = math.atan(uav_p[id][2]/math.sqrt((jammer_p[i][0]-uav_p[id][0])**2+(jammer_p[i][1]-uav_p[id][1])**2))
            angle = s / math.pi * 180
            plos = 1/(1+self.a*math.exp(-self.b*(angle-self.a)))
            Plos.append(plos)


        for i in range(self.num_j):
            y = 1/(Plos[i]+(1-Plos[i])*self.k)
            d_il= dis_jl[i]
            g_jl = y*self.beita/d_il**2
            G_jl.append(g_jl)
        #干扰器总干扰功率
        for i in range(self.num_j):     # 干扰器总干扰
            j = self.power_j*G_jl[i]
            sum_J = sum_J+j

        return sum_J

    def down_sinr(self,jammer_p, uav_p ,p,id):
        channel_SNR_up = []
        SINR = []
        dis_fl = []
        dis_jl = []
        G_jl = []
        G_fl = []
        sum_J = 0
        dis_i_fl = (uav_p[id][0]-uav_p[0][0])**2+(uav_p[id][1]-uav_p[0][1])**2+(uav_p[id][2]-uav_p[0][2])**2
        dis_fl.append(dis_i_fl)

        for j in range(self.num_j):
            dis_i_jl = (jammer_p[j][0]-uav_p[id][0])**2+(jammer_p[j][1]-uav_p[id][1])**2+(uav_p[id][2])**2
            dis_i_jl = math.sqrt(dis_i_jl)
            dis_jl.append(dis_i_jl-self.Qm)
        Plos = []
        for i in range(self.num_j):
            s = math.atan(uav_p[id][2]/math.sqrt((jammer_p[i][0]-uav_p[id][0])**2+(jammer_p[i][1]-uav_p[id][1])**2))
            angle = s / math.pi * 180
            plos = 1/(1+self.a*math.exp(-self.b*(angle-self.a)))
            Plos.append(plos)

        d_il = dis_fl[0]
        g_fl = self.beita / d_il
        G_fl.append(g_fl)

        for i in range(self.num_j):
            y = 1 / (Plos[i] + (1 - Plos[i]) * self.k)
            d_il = dis_jl[i]
            g_jl = y * self.beita / d_il ** 2
            G_jl.append(g_jl)
            # 干扰器总干扰功率
        for i in range(self.num_j):  # 干扰器总干扰
            j = self.power_j * G_jl[i]
            sum_J = sum_J + j

        sinr = p * G_fl[0] / (sum_J + self.N_B)
        SINR = 10 * math.log(sinr)
        return SINR







    def ofdma_t_up_(self, jammer_p, uav_p ,p):

        sum_J = 0
        sum_R = 0

        channel_SNR_up = []
        SINR = []
        comm_rate_up = []
        noise = []
        G_fl = []
        G_jl = []

        dis_fl = []
        dis_jl = []
        #计算底层无人机与顶层无人机距离
        for i in range(self.f_uav_num):
            dis_i_fl = (uav_p[i+1][0]-uav_p[0][0])**2+(uav_p[i+1][1]-uav_p[0][1])**2+(uav_p[i+1][2]-uav_p[0][2])**2
            dis_fl.append(dis_i_fl)
        #计算干扰器与顶层无人机的距离
        for i in range(self.num_j):
            dis_i_jl = (jammer_p[i][0]-uav_p[0][0])**2+(jammer_p[i][1]-uav_p[0][1])**2+(uav_p[0][2])**2
            dis_i_jl = math.sqrt(dis_i_jl)
            dis_jl.append(dis_i_jl-self.Qm)

        Plos = []
        for i in range(self.num_j):
            s = math.atan(uav_p[0][2]/math.sqrt((jammer_p[i][0]-uav_p[0][0])**2+(jammer_p[i][1]-uav_p[0][1])**2))
            angle = s / math.pi * 180
            plos = 1/(1+self.a*math.exp(-self.b*(angle-self.a)))
            Plos.append(plos)


        # # 计算白噪声
        # for i in range(self.f_uav_num):
        #     noise.append(10**(-3)*b[i]*10**(self.n0/10))

        #计算信道增益
        for i in range(self.f_uav_num):
            d_il= dis_fl[i]
            g_fl = self.beita/d_il
            G_fl.append(g_fl)

        for i in range(self.num_j):
            y = 1/(Plos[i]+(1-Plos[i])*self.k)
            d_il= dis_jl[i]
            g_jl = y*self.beita/d_il**2
            G_jl.append(g_jl)
        #干扰器总干扰功率
        for i in range(self.num_j):     # 干扰器总干扰
            j = self.power_j*G_jl[i]
            sum_J = sum_J+j

        # 通信模型,上行信道
        # 计算每个底层无人机与顶层无人机之间的SNR
        for i in range(self.f_uav_num):
            sinr = p[i]* G_fl[i] / (sum_J + self.N_B)
            channel_SNR_up.append(sinr)
            SINR.append(10*math.log(sinr))
        channel_SNR_up= np.array(channel_SNR_up).reshape(self.f_uav_num, 1)
        SINR = np.array(SINR).reshape(self.f_uav_num, 1)

        # 计算每个底层无人机与顶层无人机之间数据的传输速率
        for i in range(self.f_uav_num):
            rate_up= 0.001*self.subbandwidth* math.log2(1+channel_SNR_up[i])  #标量bit/s
            comm_rate_up.append(rate_up)

        comm_rate_up= np.array(comm_rate_up).reshape(self.f_uav_num, 1)
        for i in range(self.f_uav_num):
            sum_R = sum_R + comm_rate_up[i][0]

        return sum_R, SINR, comm_rate_up


    def safe_dis(self, start_loaction,dmin):
        uav1 = start_loaction[0]
        uav2 = start_loaction[1]
        uav3 = start_loaction[2]
        uav4 = start_loaction[3]
        sum = 0
        d23 = math.sqrt((uav2[0]-uav3[0])**2+(uav2[1]-uav3[1])**2+(uav2[2]-uav3[2])**2)
        d24 = math.sqrt((uav2[0]-uav4[0])**2+(uav2[1]-uav4[1])**2+(uav2[2]-uav4[2])**2)
        d34 = math.sqrt((uav3[0]-uav4[0])**2+(uav3[1]-uav4[1])**2+(uav3[2]-uav4[2])**2)
        d = [d23,d24,d34]
        for i in range(3):
            if d[i] < dmin:
                sum = sum + 1
        return sum


#     def safe_dis(self, start_loaction,i,dmin):
#         uav1 = start_loaction[0]
#         uav2 = start_loaction[1]
#         uav3 = start_loaction[2]
#         uav4 = start_loaction[3]
#         sum = 0
        
#         if i == 1:
#             d = [math.sqrt((uav2[0]-uav3[0])**2+(uav2[1]-uav3[1])**2+(uav2[2]-uav3[2])**2), math.sqrt((uav2[0]-uav4[0])**2+(uav2[1]-uav4[1])**2+(uav2[2]-uav4[2])**2)]
#             for j in range(2):
#                 if d[j] < dmin:
#                     sum = sum +1
                    
#         if i == 2:
#             d = [math.sqrt((uav3[0]-uav2[0])**2+(uav3[1]-uav2[1])**2+(uav3[2]-uav2[2])**2), math.sqrt((uav3[0]-uav4[0])**2+(uav3[1]-uav4[1])**2+(uav3[2]-uav4[2])**2)]
#             for j in range(2):
#                 if d[j] < dmin:
#                     sum = sum +1
                    
#         if i == 3:
#             d = [math.sqrt((uav4[0]-uav2[0])**2+(uav4[1]-uav4[1])**2+(uav4[2]-uav2[2])**2), math.sqrt((uav4[0]-uav3[0])**2+(uav4[1]-uav3[1])**2+(uav4[2]-uav3[2])**2)]
#             for j in range(2):
#                 if d[j] < dmin:
#                     sum = sum +1
#         return sum




