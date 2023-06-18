import random
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from abc import abstractmethod
from collections import defaultdict

class QAgent:
    def __init__(self, ):
        pass

    @abstractmethod
    def select_action(self, ob):
        pass

class my_QAgent(QAgent):
    def __init__(self):
        super().__init__()
        self.alpha = 0.2  #learning rate
        self.gamma = 0.99 #衰减系数
        self.q_val = defaultdict(lambda:[0.0 , 0.0 , 0.0 , 0.0])

    def select_action(self,obs):
        '''select an action without exploration'''
        q_action = self.q_val[str(obs)]	#TODO:check whether correct
        max_val =q_action[0]
        max_actions = []
        for i in range(4):
            if q_action[i] > max_val:
                max_actions.clear()
                max_actions.append(i)
                max_val = q_action[i]
            elif q_action[i] == max_val:
                max_actions.append(i)
        return random.choice(max_actions)

    def update(self,obs_before,obs_next,action:int,reward):
        '''update the Q function'''
        qs_before = self.q_val[str(obs_before)]
        qs_next = self.q_val[str(obs_next)]
        image_act = self.select_action(obs_next)
        qs_before[action] += self.alpha*(reward+self.gamma*qs_next[image_act]-qs_before[action])
        # qs_before[action] = np.clip(qs_before[action],-100,100)
		

class Model:
    def __init__(self, width, height, policy):
        self.width = width
        self.height = height
        self.policy = policy
        pass

    @abstractmethod
    def store_transition(self, s, a, r, s_):
        pass

    @abstractmethod
    def sample_state(self):
        pass

    @abstractmethod
    def sample_action(self, s):
        pass

    @abstractmethod
    def predict(self, s, a):
        pass

class DynaModel(Model):
    def __init__(self, width, height, policy):
        Model.__init__(self, width, height, policy)
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.buffer = []
        self.action_dic = defaultdict(lambda:[])

    def store_transition(self, s, a, r, s_):
        self.buffer.append([s, a, r, s_])
        if a not in self.action_dic[str(s)]:
            self.action_dic[str(s)].append(a)

    def sample_state(self):
        idx = np.random.randint(0, len(self.buffer))
        return self.buffer[idx][0], idx

    def sample_action(self, s):
        if len(self.action_dic[str(s)]) == 0:
            return self.policy.select_action(s)
        else:
            return random.choice(self.action_dic[str(s)])

    def predict(self, s, a):
        max = 0
        sList= [self.sample_state()[0]]
        for item in self.buffer:
            if str(item[0]) == str(s) and item[1] == a:
                if item[2] > max:
                    sList.clear()
                    sList.append(item[3])
                    max = item[2]
                elif item[2] == max:
                    sList.append(item[3])
        return random.choice(sList)

    def train_transition(self):
        s, a, r, s_ = random.choice(self.buffer)
        lastQ = self.q_table[str(s)][a]
        nextQ = r + self.policy.discountFactor * max(self.q_table[str(s_)])
        self.q_table[str(s)][a] += self.policy.learningRate * (nextQ - lastQ)


class NetworkModel(Model):
    def __init__(self, width, height, policy):
        Model.__init__(self, width, height, policy)
        self.x_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='x')
        self.x_next_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='x_next')
        self.a_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='a')
        self.r_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='r')
        h1 = tf.layers.dense(tf.concat([self.x_ph, self.a_ph], axis=-1), units=256, activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, units=256, activation=tf.nn.relu)
        self.next_x = tf.layers.dense(h2, units=3, activation=tf.nn.tanh) * 1.3 + self.x_ph
        self.x_mse = tf.reduce_mean(tf.square(self.next_x - self.x_next_ph))
        self.opt_x = tf.train.RMSPropOptimizer(learning_rate=1e-5).minimize(self.x_mse)
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.variables_initializer(tf.global_variables()))
        self.buffer = []
        self.sensitive_index = []

    def norm_s(self, s):
        return s

    def de_norm_s(self, s):
        s = np.clip(np.round(s), 0, self.width - 1).astype(np.int32)
        s[2] = np.clip(s[2], 0, 1).astype(np.int32)
        return s

    def store_transition(self, s, a, r, s_):
        s = self.norm_s(s)
        s_ = self.norm_s(s_)
        self.buffer.append([s, a, r, s_])
        if s[-1] - s_[-1] != 0:
            self.sensitive_index.append(len(self.buffer) - 1)

    def train_transition(self, batch_size):
        s_list = []
        a_list = []
        r_list = []
        s_next_list = []
        for _ in range(batch_size):
            idx = np.random.randint(0, len(self.buffer))
            s, a, r, s_ = self.buffer[idx]
            s_list.append(s)
            a_list.append([a])
            r_list.append(r)
            s_next_list.append(s_)

        # if len(self.sensitive_index) > 0:
        #     for _ in range(batch_size):
        #         idx = np.random.randint(0, len(self.sensitive_index))
        #         idx = self.sensitive_index[idx]
        #         s, a, r, s_ = self.buffer[idx]
        #         s_list.append(s)
        #         a_list.append([a])
        #         r_list.append(r)
        #         s_next_list.append(s_)
        
        x_mse = self.sess.run([self.x_mse,  self.opt_x], feed_dict={
            self.x_ph: s_list, self.a_ph: a_list, self.x_next_ph: s_next_list
        })[:1]
        return x_mse

    def sample_state(self):
        idx = np.random.randint(0, len(self.buffer))
        s, a, r, s_ = self.buffer[idx]
        return self.de_norm_s(s), idx

    def sample_action(self, s):
        return self.policy.select_action(s)

    def predict(self, s, a):
        s_ = self.sess.run(self.next_x, feed_dict={self.x_ph: [s], self.a_ph: [[a]]})
        return self.de_norm_s(s_[0])
