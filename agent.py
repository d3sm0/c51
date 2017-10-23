import numpy as np
from models import DQN
import tensorflow as tf
from tf_utils import transfer_learning, get_trainable_variables
from utils import LinearSchedule
from memory import ReplayBuffer


class Agent(object):
    def __init__(self, obs_dim, acts_dim, buffer_size=int(10e6), max_steps=100000, expl_fraction=.1, final_eps=.01,
                 num_cpu=4):
        self.acts_dim = acts_dim

        self.target = DQN(name='target', obs_dim=obs_dim, acts_dim=acts_dim)
        self.agent = DQN(name='agent', obs_dim=obs_dim, acts_dim=acts_dim)
        self.memory = ReplayBuffer(size=buffer_size)
        self.scheduler = LinearSchedule(init_value=1., final_value=final_eps, max_steps=(max_steps * expl_fraction))
        self.__sync_op = transfer_learning(to_tensors=get_trainable_variables(scope='target'),
                                           from_tensors=get_trainable_variables('agent'))
        self.sess = self.__make_session(num_cpu=num_cpu)
        self.sess.run(tf.global_variables_initializer())
        self.update_target()

    def update_target(self):
        self.sess.run(self.__sync_op)

    def get_action(self, obs, schedule):
        self.eps = self.scheduler.value(t=schedule)
        if self.eps < np.random.random():
            act = self.sess.run(self.agent.next_action, feed_dict={self.agent.obs: obs})[0]
        else:
            act = np.random.randint(low=0, high=self.acts_dim)
        return act
    def get_p(self, obs):
        return self.sess.run(self.agent.p, feed_dict={self.agent.obs:obs})[0]
    def train(self, obs, acts, rws, obs1, dones):
        # =============
        # TODO this should be done inside the TF graph....
        # =============
        thtz = self.sess.run([self.target.ThTz],
                             feed_dict={self.target.obs: obs1, self.target.rws: rws, self.target.dones: dones})[0]
        loss, _ = self.sess.run([self.agent.cross_entropy, self.agent.train_op],
                                feed_dict={self.agent.obs: obs, self.agent.acts: acts, self.agent.thtz: thtz})
        return loss

    def __make_session(self, num_cpu, memory_fraction=.25):
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu,
            log_device_placement = False
        )
        tf_config.gpu_options.allow_growth = True
        # tf_config.gpu_options.per_rpocess_gpu_memory_fraction = memory_fraction
        return tf.Session(config=tf_config)

    def close(self):
        self.sess.close()
