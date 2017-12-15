import numpy as np

from memory import ReplayBuffer
from models import DQN
from utils.misc import LinearSchedule
from utils.tf_utils import transfer_learning, get_trainable_variables, make_session, init_graph


class Agent(object):
    def __init__(self, obs_dim, acts_dim, buffer_size=int(1e5), max_steps=100000, expl_fraction=.1, final_eps=.001,
                 num_cpu=4, topology="linear"):
        self.acts_dim = acts_dim
        self.eps = 1.
        self.target = DQN(name='target', obs_dim=obs_dim, acts_dim=acts_dim, topology=topology)
        self.agent = DQN(name='agent', obs_dim=obs_dim, acts_dim=acts_dim, topology=topology,
                         target_thtz=self.target.ThTz)
        self.memory = ReplayBuffer(size=buffer_size)
        self.scheduler = LinearSchedule(init_value=1., final_value=final_eps, max_steps=(max_steps * expl_fraction))
        self.__sync_op = transfer_learning(to_tensors=get_trainable_variables(scope='target'),
                                           from_tensors=get_trainable_variables('agent'))
        self.sess = make_session(num_cpu=num_cpu)
        self._reset_graph()
        self.update_target()

    def get_train_summary(self, feed_dict):

        return self.sess.run([self.agent._summary_op, self.agent._global_step], feed_dict=feed_dict)

    def _reset_graph(self):
        init_graph(sess=self.sess)

    def update_target(self):
        self.sess.run(self.__sync_op)

    def step(self, obs, schedule):
        self.eps = self.scheduler.value(t=schedule)
        if self.eps < np.random.random():
            act = self.sess.run(self.agent.next_action, feed_dict={self.agent.obs: obs})[0]
        else:
            act = np.random.randint(low=0, high=self.acts_dim)
        return act

    def get_p(self, obs):
        return self.sess.run(self.agent.p, feed_dict={self.agent.obs: obs})[0]

    def sample(self, batch_size):
        return self.memory.sample(batch_size=batch_size)

    def train(self, obs, acts, rws, obs1, dones):

        # thtz = self.sess.run([self.target.ThTz],
        #                      feed_dict={self.target.obs: obs1, self.target.rws: rws, self.target.dones: dones})[0]

        feed_dict = {
            self.target.obs: obs1, self.target.rws: rws, self.target.dones: dones, self.agent.obs: obs,
            self.agent.acts: acts, self.agent.rws: rws
        }
        loss, _ = self.sess.run([self.agent.cross_entropy, self.agent.train_op],
                                feed_dict=feed_dict)

        return loss, feed_dict

    def close(self):
        self.sess.close()
