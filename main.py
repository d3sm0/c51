import gym
from agent import Agent

from utils import PlotMachine
MAX_STEPS = 100000
TRAIN_FREQ = 5
UPDATE_TARGET_FREQ = 500
BATCH_SIZE = 64
PRINT_FREQ = 500
BUFFER_SIZE = 50000
# TODO:
"""
- add distribution plotter
- add lstm
- add prioritized replay memory
- add a way to manage network config, training config 
- add tensorboard
"""


def main():
    env = gym.make('CartPole-v0')
    agent = Agent(obs_dim=env.observation_space.shape[0], acts_dim=env.action_space.n, max_steps=MAX_STEPS,
                  buffer_size=BUFFER_SIZE)
    plotter = PlotMachine(agent = agent, p_params = {'v_min':0, 'v_max':25,'n_atoms':11} , n_actions = env.action_space.n, action_set = None)
    ob = env.reset()
    ep_rw = 0
    from tf_utils import load_model

    load_model(sess = agent.sess, load_path='logs')
    for t in range(MAX_STEPS):
        if ep_rw > 150:
            env.render()
            plotter.plot_dist(obs = [ob])
        act = agent.get_action(obs=[ob], schedule=t)
        ob1, r, done, _ = env.step(action=act)
        agent.memory.add(step=(ob, act, r, ob1, float(done)))
        ob = ob1.copy()
        ep_rw += r
        if done:
            ob = env.reset()
            if t % TRAIN_FREQ == 0:
                batch = agent.memory.sample(batch_size=BATCH_SIZE)
                loss = agent.train(*batch)
                if t % PRINT_FREQ:
                    print('EP {}, loss {}, eps {}, rw {}'.format(t, loss, agent.eps, ep_rw))
                    from tf_utils import save, get_trainable_variables
                    save(sess = agent.sess, save_path='logs', var_list=get_trainable_variables(scope='target'))
            if t % UPDATE_TARGET_FREQ == 0:
                agent.update_target()
            ep_rw = 0

if __name__ == '__main__':
    main()
