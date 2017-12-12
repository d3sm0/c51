import gym

from agent import Agent
# from utils import PlotMachine
from utils.logger import Logger
from utils.tf_utils import set_global_seed


def train(env_id, max_steps, topology="linear", batch_size=64, train_freq=1, update_freq=500, save_freq=5000):
    env = gym.make("CartPole-v0")
    #
    agent = Agent(obs_dim=env.observation_space.shape[0], acts_dim=env.action_space.n, topology=topology,
                  max_steps=max_steps)
    # # plotter = PlotMachine(agent=agent, v_min=agent_config['v_min'], v_max=agent_config['v_max'],
    #                       nb_atoms=agent_config['nb_atoms'],
    #                       n_actions=env.action_space.n, action_set=None)

    logger = Logger(log_dir='logs', var_list=agent.target._params)
    ep, ep_rw = 0, 0

    ob = env.reset()
    try:
        for t in range(max_steps):
            act = agent.step(obs=[ob], schedule=t)
            #             plotter.plot_dist(obs=[ob])
            ob1, r, done, _ = env.step(action=act)
            agent.memory.add(step=(ob, act, r, ob1, float(done)))
            ob = ob1.copy()
            ep_rw += r
            if done:
                env.reset()
                ep += 1
                ep_rw = 0
            if t % train_freq == 0:
                batch = agent.sample(batch_size=batch_size)
                loss, feed_dict = agent.train(*batch)
            if t % update_freq == 0:
                agent.update_target()
                summary, global_step = agent.get_train_summary(feed_dict=feed_dict)
                ep_stats = {
                    'loss': loss,
                    'agent_eps': agent.eps,
                    'ep_rw': ep_rw,
                    'total_ep': ep,
                    'total_steps': t
                }
                logger.log(ep_stats, total_ep=ep)
                logger.dump(stats=ep_stats, tf_summary=summary, global_step=t)

            if t % save_freq == 0:
                logger.save_model(sess=agent.sess, global_step=t)

    except KeyboardInterrupt:
        logger.save_model(sess=agent.sess, global_step=t)
        agent.close()
        print('Closing experiment. File saved at {}'.format(logger.save_path))


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'linear'],
                        default='linear')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'],
                        default='constant')
    parser.add_argument('--logdir', help='Directory for logging', default='logs')
    parser.add_argument('--max-timesteps', type=int, default=int(10e6))
    args = parser.parse_args()
    set_global_seed(args.seed)
    train(args.env, max_steps=args.max_timesteps, topology=args.policy)


if __name__ == '__main__':
    main()
