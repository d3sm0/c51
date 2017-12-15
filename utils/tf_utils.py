import os

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def build_z(v_min, v_max, n_atoms):
    dz = (v_max - v_min) / (n_atoms - 1)
    z = tf.range(v_min, v_max + dz / 2, dz, dtype=tf.float32, name='z')
    return z, dz


def p_to_q(p_dist, v_min, v_max, n_atoms):
    z, _ = build_z(v_min, v_max, n_atoms)
    return tf.tensordot(p_dist, z, axes=[[-1], [-1]])


def set_global_seed(seed=1234):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    pass


def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


def transfer_learning(to_tensors, from_tensors, tau=1.):
    update_op = []
    for from_t, to_t in zip(sorted(from_tensors, key=lambda v: v.name),
                            sorted(to_tensors, key=lambda v: v.name)):
        update_op.append(
            # C <- C * tau + C_old * (1-tau)
            tf.assign(to_t, tf.multiply(from_t, tau) + tf.multiply(to_t, 1. - tau))
        )
    return update_op  # tf.group(*update_op)


def get_pa(p, acts, batch_size):
    cat_idx = tf.transpose(
        tf.reshape(tf.concat([tf.range(batch_size), acts], axis=0), shape=[2, batch_size]))
    p_target = tf.gather_nd(params=p, indices=cat_idx)
    return p_target


def load_model(sess, load_path, var_list=None):
    ckpt = tf.train.load_checkpoint(ckpt_dir_or_file=load_path)
    saver = tf.train.Saver(var_list=var_list)
    try:
        saver.restore(sess=sess, save_path=ckpt)
    except Exception as e:
        tf.logging.error(e)


def save(sess, save_path, var_list=None):
    os.makedirs(save_path, exist_ok=True)
    saver = tf.train.Saver(var_list=var_list)
    try:
        saver.save(sess=sess, save_path=os.path.join(save_path, 'model.ckpt'),
                   write_meta_graph=False)
    except Exception as e:
        tf.logging.error(e)


def create_saver(var_list):
    return tf.train.Saver(var_list=var_list, save_relative_paths=True, reshape=True)


def create_writer(path, suffix):
    return tf.summary.FileWriter(logdir=path, flush_secs=360, filename_suffix=suffix)


def create_summary():
    return tf.summary.Summary()


def fc(x, h_size, name, act=tf.nn.relu, std=0.1):
    with tf.variable_scope(name):
        input_size = x.get_shape()[1]
        w = tf.get_variable('w', (input_size, h_size), initializer=tf.random_normal_initializer(stddev=std))
        b = tf.get_variable('b', (h_size), initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        return act(z)


def init_graph(sess):
    sess.run(tf.global_variables_initializer())
    tf.logging.info('Graph initialized')


def make_config(num_cpu, memory_fraction=.25):
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu,
        log_device_placement=False
    )
    tf_config.gpu_options.allow_growth = True
    return tf_config


def make_session(num_cpu=1):
    return tf.Session(config=make_config(num_cpu=num_cpu))


# def make_session(num_cpu, memory_fraction=.25):
#     tf_config = tf.ConfigProto(
#         inter_op_parallelism_threads=num_cpu,
#         intra_op_parallelism_threads=num_cpu,
#         log_device_placement = False
#     )
#     tf_config.gpu_options.allow_growth = True
#     # tf_config.gpu_options.per_rpocess_gpu_memory_fraction = memory_fraction
#     return tf.Session(config=tf_config)
def fc_noisy(x, h_size, name, reuse=False, act=lambda x: x):
    d = x.get_shape().as_list()[1]

    mu_0 = tf.random_uniform_initializer(minval=-1 / np.power(d, .5), maxval=1 / np.power(d, .5))
    sigma_0 = tf.constant_initializer(.4 / np.power(d, .5))

    p = tf.random_normal([d, 1])
    q = tf.random_normal([1, h_size])
    f_p = tf.multiply(tf.sign(p), tf.pow(tf.abs(p), .5))
    f_q = tf.multiply(tf.sign(q), tf.pow(tf.abs(q), .5))

    w_eps = tf.multiply(f_p, f_q)
    b_eps = tf.squeeze(f_q)

    with tf.variable_scope(name, reuse=reuse):
        w_mu = tf.get_variable('w_mu', shape=[d, h_size], initializer=mu_0)
        w_sigma = tf.get_variable('w_sigma', shape=[d, h_size], initializer=sigma_0)
        w = w_mu + tf.multiply(w_sigma, w_eps)

        b_mu = tf.get_variable('b_mu', shape=[h_size], initializer=mu_0)
        b_sigma = tf.get_variable('b_sigma', shape=[h_size], initializer=sigma_0)
        b = b_mu + tf.multiply(b_sigma, b_eps)
        z = tf.matmul(x, w) + b

    return act(z)
