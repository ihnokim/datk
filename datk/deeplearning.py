import tensorflow as tf
import numpy as np
import pandas as pd
import datk
import os
import matplotlib.pyplot as plt
from IPython import display
import pylab as pl


def collect_data(df, collector):
    # collector(row) should return (x, y)
    xret = []
    yret = []
    for row in df.itertuples():
        xrow, yrow = collector(row)
        xret.append(xrow)
        yret.append(yrow)
    return np.array(xret), np.array(yret)


def rsq(v1, v2):
    mu1, var1 = tf.nn.moments(v1, axes=[0])
    mu2, var2 = tf.nn.moments(v2, axes=[0])
    d1 = v1 - mu1 * tf.ones(tf.shape(v1))
    d2 = v2 - mu2 * tf.ones(tf.shape(v2))
    return tf.square(tf.divide(tf.reduce_mean(tf.multiply(d1, d2), 0), tf.sqrt(tf.multiply(var1, var2))))


class NeuralNetwork():
    def __init__(self, hyper_param={}, GPU_N='0'): # task = 'classification', 'regression'
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU_N
        self.hyper_param = hyper_param
        self.tensor = {}
    
    def __setitem__(self, key, value):
        self.hyper_param[key] = value
    
    def __getitem__(self, key):
        return self.hyper_param[key]
    
    def build(self):
        tf.reset_default_graph()
        self.tensor['x'] = tf.placeholder(tf.float32, shape=[None, self.hyper_param['layer_size'][0]])
        self.tensor['x_sigma'] = tf.placeholder(tf.float32, shape=[None, self.hyper_param['layer_size'][0]])
        self.tensor['noise'] = tf.placeholder(tf.float32)
        self.tensor['keep_prob'] = tf.placeholder(tf.float32)
        self.tensor['y'] = tf.placeholder(tf.float32, shape = [None, self.hyper_param['layer_size'][-1]])
        self.tensor['weight'] = []
        self.tensor['bias'] = []
        self.tensor['hidden_layer'] = [self.tensor['x']]
        for n in range(0, len(self.hyper_param['layer_size']) - 1): # other initialization methods should be implemented here
            self.tensor['weight'].append(tf.Variable(tf.truncated_normal([self.hyper_param['layer_size'][n], self.hyper_param['layer_size'][n+1]], stddev=0.1)))
            self.tensor['bias'].append(tf.Variable(tf.constant(0.0, shape=[self.hyper_param['layer_size'][n+1]])))
            self.tensor['hidden_layer'].append(tf.nn.dropout(tf.nn.relu(tf.matmul(self.tensor['hidden_layer'][n], self.tensor['weight'][n]) + self.tensor['bias'][n]), self.tensor['keep_prob']))
        self.tensor['hidden_layer'] = self.tensor['hidden_layer'][1: -1]
        self.tensor['y_pred'] = tf.add(tf.matmul(self.tensor['hidden_layer'][-1], self.tensor['weight'][-1]), self.tensor['bias'][-1])
        mse = tf.reduce_mean(tf.square(self.tensor['y'] - self.tensor['y_pred']), 0)
        # self.tensor['rmse'] = tf.sqrt(mse)
        reg = self.hyper_param['weight_decay'] * tf.reduce_sum(tf.square(self.tensor['weight'][0]))
        for n in range(1, len(self.tensor['weight'])):
            reg = tf.add(reg, self.hyper_param['weight_decay'] * tf.reduce_sum(tf.square(self.tensor['weight'][n])))
        self.tensor['cost'] = tf.reduce_mean(mse) + reg
        self.tensor['rsq'] = rsq(self.tensor['y'], self.tensor['y_pred'])
        self.tensor['optimizer'] = tf.train.AdamOptimizer(self.hyper_param['learning_rate']).minimize(self.tensor['cost'])
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def learn(self, x, y, epoch=10, valid_ratio=0.1, visualize=True): # epoch?
        x, y = np.array(x), np.array(y)
        perm_index = np.random.permutation(range(len(x)))
        n_valid = int(len(x) * valid_ratio)
        n_train = len(x) - n_valid
        x_valid, y_valid = x[perm_index[:n_valid], :], y[perm_index[:n_valid], :]
        x_train, y_train = x[perm_index[n_valid:], :], y[perm_index[n_valid:], :]
        
        self.x_mu, self.x_sigma = (np.mean(x_train, axis=0), np.std(x_train, axis=0))
        self.y_mu, self.y_sigma = (np.mean(y_train, axis=0), np.std(y_train, axis=0))
        
        x_valid, y_valid = (x_valid - self.x_mu) / self.x_sigma, (y_valid - self.y_mu) / self.y_sigma
        x_train, y_train = (x_train - self.x_mu) / self.x_sigma, (y_train - self.y_mu) / self.y_sigma
        
        epochs = []
        cost_trains = []
        cost_valids = []
        rsq_trains = []
        rsq_valids = []
        
        for e in range(epoch):
            _, cost_train, rsq_train = self.sess.run([self.tensor['optimizer'], self.tensor['cost'], self.tensor['rsq']], feed_dict={self.tensor['x']: x_train, self.tensor['y']: y_train, self.tensor['keep_prob']: self.hyper_param['keep_prob']})
            cost_valid, rsq_valid = self.sess.run([self.tensor['cost'], self.tensor['rsq']], feed_dict={self.tensor['x']: x_valid, self.tensor['y']: y_valid, self.tensor['keep_prob']: 1.0})
            '''
            if 'early_stop' in self.hyper_param:
                # do something
            '''
            if e % int(epoch / 10) == 0 or e + 1 == epoch:
                if visualize:
                    epochs.append(e)
                    cost_trains.append(cost_train)
                    rsq_trains.append(rsq_train)
                    cost_valids.append(cost_valid)
                    rsq_valids.append(rsq_valid)
                    y_pred_train = self.sess.run(self.tensor['y_pred'], feed_dict={self.tensor['x']: x_train, self.tensor['keep_prob']: 1.0}) * self.y_sigma + self.y_mu
                    y_true_train = y_train * self.y_sigma + self.y_mu
                    y_pred_valid = self.sess.run(self.tensor['y_pred'], feed_dict={self.tensor['x']: x_valid, self.tensor['keep_prob']: 1.0}) * self.y_sigma + self.y_mu
                    y_true_valid = y_valid * self.y_sigma + self.y_mu
                    
                    pred_min, pred_max = (np.min(y_pred_train), np.max(y_pred_train))
                    true_min, true_max = (np.min(y_true_train), np.max(y_true_train))

                    fig = plt.figure(figsize=(18, 9))
                    ax1 = fig.add_subplot(121)
                    ax2 = fig.add_subplot(222)
                    ax3 = fig.add_subplot(224)
                    
                    # need to change coloring strategy
                    ax1.plot([pred_min, pred_max], [true_min, true_max], 'black')
                    ax1.plot(y_pred_train, y_true_train, 'x', label='train')
                    ax1.plot(y_pred_valid, y_true_valid, 'o', label='valid')
                    ax2.plot(epochs, cost_trains, '--', label='train')
                    ax2.plot(epochs, cost_valids, '-', label='valid')
                    ax3.plot(epochs, rsq_trains, '--', label='train')
                    ax3.plot(epochs, rsq_valids, '-', label='valid')

                    ax1.legend(loc='best')
                    ax2.legend(loc='best')
                    ax3.legend(loc='best')

                    ax2.set_title('cost', fontsize=10)
                    ax3.set_title('rsq', fontsize=10)

                    # ax3.set_xticks(range(0, epoch + 1, 1000))
                    fig.show()
                    display.clear_output(wait=True)
                    display.display(pl.gcf())
                    pl.gcf().clear()
                else:
                    print('epoch:', '%06d' % (e + 1), 'cost =', '{:.6f}'.format(cost_valid), 'rsq =', ['{:.6f}'.format(rsq_valid[i]) for i in range(len(rsq_valid))])
    def rsq(self, x, y):
        x = np.array(x)
        y = np.array(y)
        return [datk.rsq(self.inference(x)[:, i], y[:, i]) for i in range(y.shape[1])]
    
    def rmse(self, x, y):
        x = np.array(x)
        y = np.array(y)
        return [datk.rmse(self.inference(x)[:, i], y[:, i]) for i in range(y.shape[1])]
    
    def inference(self, x):
        x = np.array(x)
        if len(x.shape) < 2:
            x = np.array([list(x)])
            return self.sess.run(self.tensor['y_pred'], feed_dict={self.tensor['x']: (x - self.x_mu) / self.x_sigma, self.tensor['keep_prob']: 1.0})[0] * self.y_sigma + self.y_mu
        return self.sess.run(self.tensor['y_pred'], feed_dict={self.tensor['x']: (x - self.x_mu) / self.x_sigma, self.tensor['keep_prob']: 1.0}) * self.y_sigma + self.y_mu
