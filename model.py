import json
import os
os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.layers import Input, Dense, Lambda, BatchNormalization, Activation
from keras.models import Model
from keras import backend as K
from keras import objectives
import theano
import theano.tensor as T
import math
from sklearn.cluster import KMeans
from sklearn import metrics as mtr
import metrics
import warnings
import matplotlib.pyplot as plt
import random


warnings.filterwarnings("ignore")
theano.config.floatX = 'float32'

class My_Model():
    def __init__(self, batch_size, num_views, latent_dim, intermediate_dim, config, weights_path, dataset,
                 save=0, ispretrain=True,
                 **kwargs):
        self.batch_size = batch_size
        self.num_views = num_views
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.ispretrain = ispretrain
        self.dataset = dataset
        self.init = 'variancescaling'
        self.weights_path = weights_path
        self.original_dim, self.epoch, self.n_centroid, self.lr_nn, self.lr_gmm, self.decay_n, self.decay_nn, self.decay_gmm, self.alpha, self.beta, self.datatype = config
        self.sample_output, self.gamma_output, self.my_model = self.build()
        self.ispretrain = ispretrain
        self.save = save

    def encoder(self, original_dim):
        means = []
        inputs = []
        vars = []
        for i in range(0, self.num_views):
            input = Input(batch_shape=(self.batch_size, original_dim[i]), name='input_%d' %i) # batch_size个样本 * d
            layer1 = Dense(self.intermediate_dim[0], init=self.init, name='encode1_%d' % i)(input)
            layer1 = BatchNormalization(name='bn1_%d' % i)(layer1)
            layer1 = Activation('relu')(layer1)
            layer2 = Dense(self.intermediate_dim[1], init=self.init, name='encode2_%d' % i)(layer1)
            layer2 = BatchNormalization(name='bn2_%d' % i)(layer2)
            layer2 = Activation('relu')(layer2)
            layer3 = Dense(self.intermediate_dim[2], init=self.init, name='encode3_%d' % i)(layer2)
            layer3 = BatchNormalization(name='bn3_%d' % i)(layer3)
            layer3 = Activation('relu')(layer3)

            z_means = Dense(self.latent_dim, init=self.init, activation=None, name='mean_%d' %i)(layer3)
            z_vars = Dense(self.latent_dim, init=self.init, activation=None,name='var_%d' %i)(layer3)
            means.append(z_means)
            vars.append(z_vars)
            inputs.append(input)
        return means, vars, inputs

    def decoder(self, latent, original_dim):
        reconst=[]
        for i in range(0, self.num_views):
            layer4 = Dense(self.intermediate_dim[-1], init=self.init, activation='relu', name='decode1_%d' %i)(latent)
            layer5 = Dense(self.intermediate_dim[-2], init=self.init, activation='relu', name='decode2_%d' %i)(layer4)
            layer6 = Dense(self.intermediate_dim[-3], init=self.init, activation='relu', name='decode3_%d' %i)(layer5)
            decoded= Dense(original_dim[i], init=self.init, activation=self.datatype, name='reconst_%d' %i)(layer6)
            reconst.append(decoded)
        return reconst

    def build(self):
        self.gmmpara_init()
        self.get_zeta()
        means, vars, self.inputs = self.encoder(self.original_dim)
        z_mean = Lambda(self.mixture_u, output_shape=(self.latent_dim,))(means)
        z_log_var = Lambda(self.mixture_var, output_shape=(self.latent_dim,))(vars)
        single_z = []
        for i in range(0, self.num_views):
            sz = Lambda(self.sampling, output_shape=(self.latent_dim,))([means[i], vars[i]])
            single_z.append(sz)
        z = Lambda(self.mixture_z, output_shape=(self.latent_dim,))(single_z)
        x_decoded = self.decoder(z, self.original_dim)
        output = x_decoded
        output.append(z)
        output.append(z_mean)
        output.append(z_log_var)

        for i in range(0, self.num_views):
            output.append(single_z[i])
        for i in range(0, self.num_views):
            output.append(means[i])
        for i in range(0, self.num_views):
            output.append(vars[i])

        gamma = Lambda(self.get_gamma, output_shape=(self.n_centroid,))(z)

        vade_loss = Lambda(self.vade_loss_function, name='vade_loss', output_shape=([],))(output)
        multiview_output = [vade_loss]

        return Model(self.inputs, single_z[0]), Model(self.inputs, gamma), Model(self.inputs, multiview_output)

    def gmmpara_init(self):
        theta_init = np.ones(self.n_centroid) / self.n_centroid
        u_init = np.zeros((self.latent_dim, self.n_centroid))
        lambda_init = np.ones((self.latent_dim, self.n_centroid))

        self.theta_p = theano.shared(np.asarray(theta_init, dtype=theano.config.floatX), name="pi")
        self.u_p = theano.shared(np.asarray(u_init, dtype=theano.config.floatX), name="u")
        self.lambda_p = theano.shared(np.asarray(lambda_init, dtype=theano.config.floatX), name="lambda")

    def get_zeta(self):
        zeta_init = np.ones(self.num_views) / self.num_views
        self.zeta = theano.shared(np.asarray(zeta_init, dtype=theano.config.floatX), name="zeta")

    def compute_epsilon(self, args):
        z, z_mean, z_log_var = args
        epsilon = (z - z_mean) / K.exp(z_log_var / 2)
        return epsilon

    def compute_single_z(self, args):
        means, vars, epsilon = args
        sz = means + K.exp(vars / 2) * epsilon
        return sz

    def mixture_u(self, args):
        u = 0
        for i in range(0, self.num_views):
            u += self.zeta[i] * args[i]
        return u

    def mixture_var(self, args):
        var = 0
        for i in range(0, self.num_views):
            var += self.zeta[i] * K.exp(args[i])
        return K.log(var)

    def mixture_z(self, args):
        z = 0
        for i in range(0, self.num_views):
            z += self.zeta[i] * args[i]
        return z

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def get_gamma(self, tempz): # q(c|x(v))
        temp_Z = T.transpose(K.repeat(tempz, self.n_centroid), [0, 2, 1])
        temp_u_tensor = T.repeat(self.u_p.dimshuffle('x', 0, 1), self.batch_size, axis=0)

        temp_lambda_tensor = T.repeat(self.lambda_p.dimshuffle('x', 0, 1), self.batch_size, axis=0)
        # version1
        temp_theta_tensor = self.theta_p.dimshuffle('x', 'x', 0) * T.ones(
            (self.batch_size, self.latent_dim, self.n_centroid))
        temp_p_c_z = K.exp(K.sum((K.log(temp_theta_tensor) - 0.5 * K.log(2 * math.pi * temp_lambda_tensor) - \
                                  K.square(temp_Z - temp_u_tensor) / (2 * temp_lambda_tensor)), axis=1)) + 1e-10
        return temp_p_c_z / K.sum(temp_p_c_z, axis=-1, keepdims=True)

    def vade_loss_function(self, args):
        inputs = self.inputs
        idx = self.num_views
        reconst, z, z_mean, z_log_var = args[:idx], args[-3-3*idx], args[-2-3*idx], args[-1-3*idx]
        single_z = args[-3*idx:-2*idx]
        means = args[-2*idx:-idx]
        vars = args[-idx:]
        Z = T.transpose(K.repeat(z, self.n_centroid), [0, 2, 1]) # n*d*k -> n*k*d
        z_mean_t = T.transpose(K.repeat(z_mean, self.n_centroid), [0, 2, 1])
        z_log_var_t = T.transpose(K.repeat(z_log_var, self.n_centroid), [0, 2, 1])
        u_tensor3 = T.repeat(self.u_p.dimshuffle('x', 0, 1), self.batch_size, axis=0)
        lambda_tensor3 = T.repeat(self.lambda_p.dimshuffle('x', 0, 1), self.batch_size, axis=0)

        #version1
        theta_tensor3 = self.theta_p.dimshuffle('x', 'x', 0) * T.ones(
            (self.batch_size, self.latent_dim, self.n_centroid))

        p_c_z = K.exp(K.sum((K.log(theta_tensor3) - 0.5 * K.log(2 * math.pi * lambda_tensor3) - \
                             K.square(Z - u_tensor3) / (2 * lambda_tensor3)), axis=1)) + 1e-10

        gamma = p_c_z / K.sum(p_c_z, axis=-1, keepdims=True)
        gamma_t = K.repeat(gamma, self.latent_dim) # n*k*d

        loss = 0
        reconst_loss=0
        num_views = self.num_views
        for i in range(0, num_views):
            r_loss = self.original_dim[i] * objectives.mean_squared_error(inputs[i], reconst[i])
            reconst_loss += r_loss
        loss += reconst_loss

        SGVB_loss = self.alpha * (K.sum(0.5 * gamma_t * (
                    self.latent_dim * K.log(math.pi * 2) + K.log(lambda_tensor3) + K.exp(
                z_log_var_t) / lambda_tensor3 + K.square(z_mean_t - u_tensor3) / lambda_tensor3), axis=(1, 2)) \
                               - 0.5 * K.sum(z_log_var + 1, axis=-1) \
                               - K.sum(
                    K.log(K.repeat_elements(self.theta_p.dimshuffle('x', 0), self.batch_size, 0)) * gamma, axis=-1) \
                               + K.sum(K.log(gamma) * gamma, axis=-1))
        loss += SGVB_loss

        temperature = 0.1
        loss_CL = 0.0

        for i in range(1):
            for j in range(i + 1, num_views):
                single_z[i] = K.l2_normalize(single_z[i], axis=-1)
                single_z[j] = K.l2_normalize(single_z[j], axis=-1)
                out = K.concatenate([single_z[i], single_z[j]], axis=0)

                sim_matrix = K.exp(K.dot(out, K.transpose(out)) / temperature)

                mask = 1 - K.eye(2 * self.batch_size)
                sim_matrix = sim_matrix * mask  # Mask the diagonal

                mask_flattened = K.flatten(mask)
                sim_matrix_flattened = K.flatten(sim_matrix)
                sim_matrix = K.switch(K.equal(mask_flattened, 1), sim_matrix_flattened, 0)
                sim_matrix = K.reshape(sim_matrix, (2 * self.batch_size, -1))

                # Compute positive similarity [batch_size]
                pos_sim = K.exp(K.sum(single_z[i] * single_z[j], axis=-1) / temperature)
                pos_sim = K.concatenate([pos_sim, pos_sim], axis=0)  # [2*batch_size]

                # Compute pair loss
                sim_matrix_sum = K.sum(sim_matrix, axis=-1)
                sim_matrix_sum = K.clip(sim_matrix_sum, K.epsilon(), 1e10)
                pair_loss = -K.mean(K.log(pos_sim / sim_matrix_sum))
                loss_CL += pair_loss

        loss_CL = K.mean(loss_CL)

        loss_IP = 0.0

        for i in range(1):
            for j in range(i + 1, num_views):
                loss_IP += objectives.mean_squared_error(single_z[i], single_z[j])

        loss_IP = K.mean(loss_IP)

        loss += self.beta * loss_CL
        loss += self.beta * 0.4 * loss_IP

        return loss

    def floatX(self, X):
        return np.asarray(X, dtype=theano.config.floatX)

    def load_pretrain_weights(self, vade, weights_path, dataset, inputs, Y):
        vade.load_weights(weights_path + dataset + '.h5')
        sample = self.sample_output.predict(inputs, batch_size=self.batch_size) # 拿到的隐变量z_mean

        kmeans = KMeans(n_clusters=self.n_centroid, n_init=20)
        kmeans.fit(sample)
        self.u_p.set_value(self.floatX(kmeans.cluster_centers_.T)) # u_p: K*d -> 转置 d*K

        return vade

    def compile(self, inputs, Y):

        if self.ispretrain is True:
            self.my_model = self.load_pretrain_weights(self.my_model, self.weights_path, self.dataset, inputs, Y)

        adam_nn = Adam(lr=self.lr_nn, epsilon=1e-4)
        adam_gmm = Adam(lr=self.lr_gmm, epsilon=1e-4)

        # 再调用的tools/training下的compile
        self.my_model.compile(optimizer=adam_nn, loss=lambda y_true, y_pred: y_pred,
                                add_trainable_weights=[self.theta_p, self.u_p, self.lambda_p, self.zeta],
                                add_optimizer=adam_gmm)

        self.epoch_begin = EpochBegin(self.sample_output, self.decay_n, self.gamma_output, self.decay_nn, self.decay_gmm, adam_nn, adam_gmm,
                                      inputs, Y, self.u_p, self.lambda_p, self.theta_p, self.zeta,
                                      self.batch_size, self.n_centroid, self.ispretrain, self.dataset, self.save)

    def train(self, inputs):
        none = np.zeros([np.shape(inputs[0])[0]])
        self.my_model.fit(x=inputs, y=none, shuffle=True, nb_epoch=self.epoch, batch_size=self.batch_size,
                            callbacks=[self.epoch_begin])

class EpochBegin(Callback):
    def __init__(self, sample_output, decay_n, gamma_output, decay_nn, decay_gmm, adam_nn, adam_gmm, inputs, Y, u_p, lambda_p, theta_p,
                 zeta, batch_size, n_centroid, ispretrain, dataset, save):
        self.sample_output = sample_output
        self.decay_n = decay_n
        self.gamma_output = gamma_output # 聚类结果gamma == q( c|x(v) )
        self.decay_nn = decay_nn
        self.decay_gmm = decay_gmm
        self.adam_nn = adam_nn
        self.adam_gmm = adam_gmm
        self.inputs = inputs
        self.Y = Y
        # 三个都是高斯混合模型GMM的参数:
        self.u_p = u_p
        self.lambda_p = lambda_p
        self.theta_p = theta_p

        self.zeta = zeta # 加权融合均值和方差时每个view的权重向量

        self.best_results = {'epoch': -1, 'acc': -np.inf, 'nmi': -np.inf, 'purity': -np.inf, 'ari': -np.inf}
        self.metrics_history = []

        self.n_centroid = n_centroid
        self.batch_size = batch_size
        self.ispretrain = ispretrain
        self.dataset = dataset
        self.save = save

    def on_epoch_begin(self, epoch, logs={}):
        self.epochBegin(epoch)

    def plot_embedding(self, data, label, id):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)

        fig = plt.figure()
        ax = plt.subplot(111)
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], str(label[i]),
                     color=plt.cm.Set1(label[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})

        plt.axis('off')
        plt.savefig('vis_%d' %id)
        return fig

    def epochBegin(self, epoch):
        if epoch % self.decay_n == 0 and epoch != 0:
            self.lr_decay()

        sample = self.sample_output.predict(self.inputs, batch_size=self.batch_size)
        kmeans = KMeans(n_clusters=self.n_centroid, n_init=20)
        kmeans.fit(sample)
        pred = kmeans.predict(sample)

        # if epoch % self.decay_n == 0 and epoch != 0:
        #     # 计算百分比
        #     unique, counts = np.unique(pred, return_counts=True)
        #     percentages = (counts / pred.size) * 100
        #     for value, percentage in zip(unique, percentages):
        #         print("类别 {} 占 {:.2f}%".format(value, percentage))

        acc = self.cluster_acc(pred, self.Y)

        Y = np.reshape(self.Y, [self.Y.shape[0]])
        nmi = metrics.nmi(Y, pred)
        ari = metrics.ari(Y, pred)
        purity = self.purity_score(Y, pred)
        global accuracy
        accuracy = []
        accuracy += [acc[0]]

        # 保留最优结果
        if self.ispretrain:
            self.metrics_history.append({'epoch': epoch, 'acc': acc[0], 'nmi': nmi, 'purity': purity, 'ari': ari})
            if acc[0] > self.best_results['acc']:
                self.best_results = {'epoch': epoch, 'acc': acc[0], 'nmi': nmi, 'purity': purity, 'ari': ari}

        # if epoch > 0 and self.ispretrain:
        #     print('ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, Purity: {:.4f}'.format(acc[0], nmi, ari, purity))

    def on_train_end(self, logs=None):
        if self.ispretrain:
            print("\nBest Metrics:")
            print("ACC: {:.4f}, NMI: {:.4f}, Purity: {:.4f}, ARI: {:.4f}".format(
                self.best_results['acc'], self.best_results['nmi'],
                self.best_results['purity'], self.best_results['ari']
            ))
            if self.save==1: # sava==1才保留每个epoch的结果
                file_path = "F:/A_my/myModel/shuffled_pretrain/result/"
                file_path += self.dataset + ".txt"
                with open(file_path, "w") as file:
                    for entry in self.metrics_history:
                        file.write(json.dumps(entry) + "\n")
                    file.write(json.dumps(self.best_results) + "\n")
                print("Metrics history saved to {}".format(file_path))

    def purity_score(self, y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = mtr.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    def cluster_acc(self, Y_pred, Y):
        from sklearn.utils.linear_assignment_ import linear_assignment
        assert Y_pred.size == Y.size
        D = max(Y_pred.max(), Y.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(Y_pred.size):
            w[Y_pred[i], Y[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, w

    def lr_decay(self):
        self.adam_nn.lr.set_value(self.floatX(self.adam_nn.lr.get_value() * self.decay_nn))
        self.adam_gmm.lr.set_value(self.floatX(self.adam_gmm.lr.get_value() * self.decay_gmm))
        print ('lr_nn:%f' % self.adam_nn.lr.get_value())
        print ('lr_gmm:%f' % self.adam_gmm.lr.get_value())

    def floatX(self, X):
        return np.asarray(X, dtype=theano.config.floatX)
