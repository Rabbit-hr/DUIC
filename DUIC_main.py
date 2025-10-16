import numpy as np
from tqdm.keras import TqdmCallback
from keras.models import Model
from keras.optimizers import SGD
from sklearn.cluster import KMeans
import metrics
from datasets import load_data
from datasets import pkl_read,pkl_save
from PIGC.Section3_2_1_User_intent_parsing import calculate_intent_alignment_score
from PIGC.Section3_2_2_Intention_distribution_generating import intention_distribution_generator
from PIGC.Section3_2_3_Distribution_Results_generating import Clustering_result_generating
from UACE.Section3_3_User_Aligned_Cluster_Explanation import CR_leaner,print_top_r
import os


class Descriptive_Intention_Guided_Understandable_Clustering(object):
    def __init__(self, dims, save_weights_path,y,intention_alignment_scores,n_clusters, pre_training_epochs,intent_guiding_epochs, pretrain_batch_size, init='glorot_uniform'):
        super(Descriptive_Intention_Guided_Understandable_Clustering, self).__init__()
        self.dims = dims
        self.input_dim = dims[0]
        self.save_weights_path = save_weights_path
        self.n_clusters = n_clusters
        self.y = y
        self.intention_alignment_scores = intention_alignment_scores
        self.pre_training_epochs = pre_training_epochs
        self.intent_guiding_epochs = intent_guiding_epochs
        self.pretrain_batch_size = pretrain_batch_size
        self.theta_leaner, self.pre_m, self.intent_guide_module = intention_distribution_generator(dims=self.dims,n_clusters=self.n_clusters)
        #clustering Layers
        clustering_layer = Clustering_result_generating(self.n_clusters, name='clustering')(self.theta_leaner.output)
        self.clustering_optimizer = Model(inputs=self.theta_leaner.inputs, outputs=clustering_layer)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.clustering_optimizer.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y=None, maxiter=2e4, batch_size=256, tol=1e-3, update_interval=140, save_dir='./results'):
        # Eq.(9)-Eq.(11)
        print('Update interval', update_interval)
        save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
        print('Save interval', save_interval)

        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.theta_leaner.predict(x))
        y_pred_last = np.copy(y_pred)
        self.clustering_optimizer.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.clustering_optimizer.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p
                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    vi  = np.round(metrics.variation_of_information(y, y_pred),5)
                    loss = np.round(loss, 5)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f,vi = %.5f' % (ite, acc, nmi, ari,vi), ' ; loss=',
                          loss)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break
            idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
            loss = self.clustering_optimizer.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
            ite += 1
        # optimize theta_distribution
        optimized_theta = np.copy(self.theta_leaner.predict(x, verbose=0))
        return y_pred, optimized_theta

    def pretrain(self, x, batch_size=256):
        if self.save_weights_path:
            # load weights
            if os.path.exists(self.save_weights_path):
                self.pre_m.load_weights(self.save_weights_path)
                print(f"Networks weights loaded from {self.save_weights_path}")
            else:
                print('...pretraining...')
                self.pre_m.fit(x,
                                 shuffle=True,
                                 epochs=self.pre_training_epochs,
                                 batch_size=batch_size,
                                 verbose=0,
                                 callbacks=[TqdmCallback(verbose=0)])
                self.pre_m.save_weights(self.save_weights_path)
        theta = self.theta_leaner.predict(x)
        return theta

    def intention_optimize_theta_leaner(self, x, intention_guiding_vec, batch_size=256):
        training_epochs = self.intent_guiding_epochs
        print('...training_intention_embedder...')
        self.intent_guide_module.fit(x=x,
                                y=intention_guiding_vec,
                                epochs=training_epochs,
                                batch_size=batch_size,
                                verbose=0,
                                callbacks=[TqdmCallback(verbose=0)])
        intention_optimized_theta = self.theta_leaner.predict(x)
        return intention_optimized_theta  # optimized theta with user intention

    def get_clustering_result(self,x):
        x_theta= self.pretrain(x=x)
        km = KMeans(n_clusters=self.n_clusters, n_init=20)
        print('...pretraining...')
        y_pred = km.fit_predict(x_theta)
        print('without_intention_guiding')
        print(' ' * 8 + '|==> acc: %.4f,  nmi: %.4f  ,  ari: %.4f ,  vi: %.4f<==|'
              % (metrics.acc(y, y_pred), metrics.nmi(y, y_pred), metrics.ari(y, y_pred), metrics.variation_of_information(y, y_pred)))

        intention_e = self.intention_optimize_theta_leaner(x=x, intention_guiding_vec = self.intention_alignment_scores)
        km = KMeans(n_clusters=self.n_clusters, n_init=20)
        ig_y = km.fit_predict(intention_e)
        #intention_guiding_optimized_theta
        print('Intention guided clustering...')
        print(' ' * 8 + '|==> Accuracy: %.4f, acc: %.4f,  nmi: %.4f  ,  ari: %.4f ,  vi: %.4f<==|'
              % (metrics.Accuracy(self.y, ig_y), metrics.acc(self.y, ig_y),
                 metrics.nmi(self.y, ig_y), metrics.ari(self.y, ig_y),metrics.variation_of_information(y, ig_y)))
        return intention_e,x_theta

if __name__ == "__main__":
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='20NG3',choices=['20NG3','BBCSport'])
    parser.add_argument('--pretrain_batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--pretrain_epochs', default=50, type=int)
    parser.add_argument('--intent_guiding_epochs',default=100, type=int)
    parser.add_argument('--update_interval', default=None, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--weights_save_dir', default='results')
    parser.add_argument('--vec_dim', default='vec_dim')
    parser.add_argument('--clustering_o_batch_size', default='256',type=int)
    args = parser.parse_args()
    print(args)

    #load_data
    x, y, term_list,term_vec = load_data(args.dataset)
    init = 'glorot_uniform'
    pretrain_optimizer = 'adam'

    if args.dataset == '20NG3':
        update_interval = 140
        pretrain_epochs = 100
        intention_descriptors = ['science', 'technology', 'health', 'sports', 'entertainment', 'politics','education', 'business', 'art', 'environment']
    elif args.dataset == 'BBCSport':
        update_interval = 140
        pretrain_epochs = 100
        intention_descriptors = ["soccer", "basketball", "baseball","tennis", "golf", 'football',"volleyball", "cricket", "rugby",'athletics','table','american']


    if args.update_interval is not None:
        update_interval = args.update_interval
    if args.pretrain_epochs is not None:
        pretrain_epochs = args.pretrain_epochs

    #load query vectors
    if os.path.exists('intention_information/'+args.dataset+'_intention_alignment_score.pkl'):
        intention_alignment_score = pkl_read('intention_information/'+args.dataset+'_intention_alignment_score.pkl')
    else:
        print('...creating intention_alignment_score...')
        intention_alignment_score = calculate_intent_alignment_score(query_list=intention_descriptors,x=x,
                                                                term_vec=term_vec,score_save_path='intention_information/'+args.dataset+'_intention_alignment_score.pkl')
    n_clusters = len(np.unique(y))
    dims = [x.shape[-1], 500, 2000, 256, n_clusters]

    duic = Descriptive_Intention_Guided_Understandable_Clustering(dims=dims, y=y, save_weights_path='model_weights/'+args.dataset+'_pre_network_weights.h5', intention_alignment_scores=intention_alignment_score, n_clusters=n_clusters,
                         pre_training_epochs=args.pretrain_epochs,intent_guiding_epochs=args.intent_guiding_epochs, init=init, pretrain_batch_size=256)
    duic.compile(optimizer=SGD(0.01, 0.9), loss='kld')
    intent_theta, theta_em = duic.get_clustering_result(x=x)

    """optimize clustering"""
    print('...optimize_clustering_with_loss_C...')
    y_p, intent_theta = duic.fit(x=x, y=y, tol=args.tol, maxiter=args.maxiter, batch_size=args.clustering_o_batch_size,update_interval=update_interval)
    print(' ' * 8 + '|==> acc: %.4f,  nmi: %.4f  ,  ari: %.4f ,  vi: %.4f<==|'
          % (metrics.acc(y, y_p), metrics.nmi(y, y_p), metrics.ari(y, y_p),metrics.variation_of_information(y, y_p)))

    pkl_save('data/'+args.dataset+'/theta.pkl',intent_theta)
    # learn cluster-level representations CR
    CR = CR_leaner(theta=intent_theta,raw_data = x,n_clusters=n_clusters,terms_vec=term_vec,CR_learn_epochs=200)
    top_r_term_vectors = print_top_r(CR=CR,terms_vec=term_vec,term_list=term_list,top_r=10)#top_r = 10  # Find the 10 most similar terms