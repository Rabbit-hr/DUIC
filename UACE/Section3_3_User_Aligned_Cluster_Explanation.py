import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np


def softmax_rowwise(X):
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)

def CR_leaner(theta,raw_data,n_clusters,terms_vec,CR_learn_epochs):
    # Unified and norm data types:
    raw_data = raw_data.astype('float32')
    terms_vec = terms_vec.astype('float32')
    theta = theta.astype('float32')
    raw_data = (raw_data - np.mean(raw_data, axis=0)) / np.std(raw_data, axis=0)
    terms_vec = (terms_vec - np.mean(terms_vec, axis=0)) / np.std(terms_vec, axis=0)

    # Initialize the learnable topic_vec matr
    #Eq.(12) cluster_level_representation CR
    CR = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(n_clusters, 300), dtype=tf.float32))

    @tf.function
    def train_step(terms_vec, theta, raw_data):
        with tf.GradientTape() as tape:
            # calculate cluster-terms distribution B Eq.(13)
            C_calculated = tf.matmul(CR, tf.transpose(terms_vec))
            B = tf.nn.softmax(C_calculated, axis=-1)
            # IS * cluster-terms distribution  #Eq.(15)
            x_hat = tf.matmul(theta, B)
            #Eq.(16)
            loss_recon = tf.reduce_mean(tf.square(raw_data - x_hat))  # use MSE to calculate loss

            # calculate gradients
        gradients = tape.gradient(loss_recon, [CR])
        optimizer.apply_gradients(zip(gradients, [CR]))
        return loss_recon

    optimizer = Adam(learning_rate=0.01)
    for epoch in range(CR_learn_epochs):
        loss_value = train_step(terms_vec, theta, raw_data)
        if epoch % 50 == 0:
            print(f"Epoch {epoch + 1}/{CR_learn_epochs}, Loss: {loss_value.numpy()}")
    final_CR = CR.numpy()
    return final_CR

def print_top_r(CR,terms_vec,term_list,top_r):#Eq.(17)
    cos_sim_matrix = np.dot(CR, np.transpose(terms_vec))
    cos_sim_matrix = softmax_rowwise(cos_sim_matrix)
    most_similar_indices = np.argsort(cos_sim_matrix, axis=-1)[:, ::-1][:, :top_r]
    top_r_terms = []
    # Traverse the similar word index of each query word vector and find the corresponding word according to the index
    for i, indices in enumerate(most_similar_indices):
        print("\n")
        print(f"Topic {i + 1}:")

        similar_words = [term_list[idx] for idx in indices]
        top_r_word_vectors = np.array([terms_vec[idx] for idx in indices])
        top_r_terms.append(top_r_word_vectors)
        print(f"{similar_words}")
    top_k_terms = np.array(top_r_terms)
    return top_k_terms
