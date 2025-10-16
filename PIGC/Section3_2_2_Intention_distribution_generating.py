from keras import backend as K
from keras.layers import Dense, Input,Lambda
from keras.models import Model
from tensorflow.keras.optimizers import Adam

# Re-parameterization Eq.(4)
def re_sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon

def ourself_softmax_activation(d):#Input D:
    numerator = K.exp(d)
    denominator = K.sum(K.exp(d), axis=-1, keepdims=True)
    result = numerator / (denominator + K.epsilon())
    return result

def loss_intent_dot(intention_score, loss_log_part):
    intention_score = l2_normalize_rows(intention_score)
    loss_log_part = l2_normalize_rows(loss_log_part)

    sim_matrix = K.dot(intention_score, K.transpose(intention_score))
    A_intent = sim_matrix / (K.sqrt(K.sum(K.square(intention_score), axis=1, keepdims=True)) *
                      K.transpose(K.sqrt(K.sum(K.square(intention_score), axis=1, keepdims=True))) + K.epsilon())
    #Eq.(7)
    A_intent = K.softmax(A_intent, axis=-1)
    norm = K.sqrt(K.sum(K.square(A_intent), axis=1, keepdims=True))
    A_intent = A_intent / (norm + K.epsilon())  # 归一化

    # Final loss
    loss_intent = K.mean(K.sum(loss_log_part * A_intent, axis=-1))
    return loss_intent
def l2_normalize_rows(matrix):
    l2_norm = K.sqrt(K.sum(K.square(matrix), axis=1, keepdims=True))
    normalized_matrix = matrix / (l2_norm + K.epsilon())
    return normalized_matrix

def intention_distributions_leaner(theta):
    norm = K.sqrt(K.sum(K.square(theta), axis=1, keepdims=True))
    theta = theta / (norm + K.epsilon())

    dot_product = K.dot(theta, K.transpose(theta))
    norm = K.sqrt(K.sum(K.square(theta), axis=1, keepdims=True))
    similarity_matrix = dot_product / (norm * K.transpose(norm) + K.epsilon())

    norm = K.sqrt(K.sum(K.square(similarity_matrix), axis=1, keepdims=True))
    similarity_matrix = similarity_matrix / (norm + K.epsilon())
    # Multiply each value of the matrix by temperature coefficient, calculate exp(-d(zi,zj))
    τ = 2.0 #τ：temperature parameters
    scaled_matrix = -(1/τ) * similarity_matrix
    #Eq.(7)
    loss_ = ourself_softmax_activation(scaled_matrix)
    loss_log_part = K.log(loss_ + K.epsilon())
    norm = K.sqrt(K.sum(K.square(loss_log_part), axis=1, keepdims=True))
    result = loss_log_part / (norm + K.epsilon())

    return result

#stick-breaking
def Stick_Breaing(args):
    eta = args  # Input eta η
    theta_list = []
    remaining_stick = K.ones_like(eta[:, 0])
    K_dim = K.int_shape(eta)[-1]
    for i in range(K_dim):
        if i == 0:
            # First step: directly take eta's first value as the first theta_k
            theta_k = eta[:, i]
        elif i != K_dim - 1:
            # Intermediate steps: theta_k = eta[i] × remaining stick length
            theta_k = eta[:, i] * remaining_stick
        else:
            # Final step: take the remaining stick length as theta_k
            theta_k = remaining_stick
            theta_k = K.clip(theta_k, 1e-5, 1)
        # Update the remaining stick length: current length × (1 - eta[i])
        remaining_stick = remaining_stick * (1 - eta[:, i])
        # Append the current theta_k to the result list
        theta_list.append(theta_k)
    theta = K.stack(theta_list, axis=1)
    return theta

def initial_distribution_leaner(dims,n_clusters,act='relu', init='glorot_uniform'):
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],))

    h = x
    for i in range(n_stacks - 1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)
    mean = Dense(dims[-1])(h)
    log_var = Dense(dims[-1])(h)
    #Eq.(3)
    h = Lambda(re_sampling, output_shape=(dims[-1],))([mean, log_var])
    # calculate distribution theta
    # η = sigmoid(W ^ T * x), Eq.(4);
    eta = Dense(n_clusters, activation="sigmoid", kernel_initializer=init)(h)
    # θ = f_SB(η):stick-breaking
    theta = Lambda(Stick_Breaing, output_shape=(n_clusters,))(eta)

    #reconstruction
    representation = theta
    for i in range(n_stacks - 1, 0, -1):
        representation = Dense(dims[i], activation=act, kernel_initializer=init)(representation)
    reconstruction = Dense(dims[0], kernel_initializer=init)(representation)

    # construct model
    theta_generator = Model(inputs=x, outputs=theta)
    init_distribution = Model(inputs=x, outputs=reconstruction)

    # create loss Eq.(6):
    reconstruction_loss = K.sum((x - reconstruction) ** 2, 0)
    kl_loss = - K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)
    loss_init =reconstruction_loss + 0.01*K.mean(kl_loss)
    init_distribution.add_loss(loss_init)
    init_distribution.compile(optimizer='adam')

    return theta_generator, init_distribution


def intention_distribution_generator(dims,n_clusters):
    theta_generator,init_distribution = initial_distribution_leaner(act='relu', n_clusters=n_clusters, init='glorot_uniform',dims=dims)
    output_theta = theta_generator.output
    sim_layer = Lambda(intention_distributions_leaner)(output_theta)
    intent_guider = Model(inputs=theta_generator.input, outputs=sim_layer)
    opti = Adam(learning_rate=0.01)
    intent_guider.compile(optimizer=opti, loss=loss_intent_dot)

    return theta_generator,init_distribution,intent_guider