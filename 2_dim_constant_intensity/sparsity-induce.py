import argparse
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.linalg import sqrtm
import random

import sklearn
import matplotlib.pylab as pl
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"  # specify which GPU(s) to be used

parser = argparse.ArgumentParser()
# ICNN OT Parameters
parser.add_argument('--DATASET_X', type=str, default='Yn', help='which dataset to use for X')
parser.add_argument('--DATASET_Y', type=str, default='Xn', help='which dataset to use for Y')

parser.add_argument('--SHOW_THE_PLOT', type=bool, default=False, help='Boolean option to show the plots or not')
parser.add_argument('--DRAW_THE_ARROWS', type=bool, default=False, help='Whether to draw transport arrows or not')

parser.add_argument('--TRIAL', type=int, default=1, help='the trail no.')

parser.add_argument('--LAMBDA', type=float, default=1, help='Regularization constant for positive weight constraints')

parser.add_argument('--NUM_NEURON', type=int, default=64, help='number of neurons per layer')

parser.add_argument('--NUM_LAYERS', type=int, default=4, help='number of hidden layers before output')

parser.add_argument('--LR', type=float, default=1e-4, help='learning rate')


parser.add_argument('--ITERS', type=int, default=80000, help='number of iterations of training')

parser.add_argument('--BATCH_SIZE', type=int, default=128, help='size of the batches')

parser.add_argument('--SCALE', type=float, default=5.0, help='scale for the gaussian_mixtures')
parser.add_argument('--VARIANCE', type=float, default=0.5, help='variance for each mixture')

parser.add_argument('--N_TEST', type=int, default=500, help='number of test samples')
parser.add_argument('--N_PLOT', type=int, default=220, help='number of samples for plotting')
parser.add_argument('--N_CPU', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--INPUT_DIM', type=int, default=2, help='dimensionality of the input x')
parser.add_argument('--N_GENERATOR_ITERS', type=int, default=10, help='number of training steps for discriminator per iter')

# SICNN Parameters
parser.add_argument('--SPARSITY_INDUCING_INTENSITY', type=float, default = 0.0015, help = 'The sparsity induced intensity lambda used in SICNN')
parser.add_argument('--PENALITY', type=str, default='stvs', help='penalty function') # Choice between 'stvs', 'l1', and 'ovk'


opt = parser.parse_args()
print(opt)



def main():

    # specify the convex function class
    print("specify the convex function class")
    
    hidden_size_list = [opt.NUM_NEURON for i in range(opt.NUM_LAYERS)]

    hidden_size_list.append(1)

    print(hidden_size_list)
    
    fn_model = Kantorovich_Potential(opt.INPUT_DIM, hidden_size_list)  
    gn_model = Kantorovich_Potential(opt.INPUT_DIM, hidden_size_list)

    SET_PARAMS_NAME = str(opt.BATCH_SIZE) + '_batch' + str(opt.NUM_NEURON) + '_neurons' + str(opt.NUM_LAYERS) + '_layers'
    EXPERIMENT_NAME = opt.DATASET_X

    # Directory to store the images
    
    os.makedirs("figures/{0}/{1}_batch/{2}_layers/{3}_neurons/{4}_scale/{5}_variance/weightRegular_{6}/LR_{7}/trial_{8}".format(EXPERIMENT_NAME,
                                    opt.BATCH_SIZE, opt.NUM_LAYERS, opt.NUM_NEURON, 
                                                                    opt.SCALE, opt.VARIANCE,
                                                                        opt.LAMBDA, opt.LR, opt.TRIAL), exist_ok=True)

    # Define the test set
    print ("Define the test set")
    X_test = next(sample_data_gen(opt.DATASET_X, opt.N_TEST, opt.SCALE, opt.VARIANCE))

    Y_test = next(sample_data_gen(opt.DATASET_Y, opt.N_TEST, opt.SCALE, opt.VARIANCE))

    saver = tf.compat.v1.train.Saver()

    # Running the optimization
    with tf.compat.v1.Session() as sess:
        
        # Initialize the computation (assuming ComputeOT is properly defined)
        compute_OT = ComputeOT(sess, opt.INPUT_DIM, fn_model, gn_model, opt.LR, 
                            opt.SPARSITY_INDUCING_INTENSITY, opt.PENALITY)  # Initializing

        # Run the learning process
        compute_OT.learn(opt.BATCH_SIZE, opt.ITERS, opt.N_GENERATOR_ITERS, opt.SCALE, 
                        opt.VARIANCE, opt.DATASET_X, opt.DATASET_Y, opt.N_PLOT, 
                        EXPERIMENT_NAME, opt)  # Learning the optimal map

        # Ensure the directory exists before saving the model
        save_dir = "saving_model/{0}".format(EXPERIMENT_NAME)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the model checkpoint
        saver.save(sess, os.path.join(save_dir, "model-{0}.ckpt".format(SET_PARAMS_NAME + str(opt.ITERS) + '_iters')))

        print("Final Learned Wasserstein distance: {0}".format(compute_OT.compute_W2(X_test, Y_test)))


        
class ComputeOT:
    
    def __init__(self, sess, input_dim, f_model, g_model,lr, intensity, penalty):
        
        self.sess = sess 
        
        self.f_model = f_model
        self.g_model = g_model

        self.input_dim = input_dim
        self.intensity = intensity
        
        self.x = tf.compat.v1.placeholder(tf.float32, [None, input_dim])
        self.y = tf.compat.v1.placeholder(tf.float32, [None, input_dim])

        self.fx = self.f_model.forward(self.x)
        
        self.gy = self.g_model.forward(self.y) 

        [self.grad_fx] = tf.gradients(self.fx,self.x)
        
        [self.grad_gy] = tf.gradients(self.gy,self.y)
        
        self.f_grad_gy = self.f_model.forward(self.grad_gy)
        self.y_dot_grad_gy = tf.reduce_sum(tf.multiply(self.y, self.grad_gy), axis=1, keepdims=True)

        self.S = 0
        
        if penalty == 'l1':

            self.tau = tf.reduce_sum(tf.abs(self.grad_gy - self.y), axis=1)
            self.S = tf.reduce_sum(self.tau)

        elif penalty == 'stvs':

            gamma = 100  # Example value for gamma, altered based on the length of the displacement vector
            sigma_z = tf.asinh(tf.abs(self.grad_gy - self.y) / (2 * gamma))
            self.tau = gamma**2 * (sigma_z + 0.5 - 0.5 * tf.exp(-2 * sigma_z))
            self.S = tf.reduce_sum(self.tau)

        elif penalty == 'ovk':

            z = self.grad_gy - self.y
            z_sorted = tf.sort(tf.abs(z), direction='DESCENDING')  
            top_k = z_sorted[:1] 
            rest = z_sorted[1:]  
            top_norm = tf.reduce_sum(tf.square(top_k))
            rest_norm = tf.reduce_sum(tf.square(rest) / 2) 
            self.tau = top_norm + rest_norm
            self.S = tf.reduce_sum(self.tau)

        self.g_sparsity_penalty = self.intensity * self.S
        self.x_squared = tf.reduce_sum(tf.multiply(self.x, self.x), axis=1, keepdims=True)
        self.y_squared = tf.reduce_sum(tf.multiply(self.y, self.y), axis=1, keepdims=True)

        self.f_loss = tf.reduce_mean(self.fx - self.f_grad_gy)
        self.g_loss = tf.reduce_mean(self.f_grad_gy - self.y_dot_grad_gy) + self.g_sparsity_penalty

        self.f_postive_constraint_loss = self.f_model.positive_constraint_loss
        self.g_postive_constraint_loss = self.g_model.positive_constraint_loss
        
        if opt.LAMBDA>0:

            self.f_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = lr, beta1=0.5, beta2=0.9).minimize(self.f_loss, var_list= self.f_model.var_list)
            self.g_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9).minimize(self.g_loss + opt.LAMBDA * self.g_postive_constraint_loss, var_list=self.g_model.var_list)
        
        else:

            self.f_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(self.f_loss, var_list= self.f_model.var_list)
            self.g_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(self.g_loss, var_list=self.g_model.var_list)

        self.W2 = tf.reduce_mean(self.f_grad_gy - self.fx  - self.y_dot_grad_gy + 0.5*self.x_squared + 0.5*self.y_squared)
        
        self.init = tf.compat.v1.global_variables_initializer()


    def learn(self, batch_size, iters, inner_loop_iterations, scale, variance, dataset_x, dataset_y, plot_size, experiment_name, opt):
       
        print_T = 1000

        save_figure_iterations = 10000
        
        self.sess.run(self.init)
        data_gen_x = sample_data_gen(dataset_x, batch_size, scale, variance)
        print ("data_gen_x created")
        data_gen_y = sample_data_gen(dataset_y, batch_size, scale, variance)
        print ("data_gen_y created")
        
        # This data will be used for plotting
        X_plot = next(sample_data_gen(dataset_x, plot_size, scale, variance))
        Y_plot = next(sample_data_gen(dataset_y, plot_size, scale, variance))
        print("Plotting data created")

        if opt.LAMBDA > 0:
            trainable_g_list = [self.g_optimizer]
            trainable_f_list = [self.f_optimizer , self.f_model.proj]

        else:
            trainable_g_list = [self.g_optimizer , self.g_model.proj]
            trainable_f_list = [self.f_optimizer , self.f_model.proj]

        
        for iteration in range(iters):
            
            for j in range(inner_loop_iterations):
                x_train = next(data_gen_x)
                y_train = next(data_gen_y)

                #Training the g neural network
                _ = self.sess.run(trainable_g_list,feed_dict={self.x: x_train, self.y:y_train})

            x_train = next(data_gen_x)
            y_train = next(data_gen_y)

            #Training the f_neural network
            _ = self.sess.run( trainable_f_list, feed_dict={self.x: x_train, self.y: y_train})

            if iteration % print_T == 0:
                f_loss, g_loss, g_sparsity_penalty, W2 = self.sess.run([self.f_loss, self.g_loss, self.g_sparsity_penalty, self.W2],feed_dict={self.x: x_train, self.y: y_train}) 
                print ("Iterations = %i, f_loss = %.4f, g_loss = %.4f, g_sparsity_penalty = %.4f, W2 = %.4f" %(iteration,f_loss,g_loss, g_sparsity_penalty, W2))
            
            if  (iteration+1) % save_figure_iterations == 0:
                self.save_the_figure(iteration+1, X_plot, Y_plot, experiment_name, opt)
    
    
    def transport_X_to_Y(self, X):
        
        T_X_to_Y = self.sess.run(self.grad_fx, feed_dict={self.x: X}) 
        
        return T_X_to_Y

    def transport_Y_to_X(self, Y):
        
        T_Y_to_X = self.sess.run(self.grad_gy, feed_dict={self.y: Y}) 
        
        return T_Y_to_X
    
    def eval_gy(self, Y):
        
        _gy = self.sess.run(self.gy, feed_dict={self.y: Y}) 
        
        return _gy

    def compute_W2(self, X, Y):

        return self.sess.run(self.W2, feed_dict={self.x: X, self.y: Y})

    def save_the_figure(self, iteration, X_plot, Y_plot, experiment_name, opt):
       
        (plot_size,_) = np.shape(X_plot)

        X_pred = self.transport_Y_to_X(Y_plot)


        fig = plt.figure()


        plt.scatter(Y_plot[:,0], Y_plot[:,1], color='orange', 
                    alpha=0.5, label=r'$Y$')
        plt.scatter(X_plot[:,0], X_plot[:,1], color='red', 
                    alpha=0.5, label=r'$X$')
        plt.scatter(X_pred[:,0], X_pred[:,1], color='purple', 
                    alpha=0.5, label=r'$\nabla g(Y)$')
        
        pairs_array = np.hstack((Y_plot, X_pred))
        plt.legend()

        for i in range(plot_size):
                drawArrow(Y_plot[i,:], X_pred[i,:])

        '''
        if opt.DRAW_THE_ARROWS:

            for i in range(plot_size):
                drawArrow(Y_plot[i,:], X_pred[i,:])
        '''
        if opt.SHOW_THE_PLOT:

            plt.show()



        fig.savefig("figures/{0}/{1}_batch/{2}_layers/{3}_neurons/{4}_scale/{5}_variance/weightRegular_{6}/LR_{7}/trial_{8}/New_GT_{9}.png"
                                        .format(experiment_name, opt.BATCH_SIZE, opt.NUM_LAYERS, opt.NUM_NEURON,
                                                                         opt.SCALE, opt.VARIANCE,opt.LAMBDA, opt.LR ,opt.TRIAL,
                                                                          str(iteration)))
        np.save("figures/{0}/{1}_batch/{2}_layers/{3}_neurons/{4}_scale/{5}_variance/weightRegular_{6}/LR_{7}/trial_{8}/New_GT_{9}.npy"
                                        .format(experiment_name, opt.BATCH_SIZE, opt.NUM_LAYERS, opt.NUM_NEURON,
                                                                         opt.SCALE, opt.VARIANCE,opt.LAMBDA, opt.LR ,opt.TRIAL,
                                                                          str(iteration)), pairs_array)
        print("Plot saved at iteration {0}".format(iteration))
        
class Kantorovich_Potential:
    ''' 
        Modelling the Kantorovich potential as Input convex neural network (ICNN)
        input: y
        output: z = h_L
        Architecture: h_1     = ReLU^2(A_0 y + b_0)
                      h_{l+1} =   ReLU(A_l y + b_l + W_{l-1} h_l)
        Constraint: W_l > 0
    '''
    def __init__(self,input_size, hidden_size_list):

        # hidden_size_list always contains 1 in the end because it's a scalar output
        self.input_size = input_size
        self.num_hidden_layers = len(hidden_size_list)
        
        # list of matrices that interacts with input
        self.A = []
        for k in range(0, self.num_hidden_layers):
            self.A.append(tf.Variable(tf.random.uniform([self.input_size, hidden_size_list[k]], maxval=0.1), dtype=tf.float32))

        # list of bias vectors at each hidden layer 
        self.b = []
        for k in range(0, self.num_hidden_layers):
            self.b.append(tf.Variable(tf.zeros([1, hidden_size_list[k]]),dtype=tf.float32))

        # list of matrices between consecutive layers
        self.W = []
        for k in range(1, self.num_hidden_layers):
            self.W.append(tf.Variable(tf.random.uniform([hidden_size_list[k-1], hidden_size_list[k]], maxval=0.1), dtype=tf.float32))
        
        self.var_list = self.A +  self.b + self.W

        self.positive_constraint_loss = tf.add_n([tf.nn.l2_loss(tf.nn.relu(-w)) for w in self.W])

        self.proj = [w.assign(tf.nn.relu(w)) for w in self.W] #ensuring the weights to stay positive

    def forward(self, input_y):
        
        # Using ReLU Squared
        z = tf.nn.leaky_relu(tf.matmul(input_y, self.A[0]) + self.b[0], alpha=0.2)
        z = tf.multiply(z,z)

        # # If we want to use ReLU and softplus for the input layer
        # z = tf.matmul(input_y, self.A[0]) + self.b[0]
        # z = tf.multiply(tf.nn.relu(z),tf.nn.softplus(z))
        
        # If we want to use the exponential layer for the input layer
        ## z=tf.nn.softplus(tf.matmul(input_y, self.A[0]) + self.b[0])
        
        for k in range(1,self.num_hidden_layers):
            
            z = tf.nn.leaky_relu(tf.matmul(input_y, self.A[k]) + self.b[k] + tf.matmul(z, self.W[k-1]))

        return z


def sample_data_gen(DATASET, BATCH_SIZE, SCALE , VARIANCE):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    
    data_folder_sicnn = os.path.join(current_folder, 'data')

    if DATASET == 'X':
        while True:
            dataset_X = np.load(os.path.join(data_folder_sicnn, 'X.npy'))
            yield dataset_X

    elif DATASET == 'Y':
        while True:
            dataset_Y = np.load(os.path.join(data_folder_sicnn, 'Y.npy'))
            yield dataset_Y

    elif DATASET == 'Xn':
        while True:
            dataset_Xn = np.load(os.path.join(data_folder_sicnn, 'Xn.npy'))
            yield dataset_Xn

    elif DATASET == 'Yn':
        while True:
            dataset_Yn = np.load(os.path.join(data_folder_sicnn, 'Yn.npy'))
            yield dataset_Yn


def generate_uniform_around_centers(centers,variance):

    num_center = len(centers)

    return centers[np.random.choice(num_center)] + variance*np.random.uniform(-1,1,(2))

def generate_cross(centers,variance):

    num_center = len(centers)
    x = variance*np.random.uniform(-1,1)
    y = (np.random.randint(2)*2 -1)*x

    return centers[np.random.choice(num_center)] + [x,y]

def drawArrow(A, B):
    plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1], color=[0.5,0.5,1], alpha=0.3)
              #head_width=0.01, length_includes_head=False)


if __name__=='__main__':
    main()     