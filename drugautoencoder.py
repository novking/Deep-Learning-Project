"""
 This tutorial introduces denoising auto-encoders (dA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image

# infile = 'drugtrain_05pl1.txt'
infile = 'drugtrain_08pl2pf5.txt'
imgRows = 8*2        # the number of paths per instance, must be hardcoded
# imgRows = 5
n_hidden = 1000
epochs = 100
batch_size = 20
corruption_level = 0
learn_rate = 0.1
num_preds = 13      # of unique predicates in file, used to decode data

float_type = theano.config.floatX

class dA(object):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None
    ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=float_type
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=float_type
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=float_type
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=float_type) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """

        # let's try some alternatives...        

        # plain softmax doesn't do so well
        # return T.nnet.softmax(T.dot(hidden, self.W_prime) + self.b_prime)

        # try adjusting softmax by number of 1's expected
        # note this will probably cause issues for instances with few paths...
        # results in NAN costs!!!!
        # return T.mul(T.nnet.softmax(T.dot(hidden, self.W_prime) + self.b_prime), 2 * imgRows)

        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
        
    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        
        # alternative: mean squared error
        # results in very poor filters and reconstructions, although cost doesn't look too bad
        # L = T.sum(T.sqr(z-self.x), axis=1)
        
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)

    def get_reconstruction(self,input):
        y = self.get_hidden_values(input)
        z = self.get_reconstructed_input(y)
        return z
        
def load_data(dataset):
    print("Loading data...")
    data_x = numpy.loadtxt(dataset, numpy.int8, "#", ",")
    slice_train = round(data_x.shape[0]*.9)
    train_x = data_x[:slice_train]
    val_x = data_x[slice_train:]
    # shared_x = theano.shared(data_x, borrow=True)
    shared_x = theano.shared(numpy.asarray(train_x, dtype=float_type),
                             borrow=True)
    shared_val = theano.shared(numpy.asarray(val_x, dtype=float_type),
                             borrow=True)
    print("Data loaded.")
    return shared_x, shared_val

def save_encoding(da, encode, encodefile):
    x = T.matrix('x')  # the data is presented as rasterized images

    encode_da = theano.function(
        [x],
        da.get_hidden_values(x)
    )

    y = encode_da(encode)    
    # print("encode(x)=", y)
    numpy.savetxt(encodefile, y)    

def save_reconstruction(da, x_in, file):
    x = T.matrix('x')
    # recon = T.matrix('recon')
    recon_da = theano.function([x], da.get_reconstruction(x))
    y = recon_da(x_in)
    # print("recon=",y)
    path_ids = decode(y)
    # print("path ids=",path_ids)
    # numpy.savetxt(file, path_ids, fmt='%1.2f')
    numpy.savetxt(file, path_ids, fmt='%d')
        
def decode(x_in,num_paths=imgRows,num_preds=num_preds):
    # BUG: doesn't check for paths where all values are 0
    # create a matrix for each segment of the path
    segs = numpy.split(x_in,num_paths,axis=1)
    out = numpy.empty((x_in.shape[0],0),dtype='uint16')
    for s in segs:
        # print("s:", s)
        # find max entry of pred portion
        pred = numpy.argmax(s[:,0:num_preds],1)
        # print("pred:",pred.shape, pred)
        # find max entry of value portion
        obj = numpy.argmax(s[:,num_preds:],1)
        # note because we split the matrix between preds and objs, the
        # argmax is the correct index. No need to subtract the num_preds
        
        # print("obj",obj.shape, obj)
        pred.shape=(pred.shape[0],1)
        obj.shape=(obj.shape[0],1)
        # combine
        # pair = numpy.concatenate([pred,obj], axis=1)
        # out = numpy.concatenate([out,pair],axis=1)
        out = numpy.concatenate([out,pred,obj],axis=1)
    # print("out",out.shape,out)
    return out
    
def test_dA(learning_rate=learn_rate, training_epochs=epochs,
            dataset=infile,
            batch_size=50, output_folder='dA_plots'):

    """

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    train_set_x, val_set_x = load_data(dataset)
    # train_set_x, train_set_y = datasets[0]
    inputRows, inputCols = train_set_x.get_value(borrow=True).shape
    print("Train is ", inputRows, "x", inputCols)
    print("Validation is ", val_set_x.get_value(borrow=True).shape[0], "x", 
          val_set_x.get_value(borrow=True).shape[1])
        
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_val_batches = val_set_x.get_value(borrow=True).shape[0] // batch_size
    # start-snippet-2
    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    # end-snippet-2

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=inputCols,
        n_hidden=n_hidden
    )

    cost, updates = da.get_cost_updates(
        corruption_level=corruption_level,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    valid_da = theano.function(
        [index],
        cost,
        givens={
                x: val_set_x[index * batch_size: (index + 1) * batch_size   ]
         }
    )
    
#    encode_da = theano.function(
#        [x],
#        da.get_hidden_values(x)
#    )

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in range(training_epochs):
        # go through trainng set
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        # create an image of the weights after every n epochs and at th eend
        if (epoch % 5) == 0 or epoch==(training_epochs-1):
            image = Image.fromarray(
                tile_raster_images(X=da.W.get_value(borrow=True).T,
                                   img_shape=(imgRows,inputCols//imgRows), tile_shape=(10, 10),
                                   tile_spacing=(1, 1),maxWidth=400))
            image.save('filters_epoch_'+str(epoch)+'.png')

        val_cost = [] 
        for batch_index in range(n_val_batches):
            val_cost.append(valid_da(batch_index))
                
        print('Training epoch %d, cost ' % epoch, numpy.mean(c), ', val cost ', numpy.mean(val_cost))

        
    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print(('The no corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.)), file=sys.stderr)
 
    full_data = numpy.concatenate([train_set_x.get_value(borrow=True), 
                         val_set_x.get_value(borrow=True)])

    save_encoding(da, full_data, 'encodedrugs.txt')
    save_reconstruction(da, full_data, 'recodrugs.txt')
    
    # start-snippet-3
    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################

#    rng = numpy.random.RandomState(123)
#    theano_rng = RandomStreams(rng.randint(2 ** 30))
#
#    da = dA(
#        numpy_rng=rng,
#        theano_rng=theano_rng,
#        input=x,
#        n_visible=inputCols,
#        n_hidden=200
#    )
#
#    cost, updates = da.get_cost_updates(
#        corruption_level=0.3,
#        learning_rate=learning_rate
#    )
#
#    train_da = theano.function(
#        [index],
#        cost,
#        updates=updates,
#        givens={
#            x: train_set_x[index * batch_size: (index + 1) * batch_size]
#        }
#    )
#
#    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

#    # go through training epochs
#    for epoch in range(training_epochs):
#        # go through trainng set
#        c = []
#        for batch_index in range(n_train_batches):
#            c.append(train_da(batch_index))
#
#        print('Training epoch %d, cost ' % epoch, numpy.mean(c))
#
#    end_time = timeit.default_timer()
#
#    training_time = (end_time - start_time)
#
#    print(('The 30% corruption code for file ' +
#                          os.path.split(__file__)[1] +
#                          ' ran for %.2fm' % (training_time / 60.)), file=sys.stderr)
#    # end-snippet-3
#
#    # start-snippet-4
#    image = Image.fromarray(tile_raster_images(
#        X=da.W.get_value(borrow=True).T,
#        img_shape=(5,inputCols/5), tile_shape=(10, 10),
#        tile_spacing=(1, 1)))
#    image.save('filters_corruption_30.png')
#    # end-snippet-4
#
#    os.chdir('../')


if __name__ == '__main__':
    test_dA(batch_size=batch_size)
