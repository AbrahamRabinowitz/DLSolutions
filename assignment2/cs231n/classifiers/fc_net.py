from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.dims = [input_dim] + hidden_dims + [num_classes]
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        for i in range(len(self.dims) - 1):
          self.params[f'W{i+1}'] = np.random.normal(scale = weight_scale, size = (self.dims[i], self.dims[i + 1]))
          self.params[f'b{i + 1}'] = np.zeros(self.dims[i + 1])
        
        if normalization:
          for i in range(len(self.dims)-2):
            self.params[f'gamma{i+1}'] = np.ones(self.dims[i+1])
            self.params[f'beta{i+1}'] = np.zeros(self.dims[i+1])

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        cache = []
        if self.normalization == 'batchnorm':
          scores = X
          for i in range(len(self.dims)-1):
            if i < len(self.dims)- 2:
              scores, temp_cache = affine_batch_relu_forward(scores, 
              self.params[f'W{i+1}'], self.params[f'b{i+1}'],self.params[f'gamma{i+1}'],
              self.params[f'beta{i+1}'],self.bn_params[i] )
              cache.append(temp_cache)
            else:
              scores, temp_cache = affine_forward(scores,self.params[f'W{i+1}'],self.params[f'b{i +1}'] )
              cache.append(temp_cache)
            pass
        
        if self.normalization == 'layernorm':
          scores = X
          for i in range(len(self.dims)-1):
            if i < len(self.dims)- 2:
              scores, temp_cache = affine_layer_relu_forward(scores, 
              self.params[f'W{i+1}'], self.params[f'b{i+1}'],self.params[f'gamma{i+1}'],
              self.params[f'beta{i+1}'],self.bn_params[i] )
              cache.append(temp_cache)
            else:
              scores, temp_cache = affine_forward(scores,self.params[f'W{i+1}'],self.params[f'b{i +1}'] )
              cache.append(temp_cache)
            pass
        
        if not self.normalization:
          scores = X
          for i in range(len(self.dims)-1):
            if i < len(self.dims)- 2:
              if not self.use_dropout:
                scores, temp_cache = affine_relu_forward(scores,self.params[f'W{i+1}'], self.params[f'b{i+1}'])
                cache.append(temp_cache)
              else:
                scores, temp_cache = affine_relu_dropout_forward(scores,self.params[f'W{i+1}'], self.params[f'b{i+1}'],self.dropout_param)
                cache.append(temp_cache)
            else:
              scores, temp_cache = affine_forward(scores,self.params[f'W{i+1}'],self.params[f'b{i +1}'] )
              cache.append(temp_cache)
            pass


        
        
        
        # if not self.normalization:
        #   X_values = {}
        #   X_values['X2'], temp_cache = affine_forward(X, self.params[f'W{1}'], self.params[f'b{1}'])
        #   X_values['X1'] = temp_cache[0]
        #   X_values['Xrelu2'], X_values['X2'] = relu_forward(X_values['X2'])        
        #   for i in range(2, len(self.dims)-1):
        #     X_values[f'X{i+1}'], temp_cache = affine_forward(X_values[f'Xrelu{i}'], self.params[f'W{i}'], self.params[f'b{i}'])
        #     X_values[f'Xrelu{i+1}'], X_values[f'X{i+1}']= relu_forward(X_values[f'X{i+1}'])
        
          d = len(self.dims) - 1
          # scores, temp_cache = affine_forward(X_values[f'Xrelu{d}'], self.params[f'W{d}'], self.params[f'b{d}'])
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        reg_loss = 0.0
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        for j in range(1, len(self.dims)):
          reg_loss += (self.params[f'W{j}']**2).sum()
        reg_loss *= 0.5 *self.reg
        loss, dout = softmax_loss(scores,y)
        loss += reg_loss 

        if self.normalization == 'batchnorm':
          for j in reversed(range(1,len(self.dims))):
            if j == len(self.dims) - 1:
              temp_cache = cache[j-1]
              dout, grads[f'W{j}'], grads[f'b{j}'] = affine_backward(dout, temp_cache)
              grads[f'W{j}'] += self.reg*self.params[f'W{j}']
            else:
              temp_cache = cache[j-1]
              dout, grads[f'W{j}'], grads[f'b{j}'],grads[f'gamma{j}'], grads[f'beta{j}'] = affine_batch_relu_backward(dout, temp_cache)
              grads[f'W{j}'] += self.reg * self.params[f'W{j}']
        
        if self.normalization == 'layernorm':
          for j in reversed(range(1,len(self.dims))):
            if j == len(self.dims) - 1:
              temp_cache = cache[j-1]
              dout, grads[f'W{j}'], grads[f'b{j}'] = affine_backward(dout, temp_cache)
              grads[f'W{j}'] += self.reg*self.params[f'W{j}']
            else:
              temp_cache = cache[j-1]
              dout, grads[f'W{j}'], grads[f'b{j}'],grads[f'gamma{j}'], grads[f'beta{j}'] = affine_layer_relu_backward(dout, temp_cache)
              grads[f'W{j}'] += self.reg * self.params[f'W{j}']

        if not self.normalization :
          for j in reversed(range(1,len(self.dims))):
            if j == len(self.dims) - 1:
              temp_cache = cache[j-1]
              dout, grads[f'W{j}'], grads[f'b{j}'] = affine_backward(dout, temp_cache)
              grads[f'W{j}'] += self.reg*self.params[f'W{j}']
            else:
              if not self.use_dropout:
                temp_cache = cache[j-1]
                dout, grads[f'W{j}'], grads[f'b{j}'] = affine_relu_backward(dout, temp_cache)
                grads[f'W{j}'] += self.reg * self.params[f'W{j}']
              else:
                temp_cache = cache[j-1]
                dout, grads[f'W{j}'], grads[f'b{j}'] = affine_relu_dropout_backward(dout, temp_cache)
                grads[f'W{j}'] += self.reg * self.params[f'W{j}']




        # if not self.normalization:
        #   dout, grads[f'W{d}'], grads[f'b{d}'] = affine_backward(dout,(X_values[f'Xrelu{d}'],self.params[f'W{d}'], self.params[f'b{d}']))
        #   grads[f'W{d}'] += self.reg * self.params[f'W{d}']
        #   for j in reversed(range(2,d+1)):
        #     if j == 2:
        #       dout = relu_backward(dout, X_values[f'X{j}'])
        #       dout, grads[f'W{j-1}'], grads[f'b{j-1}'] = affine_backward(dout,(X_values[f'X{j-1}'],self.params[f'W{j-1}'], self.params[f'b{j-1}']))
        #     else:
        #       dout = relu_backward(dout, X_values[f'X{j}'])
        #       dout, grads[f'W{j-1}'], grads[f'b{j-1}'] = affine_backward(dout,(X_values[f'Xrelu{j-1}'],self.params[f'W{j-1}'], self.params[f'b{j-1}']))
        #     grads[f'W{j-1}'] += self.reg * self.params[f'W{j-1}'] 
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
