#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 17:56:16 2017

@author: vu
"""
import numpy as np
import pandas as pd
import pickle
import math

import theano.tensor as T
from theano import function, printing
import theano

from theano import config
# config.device = 'cpu'
# config.gcc.cxxflags = "-D_hypot=hypot"
config.compute_test_value = 'off'
import os
os.environ["THEANO_FLAGS"] = "exception_verbosity=high,on_opt_error=optimizer_excluding=ShapeOpt:local_lift_transpose_through_dot:scan_opt"
from theano.compile.nanguardmode import NanGuardMode
# config.NanGuardMode.action == 'pdb'


class EmissionModel:
    """ Simple emission model without CNN
    word embedding layer -> ReLU layer -> softmax layer
    """
    
    def init_weights_bias(self, vocab_input_size, layer_size, vocab_output_size, seed=1402):
        random_state = np.random.RandomState(seed)
        
        size_list = np.concatenate(([vocab_input_size], layer_size, [vocab_output_size]), axis=0)
        w = []
        b = []
        
        for i in range(len(size_list) - 1):
            w.append(theano.shared(
                    value=np.asarray(
                        random_state.uniform(low=-1.0, high=1.0, size=(size_list[i+1], size_list[i])), 
                        dtype=theano.config.floatX
                    ), borrow=True
            ))
            b.append(theano.shared(
                    value=np.asarray(
                        random_state.uniform(low=-1.0, high=1.0, size=(size_list[i+1], 1)), 
                        dtype=theano.config.floatX
                    ), 
                    borrow=True,
                    broadcastable=(False,True)
            ))
        
        return w, b
    
    def softmax(self, x):
        x = T.transpose(x)
        e_x = T.exp(x - x.max(axis=1, keepdims=True)) 
        out = e_x / e_x.sum(axis=1, keepdims=True)
        return T.transpose(out)
    
    #[7,512]
    def __init__(self, vocab_input_size, layer_size, vocab_output_size, baum_welch_model, 
                 epoch=1, batch=1, learning_rate = .01, seed=1412):
        
        self.epoch = epoch
        self.batch = batch
        self.learning_rate = learning_rate
        self.seed = seed
        self.emission_posteriors = []
        self.transition_posteriors = []
        self.baum_welch_model = baum_welch_model
        
        self.vocab_input_size = vocab_input_size
        self.d_embedding_size = layer_size[0]
        
        x_training_input = T.matrix().astype(config.floatX)
        
        self.w, self.b = self.init_weights_bias(vocab_input_size, layer_size, vocab_output_size, seed)
        
        # Word embedding layer
        word_embedding_layer = T.dot(self.w[0], x_training_input) # [7, 10] * [10, 5] = [7, 5]
        
        # ReLU layer
        z_relu_layer = T.dot(self.w[1], word_embedding_layer) + self.b[1] # [512, 7] * [7, 5] = [512, 5]
        z_relu_layer_shape = T.shape(z_relu_layer)
        relu_layer = T.nnet.relu(T.flatten(z_relu_layer))
        relu_layer_reshaped = T.reshape(relu_layer, z_relu_layer_shape) # [512, 5]
        
        # Softmax layer
        z_softmax_layer = T.dot(self.w[2], relu_layer_reshaped) + self.b[2] # [12, 512] * [512, 5] = [12, 5]
#         softmax_layer = T.transpose(T.nnet.softmax(T.transpose(z_softmax_layer))) # Output: [12, 5]
        softmax_layer = T.nnet.softmax(z_softmax_layer) # Output: [12, 5]
        softmax_layer_clipped = T.clip(softmax_layer, 1e-35, 1.0 - 1e-35)
        
        # Calculate new gradient
        posteriors = T.matrix().astype(config.floatX)
        
        cost = T.sum(T.transpose(posteriors) * T.log(softmax_layer_clipped))
#         cost = T.sum(T.transpose(posteriors) * T.log(softmax_layer))
        # TODO: use dw[] and db[] abstractly 
        dw0,dw1,dw2,db1,db2 = T.grad(
            cost=cost, wrt=[self.w[0], self.w[1], self.w[2], self.b[1], self.b[2]]
        )

        # Update w and b
        updates = [
            (self.w[0], self.w[0] - self.learning_rate * dw0), 
            (self.w[1], self.w[1] - self.learning_rate * dw1), 
            (self.b[1], self.b[1] - self.learning_rate * db1),
            (self.w[2], self.w[2] - self.learning_rate * dw2), 
            (self.b[2], self.b[2] - self.learning_rate * db2)
        ]
        
        # Compile model
        self.test = theano.function(
            inputs=[x_training_input], 
            outputs=[word_embedding_layer, softmax_layer]
        ) 
        self.train_mini_batch_function = theano.function(
            inputs=[x_training_input, posteriors], 
            outputs=softmax_layer, 
            updates=updates,
            mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False)
        )
        self.test_values = theano.function(
            inputs=[x_training_input], 
            outputs=softmax_layer,
            mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False)
        )
        
    def train_mini_batch(self, testing_target, testing_source):
        one_hot_input = np.eye(self.vocab_input_size)[testing_target].T
        one_hot_input = np.asarray(one_hot_input).astype(config.floatX)
#         print("one_hot_input", one_hot_input, np.shape(one_hot_input))
        softmax_matrix = self.test_values(one_hot_input)
        print("softmax_matrix", softmax_matrix, np.shape(softmax_matrix))
        
#         softmax_matrix = np.clip(softmax_matrix, 1e-35, 1.0 - 1e-35)
        
        emission_posterior_vout = np.zeros_like(softmax_matrix.T) # [V_f_size, e_size]
        emission_matrix = [] # [f_size, e_size]
        for indice in testing_source:
            emission_matrix.append(softmax_matrix[indice])
        print("emission_matrix", emission_matrix, np.shape(emission_matrix))
        # Normalize emission_matrix
#         emission_matrix = self.baum_welch_model.normalize_matrix(emission_matrix, axis=0)
        print("emission_matrix nomalized", emission_matrix, np.shape(emission_matrix))
        emission_posterior, transition_posterior = \
            self.baum_welch_model.calculate_baum_welch_posteriors(len(testing_target), np.transpose(emission_matrix))
        print("emission_posterior", emission_posterior, np.shape(emission_posterior))
        
        # transform emission size to [target_size, v_out]
        for i, indice in enumerate(testing_source):
            emission_posterior_vout[:, indice] = np.maximum(emission_posterior_vout[:, indice], emission_posterior[:, i])
#         print("emission_posterior_vout", emission_posterior_vout, np.shape(emission_posterior_vout))
        self.train_mini_batch_function(one_hot_input, np.asarray(emission_posterior_vout).astype(config.floatX))
        
        return emission_posterior, transition_posterior
    
    def test_mini_batch(self, testing_target, testing_source):
        one_hot_input = np.eye(self.vocab_input_size)[testing_target].T
        one_hot_input = np.asarray(one_hot_input).astype(config.floatX)
        softmax_matrix = self.test_values(one_hot_input)
        
        emission_matrix = [] # [f_size, e_size]
        for indice in testing_source:
            emission_matrix.append(softmax_matrix[indice])
        return emission_matrix
        
    def train_model_epoch(self, target_inputs, source_inputs, input_indice_shift=0):
        # TODO: add epoch functionality
        for epoch in range(self.epoch):
            self.emission_posteriors = []
            self.transition_posteriors = []
        #         for target_inputs_batch, source_inputs_batch in zip(np.split(target_inputs, self.batch), np.split(source_inputs, self.batch)):
        #             for x_target, x_source in zip(target_inputs_batch, source_inputs_batch):
            for i, x_target, x_source in zip(range(len(target_inputs)), target_inputs, source_inputs):
                if (len(x_target) == 1 or len(x_source) == 1):
                    self.emission_posteriors.append(np.zeros((len(x_target), len(x_source))))
                    self.transition_posteriors.append(np.zeros((len(1), len(x_source), len(x_source))))
                    continue
                
                xx_target = [int(x)+input_indice_shift for x in x_target]
                xx_source = [int(x)+input_indice_shift for x in x_source]
                print("\n+++++++++ The sentence ", i)
                print("xx_source: ", len(xx_source), " => ", xx_source)
                print("xx_target: ", len(xx_target), " => ", xx_target)
                emis_posterior, trans_posterior = self.train_mini_batch(xx_target, xx_source)
                self.emission_posteriors.append(emis_posterior)
                self.transition_posteriors.append(trans_posterior)
            
            # Update Non-negative set of BW model
            self.baum_welch_model.update_non_negative_transition_set(self.emission_posteriors, self.transition_posteriors)
            print(self.baum_welch_model.non_negative_set)
            
    def train_model(self, target_inputs, source_inputs, input_indice_shift=0):
        # TODO: add epoch functionality
#         for i in range(self.epoch):
        self.emission_posteriors = []
        self.transition_posteriors = []
#         for target_inputs_batch, source_inputs_batch in zip(np.split(target_inputs, self.batch), np.split(source_inputs, self.batch)):
#             for x_target, x_source in zip(target_inputs_batch, source_inputs_batch):
        for i, x_target, x_source in zip(range(len(target_inputs)), target_inputs, source_inputs):
            if (len(x_target) == 1 or len(x_source) == 1):
                self.emission_posteriors.append(np.zeros((len(x_target), len(x_source))))
                self.transition_posteriors.append(np.zeros((len(1), len(x_source), len(x_source))))
                continue
            xx_target = [int(x)+input_indice_shift for x in x_target]
            xx_source = [int(x)+input_indice_shift for x in x_source]
            print("\n+++++++++ The sentence ", i)
            print("xx_source: ", len(xx_source), " => ", xx_source)
            print("xx_target: ", len(xx_target), " => ", xx_target)
            emis_posterior, trans_posterior = self.train_mini_batch(xx_target, xx_source)
            self.emission_posteriors.append(emis_posterior)
            self.transition_posteriors.append(trans_posterior)
            
        return self.transition_posteriors
    
    def get_alignment(self, target_inputs, source_inputs, input_indice_shift=0):
        alignments = []
#         for target_inputs_batch, source_inputs_batch in zip(np.split(target_inputs, self.batch), np.split(source_inputs, self.batch)):
#             for x_target, x_source in zip(target_inputs_batch, source_inputs_batch):
        for i, x_target, x_source in zip(range(len(target_inputs)), target_inputs, source_inputs):
            align = [0]
            if (len(x_target) == 1 or len(x_source) == 1):
                alignments.append(align)
                continue
            xx_target = [int(x)+input_indice_shift for x in x_target]
            xx_source = [int(x)+input_indice_shift for x in x_source]
            print("\n+++++++++ The sentence ", i)
            print("xx_source: ", len(xx_source), " => ", xx_source)
            print("xx_target: ", len(xx_target), " => ", xx_target)
            emis_matrix = self.test_mini_batch(xx_target, xx_source)
            trans_matrix = self.baum_welch_model.generate_transition_distant_matrix(len(xx_target))
            
            # TODO: calculate aligment [VITERBI]
            for ind, t in enumerate(range(1, len(x_source))):
                mul = emis_matrix[ind] * trans_matrix[align[-1]]
                print("max", np.argmax(mul), ":", mul)
                align.append(np.argmax(mul))
            exporting_align = []
            for ia, a in enumerate(align):
                if (a < len(xx_target) or a==0):
                    exporting_align.append(str(ia+1) + "-" + str(a+1)) # indice starts from 1
            print(exporting_align, len(exporting_align))
            alignments.append(exporting_align)
            alignments.append(align)
            
class BaumWelchModel:
    
    def normalize_matrix(self, x, axis=1, whole_matrix=False):
        """Compute softmax values for each sets of scores in x.
            axis=1: row
            axis=0: column 
        Input
        -----
        
        Output
        ------
        """
        if len(np.shape(x)) == 1 or whole_matrix:
#             e_x = np.exp(x - np.max(x))
            e_x = x
            return e_x / np.sum(e_x)
        if axis == 0:
#             e_x = np.exp( np.subtract(x, np.max(x, axis=axis)[None, :]) )
            e_x = x
            return e_x / np.sum(e_x, axis=axis)[None, :]
        else: 
#             e_x = np.exp( np.subtract(x, np.max(x, axis=axis)[:, None]) )
            e_x = x
            return e_x / np.sum(e_x, axis=axis)[:, None]
        
    def generate_transition_distant_matrix(self, sentence_length, 
                                           po=0., nomalized=True):
        """ Generate a transition matrix based on jump distance 
        in the latent sentence.
        We extend the latent sentence for 2*length in which each word has 
        an empty word to represent no-alignment state.
        where [sentence_length:end] elements are empty words considered as 
        latent words having no direct aligment.

        Input
        -----
        sentence_length: the length of latent sentence
                      int value
        non_negative_set: random non-negative set as max_distance size
        po: default value for A->A_empty_word

        Output
        ------
        trans_distant_matrix
        """
        if po==0.:
            po = self.po
        trans_distant_matrix = np.zeros((2*sentence_length, 2*sentence_length))

        for i in range(sentence_length):
            for j in range(sentence_length):
                indice = j - i + self.max_distance + 1
                if indice < 0:
                    p_ = self.non_negative_set[0]
                elif (indice > 2*self.max_distance + 2):
                    p_ = self.non_negative_set[-1]
                else:
                    p_ = self.non_negative_set[indice]
                trans_distant_matrix[i][j] = p_

        print(trans_distant_matrix)

        for i in range(sentence_length):
            trans_distant_matrix[i+sentence_length][i+sentence_length] = po
            trans_distant_matrix[i+sentence_length][i] = po

            sum_d = np.sum(trans_distant_matrix[i, :sentence_length])
            trans_distant_matrix[i, :sentence_length] = \
                    np.divide(
                        trans_distant_matrix[i, :sentence_length], 
                        sum_d
                    )
            trans_distant_matrix[i, sentence_length:] = \
                    np.copy(trans_distant_matrix[i, :sentence_length])

        return trans_distant_matrix
    
    def generate_transition_matrix(self, sentence_length, po=0., 
                                   nomalized=True):
        """ Generate a transition matrix based on jump distance in the latent sentence.

        Input
        -----
        sentence_length: the length of latent sentence
                      int value
        non_negative_set: random non-negative set as max_distance size
        po: default value for A->A_empty_word

        Output
        ------
        trans_matrix
        """
        if po==0.:
            po = self.po
        trans_matrix = np.zeros((sentence_length, sentence_length))

        for i in range(sentence_length):
            for j in range(sentence_length):
                indice = j - i + self.max_distance + 1
                if indice < 0:
                    p_ = self.non_negative_set[0]
                elif (indice > 2*self.max_distance + 2):
                    p_ = self.non_negative_set[-1]
                else:
                    p_ = self.non_negative_set[indice]
                trans_matrix[i][j] = p_
        if nomalized:
            return self.normalize_matrix(trans_matrix, axis=1)
        return trans_matrix
        
    def __init__(self, max_distance, po=0.3, seed=1402):
        np.random.seed(seed)
        self.max_distance = max_distance
        self.non_negative_set = np.random.randint(
                                    low=1, high=100, 
                                    size=[max_distance + max_distance + 3]
        )
        self.po = po
        
    def calc_forward_messages(self, unary_matrix, transition_matrix, emission_matrix):
        """Calcualte the forward messages ~ alpha values.
        
        
        Input
        -----
        unary_matrix: emission posteriors - marginal probabilities ~ initial matrix.
                      size ~ [1, target_len]
        transition_matrix: size ~ [target_len, target_len]
        emission_matrix: size ~ [target_len, source_len]

        Return
        ------
        alpha
        """

        # TODO: verify matrix length
        source_len = np.shape(emission_matrix)[1]
        target_len = np.shape(emission_matrix)[0]

        alpha = np.zeros(np.shape(emission_matrix))
        print("emission_matrix[:,0]", emission_matrix[:, 0])
        print("unary_matrix", unary_matrix)
        alpha.T[0] = np.multiply(emission_matrix[:,0], unary_matrix)
        print("alpha.T[0]", alpha.T[0])
        
        for t in np.arange(1, source_len):
            for i in range(target_len):
                sum_al = 0.0;
                for j in range(target_len):
                    sum_al += alpha[j][t-1] * transition_matrix[j][i]

                alpha[i][t] = emission_matrix[i][t] * sum_al

        return alpha
    
    
    def calc_backward_messages(self, transition_matrix, emission_matrix):
        """Calcualte the backward messages ~ beta values.

        Return
        ------
        beta
        """
        # TODO: verify matrix length
        source_len = np.shape(emission_matrix)[1]
        target_len = np.shape(emission_matrix)[0]

        beta = np.zeros(np.shape(emission_matrix))
        beta[:,-1] = [1]*target_len

        for t in reversed(range(source_len-1)):
            for i in range(target_len):
                for j in range(target_len):
                    beta[i][t] += beta[j][t+1] * transition_matrix[i][j] * emission_matrix[j][t+1]

        return beta

    def calc_posterior_matrix(self, alpha, beta, transition_matrix, emission_matrix):
        """Calcualte the gama and epsilon values in order to reproduce 
        better transition and emission matrix.
        
        gamma: P(e_aj|f_j)
        epsilon: P(e_aj,e_a(j+1)|f_j)

        Return
        ------
        unary_matrix, posterior_gamma, posterior_epsilon
        """
        # TODO: verify matrix length
        source_len = np.shape(alpha)[1]
        target_len = np.shape(alpha)[0]

        gamma = np.multiply(alpha, beta)
        epsilon = np.zeros((source_len-1, target_len, target_len))

        # Normalization on columns
        gamma = self.normalize_matrix(gamma, axis=0)

        for t in range(source_len-1):   
            for i in range(target_len):
                for j in range(target_len):
                    epsilon[t][i][j] = alpha[i][t] * transition_matrix[i][j] * \
                                        beta[j][t+1] * emission_matrix[j][t+1]
            # Normalization
            epsilon[t] = self.normalize_matrix(epsilon[t], whole_matrix=True)

        # Update unary matrix
        # Normalization unary
        new_unary_matrix = np.copy(gamma[:,0])
        #self.normalize_matrix(np.copy(gamma[:,0]), axis=1)

        return new_unary_matrix, gamma, epsilon


    def calculate_baum_welch_posteriors(self, sentence_length, emission_matrix, unary_matrix=None):
        if unary_matrix == None:
            unary_matrix = [0.01]*sentence_length
            unary_matrix[0] = 1 - np.sum(unary_matrix) + 0.01
        transition_matrix = self.generate_transition_matrix(sentence_length)
#         emission_matrix = self.normalize_matrix(emission_matrix, axis=0)
        
        alpha = self.calc_forward_messages(unary_matrix, 
                                           transition_matrix, emission_matrix)
        beta = self.calc_backward_messages(transition_matrix, emission_matrix)

        new_unary_matrix, emission_posterior, transition_posterior \
            = self.calc_posterior_matrix(alpha, beta, 
                                         transition_matrix, emission_matrix)
        return emission_posterior, transition_posterior # gamma, epsilon
    
    def update_non_negative_transition_set(self, emission_posteriors, 
                                           transition_posteriors):
        # 1: update non-negative set: s[-1] = 
        # 1.1: calculate new transition matrix
        transition_list = []
        for gamma, epsilon in zip(emission_posteriors, transition_posteriors):
            source_len = np.shape(gamma)[1]
            target_len = np.shape(gamma)[0]
            if (source_len <= 1 or target_len <= 1):
                transition_list.append([0])
                continue
            new_transition_matrix = np.zeros((target_len, target_len))
            
            for i in range(target_len):
                sum_gamma = np.sum(gamma[i, :])
                for j in range(target_len):
                    sum_ep = np.sum(epsilon[:, i, j])
                    new_transition_matrix[i][j] = sum_ep/sum_gamma
            # Normalization
            new_transition_matrix = self.normalize_matrix(new_transition_matrix, axis=1)
            transition_list.append(new_transition_matrix)
        
        # 1.2: update
        new_non_negative_set = np.zeros(self.max_distance + self.max_distance + 3)
        n_non_negative_set = np.ones(self.max_distance + self.max_distance + 3)
        
        for trans_matrix in transition_list:
            for i in range(len(trans_matrix)):
                for j in range(len(trans_matrix)):
                    if (len(trans_matrix)<=1 or math.isnan(trans_matrix[i][j])):
                        continue
                    indice = j - i + self.max_distance + 1
                    if indice < 0:
                        n_non_negative_set[0] += 1
                        new_non_negative_set[0] += trans_matrix[i][j]
                    elif (indice > 2*self.max_distance + 2):
                        n_non_negative_set[-1] += 1
                        new_non_negative_set[-1] += trans_matrix[i][j]
                    else:
                        n_non_negative_set[indice] += 1
                        new_non_negative_set[indice] += trans_matrix[i][j]
        
        self.old_non_negative_set = np.copy(self.non_negative_set)
        self.non_negative_set = np.array(np.divide(new_non_negative_set + self.non_negative_set, n_non_negative_set))
        


