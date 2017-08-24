#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 1 17:56:16 2017

@author: vu
"""
import numpy as np
import pandas as pd
import pickle
import math
import numbers

import theano.tensor as T
from theano import function, printing
import theano

from theano import config
# config.device = 'cpu'
# config.gcc.cxxflags = "-D_hypot=hypot"
config.compute_test_value = 'off'
#import os
#os.environ["THEANO_FLAGS"] = "exception_verbosity=high,on_opt_error=optimizer_excluding=ShapeOpt:local_lift_transpose_through_dot:scan_opt"
from theano.compile.nanguardmode import NanGuardMode
# config.NanGuardMode.action == 'pdb'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class EmissionModel:
    """ Simple emission model without CNN
    word embedding layer -> ReLU layer -> softmax layer
    """
    
    def save_obj(self, obj, path):
        print("Saving file ... " + path)
        f = open(path, 'wb')
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print("File Saved !")
        
    def load_obj(self, path):
        print("Loading file ... " + path)
        f = open(path, 'rb')
        obj = pickle.load(f)
        f.close()
        print("File loaded !")
        return obj
    
    def save_align(self, alignments, path):
        print("Saving alignment to ... " + path)
        f = open(path, 'w', encoding='utf8')
        for s in alignments:
            for ss in s:
                f.write(ss + " ") 
            f.write("\n") 
        f.close()
        print("File Saved !")
    
    def save_params(self, params_path_prefix):
        w_saved = []
        b_saved = []
        for w in self.w:
            w_saved.append(w.get_value())
        for b in self.b:
            b_saved.append(b.get_value())
        self.save_obj(w_saved, params_path_prefix + "_w.pickle")
        self.save_obj(b_saved, params_path_prefix + "_b.pickle")
        
    def load_params(self, params_path_prefix):
        f_w = open(params_path_prefix+ "_w.pickle", 'rb')
        w_saved = pickle.load(f_w)
        f_w.close()
        f_b = open(params_path_prefix+ "_b.pickle", 'rb')
        b_saved = pickle.load(f_b)
        f_b.close()
        # TODO: check w and b sizes
        w = []
        b = []
        for ww in w_saved:
            w.append(theano.shared(
                    value=np.asarray(ww, dtype=theano.config.floatX), 
                    borrow=True
            ))
        for bb in b_saved:
            b.append(theano.shared(
                    value=np.asarray(bb, dtype=theano.config.floatX), 
                    borrow=True,
                    broadcastable=(False,True)
            ))
        return w, b
    
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
                 target_tokenizer, source_tokenizer,
                 epoch=1, batch=1, learning_rate = .01, seed=1412, 
                 params_path_prefix=None, out_prefix=None):
        
        self.epoch = epoch
        self.batch = batch
        self.learning_rate = learning_rate
        self.seed = seed
        self.emission_posteriors = []
        self.transition_posteriors = []
        self.baum_welch_model = baum_welch_model
        self.target_tokenizer = target_tokenizer
        self.source_tokenizer = source_tokenizer
        
        self.vocab_input_size = vocab_input_size
        self.d_embedding_size = layer_size[0]
        
        self.params_path_prefix = params_path_prefix
        self.out_prefix = out_prefix
        
        x_training_input = T.matrix().astype(config.floatX)
        
        if (self.params_path_prefix == None):
            self.w, self.b = self.init_weights_bias(vocab_input_size, layer_size, vocab_output_size, seed)
        else:
            self.w, self.b = self.load_params(params_path_prefix)
            
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
#        print("softmax_matrix", softmax_matrix, np.shape(softmax_matrix))
        
        emission_posterior_vout = np.zeros_like(softmax_matrix.T) # [V_f_size, e_size]
        emission_matrix = [] # [f_size, e_size]
        for indice in testing_source:
            emission_matrix.append(softmax_matrix[indice])
#        print("emission_matrix", emission_matrix, np.shape(emission_matrix))
        # Normalize emission_matrix
#         emission_matrix = self.baum_welch_model.normalize_matrix(emission_matrix, axis=0)
#        print("emission_matrix nomalized", emission_matrix, np.shape(emission_matrix))
        emission_posterior, transition_posterior = \
            self.baum_welch_model.calculate_baum_welch_posteriors(len(testing_target), np.transpose(emission_matrix))
#        print("emission_posterior", emission_posterior, np.shape(emission_posterior))
        
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
        
    def train_model_epoch(self, target_inputs, source_inputs, 
                          aer_target_inputs=None, aer_source_inputs=None, 
                          input_indice_shift=0, align_indice_sift=0,
                          target_AER=None):
        if (aer_target_inputs==None):
            aer_target_inputs=target_inputs
        if (aer_source_inputs==None):
            aer_source_inputs=source_inputs
        # TODO: add epoch functionality
        for epoch in range(self.epoch):
            print("******** Epoch ", epoch, " ***********")
            self.emission_posteriors = []
            self.transition_posteriors = []
        #         for target_inputs_batch, source_inputs_batch in zip(np.split(target_inputs, self.batch), np.split(source_inputs, self.batch)):
        #             for x_target, x_source in zip(target_inputs_batch, source_inputs_batch):
            with open(source_inputs, encoding="utf8") as source_f, open(target_inputs, encoding='utf8') as target_f:
                i = 0
                for source_line, target_line in zip(source_f, target_f):
                    x_source = self.source_tokenizer.texts_to_sequences([source_line.strip()])[0]
                    x_target = self.target_tokenizer.texts_to_sequences([target_line.strip()])[0]
                    
                    if (len(x_target) < 1 or len(x_source) < 1):
                        continue
                    
                    if (len(x_target) == 1 or len(x_source) == 1):
                        self.emission_posteriors.append(np.zeros((len(x_target), len(x_source))))
                        self.transition_posteriors.append(np.zeros((1, len(x_source), len(x_source))))
                        continue
                    
                    xx_target = [int(x)+input_indice_shift for x in x_target]
                    xx_source = [int(x)+input_indice_shift for x in x_source]
#                    if (i%1000==0):
                    print("\n+++++++++ The sentence ", i, " epoch ", epoch)
                    print("xx_source: ", len(xx_source), " => ", xx_source)
                    print("xx_target: ", len(xx_target), " => ", xx_target)
                    emis_posterior, trans_posterior = self.train_mini_batch(xx_target, xx_source)
                    self.emission_posteriors.append(emis_posterior)
                    self.transition_posteriors.append(trans_posterior)
                    i += 1
            
            # Update Non-negative set of BW model
            self.baum_welch_model.update_non_negative_transition_set(self.emission_posteriors, self.transition_posteriors)
            print("New non_negative_set", self.baum_welch_model.non_negative_set)
            
            if self.out_prefix != None:
                align = self.get_alignment(target_inputs=aer_target_inputs, 
                                  source_inputs=aer_source_inputs, 
                                  input_indice_shift=input_indice_shift,
                                  align_indice_sift=align_indice_sift)
                self.save_params(self.out_prefix + "_params_epoch_" + str(epoch))
                self.save_obj(align, self.out_prefix + "_alignment_epoch_" + str(epoch))
                
            if target_AER != None:
                if self.out_prefix == None:
                    align = self.get_alignment(target_inputs=aer_target_inputs, 
                                  source_inputs=aer_source_inputs, 
                                  input_indice_shift=input_indice_shift,
                                  align_indice_sift=align_indice_sift)
                # TODO: calculate AER
                print("Epoch ", epoch, "Alignment score:", \
                      self.calculate_AER_score(result=align, target_AER=target_AER, align_indice_sift=align_indice_sift))
                
    def get_alignment(self, target_inputs, source_inputs, 
                      input_indice_shift=0, align_indice_sift=0):
        print("Calculating alignments ...")
        alignments = []
        with open(source_inputs, encoding="utf8") as source_f, open(target_inputs, encoding='utf8') as target_f:
                i = 0
                for source_line, target_line in zip(source_f, target_f):
                    x_source = self.source_tokenizer.texts_to_sequences([source_line.strip()])[0]
                    x_target = self.target_tokenizer.texts_to_sequences([target_line.strip()])[0]
                    
                    align = [0]
                    if (len(x_target) == 1 or len(x_source) == 1):
                        alignments.append(align)
                        continue
                    xx_target = [int(x)+input_indice_shift for x in x_target]
                    xx_source = [int(x)+input_indice_shift for x in x_source]
        #            print("\n+++++++++ The sentence ", i)
        #            print("xx_source: ", len(xx_source), " => ", xx_source)
        #            print("xx_target: ", len(xx_target), " => ", xx_target)
                    emis_matrix = self.test_mini_batch(xx_target, xx_source)
                    trans_matrix = self.baum_welch_model.generate_transition_distant_matrix(len(xx_target))
                    
                    # Calculate aligment [VITERBI]
                    for ind, t in enumerate(range(1, len(x_source))):
                        mul = np.array([emis_matrix[ind], emis_matrix[ind]]).flatten() * trans_matrix[align[-1]]
        #                print("max", np.argmax(mul), ":", mul)
                        align.append(np.argmax(mul))
                    
                    exporting_align = []
                    for ia, a in enumerate(align):
                        if (a < len(xx_target) or a==0 or len(align)==1):
                            exporting_align.append(str(ia+align_indice_sift) + "-" + str(a+align_indice_sift)) # indice starts from 0
                    
                    assert(align_indice_sift >= 0)
                    
                    if(len(exporting_align) < 1):
                        exporting_align.append(align_indice_sift + "-" + align_indice_sift)
        #            print("Align ", i, " : ", exporting_align, len(exporting_align))
                    alignments.append(exporting_align)
                    i += 1
            
        return alignments
    
    def calculate_AER(self, S, P, A):
        S, P, A = np.array(S), np.array(P), np.array(A)
        s_a, p_a, len_s, len_a = 0, 0, 0, 0
        for s, p, a in zip(S, P, A):
            s, p, a = np.array(s), np.array(p), np.array(a)
            s_a += len(list(set(s).intersection(a)))
            p_a += len(list(set(p).intersection(a)))
            len_s += len(s[s != ""])
            len_a += len(a[a != ""])
#        print ("s_a", s_a)
        p_a += s_a
#        print ("p_a", p_a)
        aer = (s_a + p_a) / (len_s + len_a)
#        print ("aer", 1.-aer)
        
        return 1. - aer
    
    def calculate_AER_score(self, result, target_AER, align_indice_sift=0):
        target_file = open(target_AER) # Index starts from 1

        target_lines = target_file.readlines()
        target_lines = [str(line[:-1]) for line in target_lines]
        target_lines = np.reshape(target_lines, (2501, 2))
        
        sure = target_lines[:,0]
        possible = target_lines[:,1]
        
        S, P, A = [], [], []
        
        # Split and plus 1 to result indexes
        for a, s_, p_ in zip(result, sure, possible):
            for i, number in enumerate(a):
                if(isinstance(number, numbers.Number)):
                    a[i] = str(align_indice_sift) + "-" + str(align_indice_sift)
                else:
                    n1 = int(number.split("-")[0]) + align_indice_sift
                    n2 = int(number.split("-")[1]) + align_indice_sift
                    number = str(n1) + "-" + str(n2)
                    a[i] = number
            S.append(s_.strip().split(" "))
            P.append(p_.strip().split(" "))
            A.append(a)
        return self.calculate_AER(S, P, A)
    
            
class BaumWelchModel:
    
    def add_matrix(self, a, b, max_size=None):
        # compatible with two matrices different shape
        c = np.zeros(np.max([np.shape(a), np.shape(b)], axis=0))
        c[:np.shape(a)[0], :np.shape(a)[1]] += a
        c[:np.shape(b)[0], :np.shape(b)[1]] += b
        if(max_size != None):
            x , y = np.shape(c)[0], np.shape(c)[1]
            if(max_size[0] > 0):
                x = np.min([np.shape(c)[0], max_size[0]])
            if(max_size[1] > 0):
                y = np.min([np.shape(c)[0], max_size[0]])
            return c[:x, :y]
        return c
    
    def add_vector(self, a, b, max_size=None):
        # compatible with two vectors different shape
        c = np.zeros(np.max([np.shape(a), np.shape(b)], axis=0))
        c[:len(a)] += a
        c[:len(b)] += b
        if(max_size != None and max_size > 0):
            s = np.min([np.shape(c), (max_size, 1)], axis=0)
            print("bum")
            return c[:s[0]]
        return c
    
    def add_matrix_list(self, a):
        sum_matrix = [[0]]
        for aa in a:
            sum_matrix = self.add_matrix(aa, sum_matrix)
        return sum_matrix
    
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
                    p_ = 1e-10#self.non_negative_set[0]
                elif (indice > 2*self.max_distance + 2):
                    p_ = 1e-10#self.non_negative_set[-1]
                else:
                    p_ = self.non_negative_set[indice]
                trans_distant_matrix[i][j] = p_

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
                elif (indice >= 2*self.max_distance + 2):
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
#        print("emission_matrix[:,0]", emission_matrix[:, 0])
#        print("unary_matrix", unary_matrix)
        alpha.T[0] = np.multiply(emission_matrix[:,0], unary_matrix)
#        print("alpha.T[0]", alpha.T[0])
        
        for t in np.arange(1, source_len):
            for i in range(target_len):
                sum_al = 0.0
                for j in range(target_len):
                    sum_al += alpha[j][t-1] * transition_matrix[j][i]

                alpha[i][t] = emission_matrix[i][t] * sum_al
        
        norm_alpha = np.sum(alpha, axis=0)
        norm_alpha = np.clip(norm_alpha, a_min=1e-34, a_max=np.max(norm_alpha))
        return np.divide(alpha, norm_alpha), norm_alpha
    
    def calc_backward_messages(self, transition_matrix, emission_matrix, norm_alpha):
        """Calcualte the backward messages ~ beta values.
        Return
        ------
        beta
        """
        # TODO: verify matrix length
        source_len = np.shape(emission_matrix)[1]
        target_len = np.shape(emission_matrix)[0]
        assert(len(norm_alpha) == source_len) # = t_size

        beta = np.zeros(np.shape(emission_matrix))
        beta[:,-1] = [1]*target_len

        for t in reversed(range(source_len-1)):
            for i in range(target_len):
                for j in range(target_len):
                    beta[i][t] += beta[j][t+1] * transition_matrix[i][j] * emission_matrix[j][t+1]
                    
        beta[:,:-1] = np.divide(beta[:,:-1], norm_alpha[1:])
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
            unary_matrix = [0.02]*sentence_length
            unary_matrix[0] = 1 - np.sum(unary_matrix) + 0.02
        transition_matrix = self.generate_transition_matrix(sentence_length)
#         emission_matrix = self.normalize_matrix(emission_matrix, axis=0)
        
        alpha, norm_alpha = self.calc_forward_messages(unary_matrix, 
                                           transition_matrix, emission_matrix)
        beta = self.calc_backward_messages(transition_matrix, emission_matrix, norm_alpha)

        new_unary_matrix, emission_posterior, transition_posterior \
            = self.calc_posterior_matrix(alpha, beta, 
                                         transition_matrix, emission_matrix)
        return emission_posterior, transition_posterior # gamma, epsilon
    
    def update_non_negative_transition_set(self, emission_posteriors, 
                                           transition_posteriors):
        # 1: update non-negative set: s[-1] = 
        # 1.1: calculate new transition matrix
        sum_ep = [[0]]
        sum_gamma = [0]
        for gamma, epsilon in zip(emission_posteriors, transition_posteriors):
            if (np.shape(gamma)[0] <= 1 or np.shape(gamma)[1] <= 1):
                continue
            sum_ep = self.add_matrix(self.add_matrix_list(epsilon), sum_ep)
            sum_gamma = self.add_vector(np.sum(gamma, axis=1), sum_gamma)
        
        # 1.2: update
        new_non_negative_set = np.zeros(self.max_distance + self.max_distance + 3)
        new_non_negative_set_gamma = np.zeros(self.max_distance + self.max_distance + 3)
        new_non_negative_set[0], new_non_negative_set[-1] = 1, 1
        new_non_negative_set_gamma[0], new_non_negative_set_gamma[-1] = 1, 1
        
        for i in range(len(sum_ep)):
            for j in range(len(sum_ep)):
                indice = j - i + self.max_distance + 1
                if indice < 0:
                    continue
#                     new_non_negative_set[0] += sum_ep[i][j]
#                     new_non_negative_set_gamma[0] += sum_gamma[i]
                elif (indice > 2*self.max_distance + 2):
                    continue
#                     new_non_negative_set[-1] += sum_ep[i][j]
#                     new_non_negative_set_gamma[-1] += sum_gamma[i]
                else:
                    new_non_negative_set[indice] += sum_ep[i][j]
                    new_non_negative_set_gamma[indice] += sum_gamma[i]
        
        self.old_non_negative_set = np.copy(self.non_negative_set)
        self.non_negative_set = np.array(np.divide(new_non_negative_set, new_non_negative_set_gamma))
        for i, old_n, new_n in zip(range(len(self.non_negative_set)), self.old_non_negative_set, self.non_negative_set):
            if (np.isnan(new_n)):
                self.non_negative_set[i] = np.copy(old_n)

# *****************************************************************
# ************************ Testing Zone ***************************
# *****************************************************************

def get_tokenizer(source_file, target_file):
    # Read training file
    # Read vocab en - source
    #"/vol/work2/2017-NeuralAlignments/data/en-cz/formatted/testing/testing.en-cz.en"
    #"E:/Working/Intership2017/data/en-cz/formatted/testing/testing.en-cz.en"
    # Read vocab cz - target
    #"/vol/work2/2017-NeuralAlignments/data/en-cz/formatted/testing/testing.en-cz.cz"
    #"E:/Working/Intership2017/data/en-cz/formatted/testing/testing.en-cz.cz"
    source_tokenizer = Tokenizer(lower=False, filters='\t\n')
    target_tokenizer = Tokenizer(lower=False, filters='\t\n')
    with open(source_file, encoding="utf8") as source_f, open(target_file, encoding='utf8') as target_f:
        source = []
        target = []
        for source_line, target_line in zip(source_f, target_f):
            source.append(source_line.strip())
            target.append(target_line.strip())
            if (len(target) >= 2000):
                source_tokenizer.fit_on_texts(source)
                target_tokenizer.fit_on_texts(target)
                source, target = [], []
        if (len(target) > 0):
            source_tokenizer.fit_on_texts(source)
            target_tokenizer.fit_on_texts(target)
            source, target = [], []
    source_indices = source_tokenizer.word_index
    target_indices = target_tokenizer.word_index
    
    return source_tokenizer, len(source_indices), target_tokenizer, len(target_indices)

source_tokenizer, n_source_indices, target_tokenizer, n_target_indices = get_tokenizer(
        "/vol/work2/2017-NeuralAlignments/data/en-cz/formatted/testing/testing.en-cz.en",
        "/vol/work2/2017-NeuralAlignments/data/en-cz/formatted/testing/testing.en-cz.cz"
        )

print(n_source_indices)
print(n_target_indices)

# BW model variables
max_distance = 10
baum_welch_model = BaumWelchModel(max_distance, seed=1111)
print("non_negative_set", baum_welch_model.non_negative_set)

# Emission model variables
vocab_input_size = n_target_indices
d_embedding = 128
layer_size = [d_embedding, 512]
vocab_output_size = n_source_indices
emission_model = EmissionModel(vocab_input_size=vocab_input_size, layer_size=layer_size, 
                               vocab_output_size=vocab_output_size, baum_welch_model=baum_welch_model,
                               target_tokenizer=target_tokenizer, source_tokenizer=source_tokenizer,
                               out_prefix="/vol/work2/2017-NeuralAlignments/exp-bach/en-cz/HMM/test/epoch100-alltestset-3/2308_")
emission_model.epoch = 100
trans_posteriors = emission_model.train_model_epoch(target_inputs="/vol/work2/2017-NeuralAlignments/exp-bach/en-cz/GIZA++2/corp.merg.en-cz.cln.cz", 
                                                    source_inputs="/vol/work2/2017-NeuralAlignments/exp-bach/en-cz/GIZA++2/corp.merg.en-cz.cln.en", 
                                                    aer_target_inputs="/vol/work2/2017-NeuralAlignments/data/en-cz/formatted/testing/testing.en-cz.cz", 
                                                    aer_source_inputs="/vol/work2/2017-NeuralAlignments/data/en-cz/formatted/testing/testing.en-cz.en",
                                                    input_indice_shift=-1,
                                                    align_indice_sift=1,
                                                    target_AER="/vol/work2/2017-NeuralAlignments/data/en-cz/formatted/testing/testing.en-cz.aligment")
# print("trans_posteriors", np.shape(trans_posteriors), trans_posteriors)
