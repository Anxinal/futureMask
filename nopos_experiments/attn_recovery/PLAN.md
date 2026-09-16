import math

import torch.nn as nn 
import numpy as np
import masks 

# the both classes are psuedocodes for the experiment architectures, we only use one layer for both architectures. Even though it means severe underfitting when the task is being trained, it is fine as our objective is to check the recovery of self attention 
class simple_transformer(nn.modules):

    def __init__(self, inputdim = 1024):
        self.inputdim = inputdim
        self.selfattention = attention()
        self.ffn = nn.Linear(4 * inputdim, inputdim) 


    def forward(self,inputs):
        Q, K, V = nn.Linear(self.inputdim, self.inputdim)
        query = Q(inputs)
        key = K(inputs)
        raw_attention = query * np.transpose(key)
        mask = None 
        ## only one multihead
        attention = softmax(raw_attention)
        raw_output = attention * V #route this output as the new input for masked transformer later
        raw_output += inputs #skip connection
        result = self.layernorm(self.ffn(raw_output))
        return result #same as plain transformer during the training stage



class masked_simple_transformer(nn.module):
        def __init__(self, inputdim ,maskconfigs):
            self.inputdim = inputdim
            self.maskconfig = maskconfigs
    def forward(self, inputs: np.array):
        for i in self.self_attention:
            mutliheads_input = [inputs, inputs] 
            mutltihead_projections = [Q, K = nn.Linear(self.inputdim, self.inputdim) for _ in range(2)]
            # This is unconventional but we are sticking to the theoratical construction, alternatively, we can also try the standard multihead input/query_key projections where the dimension is split in half
            concencated = []
            for i in range(2):
                input = mutliheads_input[i]
                Q,K,V = mutltihead_projections[i]
                query = Q(inputs)
                key = K(inputs)
                raw_attention = query * np.transpose(key)
                raw_attention = self.maskconfig[i].applymask()
                concencated = concencated.concencate(raw_attention)
            raw_output = attention * V 
            return raw_output
            #route this output as the new input for masked transformer later
            mask = None 
             # only one multihead
        attention = softmax(raw_attention)

class trainer:
    def __init__(self,first = simple_transformer(), second= masked_simple_transformer):
         self.first = first 
         self.second = second
    def train_first_model(self):
         # same as training a usual language model with next token prediction task
        learning_rate = 1e_4
        data_type = "fp16"
        epochs = 20
        dataset = "wikitext103"
        lr_scheduler = "inv-sqrt"
        checkpoints = None # but if it is necessary to save the model for later validation, only save the best 


         # after the training finishes, lock the parameters and keep the output vectors
    def train_second_model(self, inputs):
         # details about batch omitted but batch training should be applied instead. This is just to show the update logic
         output = self.second.forward(inputs)
         expected = self.first.forward(inputs)
         loss = nn.L1Loss(output, expected)
         loss.backward() # only on second because the parameters of the first should be fixed after training
    def validation(self,validation_inputs):
        validation_count = 10000 # as a placeholder, the actual number should depend on the training validation ratio
        MAE_result = 0
        def L2norm(x, y):
             return math.sqrt(sum([x[i]^2 + y[i]^2 for i in range(len(x))]))
        for i in range(validation_count):
            output = self.second.forward(validation_inputs)
            expected = self.first.forward(validation_inputs)
            MAE_result += abs(L2norm(output, expected))
        return MAE_result / math.sqrt()
        

#model_1: simple_transformer()
#model_2: masked_simple_transformer maskconfig = [masks.CausalMask, masks.FutureOnlyMask]
#model_3: mased_simple_transformer maskconfig = [masks.CausalMask, masks.CausalOnlyMask]
#task order train_first -> train_second(model_2) -> validation -> train_second(model_3) -> validation -> summarise mae and accuracy and compare between the two models 
