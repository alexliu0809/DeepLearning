
# coding: utf-8

# In[ ]:

import utils
import numpy as np
import edf
from time import time
import pickle
import os

train_data, trcnt = utils.load_data_onechar('data/ptb.train.txt', False)
valid_data, vacnt = utils.load_data_onechar('data/ptb.valid.txt', False)
test_data, tecnt = utils.load_data_onechar('data/ptb.test.txt', False)

#train_data, trcnt = utils.load_data_onechar('data/ptb.train.short.txt', False)
#valid_data, vacnt = utils.load_data_onechar('data/ptb.valid.short.txt', False)
#test_data, tecnt = utils.load_data_onechar('data/ptb.test.short.txt', False)


# In[ ]:

##Log Func
def Log(msg):
    f = open("Log.txt","a")
    f.write(msg + "\n")
    f.close()


# In[ ]:

############################################## Normal SGD ##############################################
def NormalSGD(eta, epoch=10):
    #print("SGD With Learning Rate %.6f:\n" % eta)
    hidden_dim = 200
    n_vocab = utils.n_vocab
    batch = 50
    parameters = []
    model = 'Models/SGD/model_SGD_%.6f_.pkl' % eta
    #print(model)
    eta = eta
    decay = 0.9

    inp = edf.Value()

    edf.params = []
    C2V = edf.Param(edf.xavier((n_vocab, hidden_dim)))

    # forget gate
    Wf = edf.Param(edf.xavier((2*hidden_dim, hidden_dim)))
    bf = edf.Param(np.zeros((hidden_dim)))
    # input gate
    Wi = edf.Param(edf.xavier((2*hidden_dim, hidden_dim)))
    bi = edf.Param(np.zeros((hidden_dim)))
    # carry cell
    Wc = edf.Param(edf.xavier((2*hidden_dim, hidden_dim)))
    bc = edf.Param(np.zeros((hidden_dim)))
    # output cell
    Wo = edf.Param(edf.xavier((2*hidden_dim, hidden_dim)))
    bo = edf.Param(np.zeros((hidden_dim)))

    V = edf.Param(edf.xavier((hidden_dim, n_vocab)))

    parameters.extend([C2V, Wf, bf, Wi, bi, Wc, bc, Wo, bo, V])


    # load the trained model if exist
    if os.path.exists(model):
        with open(model, 'rb') as f:
            p_value = pickle.load(f)
            idx = 0
            for p in p_value:
                parameters[idx].value = p
                idx += 1


    def LSTMCell(xt, h, c):

        f = edf.Sigmoid(edf.Add(edf.VDot(edf.ConCat(xt, h), Wf), bf))
        i = edf.Sigmoid(edf.Add(edf.VDot(edf.ConCat(xt, h), Wi), bi))
        o = edf.Sigmoid(edf.Add(edf.VDot(edf.ConCat(xt, h), Wo), bo))
        c_hat = edf.Tanh(edf.Add(edf.VDot(edf.ConCat(xt, h), Wc), bc))
        c_next = edf.Add(edf.Mul(f, c), edf.Mul(i, c_hat))
        h_next = edf.Mul(o, edf.Tanh(c_next))

        return h_next, c_next


    def BuildModel():

        edf.components = []

        B = inp.value.shape[0]
        T = inp.value.shape[1]
        h = edf.Value(np.zeros((B, hidden_dim))) 
        c = edf.Value(np.zeros((B, hidden_dim)))

        score = []

        for t in range(T-1):

            wordvec = edf.Embed(edf.Value(inp.value[:,t]), C2V) 
            xt = edf.Reshape(wordvec, [-1, hidden_dim])
            h_next, c_next = LSTMCell(xt, h, c)
            p = edf.SoftMax(edf.VDot(h_next, V))
            logloss = edf.Reshape(edf.LogLoss(edf.Aref(p, edf.Value(inp.value[:,t+1]))), (B, 1))

            if t == 0:
                loss = logloss
            else:
                loss = edf.ConCat(loss, logloss)

            score.append(p)    
            h = h_next
            c = c_next

        masks = np.zeros((B, T-1), dtype = np.int32)
        masks[inp.value[:,1:] != 0] = 1
        loss = edf.MeanwithMask(loss, edf.Value(masks)) 

        return loss, score


    def CalPerp(score):

        prob = [p.value for p in score]
        prob = np.transpose(np.stack(prob, axis = 0),(1,0,2))

        B = prob.shape[0]
        T = prob.shape[1]
        V = prob.shape[2]

        masks = np.zeros((B, T), dtype=np.int32)
        masks[inp.value[:,1:] != 0] = 1

        prob = prob.reshape(-1)
        idx = np.int32(inp.value[:,1:].reshape(-1))
        outer_dim = len(idx)
        inner_dim = len(prob)/outer_dim
        pick = np.int32(np.array(range(outer_dim))*inner_dim + idx)
        prob = prob[pick].reshape(B, T)

        return -np.sum(np.log(prob[np.nonzero(prob*masks)]))

    def Predict(max_step, prefix):

        edf.components = []

        T = max_step       
        h = edf.Value(np.zeros((1, hidden_dim))) 
        c = edf.Value(np.zeros((1, hidden_dim))) 

        prediction = []

        for t in range(T):

            if t < len(prefix):
                pred = edf.Value(prefix[t])
                prediction.append(pred)              
            else:
                prediction.append(pred)

            wordvec = edf.Embed(pred, C2V)
            xt = edf.Reshape(wordvec, [-1, hidden_dim])
            h_next,c_next = LSTMCell(xt, h, c)
            p = edf.SoftMax(edf.VDot(h_next, V))
            pred = edf.ArgMax(p)
            h = h_next
            c = c_next   

        edf.Forward()

        idx = [pred.value for pred in prediction]
        stop_idx = utils.to_index('}')

        if stop_idx in idx:
            return idx[0:idx.index(stop_idx)+1]
        else:
            return idx

    def Eval(data, cnt):

        perp = 0.
        avg_loss = 0.
        test_batches = range(0, len(data), batch)
        test_minbatches = [data[idx:idx+batch] for idx in test_batches]

        for minbatch in test_minbatches:

            x_padded = utils.make_mask(minbatch)
            inp.set(x_padded)
            loss, score = BuildModel()
            edf.Forward()
            avg_loss += loss.value
            perp += CalPerp(score)

        perp = np.exp(perp/cnt)
        avg_loss /= len(test_batches)
        return perp, avg_loss


    ############################################### training loop #####################################################

    batches = range(0, len(train_data), batch)
    minbatches = [train_data[idx:idx+batch] for idx in batches]

    epoch = epoch

    # initial Perplexity and loss
    perp, loss = Eval(valid_data, vacnt)
    Log("Initial: Perplexity: %0.5f Avg loss = %0.5f" % (perp, loss))    
    best_loss = loss
    #prefix = 'the agreements bring'  
    #generation = Predict(400, utils.to_idxs(prefix))
    #print("Initial generated sentence ")
    #print (utils.to_string(generation))

    Log("SGD With Learning Rate %.6f:\n" % eta)
    for ep in range(epoch):

        perm = np.random.permutation(len(minbatches)).tolist() 
        stime=time()

        for k in range(len(minbatches)):

            minbatch = minbatches[perm[k]]
            x_padded = utils.make_mask(minbatch)
            inp.set(x_padded)
            loss, score = BuildModel()
            edf.Forward()
            edf.Backward(loss)
            edf.GradClip(10)
            edf.SGD(eta)

        duration = (time() - stime)/60.
        
        perp, loss = Eval(valid_data, vacnt)
        Log("Epoch %d: Perplexity: %0.5f Avg loss = %0.5f [%.3f mins]" % (ep, perp, loss, duration))
        
        if (ep == epoch-1):
        # generate some text given the prefix and trained model
            prefix = 'the agreements bring'  
            generation = Predict(400, utils.to_idxs(prefix))
            Log("Epoch %d: generated sentence " % ep)
            Log(utils.to_string(generation)) 

            
        #Save the hyperparameters
        f_hyper = open("HyperParameters.txt","a")
        f_hyper.write("SGD LearningRate: %.6f Epoch: %d BestLoss: %0.5f Perplexity: %0.5f\n" % (eta, ep, loss, perp))
        if (ep == epoch -1):
            f_hyper.write("\n\n")
        f_hyper.close()   
            
        if loss < best_loss:
        # save the model
            best_loss = loss
            f = open(model, 'wb')
            p_value = []
            for p in parameters:
                p_value.append(p.value)
            pickle.dump(p_value, f)

        else:
            eta *= decay
            if np.isnan(loss):
                continue
            with open(model, 'rb') as f:
                p_value = pickle.load(f)
                idx = 0
                for p in p_value:
                    parameters[idx].value = p
                    idx += 1
        
        Log("\n")



# In[ ]:

############################################## RMSProp ##############################################
def MyRMSProp(eta, g, epoch=10):
    Log("RMSProp With Learning Rate: %.6f Decay Rate:%.4f \n" % (eta, g))
    hidden_dim = 200
    n_vocab = utils.n_vocab
    batch = 50
    parameters = []
    model = 'Models/RMSProp/model_RMSProp_%.6f_%.4f_.pkl' % (eta,g)
    #print(model)
    eta = eta
    decay = 0.9

    inp = edf.Value()


    edf.params = []
    C2V = edf.Param(edf.xavier((n_vocab, hidden_dim)))

    # forget gate
    Wf = edf.Param(edf.xavier((2*hidden_dim, hidden_dim)))
    bf = edf.Param(np.zeros((hidden_dim)))
    # input gate
    Wi = edf.Param(edf.xavier((2*hidden_dim, hidden_dim)))
    bi = edf.Param(np.zeros((hidden_dim)))
    # carry cell
    Wc = edf.Param(edf.xavier((2*hidden_dim, hidden_dim)))
    bc = edf.Param(np.zeros((hidden_dim)))
    # output cell
    Wo = edf.Param(edf.xavier((2*hidden_dim, hidden_dim)))
    bo = edf.Param(np.zeros((hidden_dim)))

    V = edf.Param(edf.xavier((hidden_dim, n_vocab)))

    parameters.extend([C2V, Wf, bf, Wi, bi, Wc, bc, Wo, bo, V])


    # load the trained model if exist
    if os.path.exists(model):
        with open(model, 'rb') as f:
            p_value = pickle.load(f)
            idx = 0
            for p in p_value:
                parameters[idx].value = p
                idx += 1


    def LSTMCell(xt, h, c):

        f = edf.Sigmoid(edf.Add(edf.VDot(edf.ConCat(xt, h), Wf), bf))
        i = edf.Sigmoid(edf.Add(edf.VDot(edf.ConCat(xt, h), Wi), bi))
        o = edf.Sigmoid(edf.Add(edf.VDot(edf.ConCat(xt, h), Wo), bo))
        c_hat = edf.Tanh(edf.Add(edf.VDot(edf.ConCat(xt, h), Wc), bc))
        c_next = edf.Add(edf.Mul(f, c), edf.Mul(i, c_hat))
        h_next = edf.Mul(o, edf.Tanh(c_next))

        return h_next, c_next


    def BuildModel():

        edf.components = []

        B = inp.value.shape[0]
        T = inp.value.shape[1]
        h = edf.Value(np.zeros((B, hidden_dim))) 
        c = edf.Value(np.zeros((B, hidden_dim)))

        score = []

        for t in range(T-1):

            wordvec = edf.Embed(edf.Value(inp.value[:,t]), C2V) 
            xt = edf.Reshape(wordvec, [-1, hidden_dim])
            h_next, c_next = LSTMCell(xt, h, c)
            p = edf.SoftMax(edf.VDot(h_next, V))
            logloss = edf.Reshape(edf.LogLoss(edf.Aref(p, edf.Value(inp.value[:,t+1]))), (B, 1))

            if t == 0:
                loss = logloss
            else:
                loss = edf.ConCat(loss, logloss)

            score.append(p)    
            h = h_next
            c = c_next

        masks = np.zeros((B, T-1), dtype = np.int32)
        masks[inp.value[:,1:] != 0] = 1
        loss = edf.MeanwithMask(loss, edf.Value(masks)) 

        return loss, score


    def CalPerp(score):

        prob = [p.value for p in score]
        prob = np.transpose(np.stack(prob, axis = 0),(1,0,2))

        B = prob.shape[0]
        T = prob.shape[1]
        V = prob.shape[2]

        masks = np.zeros((B, T), dtype=np.int32)
        masks[inp.value[:,1:] != 0] = 1

        prob = prob.reshape(-1)
        idx = np.int32(inp.value[:,1:].reshape(-1))
        outer_dim = len(idx)
        inner_dim = len(prob)/outer_dim
        pick = np.int32(np.array(range(outer_dim))*inner_dim + idx)
        prob = prob[pick].reshape(B, T)

        return -np.sum(np.log(prob[np.nonzero(prob*masks)]))

    def Predict(max_step, prefix):

        edf.components = []

        T = max_step       
        h = edf.Value(np.zeros((1, hidden_dim))) 
        c = edf.Value(np.zeros((1, hidden_dim))) 

        prediction = []

        for t in range(T):

            if t < len(prefix):
                pred = edf.Value(prefix[t])
                prediction.append(pred)              
            else:
                prediction.append(pred)

            wordvec = edf.Embed(pred, C2V)
            xt = edf.Reshape(wordvec, [-1, hidden_dim])
            h_next,c_next = LSTMCell(xt, h, c)
            p = edf.SoftMax(edf.VDot(h_next, V))
            pred = edf.ArgMax(p)
            h = h_next
            c = c_next   

        edf.Forward()

        idx = [pred.value for pred in prediction]
        stop_idx = utils.to_index('}')

        if stop_idx in idx:
            return idx[0:idx.index(stop_idx)+1]
        else:
            return idx

    def Eval(data, cnt):

        perp = 0.
        avg_loss = 0.
        test_batches = range(0, len(data), batch)
        test_minbatches = [data[idx:idx+batch] for idx in test_batches]

        for minbatch in test_minbatches:

            x_padded = utils.make_mask(minbatch)
            inp.set(x_padded)
            loss, score = BuildModel()
            edf.Forward()
            avg_loss += loss.value
            perp += CalPerp(score)

        perp = np.exp(perp/cnt)
        avg_loss /= len(test_batches)
        return perp, avg_loss


    ############################################### training loop #####################################################

    batches = range(0, len(train_data), batch)
    minbatches = [train_data[idx:idx+batch] for idx in batches]

    epoch = epoch

    # initial Perplexity and loss
    perp, loss = Eval(valid_data, vacnt)
    Log("Initial: Perplexity: %0.5f Avg loss = %0.5f" % (perp, loss))    
    best_loss = loss
    prefix = 'the agreements bring'  
    generation = Predict(400, utils.to_idxs(prefix))
    Log("Initial generated sentence ")
    Log (utils.to_string(generation))


    for ep in range(epoch):

        perm = np.random.permutation(len(minbatches)).tolist() 
        stime=time()

        for k in range(len(minbatches)):

            minbatch = minbatches[perm[k]]
            x_padded = utils.make_mask(minbatch)
            inp.set(x_padded)
            loss, score = BuildModel()
            edf.Forward()
            edf.Backward(loss)
            edf.GradClip(10)
            edf.RMSProp(eta,g)

        duration = (time() - stime)/60.
        
        perp, loss = Eval(valid_data, vacnt)
        Log("Epoch %d: Perplexity: %0.5f Avg loss = %0.5f [%.3f mins]" % (ep, perp, loss, duration))
        
        if (ep == epoch-1):
        # generate some text given the prefix and trained model
            prefix = 'the agreements bring'  
            generation = Predict(400, utils.to_idxs(prefix))
            Log("Epoch %d: generated sentence " % ep)
            Log(utils.to_string(generation)) 

            
        #Save the hyperparameters
        f_hyper = open("HyperParameters.txt","a")
        f_hyper.write("RMSProp LearningRate: %.6f Decay_Rate: %.4f Epoch: %d BestLoss: %0.5f Perplexity: %0.5f\n" % (eta, g, ep, loss, perp))
        if (ep == epoch -1):
            f_hyper.write("\n\n")
        f_hyper.close()
        
        if loss < best_loss:
        # save the model
            best_loss = loss
            f = open(model, 'wb')
            p_value = []
            for p in parameters:
                p_value.append(p.value)
            pickle.dump(p_value, f)

        else:
            eta *= decay
            if np.isnan(loss):
                continue
            with open(model, 'rb') as f:
                p_value = pickle.load(f)
                idx = 0
                for p in p_value:
                    parameters[idx].value = p
                    idx += 1
        
        Log("\n")



# In[ ]:

############################################## Adam ##############################################
def MyAdam(eta, beta1, beta2, epoch=10):
    Log("Adam With Learning Rate: %.6f Beta1 %.4f Beta2 %.4f\n" % (eta,beta1,beta2))
    hidden_dim = 200
    n_vocab = utils.n_vocab
    batch = 50
    parameters = []
    model = 'Models/Adam/model_Adam_%.6f_%.4f_%.4f_.pkl' % (eta,beta1,beta2)
    #print(model)
    eta = eta
    decay = 0.9

    inp = edf.Value()

    edf.params = []
    C2V = edf.Param(edf.xavier((n_vocab, hidden_dim)))

    # forget gate
    Wf = edf.Param(edf.xavier((2*hidden_dim, hidden_dim)))
    bf = edf.Param(np.zeros((hidden_dim)))
    # input gate
    Wi = edf.Param(edf.xavier((2*hidden_dim, hidden_dim)))
    bi = edf.Param(np.zeros((hidden_dim)))
    # carry cell
    Wc = edf.Param(edf.xavier((2*hidden_dim, hidden_dim)))
    bc = edf.Param(np.zeros((hidden_dim)))
    # output cell
    Wo = edf.Param(edf.xavier((2*hidden_dim, hidden_dim)))
    bo = edf.Param(np.zeros((hidden_dim)))

    V = edf.Param(edf.xavier((hidden_dim, n_vocab)))

    parameters.extend([C2V, Wf, bf, Wi, bi, Wc, bc, Wo, bo, V])


    # load the trained model if exist
    if os.path.exists(model):
        with open(model, 'rb') as f:
            p_value = pickle.load(f)
            idx = 0
            for p in p_value:
                parameters[idx].value = p
                idx += 1


    def LSTMCell(xt, h, c):

        f = edf.Sigmoid(edf.Add(edf.VDot(edf.ConCat(xt, h), Wf), bf))
        i = edf.Sigmoid(edf.Add(edf.VDot(edf.ConCat(xt, h), Wi), bi))
        o = edf.Sigmoid(edf.Add(edf.VDot(edf.ConCat(xt, h), Wo), bo))
        c_hat = edf.Tanh(edf.Add(edf.VDot(edf.ConCat(xt, h), Wc), bc))
        c_next = edf.Add(edf.Mul(f, c), edf.Mul(i, c_hat))
        h_next = edf.Mul(o, edf.Tanh(c_next))

        return h_next, c_next


    def BuildModel():

        edf.components = []

        B = inp.value.shape[0]
        T = inp.value.shape[1]
        h = edf.Value(np.zeros((B, hidden_dim))) 
        c = edf.Value(np.zeros((B, hidden_dim)))

        score = []

        for t in range(T-1):

            wordvec = edf.Embed(edf.Value(inp.value[:,t]), C2V) 
            xt = edf.Reshape(wordvec, [-1, hidden_dim])
            h_next, c_next = LSTMCell(xt, h, c)
            p = edf.SoftMax(edf.VDot(h_next, V))
            logloss = edf.Reshape(edf.LogLoss(edf.Aref(p, edf.Value(inp.value[:,t+1]))), (B, 1))

            if t == 0:
                loss = logloss
            else:
                loss = edf.ConCat(loss, logloss)

            score.append(p)    
            h = h_next
            c = c_next

        masks = np.zeros((B, T-1), dtype = np.int32)
        masks[inp.value[:,1:] != 0] = 1
        loss = edf.MeanwithMask(loss, edf.Value(masks)) 

        return loss, score


    def CalPerp(score):

        prob = [p.value for p in score]
        prob = np.transpose(np.stack(prob, axis = 0),(1,0,2))

        B = prob.shape[0]
        T = prob.shape[1]
        V = prob.shape[2]

        masks = np.zeros((B, T), dtype=np.int32)
        masks[inp.value[:,1:] != 0] = 1

        prob = prob.reshape(-1)
        idx = np.int32(inp.value[:,1:].reshape(-1))
        outer_dim = len(idx)
        inner_dim = len(prob)/outer_dim
        pick = np.int32(np.array(range(outer_dim))*inner_dim + idx)
        prob = prob[pick].reshape(B, T)

        return -np.sum(np.log(prob[np.nonzero(prob*masks)]))

    def Predict(max_step, prefix):

        edf.components = []

        T = max_step       
        h = edf.Value(np.zeros((1, hidden_dim))) 
        c = edf.Value(np.zeros((1, hidden_dim))) 

        prediction = []

        for t in range(T):

            if t < len(prefix):
                pred = edf.Value(prefix[t])
                prediction.append(pred)              
            else:
                prediction.append(pred)

            wordvec = edf.Embed(pred, C2V)
            xt = edf.Reshape(wordvec, [-1, hidden_dim])
            h_next,c_next = LSTMCell(xt, h, c)
            p = edf.SoftMax(edf.VDot(h_next, V))
            pred = edf.ArgMax(p)
            h = h_next
            c = c_next   

        edf.Forward()

        idx = [pred.value for pred in prediction]
        stop_idx = utils.to_index('}')

        if stop_idx in idx:
            return idx[0:idx.index(stop_idx)+1]
        else:
            return idx

    def Eval(data, cnt):

        perp = 0.
        avg_loss = 0.
        test_batches = range(0, len(data), batch)
        test_minbatches = [data[idx:idx+batch] for idx in test_batches]

        for minbatch in test_minbatches:

            x_padded = utils.make_mask(minbatch)
            inp.set(x_padded)
            loss, score = BuildModel()
            edf.Forward()
            avg_loss += loss.value
            perp += CalPerp(score)

        perp = np.exp(perp/cnt)
        avg_loss /= len(test_batches)
        return perp, avg_loss


    ############################################### training loop #####################################################

    batches = range(0, len(train_data), batch)
    minbatches = [train_data[idx:idx+batch] for idx in batches]

    epoch = epoch

    # initial Perplexity and loss
    perp, loss = Eval(valid_data, vacnt)
    Log("Initial: Perplexity: %0.5f Avg loss = %0.5f" % (perp, loss))    
    best_loss = loss
    prefix = 'the agreements bring'  
    generation = Predict(400, utils.to_idxs(prefix))
    Log("Initial generated sentence ")
    Log(utils.to_string(generation))


    for ep in range(epoch):

        perm = np.random.permutation(len(minbatches)).tolist() 
        stime=time()

        for k in range(len(minbatches)):

            minbatch = minbatches[perm[k]]
            x_padded = utils.make_mask(minbatch)
            inp.set(x_padded)
            loss, score = BuildModel()
            edf.Forward()
            edf.Backward(loss)
            edf.GradClip(10)
            edf.Adam(eta,beta1,beta2)

        duration = (time() - stime)/60.
        
        perp, loss = Eval(valid_data, vacnt)
        Log("Epoch %d: Perplexity: %0.5f Avg loss = %0.5f [%.3f mins]" % (ep, perp, loss, duration))
        
        if (ep == epoch-1):
        # generate some text given the prefix and trained model
            prefix = 'the agreements bring'  
            generation = Predict(400, utils.to_idxs(prefix))
            Log("Epoch %d: generated sentence " % ep)
            Log(utils.to_string(generation)) 

            
        #Save the hyperparameters
        f_hyper = open("HyperParameters.txt","a")
        f_hyper.write("Adam LearningRate: %.6f Beta1: %.4f Beta2: %.4f Epoch: %d BestLoss: %0.5f Perplexity: %0.5f\n" % (eta, beta1, beta2, ep, loss, perp))
        if (ep == epoch -1):
            f_hyper.write("\n\n")
        f_hyper.close()
        
        
        if loss < best_loss:
        # save the model
            best_loss = loss
            f = open(model, 'wb')
            p_value = []
            for p in parameters:
                p_value.append(p.value)
            pickle.dump(p_value, f)

        else:
            eta *= decay
            if np.isnan(loss):
                continue
            with open(model, 'rb') as f:
                p_value = pickle.load(f)
                idx = 0
                for p in p_value:
                    parameters[idx].value = p
                    idx += 1
           
        Log("\n")


# In[ ]:

##################### Stage2 #####################


SGD = np.arange(3*1,dtype=np.float).reshape(3,1)
RMSProp = np.arange(3*2,dtype=np.float).reshape(3,2)
Adam = np.arange(3*3,dtype=np.float).reshape(3,3)


with open("Stage1_Best.txt", 'r') as infile:
    lines = [line.strip() for line in infile][:]
    SGDLine = lines[0:4]
    RMSPropLine = lines[4:8]
    AdamLine = lines[8:12]
    
    for i in range(3):
        SGD[i][0] = SGDLine[i+1]
        
        Paras = RMSPropLine[i+1].split()
        RMSProp[i,:] = np.array(Paras[:])
        
        Paras = AdamLine[i+1].split()
        Adam[i,:] = np.array(Paras[:])
    
    print(SGD)
    print(RMSProp)
    print(Adam)

#Round1 np.random.uniform(0.5,2)

#Round1 np.random.uniform(0.8,1.2)

#Round3
pair_numbers = 16
number_of_epoch = 2
np.random.seed(5)

for j in range(pair_numbers):
	for i in range(3): # Top 3 For Each Method
	    #SGD
	    eta = SGD[i][0] * np.random.uniform(0.99,1.01)
	    NormalSGD(eta, number_of_epoch)
	    
	    #RMSProp
	    eta = RMSProp[i][0] * np.random.uniform(0.99,1.01)
	    g_RMSProp = 1 - 10 ** np.random.uniform(-0.8,-2) #0.84 - 0.99
	    MyRMSProp(eta, g_RMSProp, number_of_epoch)
	    
	    #Adam
	    eta = Adam[i][0] * np.random.uniform(0.99,1.01)
	    beta1_Adam = 1 - 10 ** np.random.uniform(-0.8,-3) #0.84 - 0.999
	    beta2_Adam = 1 - 10 ** np.random.uniform(-0.8,-4) #0.84 - 0.9999
	    MyAdam(eta, beta1_Adam, beta2_Adam, number_of_epoch)
	
# In[ ]:



