import numpy as np, os
import lib_shaw as mod
from random import randint
from torch.autograd import Variable
import torch
from torch.utils import data as util
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

class Stacked_MMU(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size, n_vocab):
        super(Stacked_MMU, self).__init__()

        #Define model
        #self.poly = mod.GD_polynet(input_size, hidden_size, hidden_size, hidden_size, None)
        self.embedding = Parameter(torch.rand(input_size, n_vocab), requires_grad=1)
        self.mmu1 = mod.GD_MMU(n_vocab, hidden_size, memory_size, hidden_size)
        self.mmu2 = mod.GD_MMU(hidden_size, hidden_size, memory_size, hidden_size)
        self.mmu3 = mod.GD_MMU(hidden_size, hidden_size, memory_size, hidden_size)

        self.w_out1 = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)
        self.w_out2 = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)
        self.w_out3 = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)

        for param in self.parameters():
            # torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)



    def forward(self, input):
        input = self.embedding.mm(input)
        mmu1_out = self.mmu1.forward(input)
        mmu2_out = self.mmu2.forward(mmu1_out)
        mmu3_out = self.mmu3.forward(mmu2_out)

        out = self.w_out3.mm(mmu3_out)# + self.w_out2.mm(mmu2_out) + self.w_out1.mm(mmu1_out)
        out  = F.log_softmax(torch.t(out))
        return torch.t(out)





    def reset(self, batch_size):
        #self.poly.reset(batch_size)
        self.mmu1.reset(batch_size)
        self.mmu2.reset(batch_size)
        self.mmu3.reset(batch_size)

class Single_MMU(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size):
        super(Single_MMU, self).__init__()

        #Define model
        self.mmu = mod.GD_MMU(input_size, hidden_size, memory_size, hidden_size)
        self.w_out = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)


    def forward(self, input):
        #input = self.poly.forward(input)
        mmu_out = self.mmu.forward(input)
        out = self.w_out.mm(mmu_out)
        out  = F.log_softmax(out)
        return out


    def reset(self, batch_size):
        self.mmu.reset(batch_size)

class Stacked_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size):
        super(Stacked_LSTM, self).__init__()

        #Define model
        #self.poly = mod.GD_polynet(input_size, hidden_size, hidden_size, hidden_size, None)
        self.lstm1 = mod.GD_LSTM(input_size, hidden_size, memory_size, hidden_size)
        self.lstm2 = mod.GD_LSTM(hidden_size, hidden_size, memory_size, hidden_size)
        self.lstm3 = mod.GD_LSTM(hidden_size, hidden_size, memory_size, hidden_size)

        self.w_out1 = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)
        self.w_out2 = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)
        self.w_out3 = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)


    def forward(self, input):
        #input = self.poly.forward(input)
        lstm1_out = self.lstm1.forward(input)
        lstm2_out = self.lstm2.forward(lstm1_out)
        lstm3_out = self.lstm3.forward(lstm2_out)

        out = self.w_out3.mm(lstm3_out)# + self.w_out2.mm(lstm2_out) + self.w_out1.mm(lstm1_out)
        out  = F.log_softmax(torch.t(out))
        return torch.t(out)




    def reset(self, batch_size):
        #self.poly.reset(batch_size)
        self.lstm1.reset(batch_size)
        self.lstm2.reset(batch_size)
        self.lstm3.reset(batch_size)

class Single_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size):
        super(Single_LSTM, self).__init__()

        #Define model
        self.lstm = mod.GD_LSTM(input_size, hidden_size, memory_size, hidden_size)
        self.w_out = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)


    def forward(self, input):
        lstm_out = self.lstm.forward(input)
        out = self.w_out.mm(lstm_out)
        out  = F.log_softmax(out)
        return out


    def reset(self, batch_size):
        self.lstm.reset(batch_size)


book_name = "sonnet.txt"
filename = 'sonnet.pt'



class Parameters:
    def __init__(self):
        self.pop_size = 50
        self.load_checkpoint = False
        self.total_gens = 50000

        #NN specifics
        self.num_hnodes = 500
        self.num_mem = 500
        self.output_activation = 'softmax'

        # Train data
        self.batch_size = 1000
        self.num_epoch = 500
        self.seq_len = 50
        self.prediction_len = 1
        self.sample_size = 1000000

        #Dependents
        self.num_output = None
        self.num_input = None
        self.save_foldername = 'R_Book/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)

class Task_Book:
    def __init__(self, parameters):
        self.params = parameters
        self.raw_text, self.char_to_int, self.n_vocab, self.int_to_char = self.read_book()

        #model = Stacked_MMU(parameters.num_input, parameters.num_hnodes, parameters.num_mem, parameters.num_output, parameters.output_activation)
        #model = mod.GD_MMU(parameters.num_input, parameters.num_hnodes, parameters.num_mem, parameters.num_output, parameters.output_activation)
        #model = Stacked_LSTM(parameters.num_input, parameters.num_hnodes, parameters.num_mem, parameters.num_output)
        model = Stacked_MMU(parameters.num_input, parameters.num_hnodes, parameters.num_mem, parameters.num_output, self.n_vocab)
        #model = Single_MMU(parameters.num_input, parameters.num_hnodes, parameters.num_mem, parameters.num_output)
        #model = Single_LSTM(parameters.num_input, parameters.num_hnodes, parameters.num_mem, parameters.num_output)


        self.train_x, self.train_y = self.get_data(seq_len=self.params.seq_len)
        model = torch.load(filename)

        #Run backprop
        self.free_write(model)
        #self.pop[0].from_gdnet()


    def read_book(self):

        # load ascii text and covert to lowercase
        raw_text = open(book_name).read()
        raw_text = raw_text.lower()[0:100000]

        # create mapping of unique chars to integers
        chars = sorted(list(set(raw_text)))
        char_to_int = dict((c, i) for i, c in enumerate(chars))
        int_to_char = dict((i, c) for i, c in enumerate(chars))

        n_chars = len(raw_text)
        n_vocab = len(chars)
        print "Total Characters: ", n_chars
        print "Total Vocab: ", n_vocab

        self.params.num_output = n_vocab
        self.params.num_input = n_vocab

        return raw_text, char_to_int, n_vocab, int_to_char

    def get_data(self, seq_len):

        #Prepare the dataset of input to output pairs encoded as integers
        data_x = []; data_y = []
        for i in range(0, len(self.raw_text) - seq_len, 1):
            seq_in = self.raw_text[i:i + seq_len]
            seq_out = self.raw_text[i + seq_len]
            #data_x.append(mod.unsqueeze(np.array([self.char_to_int[char] for char in seq_in])))

            data_x.append(np.array([np.eye(self.n_vocab, dtype='float')[self.char_to_int[seq_out]] for char in seq_in]))
            data_y.append(np.array([self.char_to_int[seq_out]]))
            #data_y.append(mod.unsqueeze(np.eye(self.n_vocab, dtype='float')[self.char_to_int[seq_out]], axis=0))
        n_patterns = len(data_x)
        print "Total Patterns: ", n_patterns

        #Reshape (batch, seq_len, breadth)
        data_x = np.array(data_x); data_y = np.array(data_y)

        return data_x[0:self.params.sample_size,:,:], data_y[0:self.params.sample_size,:]

    def free_write(self, model):
        net_input = torch.Tensor(self.train_x[0:1,:,])
        model.cuda()
        model.reset(batch_size=1)
        recall_input = torch.Tensor(torch.ones((self.params.num_input, 1)) + 2)

        while True:
            # Run sequence of characters
            for i in range(self.params.seq_len):  # For the length of the sequence
                net_inp = Variable(torch.t(net_input[:, i, :]), requires_grad=True).cuda()
                model.forward(net_inp)

            #Free Write
            net_inp = Variable(recall_input, requires_grad=True).cuda()
            net_out = torch.t(model.forward(net_inp)).cpu().data.numpy()
            print self.int_to_char[int(np.argmax(net_out))],

            #Update
            addition = torch.Tensor(np.eye(self.n_vocab, dtype='float')[int(np.argmax(net_out))]).unsqueeze(0).unsqueeze(0)
            net_input = torch.cat((net_input[:,1:,:], addition), 1)






if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    sim_task = Task_Book(parameters)
















