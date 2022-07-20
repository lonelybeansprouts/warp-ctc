import torch

from bfctc_pytorch import *
from torch.autograd import Variable

def test_simple():
    log_probs = torch.FloatTensor([[[0.1, 0.6, 0.1, 0.1, 0.1], 
                                    [0.1, 0.1, 0.6, 0.1, 0.1]]]).transpose(0, 1).log().contiguous()
    print(log_probs)
    input_lengths = torch.LongTensor([2])
    labels = torch.LongTensor([1, 2])
    target_lengths = torch.LongTensor([2])
    neg_log_likelihood, log_alpha = bf_ctc_forward(log_probs, labels, list(input_lengths), list(target_lengths))
    print(neg_log_likelihood)
    print(log_alpha)

    grad_out=torch.FloatTensor([1.0])
    grad, log_beta = bf_ctc_backward(grad_out, log_probs, labels, list(input_lengths), list(target_lengths), neg_log_likelihood, log_alpha, True)

    print(grad)
    print(log_beta)
    assert(log_alpha[0, 1, 1] == log_beta[0, 0, 0])


test_simple()



def test_large():
    output_dim = 6
    input_lengths = torch.IntTensor([111, 222, 333, 444, 555, 666, 777, 888, 999, 10000])
    log_probs = torch.zeros((10000, 10, output_dim))
    for i, length in enumerate(list(input_lengths)):
        t_probs = torch.nn.functional.log_softmax(torch.randn((length, output_dim), dtype=torch.float32), dim=1)
        log_probs[0:length, i, :] = t_probs
    
    print(log_probs[:,0,:])
    
    target_lengths = torch.IntTensor([50, 111, 222, 333, 444, 555, 777, 888, 999, 1234])
    labels = []
    for i, length in enumerate(list(target_lengths)):
        label = torch.randint(low=0, high=output_dim, size=(length,))
        labels.append(label)
    labels = torch.cat(labels, dim=0)

    neg_log_likelihood, log_alpha = neg_log_likelihood, log_alpha = bf_ctc_forward(log_probs, labels, list(input_lengths), list(target_lengths))
        
    grad_out=torch.ones((10,), dtype=torch.float32)
    grad, log_beta = bf_ctc_backward(grad_out, log_probs, labels, list(input_lengths), list(target_lengths), neg_log_likelihood, log_alpha, True)

    print(grad[:, 0, :])

    for i in range(10):
        print(log_alpha[i, input_lengths[i]-1, target_lengths[i]-1])
        print(log_beta[i, 0, 0])
        assert(abs(log_alpha[i, input_lengths[i]-1, target_lengths[i]-1].item()-log_beta[i, 0, 0].item()) < 1.0)

test_large()


def test_loss():
    output_dim = 1024
    input_lengths = torch.IntTensor([111, 222, 333, 444, 555, 666, 777, 888, 999, 10000])
    log_probs = torch.zeros((10000, 10, output_dim))
    for i, length in enumerate(list(input_lengths)):
        t_probs = torch.nn.functional.log_softmax(torch.randn((length, output_dim), dtype=torch.float32), dim=1)
        log_probs[0:length, i, :] = t_probs
    log_probs = Variable(log_probs,requires_grad=True)

    target_lengths = torch.IntTensor([50, 111, 222, 333, 444, 555, 777, 888, 999, 1234])

    labels = []
    for i, length in enumerate(list(target_lengths)):
        label = torch.randint(low=0, high=output_dim, size=(length,))
        labels.append(label)
    labels = torch.cat(labels, dim=0)
    bf_ctc = BfCtcLoss(size_average=True)
    costs = bf_ctc(log_probs, labels,  input_lengths, target_lengths)
    costs.backward()
    print(costs)

test_loss()