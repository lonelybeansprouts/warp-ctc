import torch
from torch.autograd import Function
from torch.nn import Module

from ._bf_ctc import *  



class _BF_CTC(Function):
    @staticmethod
    def forward(ctx, acts, labels, input_lengths, target_lengths):
        acts = acts.contiguous()
        batch_size = acts.size(1)
        log_probs = torch.nn.functional.log_softmax(acts, dim=-1)
        neg_log_likelihood, log_alpha = bf_ctc_forward(log_probs, labels, list(input_lengths), list(target_lengths))
        grad_out=torch.ones((batch_size,), dtype=torch.float32)
        grads, log_beta = bf_ctc_backward(grad_out, log_probs, labels, list(input_lengths), list(target_lengths), neg_log_likelihood, log_alpha, True)
        ctx.grads = grads
        neg_log_likelihood = neg_log_likelihood.masked_fill(torch.isinf(neg_log_likelihood), 0.0)
        return neg_log_likelihood

    @staticmethod
    def backward(ctx, grad_output):
        _grad_output = grad_output.to(ctx.grads.device).view(1, grad_output.numel(), 1)
        return ctx.grads.mul_(_grad_output), None, None, None, None, None, None




class BfCtcLoss(Module):
    """
    Parameters:
        size_average (bool): normalize the loss by the batch size
            (default: `False`)
        length_average (bool): normalize the loss by the total number of frames
            in the batch. If `True`, supersedes `size_average`
            (default: `False`)
    """
    def __init__(self, size_average=False, length_average=False):
        super(BfCtcLoss, self).__init__()
        self.bf_ctc = _BF_CTC.apply
        self.size_average = size_average
        self.length_average = length_average

    def forward(self, acts, labels, input_lengths, target_lengths):
        """
        acts: Tensor of (seqLength x batch x outputDim) containing output from network
        labels: 1 dimensional Tensor containing all the targets of the batch in one sequence
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
        """
        assert len(labels.size()) == 1  # labels must be 1 dimensional
        batch_size = acts.size(1)
        neg_log_likelihood = self.bf_ctc(acts, labels, input_lengths, target_lengths)
        costs = neg_log_likelihood.sum()
        if self.size_average:
            costs = costs / batch_size
        elif self.length_average:
            total_length = torch.sum(target_lengths).item()
            costs = costs / total_length
        return costs





