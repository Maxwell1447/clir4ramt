import torch
from torch import nn
from ..train.optim import MultiNLLLoss

class BOWModule(nn.Module):
    """BOW loss to predict terms in the other language.
    """
    def __init__(self, d_hidden, voc_size, factor=1.0, label_smoothing=0.0, bow_multiplicator=1.0):
        super(BOWModule, self).__init__()
        self.voc_size = voc_size
        self.factor = factor
        self.num_spec = 4
        self.voc_linear = nn.Linear(d_hidden, voc_size)
        self.bow_multiplicator = bow_multiplicator
        self.voc_linear.weight.data *= self.bow_multiplicator # UPSCALE weights for better separability of normalized vector
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss_fct = MultiNLLLoss(label_smoothing=label_smoothing)
        # self.loss_fct = nn.NLLLoss(reduction='mean')

    def forward(self, last_hidden_state_cls, other_sentence):
        # last_hidden_state_cls : B x d
        # other_sentence : B x L
        one_hot = torch.zeros(other_sentence.size(0), self.voc_size + self.num_spec, device=other_sentence.device, dtype=torch.bool)
        # tgt : (B x V)
        tgt = one_hot.scatter_(1, other_sentence, 1)[:, self.num_spec:].reshape(other_sentence.size(0), self.voc_size)
        ## out : (B x V) x 2
        # out : B x V
        out_ = self.voc_linear(last_hidden_state_cls) #.view(other_sentence.size(0) * self.voc_size)
        # out = torch.zeros((other_sentence.size(0) * self.voc_size, 2), dtype=out_.dtype, device=other_sentence.device)
        # out[:, 1] = out_
        out = self.log_softmax(out_)
        # return self.factor * self.loss_fct(out, tgt)
        return dict(loss=self.factor * self.loss_fct(out, tgt), logprobs=out, target=tgt)
        