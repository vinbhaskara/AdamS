import math
import torch
import numpy as np
from torch.optim.optimizer import Optimizer


class AdamS(Optimizer):
    r"""Implements AdamS algorithm. Unbiased estimate of gradient that implements
    a stochastic regularizer using the variance-gradient direction.
    Based on the paper: https://arxiv.org/abs/1905.13200.
    Includes AdamW-style weight decay when `decoupled_weight_decay=True`.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        eta: The stddev of the zero-mean Gaussian which is sampled to determine 
        the signed weight of variance-gradient in the parameter update
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        decoupled_weight_decay (boolean, optional): if True, implements the AdamW 
            style of weight decay (see: https://arxiv.org/abs/1711.05101), else uses L2 penalty.
            Suggested weight_decay value when this is True is `weight_decay=0.01`.
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    """

    def __init__(self, params, lr=1e-3, eta=0.0001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, decoupled_weight_decay=False, amsgrad=False, seed=None):
        
        etas = (1.0, eta)
        randomize_eta2 = True 
        bound_stddev = False
        
        if bound_stddev is True and randomize_eta2 is True:
            raise ValueError("Both randomize_eta2 and bound_stddev are True. Please check your arguments")
            
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= eta:
            raise ValueError("Invalid eta value: {}".format(eta))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if seed is not None:
            np.random.seed(seed)
        defaults = dict(lr=lr, etas=etas, betas=betas, eps=eps,
                        weight_decay=weight_decay, decoupled_weight_decay=decoupled_weight_decay, amsgrad=amsgrad, bound_stddev=bound_stddev, randomize_eta2=randomize_eta2)
        super(AdamS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that returns the loss.
        """
        loss = None
        if closure is not None:
            loss = torch.tensor(float(closure()))
        else:
            raise RuntimeError('AdamS needs closure.')

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamS does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                bound_stddev = group['bound_stddev']
                beta1, beta2 = group['betas']
                eta1, eta2 = group['etas']
                randomize_eta2 = group['randomize_eta2']
                
                if randomize_eta2:
                    eta2 = np.random.normal(0, 1, 1)[0] * eta2 # reparametrization trick
                    
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of (ucb weighted) gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of (ucb weighted) squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Exponential moving average of loss values
                    state['r_loss'] = torch.tensor(0.0)
                    # Exponential moving average of squared loss values
                    state['s_loss'] = torch.tensor(0.0)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, r_loss, s_loss = state['exp_avg'], state['exp_avg_sq'], state['r_loss'], state['s_loss']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if not group['decoupled_weight_decay']:
                        grad.add_(group['weight_decay'], p.data)
                    else:
                        # the LR is multiplied with weight_decay to follow the latest pytorch's
                        # implementation of AdamW (unlike what's proposed in the paper that excludes the LR)
                        # Ref: https://github.com/pytorch/pytorch/blob/e92f1243b478a05dc9e4dc3e2024981b5d0cdf6a/torch/optim/adam.py#L419
                        p.data.mul_(1.0 - (group['lr'] * group['weight_decay']))

                # Compute the UCB weight term based on current loss and existing averages so far r_loss, s_loss
                bias_correction1_previous = 1 - beta1 ** (state['step'] - 1)
                if state['step'] == 1:
                    r_loss_sofar_avg = 0.0
                    s_loss_sofar_avg = 0.0
                    stddev_loss_sofar = 0.0
                else:
                    r_loss_sofar_avg = r_loss/(bias_correction1_previous)
                    s_loss_sofar_avg = s_loss/(bias_correction1_previous)
                    stddev_loss_sofar = math.sqrt(abs(s_loss_sofar_avg - r_loss_sofar_avg ** 2))

                if bound_stddev and state['step'] != 1: # at step=1, r_loss avg will be 0.0
                    # keeps the relative variance around eta2 specified by the user
                    ucb_weight = float(eta1 * abs(r_loss_sofar_avg) * stddev_loss_sofar - (eta2 * abs(r_loss_sofar_avg) - stddev_loss_sofar) * (loss - r_loss_sofar_avg))
                else:
                    ucb_weight = float(eta1 * stddev_loss_sofar + eta2 * (loss - r_loss_sofar_avg))

                # reweight the gradient using ucb_weight
                grad.mul_(ucb_weight)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Update the loss variables
                r_loss.mul_(beta1).add_(1 - beta1, loss)
                s_loss.mul_(beta1).add_(1 - beta1, loss ** 2)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

# aliases
Adams = AdamS
adams = AdamS
