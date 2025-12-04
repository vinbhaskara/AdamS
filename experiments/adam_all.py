import math
import torch
import numpy as np
from torch.optim.optimizer import Optimizer


class AdamALL(Optimizer):
    r"""Implements AdamUCB/AdamCB/AdamS algorithms.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        etas  (Tuple[float, float], optional): coefficients used for weighting 
            expected loss versus standard deviation of loss across mini-batches.
            eta[0] is the weight on the expected loss term, and eta[1] is the weight
            on the standard deviation of the loss term.
        randomize_eta2: This samples eta2 from a random Normal distribution centered around 0
            and standard deviation = etas[1]. True for AdamS.
        bound_stddev: If false, then etas are used as weights of the expectation and
            the standard deviation terms. If true, then eta[1] is used as a threshold to 
            determine when to increase or decrease the variance to keep the total relative std dev 
            as close to eta[1]. True for AdamCB.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    """

    def __init__(self, params, lr=1e-3, etas=(1.0, 1.0), randomize_eta2=False, bound_stddev=False, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, seed=None):
        
        if bound_stddev is True and randomize_eta2 is True:
            raise ValueError("Both randomize_eta2 and bound_stddev are True. Please check your arguments")
            
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if seed is not None:
            np.random.seed(seed)
        defaults = dict(lr=lr, etas=etas, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, bound_stddev=bound_stddev, randomize_eta2=randomize_eta2)
        super(AdamALL, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamALL, self).__setstate__(state)
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
            raise RuntimeError('Needs closure.')

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Does not support sparse gradients, please consider SparseAdam instead')
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
                    grad.add_(group['weight_decay'], p.data)

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
