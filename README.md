## Exploiting Uncertainty of Loss Landscape for Stochastic Optimization

Paper: [http://arxiv.org/abs/1905.13200](http://arxiv.org/abs/1905.13200)

Cite as: ``V.S. Bhaskara, and S. Desai. ``_``arXiv preprint``_`` arXiv:1905.13200 [cs.LG] (2019)``.

### Algorithm 

We introduce variants of the [Adam](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html) optimizer that either bias the updates along regions that conform across mini-batches or randomly *explore* unbiased in the parameter space along the variance-gradient. Our variants of the optimizer are shown to generalize better with improved test accuracy across multiple datasets and architectures. Particularly, our optimizer shines when the data is highly noisy or redundant or low-rank. Please refer to the [paper](http://arxiv.org/abs/1905.13200) for more details.


> NOTE: **We highly recommend using the [*AdamS*](PyTorch-Optimizers/AdamS.py) optimizer, which is an unbiased estimate of the gradients and outperforms the other variants introduced in the paper, based on our experiments.**

### Code

PyTorch implementations of the Adam optimizer variants introduced in the paper are available under [``PyTorch-Optimizers/``](PyTorch-Optimizers/).

The **AdamS** optimizer for PyTorch is available [here](PyTorch-Optimizers/AdamS.py). 

### Usage

The usage is identical to the [Adam](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html) optimizer except that the `optimizer.step()` function takes a required parameter that is a lambda function as follows:

```python
# compute output and loss
outputs = model(inputs)
loss = criterion(outputs, targets)

# optimizer
optimizer.zero_grad()
loss.backward()
optimizer.step(lambda: loss)  # pass a lambda function that returns the loss tensor
```

Voila!


### Experiments

We evaluated the optimizers on multiple models such as Logistic Regression (LR), MLPs, and CNNs on the CIFAR-10/MNIST datasets. The architecture of the networks is chosen to closely resemble the experiments published in the original Adam paper [(Kingma and Ba, 2015)](https://arxiv.org/abs/1412.6980). Code for our experiments is available under [``Experiments/``](Experiments/), and is based on the original CIFAR-10 classifier code [here](https://github.com/bearpaw/pytorch-classification).

#### Reproducing the results

* Run the shell script for each type of model (LR/MLP/CNN) under [``Experiments/``](Experiments/)
* Compute the Mean and the Standard Deviation of the training/validation metrics for each configuration across the three runs. 

Results of our training runs with the mean and the standard deviation values for each configuration is provided under [``Experiments/results_mean_std/``](Experiments/results_mean_std).

### Results

#### CNN trained on CIFAR-10 with batch size = 128 and no dropout

![CNN with Batch Size 128](Experiments/results_mean_std/images/cifar-10.jpg)

#### CNN trained on CIFAR-10 with batch size = 16 and no dropout

![CNN with Batch Size 16](Experiments/results_mean_std/images/cifar-10-bsz16.jpg)


#### Comparison of Dropout with AdamS for CNN trained on CIFAR-10 with batch size = 128 

![Comparing dropout](Experiments/results_mean_std/images/dropout.jpg)

### Update Rules

The update rules for various variants of Adam in the paper are summarized below: 

![Summary of update rules](updates.png)  

AdamUCB and AdamCB are biased estimates of the full-gradient. We recommend using AdamS which is an unbiased estimate, and outperforms other variants based on our experiments with CIFAR-10. 

Please refer to the [paper](http://arxiv.org/abs/1905.13200) for more details.



### Contribute

Feel free to create a pull request if you find any bugs or you want to contribute (e.g., more datasets, more network structures, or tensorflow/keras ports). 
