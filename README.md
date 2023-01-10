# Empirical Study on Exploring the Impact of Controlling the Objective on Disentanglement Learning During Training


## Table of Contents
1. [Overview](#overview)
2. [Objective](#objective)
3. [Experiment](#experiment)
4. [Future Work](#future-work)

### Overview
***
There exist many unsupervised or weakly-supervised generative approaches, and they use similar techniques to manipulate some hyper-parameters to modulate the learning constraints to the training objective. For example, for the modified variant of variational autoencoder (VAE, an artificial neural network architectural probabilistic graphical model), $\beta$-VAE, introduces the tunable parameter $\beta$, which is used to emphasize the learning of disentangled representations. With $\beta$ = 1, $\beta$-VAE is considered as a regular VAE, and with $\beta$ $>$ 1, the model is pushed to learn the disentangled representations if there exist some underlying independent variations in the training data. The existing work has already done experiments on different $\beta$ values, and all of them have pointed out that, a big $\beta$ may create trade-offs between generation quality and the extent of disentanglement. Therefore the problem is raised, by modifying such hyper-parameters during the training process, would that be possible to minimize this kind of trade-off.
### Screenshot
<img src="./Experiment Code/Image collection.jpg" width=700 height=700>

## Objective
***
Assume in an unsupervised manner, the learning process of a generative model can be described as two phases, learning the domain and learning the disentangled representation, and these two phases can be alternatively switched by a tunable hyperparameter. In such a scenario, can we minimize the tradeoff between generation quality and disentanglement, which can be evaluated by reconstruction loss and disentanglement metrics? Answering this question is the objective of this empirical study.

## Experiment
***
We employ $\beta$-VAE as the main experimental object. The training experiments are performed over three image-based disentanglement domains. These specifically generated domains provide knowledge of their underlying features which facilitates the use of disentanglement metrics and also makes the other tunable parts of the experiment to be more controllable. We use reconstruction loss to evaluate generation quality, and use three different disentanglement metrics to evaluate the outcome of disentanglement training.

## Future Work
***
During the experimental process, we observed that if we tune the $\beta$ value before the reconstruction loss fully converges, the model can have some good training outcomes beyond expectations. But this does not seem to be a common phenomenon. In the future, we want to explore more on this issue more and will need to conduct a large number of experiments to verify the feasibility of this conjecture. Further, this empirical study is limited to $\beta$-VAE. Whether the conclusion can be generalized to other disentanglement learning strategies still needs to be verified by large-scale experiments on multiple generative models.
## FAQs
***
A list of frequently asked questions
1. **This is a question in bold**
Answer to the first question with _italic words_. 
2. __Second question in bold__ 
To answer this question, we use an unordered list:
* First point
* Second Point
* Third point
3. **Third question in bold**
Answer to the third question with *italic words*.
4. **Fourth question in bold**
| Headline 1 in the tablehead | Headline 2 in the tablehead | Headline 3 in the tablehead |
|:--------------|:-------------:|--------------:|
| text-align left | text-align center | text-align right |
