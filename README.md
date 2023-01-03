#  Disentanglement Study 

There exist many unsupervised or weakly-supervised generative approaches, and they use similar techniques to manipulate some hyper-parameters to modulate the learning constraints to the training objective. For example, for the modified variant of variational autoencoder (VAE, an artificial neural network architectural probabilistic graphical model), $\beta$-VAE, introduces the tunable parameter $\beta$, which is used to emphasize the learning of disentangled representations. With $\beta$ = 1, $\beta$-VAE is considered as a regular VAE, and with $\beta$ $>$ 1, the model is pushed to learn the disentangled representations if there exist some underlying independent variations in the training data. The existing work has already done experiments on different $\beta$ values, and all of them have pointed out that, a big $\beta$ may create trade-offs between generation quality and the extent of disentanglement. Therefore the problem is raised, by modifying such hyper-parameters during the training process, would that be possible to minimize this kind of trade-off.

<img src="./Experiment Code/Image collection.jpg" width=700 height=700>

## Usage
rough copy for thesis, and stats
https://docs.google.com/document/d/1PTROkYoJQ6f-PrLWsHxbb0aunci0NGc0OWPPdlNa3Vs/edit?usp=sharing

## Requirements

## Citing This Work
