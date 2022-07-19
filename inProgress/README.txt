trainingcontrol.py 
# DONE write training process into loop

trainingcallback.py
# DONE have control over beta value during the training process

metricValidation.py
# DONE validate metrics between beta-VAE and VAE

VAE: 
[07:11<00:00,  1.36it/s, loss=51.10, recon_loss=51.10, aug_loss=0.000]
{'dci.informativeness_train': 0.9625, 'dci.informativeness_test': 0.33999999999999997, 'dci.disentanglement': 0.01443064446343698, 'dci.completeness': 0.019299076551980043, 'mig.discrete_score': 0.012078071262848556, 'sap.score': 0.007599999999999997}

beta-VAE:
[06:08<00:00,  1.59it/s, loss=82.40, recon_loss=67.40, reg_loss=15.00, aug_loss=0.000] beta = 4
{'dci.informativeness_train': 0.95775, 'dci.informativeness_test': 0.28600000000000003, 'dci.disentanglement': 0.019728475907970364, 'dci.completeness': 0.02398439146078013, 'mig.discrete_score': 0.021150597774475543, 'sap.score': 0.004749999999999994}

[06:48<00:00,  1.44it/s, loss=74.40, recon_loss=74.40, reg_loss=0.00107, aug_loss=0.000] beta = 3
{'dci.informativeness_train': 0.9660000000000001, 'dci.informativeness_test': 0.22099999999999997, 'dci.disentanglement': 0.014585690961003933, 'dci.completeness': 0.016505228698311475, 'mig.discrete_score': 0.013328998116705768, 'sap.score': 0.0035000000000000014}

adaGVAE:
[14:06<00:00,  1.44s/it, loss=85.60, recon_loss=85.60, reg_loss=2.68e-5, aug_loss=0.000] 
{'dci.informativeness_train': 0.9655, 'dci.informativeness_test': 0.17400000000000002, 'dci.disentanglement': 0.004920799188874761, 'dci.completeness': 0.008473833324953939, 'mig.discrete_score': 0.0128798563551259, 'sap.score': 0.0}

# DONE imshow in metricValidation.py

# TODO set up experiments
  # TODO original training as control group and 
    # For Beta Vae and AdaGVae, Beta = [0, 1, 2, 3, 4]
  # TODO beta controled training as experimental group.
    # slightly increase or decrease after each batch
      # slowly increase Beta from 0 to 4
      # slowly decrease Beta from 4 to 0
    # train generative ability upto certain point, than train disentanglement.
    # train disentanglement upto certain point, than train generative ability.
  # TODO Delay, until all the early experiments have conclusions. 
    # control metrics, make generative quality and disentanglement stay in certain range.    
    

