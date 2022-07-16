trainingcontrol.py 
# DONE write training process into loop

trainingcallback.py
# DONE have control over beta value during the training process

metricValidation.py
# TODO ASAP validate metrics between beta-VAE and VAE

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
    
