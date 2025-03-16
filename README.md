Attention block in each model needs to be modified to incorporate the modified softmax. 
The weights of the models remain intact

{model}_train_seq.sub scripts can be used to train the SoftNet parameters of the model. 
param.requires_grad is set False for all model parameters apart from the SoftNet parameters. 

Profiling for StatMax and AttSkip is done using inference scripy, but using training dataset. 
