# DA6401-A1
A1.py contains the classes Layer, NN & Optimizers. This model can run on SGD, MomentumGD, NAGD, RMSProp, Adam & NAdam optimizers.
from A1.py import * to train.py. Save both in same directory.
Use requirements.txt to get necessary packages. use pip install -r requirements.txt. 
More details in attached report, available at https://api.wandb.ai/links/megh_m-iit-madras/yggk5zqd.

The function train.py takes in the following arguments and needs to be run to train the model. 2 choices of Datasets exist. Default value indicated after argument preset.

-wp, --wandb_project	DA6401_Assignment-1

-we, --wandb_entity		megh_m-iit-madras

-d, --dataset	fashion_mnist	choices: ["mnist", "fashion_mnist"]

-e, --epochs	1	

-b, --batch_size	4

-l, --loss	cross_entropy	choices: ["mean_squared_error", "cross_entropy"]

-o, --optimizer	sgd	choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]

-lr, --learning_rate	0.1

-m, --momentum	0.5	

-beta, --beta	0.5	

-beta1, --beta1	0.5

-beta2, --beta2	0.5	

-eps, --epsilon	0.000001	

-w_d, --weight_decay	.0	

-w_i, --weight_init	random	choices: 

-nhl, --num_layers	

-sz, --hidden_size
-a, --activation	sigmoid	choices: ["identity", "sigmoid", "tanh", "ReLU"]
