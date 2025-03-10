import numpy as np
import pandas as pd
import argparse
from A1 import *

def parse_args():
    parser = argparse.ArgumentParser(description='To train network')
    
    # Wandb parameters as given in assignment statement
    parser.add_argument('-wp', '--wandb_project', type=str, default='DA6401_Assignment-1',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', type=str, default='megh_m-iit-madras',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
    
    # Dataset
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', 
                        choices=["mnist", "fashion_mnist"],
                        help='Dataset to use for model')
    
    # Training parameters
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='Number of epochs to train neural network.')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='Batch size used to train neural network.')
    
    # Loss function
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=["mean_squared_error", "cross_entropy"],
                        help='Loss function used to train neural network.')
    
    # Optimizer
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        choices=["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"],
                        help='Optimizer used to train neural network.')
    
    # Optimizer parameters
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01,
                        help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='Momentum used by momentum and nag optimizers.')
    parser.add_argument('-beta', '--beta', type=float, default=0.9,
                        help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9,
                        help='Beta1 used by adam and nadam optimizers.')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.99,
                        help='Beta2 used by adam and nadam optimizers.')
    parser.add_argument('-eps', '--epsilon', type=float, default=0.00000001,
                        help='Epsilon used by optimizers.')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0,
                        help='Weight decay used by optimizers.')
    
    # Model architecture
    parser.add_argument('-w_i', '--weight_init', type=str, default='random',
                        choices=["random", "Xavier"],
                        help='Weight initialization method.')
    parser.add_argument('-nhl', '--num_layers', type=int, default=3,
                        help='Number of hidden layers used in feedforward neural network.')
    parser.add_argument('-sz', '--hidden_size', type=int, default=64,
                        help='Number of hidden neurons in a feedforward layer.')
    parser.add_argument('-a', '--activation', type=str, default='sigmoid',
                        choices=["identity", "sigmoid", "tanh", "ReLU"],
                        help='Activation function used in hidden layers.')
    
    return parser.parse_args()
    def main():
    args = parse_args()

    # Initializing wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args) #Creating config dictionary from arguments passed
    )

    # Load data
    X_train, y_train, X_test, y_test = load_data(args.dataset)

    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = SGD(eta=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'momentum':
        optimizer = MomentumGD(eta=args.learning_rate, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'nag':
        optimizer = NAGD(eta=args.learning_rate, momentum=args.momentum,
                       weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(eta=args.learning_rate, beta=args.beta,
                          epsilon=args.epsilon, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = Adam(eta=args.learning_rate, beta1=args.beta1,
                       beta2=args.beta2, epsilon=args.epsilon,
                       weight_decay=args.weight_decay)
    elif args.optimizer == 'nadam':
        optimizer = NAdam(eta=args.learning_rate, beta1=args.beta1,
                        beta2=args.beta2, epsilon=args.epsilon,
                        weight_decay=args.weight_decay)

    # Create neural network
    input_size = X_train.shape[1]  # 784 for MNIST/Fashion MNIST
    hidden_layers = [args.hidden_size] * args.num_layers
    output_size = y_train.shape[1]  # 10 for MNIST/Fashion MNIST

    nn = NN(
        in_size=input_size,
        hidden=hidden_layers,
        out_size=output_size,
        actv=args.activation,
        weight_init=args.weight_init,
        loss=args.loss,
        optimizer=optimizer
    )

    # Training loop
    for epoch in range(args.epochs):
        # Shuffling the training data every epoch
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        # Mini-batch training
        for i in range(0, len(X_train), args.batch_size):
            X_batch = X_train_shuffled[i:i+args.batch_size]
            y_batch = y_train_shuffled[i:i+args.batch_size]
            nn.back_prop(X_batch, y_batch)

        # Calculate loss and accuracy on train and test sets
        train_loss = nn.calc_loss(X_train, y_train)
        test_loss = nn.calc_loss(X_test, y_test)

        train_predictions = np.argmax(nn.predict(X_train), axis=1)
        train_true_labels = np.argmax(y_train, axis=1)
        train_accuracy = np.mean(train_predictions == train_true_labels)

        test_predictions = np.argmax(nn.predict(X_test), axis=1)
        test_true_labels = np.argmax(y_test, axis=1)
        test_accuracy = np.mean(test_predictions == test_true_labels)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": test_accuracy
        })

        print(f"Epoch {epoch+1}/{args.epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Final evaluation
    final_train_accuracy = np.mean(np.argmax(nn.predict(X_train), axis=1) == np.argmax(y_train, axis=1))
    final_test_accuracy = np.mean(np.argmax(nn.predict(X_test), axis=1) == np.argmax(y_test, axis=1))

    print(f"Final Train Accuracy: {final_train_accuracy:.4f}")
    print(f"Final Test Accuracy: {final_test_accuracy:.4f}")

    # Log final metrics
    wandb.log({
        "final_train_accuracy": final_train_accuracy,
        "final_validation_accuracy": final_test_accuracy
        "final_train_loss": train_loss,
        "final_test_loss": test_loss
    })

    wandb.finish()

if __name__ == "__main__":
    main()
