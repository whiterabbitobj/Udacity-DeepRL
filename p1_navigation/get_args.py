import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Train or Test a Deep RL agent in Udacity's Banana Environment",
            usage="EXAMPLE COMMAND:\npython banana_agent.py --train --batch_size 64 -lr 5e-4")

    # parser.add_argument("-a", "--train",
    #         help="Set the agent into training mode.",
    #         action="store_true")
    parser.add_argument("-m", "--mode",
            help="In which mode should the Agent run? (train, demo)")
    parser.add_argument("-a", "--agent_type",
            help="Which type of Agent to use. (DQN, DDQN)")
    parser.add_argument("--gpu",
            help="Use GPU if available.",
            action="store_true")
    parser.add_argument("-ec", "--episode_count",
            help="How many episodes to train?",
            type=int,
            default=200)
    parser.add_argument("-e", "--epsilon",
            help="Starting value of Epsilon.",
            type=float,
            default=1.0)
    parser.add_argument("-ed", "--epsilon_decay",
            help="Epsilon decay value.",
            type=float,
            default=0.9999)
    parser.add_argument("-em", "--epsilon_min",
            help="Minimum value for epsilon.",
            type=float,
            default=0.1)
    parser.add_argument("-d", "--discount",
            help="Discount rate.",
            type=float,
            default=1.0)
    parser.add_argument("-lr", "--learn_rate",
            help="Alpha (Learning Rate).",
            type=float,
            default=1.0)



#########################
#   OLD CODE BELOW, DELETE ME BEFORE PRODUCTION!
#########################
    parser.add_argument("-a", "--arch",
            help="The name of the pre-trained model to use. Currently supported: densenet121, densenet169, vgg16. Default: 'densenet169'.",
            type=str,
            default="densenet169")
    parser.add_argument("-b", "--batch_size",
            help="Batch size of the dataloaders used in training/testing. Default: 64",
            type=int,
            default=64)
    parser.add_argument("-c", "--crop_size",
            help="Crop size of dataset images. Default: 224",
            type=int,
            default=224)
    parser.add_argument("-dir", "--data_dir",
            help="Base directory of dataset images. Default: flowers/",
            type=str,
            default="flowers")
    parser.add_argument("-dr", "--drop_rate",
            help="Dropout rate of the classifier network. Default 0.5",
            type=float,
            default=0.5)
    parser.add_argument("-e", "--epochs",
            help="How many epochs to run through the training. Default: 10",
            type=int,
            default=10)
    parser.add_argument("--gpu",
            help="Run on the gpu using cuda, defaults to false",
            action="store_true")
    parser.add_argument("-hs", "--hidden_sizes",
            help="The size of the hidden layers for the classifier. Default: 1000 500.",
            nargs="+",
            type=int,
            default=[1000,500])
    parser.add_argument("-l", "--loaders",
            help="The folder names of the different datasets. e.g. train, valid, test. Default: train valid test",
            type=str,
            default=['train','valid','test'])
    parser.add_argument("-lr", "--learning_rate",
            help="The learning rate of the optimizer. Default: 0.001",
            type=float,
            default=0.001)
    parser.add_argument("--mean",
            help="The average (mean) of the dataset images. Default: 0.485, 0.456, 0.406",
            nargs="+",
            type=int,
            default=[0.485, 0.456, 0.406])
    parser.add_argument("-pe", "--print_every",
            help="How many batches between status prints. Default: 3",
            type=int, default=3)
    parser.add_argument("-rr", "--rand_rot",
            help="Random rotation size added to training images. Default: 30",
            type=int,
            default=30)
    parser.add_argument("-rs", "--rescale_size",
            help="Rescale before crop of testing images. Default: 256",
            type=int,
            default=256)
    parser.add_argument("-s", "--save_dir",
            help="Directory to save model checkpoints",
            type=str,
            default="")
    parser.add_argument("--std",
            help="Standard deviation of dataset images. Default: 0.229 0.224 0.225",
            nargs="+",
            type=int,
            default=[0.229, 0.224, 0.225])
    parser.add_argument("-tr", "--testrun",
            help="How many batches to run through before quitting. For testing purposes. Default: None",
            type=int,
            default=None)
    parser.add_argument("-v", "--do_validation",
            help="How often to run a validation, as a multiplier of the print_every flag. Higher numbers means less validation runs.\
                  1 means to run every time the status is printed. 0 will not run validation. Default: None",
            type=int,
            default=None)

    return parser.parse_args()
