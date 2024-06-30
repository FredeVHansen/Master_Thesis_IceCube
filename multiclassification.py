

""""
Importing Packages.
""""

from distutils.log import debug
import logging
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import torch
from torch.optim.adam import Adam
from torch.nn.functional import one_hot, softmax

from graphnet.training.loss_functions import CrossEntropyLoss

from graphnet.data.constants import FEATURES, TRUTH

from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph

from graphnet.models.task.classification import MulticlassClassificationTask 
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.utils import (
    get_predictions,
    make_train_validation_dataloader,
    save_results,
    make_dataloader
)
from graphnet.models.graphs import GraphDefinition

import numpy as np
import pandas as pd
import csv
import argparse


print('All packages is imported')



# Configurations
torch.multiprocessing.set_sharing_strategy("file_system")

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE[:-1]

""""
Use of WandB. Can be outcommented if not needed.
""""

# Make sure W&B output directory exists
WANDB_DIR = "./wandb/"
os.makedirs(WANDB_DIR, exist_ok=True)

# Initialise Weights & Biases (W&B) run
wandb_logger = WandbLogger(
    project="", #Write project name.
    name='', #Write run name here.
    save_dir=WANDB_DIR,
    log_model=True,
)

print('WandB initialized')


""""
Setting everything up for the model.
""""

#Give the path to the database here
parser.add_argument(
    "-db",
    "--database",
    dest="path_to_db",
    type=str,
    help="<required> path(s) to database [list]",
    default="", #Write the path to the db which you intend to train on.
    # required=True,
)

#Give the output directory here
parser.add_argument(
    "-o",
    "--output",
    dest="output",
    type=str,
    help="<required> the output path [str]",
    default="", #Write the path to the directory where you want to store your model.
    # required=True,
)

#Pulsemap (should almost always be SplitInIcePulses)
parser.add_argument(
    "-p",
    "--pulsemap",
    dest="pulsemap",
    type=str,
    help="<required> the pulsemap to use. [str]",
    default="SplitInIcePulses", #This is where you can change pulsemap from SplitInIcePulses to something different if needed.
    # required=True,
)

#Choice of GPU (comment out for cpu-usage)
parser.add_argument(
    "-g",
    "--gpu",
    dest="gpu",
    type=int,
    help="<required> the name for the model. [str]",
    default=0, #Here you tell which GPU you want to run on.
    # required=True,
)

#Choice of batch_size Usually we use 512 as a standard but can be changed.
parser.add_argument(
    "-b",
    "--batch_size",
    dest="batch_size",
    type=int,
    help="<required> the name for the model. [str]",
    default=512, #Here you change the batch_size
    # required=True,
)

#Choice of numbers of epochs. 
parser.add_argument(
    "-e",
    "--epochs",
    dest="epochs",
    type=int,
    help="<required> the name for the model. [str]",
    default=150, #Number of epochs. Set it higher than what you will expect that you need as we are using early stopping.
    # required=True,
)

#Choice of number of workers
parser.add_argument(
    "-w",
    "--workers",
    dest="workers",
    type=int,
    help="<required> the number of cpu's to use. [str]",
    default=15, #Here you can change the number of workers
    # required=True,
)

#Give the name of the file you wanna create
parser.add_argument(
    "-r",
    "--run_name",
    dest="run_name",
    type=str,
    help="<required> the name for the model. [str]",
    default='', #Here you put the filename
    # required=True,
)

#State that you wanna use GPU 
parser.add_argument(
    "-a",
    "--accelerator",
    dest="accelerator",
    type=str,
    help="<required> the name for the model. [str]",
    default="gpu", #Can be changed to cpu.
    # required=True,
)

args = parser.parse_args()

print('Argparse done, defining main loop')


""""
Now we create the main functionthat will run and create our model.
""""

def main():

    # Configuration. All that we defined beforehand.
    config = {
        "db": args.path_to_db,
        "pulsemap": args.pulsemap,
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "accelerator": args.accelerator,
        "devices": [args.gpu],
        "target": "pid",
        "n_epochs": args.epochs,
        "patience": 5, #Early stopping, how many epochs without improvements before the model stops and take the best result.
    }

    config["archive"] = args.output
    config["run_name"] = "dynedge_{}_".format(config["target"]) + args.run_name
    print('before logs to wand')
    # Log configuration to W&B. If not using WandB outcomment this
    wandb_logger.experiment.config.update(config)
    print('after logs to wand')

    #Insert your training, validation and test set. How you load them can depend on what you wanna train on and the dataset
    train_selection = pd.read_csv('').reset_index(drop = True)['event_no'].ravel().tolist()
    validation_selection = pd.read_csv('').reset_index(drop = True)['event_no'].ravel().tolist()
    test_selection = pd.read_csv('').reset_index(drop = True)['event_no'].ravel().tolist()

    #Test script. Only takes the first 1000/100/100 events to check if script works. Outcomment or change the number of events when you are ready do the full training.
    train_selection = train_selection[:1000]
    validation_selection = validation_selection[:100]
    test_selection = test_selection[:100]


    # Module that defines what the GNN sees. Essential for running graphnet (still under development so keep that in mind if it doesn't work with your current graphnet).
    graph_definition = KNNGraph(
        detector = IceCubeDeepCore(),
        nb_nearest_neighbours=8
    )

    #Dataloader for training
    training_dataloader = make_dataloader(db = config['db'],
                                            selection = train_selection, 
                                            pulsemaps = config['pulsemap'],
                                            features = features,
                                            truth = truth,
                                            batch_size = config['batch_size'],
                                            num_workers = config['num_workers'],
                                            graph_definition = graph_definition,
                                            shuffle = True)

    #Dataloader for validation
    validation_dataloader = make_dataloader(db = config['db'],
                                            selection = validation_selection,
                                            pulsemaps = config['pulsemap'],
                                            features = features,
                                            truth = truth,
                                            batch_size = config['batch_size'],
                                            num_workers = config['num_workers'],
                                            graph_definition = graph_definition,
                                            shuffle = False)

    #Dataloader for test
    test_dataloader = make_dataloader(db = config['db'],
                                            selection = test_selection,
                                            pulsemaps = config['pulsemap'],
                                            features = features,
                                            truth = truth,
                                            batch_size = config['batch_size'],
                                            num_workers = config['num_workers'],
                                            graph_definition = graph_definition,
                                            shuffle = False)

    #Building the model
    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )

    #Define your task (in this case it is multiclassification).
    task = MulticlassClassificationTask(
        nb_outputs = 3,
        hidden_size=gnn.nb_outputs,
        target_labels=config["target"],
        #Loss function below is an example for multiclassification of "pid"
        loss_function=CrossEntropyLoss(options={1: 0, -1: 0, 13: 1, -13: 1, 12: 2, -12: 2, 14: 2, -14: 2, 16: 2, -16: 2}),
        transform_inference=lambda x: softmax(x,dim=-1),
    )

    #Define the model. This can be optimized depending on the task.
    model = StandardModel(
        graph_definition=graph_definition,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam, #Usually we just use Adam as the optimizer
        optimizer_kwargs={"lr": 1e-05, "eps": 1e-03}, #Give the learning rate and epsilon
        scheduler_class=PiecewiseLinearLR, #How will the learning rate change. This can be changed to your liking
        scheduler_kwargs={
            "milestones": [
                0,
                len(training_dataloader) / 2,
                len(training_dataloader) * config["n_epochs"],
            ],
            "factors": [1e-2, 1, 1e-02],
        },
        scheduler_config={
            "interval": "step",
        },
    )

    #Using the early stopping. Remember you defined this earlier.
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["patience"],
        ),
        ProgressBar(),
    ]

    #Define your trainer. Here we use pytorch Trainer.
    trainer = Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        max_epochs=config["n_epochs"],
        callbacks=callbacks,
        log_every_n_steps=1,
        logger=wandb_logger, #If not using WandB change this to None.
        #logger = None,
    )

    #Training the model.
    try:
        trainer.fit(model, training_dataloader, validation_dataloader)
    except KeyboardInterrupt:
        logger.warning("[ctrl+c] Exiting gracefully.")
        pass

     # Predict on Validation Set and save results to file. This is an example for multiclassification of "pid"
    results_val = get_predictions(
        trainer = trainer,
        model = model,
        dataloader = validation_dataloader,
        prediction_columns =[config["target"] + "_noise_pred", config["target"] + "_muon_pred", config["target"]+ "_neutrino_pred"],
        additional_attributes=[config["target"], "event_no"],
    )
    save_results(config["db"], config["run_name"] + '_validation_set', results_val, config["archive"], model)


# Main function call. Here we call the model and starts the training.
if __name__ == "__main__":
    print('Before main loop')
    main()