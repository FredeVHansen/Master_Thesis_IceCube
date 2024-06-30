
"""
Importing packages
"""

from distutils.log import debug
import logging
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import torch
from torch.optim.adam import Adam
from torch.nn.functional import one_hot, softmax

from graphnet.training.loss_functions import CrossEntropyLoss, VonMisesFisher2DLoss, LogCoshLoss, MSELoss
from graphnet.data.constants import FEATURES, TRUTH

from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import GraphDefinition
from graphnet.models.graphs import KNNGraph

from graphnet.models.task.reconstruction import EnergyReconstruction, ZenithReconstructionWithKappa, AzimuthReconstructionWithKappa
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.utils import (
    get_predictions,
    make_train_validation_dataloader,
    save_results,
    make_dataloader
)

from graphnet.training.utils import make_train_validation_dataloader

import random

import numpy as np
import pandas as pd
import csv

from graphnet.utilities.logging import Logger

logger = Logger()

print('All is imported')

# Configurations
torch.multiprocessing.set_sharing_strategy("file_system")

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE[:-1]

# Make sure W&B output directory exist. Can be removed if not wanted
WANDB_DIR = "./wandb/"
os.makedirs(WANDB_DIR, exist_ok=True)


"""
Make your training function.
"""
def train(config):
    #Insert your training, validation and test set. How you load them can depend on what you wanna train on and the dataset
    train_selection = pd.read_csv('').reset_index(drop = True)['event_no'].ravel().tolist()
    validation_selection = pd.read_csv('').reset_index(drop = True)['event_no'].ravel().tolist()
    test_selection = pd.read_csv('').reset_index(drop = True)['event_no'].ravel().tolist()
    #If you wanna shuffle you events
    random.shuffle(train_selection)
    random.shuffle(validation_selection)
    random.shuffle(test_selection)

    #Test script. Only takes the first 1000/100/100 events to check if script works. Outcomment/change when you are ready for full training.
    train_selection = train_selection[:1_000]
    validation_selection = validation_selection[:100]
    test_selection = test_selection[:100]

    print(f'Numner of training events: {len(train_selection)}')
    print(f'Numner of validation events: {len(validation_selection)}')
    print(f'Numner of test events: {len(test_selection)}')

    #For WandB. Outcomment if not wanted.
    wandb_logger.experiment.config.update(config)

    #Common variables for training and truth
    logger.info(f'features: {features}')
    logger.info(f'truth: {truth}')

    # Module that defines what the GNN sees. Essential for running graphnet (still under development so keep that in mind if it doesn't work with your current graphnet).
    graph_definition = KNNGraph(
        detector = IceCubeDeepCore(),
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
        add_global_variables_after_pooling=True,
    )

    # If elif argument for choosing the given task. You can easily add more with elif arguments if wanted. Just think about reconstruction-and loss function.
    #Energy
    if config["target"] == "energy":
        task = EnergyReconstruction(
            hidden_size = gnn.nb_outputs,
            target_labels = config["target"],
            #loss_function = LogCoshLoss(),
            loss_function = MSELoss(),
            transform_prediction_and_target = lambda x: torch.log10(x),
        )
    
    #Zenith angle
    elif config["target"] == "zenith":
        task = ZenithReconstructionWithKappa(
            hidden_size = gnn.nb_outputs,
            target_labels = config["target"],
            loss_function = VonMisesFisher2DLoss(),
        )

    #Azimuth angle
    elif config["target"] == "azimuth":
        task = AzimuthReconstructionWithKappa(
            hidden_size = gnn.nb_outputs,
            target_labels = config["target"],
            loss_function = VonMisesFisher2DLoss(),
        )

    #Define the model. This can be optimized depending on the task.
    model = StandardModel(
        graph_definition = graph_definition,
        gnn = gnn,
        tasks = [task],
        optimizer_class = Adam, #Usually we just use Adam as the optimizer
        optimizer_kwargs = {"lr": 1e-03, "eps": 1e-03}, #Give the learning rate and epsilon
        scheduler_class = PiecewiseLinearLR, #How will the learning rate change. This can be changed to your liking
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
    
    #Using early stopping.You will define this later on in the script.
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["patience"],
        ),
        ProgressBar(),
    ]

    #Define your trainer. Here we use pytorch Trainer.
    trainer = Trainer(
        default_root_dir=f'~/{config["run_name"]}',
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
        logger.warning("[ctrl+c] exiting gracefully.")
        pass

    # Predict on test Set and save results to file. This is your predictions for the model. You can also predict on other databases later on.
    #This prediction is an example of either zenith or azimuth.
    results = get_predictions(
        trainer = trainer,
        model = model,
        dataloader = test_dataloader,
        prediction_columns = [config["target"] + "_pred", config["target"] + "_kappa"],
        additional_attributes = [config["target"], "event_no", "energy"],
    )

    save_results(config["db"], config["run_name"] + "validation_set", results, config["archive"], model)

"""
Now we create the main functionthat will run and create our model.
"""
def main():
    #Tell which task you wanna do.
    for target in ["azimuth"]: #["energy"], ["zenith"]
        pulsemap = 'SplitInIcePulses'
        n_epochs = 100
        #Archive is where you wanna save this model and prediction.
        archive = ""
        #Run name is the name of your run. Below is an example of how you can build it up.
        run_name = f"your_name_{target}_task"

    #Configurations. Most of the things below is selfexplanatory and can be changed according to what you need.
        config = {
            "db": "", #This is the path to your database you wanna predict on.
            "pulsemap": pulsemap,
            "batch_size": 512,
            "num_workers": 10,
            "accelerator": "gpu",
            "devices": [1],
            "target": target,
            "n_epochs": n_epochs,
            "patience": 5,
            "archive": archive,
            "run_name": run_name,
        }

    #Train the model.
    train(config)#, wandb_logger

# Main function call. Here we call the model and starts the training.
if __name__ == "__main__":
    main()
