

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
    make_dataloader,
    save_results,
)

import numpy as np
import pandas as pd
import csv

print('All is imported')

# Configurations
torch.multiprocessing.set_sharing_strategy("file_system")

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE[:-1]



def main(
    input_path: str,
    output_path: str,
    model_path: str,
):

    #Event_no selection. If you don't need this just outcomment i
    RD_selection = pd.read_csv('').reset_index(drop = True)['event_no'].ravel().tolist()
    MC_selection = pd.read_csv('').reset_index(drop = True)['event_no'].ravel().tolist()

    #Testing selection to check that the code is running properly. Always do this before doing the full prediction.
    RD_selection = RD_selection[:100] 
    MC_selection = MC_selection[:100]

    # Configuration. Same setup as your model
    config = {
        "db": input_path,
        "pulsemap": "SplitInIcePulses",
        "batch_size": 512,
        "num_workers": 15,
        "accelerator": "gpu",
        "devices": [1], #Which GPU you wanna use. Always take the ones that is not in use.
        "target": "pid",
        "n_epochs": 1,
        "patience": 1, #Patience is not needed as we are nt training.
    }

    archive = output_path

    # Name of the run you are doing. An example is given below.
    run_name = "model_trained_on_{}__more_neccesary_information".format(
        config["target"]
    )

    # Module that defines what the GNN sees. Essential for running graphnet (still under development so keep that in mind if it doesn't work with your current graphnet).
    graph_definition = KNNGraph(
        detector = IceCubeDeepCore(),
    )

    prediction_dataloader_RD = make_dataloader(
        db = config["db"],
        pulsemaps = config["pulsemap"],
        features = features,
        truth = truth,
        selection = RD_selection,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        graph_definition = graph_definition
    )

    #Buil of the model
    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )

    #The task you have done.
    task = MulticlassClassificationTask(
        nb_outputs = 3,
        hidden_size=gnn.nb_outputs,
        target_labels=config["target"],
        loss_function=CrossEntropyLoss(options={1: 0, -1: 0, 13: 1, -13: 1, 12: 2, -12: 2, 14: 2, -14: 2, 16: 2, -16: 2}),
        transform_inference=lambda x: softmax(x,dim=-1),
    )

    #Define the model.
    model = StandardModel(
        #detector=detector,
        graph_definition=graph_definition,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-04, "eps": 1e-03},
        scheduler_class=PiecewiseLinearLR,
    )

    #Using early stopping.
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
        logger=wandb_logger,#If not using WandB change this to None.
        #logger = None,
    )

    # Load your existing model
    model.load_state_dict(model_path)

    # predict and save predictions to file
    resultsRD = get_predictions(
        trainer = trainer,
        model = model,
        dataloader = prediction_dataloader_RD,
        # State what you wanna predict. Example below is for multiclassification for "pid".
        prediction_columns = [config["target"] + "_noise_pred", config["target"] + "_muon_pred", config["target"]+ "_neutrino_pred"],
        additional_attributes=[config["target"], "event_no"],# "EventID", "SubEventID", "RunID", "SubrunID"],
    )
    #Saving the model. Here you also give the name
    resultsRD.to_csv(
        output_folder + "/{}_name_on_your_prediction_csv_file.csv".format(config["target"])
    )

#Run your main function. Here you also give the input db, output folder and model path (important that it ends with state_dict.pth)
if __name__ == "__main__":
    # Input database path
    input_db = ""
    # Output folder path
    output_folder = ""
    # Model path
    model_path = "state_dict.pth"


    main(input_db, output_folder, model_path)


