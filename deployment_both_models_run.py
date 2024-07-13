from glob import glob
from os.path import join
from typing import TYPE_CHECKING, List, Sequence
import fnmatch
#import torch
from torch.nn.functional import one_hot, softmax
import argparse


from graphnet.constants import (
    TEST_DATA_DIR,
    EXAMPLE_OUTPUT_DIR,
    PRETRAINED_MODEL_DIR,
)
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.extractors.i3featureextractor import (
    I3FeatureExtractorIceCubeUpgrade,
    I3FeatureExtractorIceCube86,
    I3FeatureExtractorIceCubeDeepCore,
    I3FeatureExtractorIceCubeUpgrade
)
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import Logger

if has_icecube_package() or TYPE_CHECKING:
    from graphnet.deployment.i3modules import (
        GraphNeTI3Deployer,
        I3InferenceModule,
    )


ERROR_MESSAGE_MISSING_ICETRAY = (
    "This example requires IceTray to be installed, which doesn't seem to be "
    "the case. Please install IceTray; run this example in the GraphNeT "
    "Docker container which comes with IceTray installed; or run an example "
    "script in one of the other folders:"
    "\n * examples/02_data/"
    "\n * examples/03_weights/"
    "\n * examples/04_training/"
    "\n * examples/05_pisa/"
    "\nExiting."
)

features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE[:-1]


def main() -> None:
    
    
    print('Running now')
    
    parser = argparse.ArgumentParser(description="Process a file.")
    
    # Add an argument for the filename
    parser.add_argument('gcd_file', type=str, help='The path to the file to process')
    
    parser.add_argument('i3_file', type=str, help='The path to the file to process')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Use the filename from the arguments
    gcd_file = args.gcd_file
    i3_file = args.i3_file
    
    """GraphNeTI3Deployer Example."""
    # configure input files, output folders and pulsemap
    pulsemap = "SplitInIcePulses"
    input_folders = [f"/groups/icecube/cjb924/workspace/work/i3_deployment/test_data"] #Change this to i3 files directory #change!
    
    model_config_LE = f"/groups/icecube/cjb924/workspace/work/i3_deployment/files_for_use/models_and_config/LE_pid_multiclassification.yml" #Model configs
    state_dict_LE = f"/groups/icecube/cjb924/workspace/work/i3_deployment/files_for_use/models_and_config/LE_pid_multiclassification_state_dict.pth" #State_dicts for models
    
    # model_config_HE = f"/groups/icecube/cjb924/workspace/work/i3_deployment/files_for_use/HE_pid_multiclassification.yml" #Model configs
    # state_dict_HE = f"/groups/icecube/cjb924/workspace/work/i3_deployment/files_for_use/HE_pid_multiclassification_state_dict.pth" #State_dicts for models
    
    model_config_HE = f"/groups/icecube/cjb924/workspace/work/i3_deployment/files_for_use/models_and_config/config.yml" #Model configs
    state_dict_HE = f"/groups/icecube/cjb924/workspace/work/i3_deployment/files_for_use/models_and_config/_test_set_state_dict.pth" #State_dicts for models
    
    output_folder = f"/groups/icecube/cjb924/workspace/work/i3_deployment/results_LE_only" #Output folder path
    
    #gcd_file = f"/groups/icecube/cjb924/workspace/work/i3_deployment/test_data/Level2_IC86.2021_data_Run00136124_0101_81_647_GCD.i3.zst" #i3 files #Question???                     #CHECK
    gcd_file = f"{gcd_file}"
    i3_file = f"{i3_file}"
    
    #input_files = []
    #for folder in input_folders:
    #    print('File in for loop', folder)
    #    input_files.extend(glob(join(folder, "*.i3.zst")))
    #    print('Checking filename')
    #    input_files = [file for file in input_files if not fnmatch.fnmatch(file, '*_IT.i3.zst')] #Test                                                                              #CHECK
    #    input_files = [file for file in input_files if not fnmatch.fnmatch(file, '*_GCD.i3.zst')] #Test                                                                             #CHECK


    print('For loop done')
    #print(input_files)

    deployment_module_LE = I3InferenceModule(
        pulsemap=pulsemap,
        features=features,
        #pulsemap_extractor=I3FeatureExtractorIceCubeUpgrade(pulsemap=pulsemap),
        pulsemap_extractor=I3FeatureExtractorIceCubeDeepCore(pulsemap=pulsemap),
        model_config=model_config_LE,
        state_dict=state_dict_LE,
        gcd_file=gcd_file,
        prediction_columns=["3_model_pid_noise_pred_LE", "3_model_pid_muon_pred_LE", "3_model_pid_neutrino_pred_LE"],
        model_name="deployment",
    )
    
    #deployment_module_HE = I3InferenceModule(
    #    pulsemap=pulsemap,
    #    features=features,
    #    #pulsemap_extractor=I3FeatureExtractorIceCubeUpgrade(pulsemap=pulsemap),
    #    pulsemap_extractor=I3FeatureExtractorIceCubeDeepCore(pulsemap=pulsemap),
    #    model_config=model_config_HE,
    #    state_dict=state_dict_HE,
    #    gcd_file=gcd_file,
    #    #prediction_columns=["pid_muon_pred_HE", "pid_neutrino_pred_HE"],
    #    prediction_columns=["5_model_pid_noise_pred", "5_model_pid_neutrino_pred_LE", "5_model_pid_neutrino_pred_HE", "5_model_pid_muon_pred_LE", "5_model_pid_muon_pred_HE"],
    #    model_name="deployment",
    #)
    
    
    print('Defining deployer')
    # Construct I3 deployer
    deployer = GraphNeTI3Deployer(
        graphnet_modules=[deployment_module_LE],#, deployment_module_HE],#, filter],
        n_workers=1,
        gcd_file=gcd_file,
    )
    print('running deployer')
    # Start deployment - files will be written to output_folder
    deployer.run(
        input_files=i3_file,
        output_folder=output_folder,
    )
    print('Jobs Done')
    
if __name__ == "__main__":
    if not has_icecube_package():
        Logger(log_folder=None).error(ERROR_MESSAGE_MISSING_ICETRAY)

    else:
        # Parse command-line arguments
        parser = ArgumentParser(
            description="""
Use GraphNeTI3Modules to deploy trained model with GraphNeTI3Deployer.
"""
        )

        args, unknown = parser.parse_known_args()

        # Run example script
        main()

