from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import Logger
from graphnet.data.extractors.i3extractor import I3Extractor


if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray, dataio  

#class I3DeploymentExtractor(I3Extractor):
#
#    def __init__(self, name: str = "deployment_predictions"):
#        """Construct I3DeploymentExtractor."""
#        # Base class constructor
#        super().__init__(name)
#
#    #def __call__(self, frame, padding_value=-1) -> dict:
#    def __call__(self, frame, padding_value=-1) -> dict:
#
#        results = {}
#        keys = ['deployment_neutrino_pred', 'deployment_noise_pred', 'deployment_muon_pred']
#        for key in keys:
#            if key in frame.keys():
#                results[key] = frame[key].value
#            else:
#                results[key] = padding_value
#        return results
    
    
class I3DeploymentExtractor(I3Extractor):

    def __init__(self, name: str = "deployment_predictions"):
        """Construct I3DeploymentExtractor."""
        # Base class constructor
        super().__init__(name)

    #def __call__(self, frame, padding_value=-1) -> dict:
    def __call__(self, frame, padding_value=0) -> dict:

        results = {}
        keys = ['deployment_pid_neutrino_pred_LE', 'deployment_pid_noise_pred_LE', 'deployment_pid_muon_pred_LE', 'deployment_pid_neutrino_pred_HE', 'deployment_pid_muon_pred_HE']
        for key in keys:
            if key in frame.keys():
                results[key] = frame[key].value
            else:
                results[key] = padding_value
        return results