"""Node definitions that crop the number of pulses in a pulsemap."""

from torch_geometric.data import Data
import torch
from torch import nn
from typing import Callable, Any, Optional, Tuple, List

from graphnet.models.graphs.nodes import NodeDefinition
#from graphnet.models.graphs.nodes.utils import fps
from graphnet.models.graphs.nodes import NodesAsPulses


class PulsesCroppedValue(NodeDefinition):
    """Represent each node as pulse with an upper limit of nodes."""

    @property
    def nb_outputs(self) -> int:
        """Return number of outputs."""
        return self.nb_inputs

    def __init__(
        self,
        max_pulses: int,
        transform: Callable[[torch.Tensor], torch.Tensor] = None,
        **kwargs: Any,
    ):
        """Construct `PulsesCroppedByCharge`.".

        Selects at most 'max_pulses' number of pulses, chosen as those with the smallest
         value after applying transform.

        Args:
            max_pulses: Maximal number of pulses allowed in a pulsemap.
             transform: Transform applied to the input tensor, before determining order.
            **kwargs: kwargs passed to NodeDefinitions constructor.
        """
        transform = transform or nn.Identity()
        super().__init__(**kwargs)  # noqa
        self.max_pulses = max_pulses
        self.transform = transform

    # abstract method(s)
    def _construct_nodes(
        self, x: torch.tensor#, include_sensor_id: bool = False
    ) -> Data:
        if x.shape[0] < self.max_pulses:
            return Data(x=x)
        return Data(
            x=x[
                torch.sort(self.transform(x), dim=0).indices,
                :,
            ]
        )


class PulsesCroppedRandomly(NodeDefinition):
    """Represent each node as pulse with an upper limit of nodes."""

    @property
    def nb_outputs(self) -> int:
        """Return number of outputs."""
        return self.nb_inputs

    def __init__(self, max_pulses: int, **kwargs: Any):
        """Construct 'PulsesCroppedRandomly'.

        Selects at most 'max_pulses' number of pulses, chosen randomly.

        Args:
            max_pulses: Maximal number of pulses allowed in the pulsemap.
            **kwargs: kwargs passed to NodeDefinitions constructor.
        """
        super().__init__(**kwargs)  # noqa
        self.max_pulses = max_pulses

    def _construct_nodes(
        self, x: torch.tensor#, include_sensor_id: bool = False
    ) -> Data:
        x, maybe_sensor_id = self._maybe_split_sensor_id(x)#, include_sensor_id)
        if x.shape[0] < self.max_pulses:
            data = Data(x=x)
            self._maybe_add_sensor_id(data, maybe_sensor_id)
            return data
        shuffled_indices = torch.randperm(x.shape[0])
        data = Data(x=x[shuffled_indices])
        data = self._maybe_add_sensor_id(
            data, shuffled_indices, maybe_sensor_id
        )
        return data

    def _maybe_add_sensor_id(  #
        self,
        data: Data,
        maybe_sensor_id: Optional[torch.Tensor] = None,
        shuffled_indices: Optional[torch.Tensor] = None,
    ) -> Data:
        if maybe_sensor_id is not None:
            if shuffled_indices is not None:
                data.sensor_id = maybe_sensor_id[shuffled_indices]
            else:
                data.sensor_id = maybe_sensor_id
        return data


class CroppedFPSNodes(NodeDefinition):
    def __init__(
        self,
        max_length: int,
        fps_features: Optional[List[int]] = None,
        start_idx: Optional[int] = None,
    ):
        super().__init__()
        self.max_length = max_length
        self.fps_features = fps_features
        self.start_idx = start_idx

    @property
    def nb_outputs(self) -> int:
        """Return number of outputs."""
        return self.nb_inputs

    def _construct_nodes(
        self, x: torch.Tensor#, include_sensor_id: bool = False
    ) -> Tuple[Data]:
        if x.shape[0] < self.max_length:
            return Data(x=x)

        fps_features = self.fps_features or list(range(x.shape[1]))
        arr = x.numpy()
        sample_indices = fps(
            arr[:, fps_features], self.max_length, self.start_idx
        )
        return Data(x=x[sample_indices])


class MaxNodesAsPulses(NodesAsPulses):
    """Represent each measured pulse of Cherenkov Radiation as a node."""

    def __init__(
        self,
        max_length: int = 256,
        input_feature_names: Optional[List[str]] = None,
    ) -> None:
        """Construct `MaxNodesAsPulses`.

        Args:
            max_length: Maximum number of pulses to keep.
            input_feature_names: (Optional) column names for input features.
        """
        self.max_length = max_length
        # Base class constructor
        super().__init__(input_feature_names=input_feature_names)

    def _construct_nodes(
        self, x: torch.Tensor#, include_sensor_id: bool = False
    ) -> Tuple[Data, List[str]]:
        return super()._construct_nodes(x=x[: self.max_length])#, include_sensor_id=include_sensor_id)