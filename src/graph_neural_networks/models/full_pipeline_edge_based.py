from pytorch_lightning import LightningModule
import torch
from torch import nn

from graph_neural_networks.models.edge_query import EdgeQuery
from graph_neural_networks.reconstruction.reconstruction_method import PathReconstructionMethod

class FullPipelineEdgeBased(LightningModule):
    def __init__(
        self,
        image_segmentator: nn.Module,
        edge_query: EdgeQuery,
        path_reconstruction_method: PathReconstructionMethod,
        path_encoder: nn.Module,
        classifier: nn.Module,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            image_segmentator: The module used to segment images into graphs, get the feature maps, and the probability map.
            edge_query: The module used to query edges from the graph.
            path_reconstruction_method: The method used to reconstruct paths centerlines between nodes.
            path_encoder: The module used to encode the reconstructed paths.
            classifier: The module used to classify edges based on the encoded paths.
            *args: Additional positional arguments to pass to the superclass.
            **kwargs: Additional keyword arguments to pass to the superclass.
        """

        super().__init__(*args, **kwargs)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["image_segmentator", "edge_query", "path_reconstruction_method", "path_encoder", "classifier"])

        self.image_segmentator = image_segmentator
        self.edge_query = edge_query
        self.path_reconstruction_method = path_reconstruction_method
        self.path_encoder = path_encoder
        self.classifier = classifier

    def forward(self, 
                # FIXME: add arguments
                ) -> torch.Tensor:
        pass
        # TODO

    def model_step(self, 
                   # FIXME: add arguments
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        pass
        # TODO
    
    def predict_step(self,
                     # FIXME: add arguments
                     ):
        pass
        # TODO