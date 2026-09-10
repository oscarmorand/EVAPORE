from path_neural_networks.models.losses.bce_with_logits_loss import BCEWithLogitsLoss
from path_neural_networks.models.losses.weighted_bce_with_logits_loss import WeightedBCEWithLogitsLoss

available_path_classification_losses: dict = {
    "BCEWithLogitsLoss": BCEWithLogitsLoss,
    "WeightedBCEWithLogitsLoss": WeightedBCEWithLogitsLoss
}