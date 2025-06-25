import logging

import torch
from torchvision import transforms

from histobench.darya_code.get_model_inference import get_model

logger = logging.getLogger(__name__)


def load_pretrained_encoder(model_name, weights_path, device):
    moco_model = get_model(
        device,
        model_type=model_name,
        inference_only=True,
    )

    # Load full MoCo state dict
    state_dict = torch.load(weights_path, map_location="cpu")["model_state_dict"]

    # Filter and rename keys that belong to encoder_q
    encoder_q_state_dict = {
        k.replace("encoder_q.", ""): v
        for k, v in state_dict.items()
        if k.startswith("encoder_q.")
    }

    # Load into encoder_q only
    encoder_q = moco_model.encoder_q
    encoder_q.load_state_dict(encoder_q_state_dict, strict=True)

    # Remove final classification layer
    encoder_q.fc = torch.nn.Identity()

    encoder_q.to(device)
    encoder_q.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    logger.info(f"Loaded encoder_q from {weights_path} (model: {model_name})")
    logger.info(f"Preprocessing pipeline: {preprocess}")
    return encoder_q, preprocess
