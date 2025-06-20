import torch
from torchvision import transforms

from histobench.darya_code.get_model_inference import get_model


def load_pretrained_encoder(model_name, weights_paths, device):
    moco_model = get_model(
        device,
        model_type=model_name,
        inference_only=True,
    )
    state_dict = torch.load(weights_paths, map_location=device)
    moco_model.load_state_dict(state_dict, strict=False)
    encoder = moco_model.encoder_q
    encoder.fc = torch.nn.Identity()  # Remove the final classification layer
    encoder.eval()
    preprocess = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return encoder, preprocess
