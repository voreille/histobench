from .moco import (
    MoCoSuperpixel,
    MoCoSuperpixelCluster,
    MoCoSuperpixelClusterBioptimus,
    MoCoV2Encoder,
)

model_cls_map = {
    "moco_v2": MoCoV2Encoder,
    "moco_superpixel": MoCoSuperpixel,
    "moco_superpixel_cluster": MoCoSuperpixelCluster,
    "moco_superpixel_cluster_bioptimus": MoCoSuperpixelClusterBioptimus,
}


def get_model(device, model_type="moco_superpixel_cluster_bioptimus", inference_only=False):
    init_queue, init_cluster_ids = None, None

    model_cls = model_cls_map[model_type]

    common_kwargs = dict(
        base_encoder="resnet50",
        output_dim=128,
        queue_size=2560,  # 32768
        momentum=0.9,
        temperature=0.7,
        init_queue=init_queue,
    )

    if model_type == "moco_superpixel_cluster":
        return model_cls(**common_kwargs, num_clusters=50, device=device)
    elif model_type == "moco_superpixel_cluster_bioptimus":
        return model_cls(**common_kwargs, init_cluster_ids=init_cluster_ids)
    else:
        return model_cls(**common_kwargs)
