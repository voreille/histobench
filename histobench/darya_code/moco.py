import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)

from .clustering import ClusterNegativeMiner, ClusterNSMoCo

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_resnet_weights(base_encoder):
    resnet_weights_map = {
        "resnet18": ResNet18_Weights.DEFAULT,
        "resnet34": ResNet34_Weights.DEFAULT,
        "resnet50": ResNet50_Weights.DEFAULT,
        "resnet101": ResNet101_Weights.DEFAULT,
        "resnet152": ResNet152_Weights.DEFAULT,
    }
    return resnet_weights_map.get(base_encoder.lower(), None)


def build_resnet_with_projection(base_encoder, output_dim, pretrained=False):
    weights = get_resnet_weights(base_encoder) if pretrained else None
    encoder = getattr(models, base_encoder)(weights=weights)
    hidden_dim = encoder.fc.in_features
    encoder.fc = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
    )
    return encoder


class MoCoV2Encoder(nn.Module):
    def __init__(
        self,
        base_encoder="resnet50",
        output_dim=128,
        queue_size=65536,
        momentum=0.999,
        temperature=0.07,
        init_queue=None,
    ):
        super().__init__()
        self.encoder_q = build_resnet_with_projection(base_encoder, output_dim)
        self.encoder_k = build_resnet_with_projection(base_encoder, output_dim)

        self.temperature = temperature
        self.momentum = momentum

        self.register_buffer(
            "queue", torch.randn(queue_size, output_dim) if init_queue is None else init_queue
        )
        if init_queue is None:
            self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self._initialize_momentum_encoder()

    def _initialize_momentum_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        end_ptr = ptr + batch_size

        if end_ptr <= self.queue.size(0):
            self.queue[ptr:end_ptr, :] = keys
        else:
            first = self.queue.size(0) - ptr
            self.queue[ptr:] = keys[:first]
            self.queue[: end_ptr % self.queue.size(0)] = keys[first:]

        self.queue_ptr[0] = (ptr + batch_size) % self.queue.size(0)

    def forward(self, x_q, x_k):
        q = F.normalize(self.encoder_q(x_q), dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = F.normalize(self.encoder_k(x_k), dim=1)
        return q, k

    def update_queue(self, keys):
        self._dequeue_and_enqueue(keys)


class MoCoSuperpixel(MoCoV2Encoder):
    def forward(self, x_q, x_k1, x_k2):
        q = F.normalize(self.encoder_q(x_q), dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k1 = F.normalize(self.encoder_k(x_k1), dim=1)
            k2 = F.normalize(self.encoder_k(x_k2), dim=1)
        return q, k1, k2

    def update_queue(self, keys, neighbor_keys):
        combined_keys = torch.cat([keys, neighbor_keys], dim=0)
        self._dequeue_and_enqueue(combined_keys)


class MoCoSuperpixelCluster(MoCoV2Encoder):
    def __init__(
        self,
        base_encoder="resnet50",
        output_dim=128,
        queue_size=65536,
        momentum=0.999,
        temperature=0.07,
        init_queue=None,
        num_clusters=50,
        device="cuda",
    ):
        super().__init__(base_encoder, output_dim, queue_size, momentum, temperature)
        self.cluster_helper = ClusterNSMoCo(num_clusters, output_dim, device)
        if init_queue is not None:
            self.register_buffer("queue", init_queue)
        else:
            self.register_buffer("queue", F.normalize(torch.randn(queue_size, output_dim), dim=1))
        self.register_buffer("queue_cluster_ids", torch.zeros(queue_size, dtype=torch.long))

    def forward(self, x_q, x_k1, x_k2, step=None, update_step=None):
        q = F.normalize(self.encoder_q(x_q), dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k1 = F.normalize(self.encoder_k(x_k1), dim=1)
            k2 = F.normalize(self.encoder_k(x_k2), dim=1)

        if (
            step is not None
            and update_step is not None
            and self.queue.shape[0] > self.cluster_helper.num_clusters
        ):
            self.cluster_helper.update_centroids(self.queue, step, update_step)

        if self.cluster_helper.initialized:
            q_cluster_ids, _ = self.cluster_helper.assign_to_centroids(q)
            k1_cluster_ids, _ = self.cluster_helper.assign_to_centroids(k1)
            k2_cluster_ids, _ = self.cluster_helper.assign_to_centroids(k2)

            negs = self.cluster_helper.get_negatives_by_cluster(q, self.queue)
            false_negatives_em = negs["false_negative_embeddings"]
            hard_negatives_em = negs["hard_negative_embeddings"]
            false_negatives_in = negs["false_negative_indices"]
            hard_negatives_in = negs["hard_negative_indices"]

        else:
            q_cluster_ids = torch.zeros(q.size(0), dtype=torch.long, device=self.device)
            k1_cluster_ids = torch.zeros(q.size(0), dtype=torch.long, device=self.device)
            k2_cluster_ids = torch.zeros(q.size(0), dtype=torch.long, device=self.device)
            false_negatives_em = [torch.tensor([], dtype=torch.long, device=self.device)] * q.size(
                0
            )
            hard_negatives_em = [torch.tensor([], dtype=torch.long, device=self.device)] * q.size(
                0
            )

        return (
            q,
            k1,
            k2,
            false_negatives_em,
            hard_negatives_em,
            false_negatives_in,
            hard_negatives_in,
            q_cluster_ids,
            k1_cluster_ids,
            k2_cluster_ids,
            self.queue,
        )

    def update_queue(self, keys, neighbor_keys, keys_id, neighbor_keys_id):
        combined_keys = torch.cat([keys, neighbor_keys], dim=0)
        combined_ids = torch.cat([keys_id, neighbor_keys_id], dim=0)
        ptr = int(self.queue_ptr)
        total = combined_keys.size(0)
        end_ptr = ptr + total

        if end_ptr <= self.queue.size(0):
            self.queue[ptr:end_ptr, :] = combined_keys
            self.queue_cluster_ids[ptr:end_ptr] = combined_ids
        else:
            first = self.queue.size(0) - ptr
            self.queue[ptr:] = combined_keys[:first]
            self.queue[: end_ptr % self.queue.size(0)] = combined_keys[first:]
            self.queue_cluster_ids[ptr:] = combined_ids[:first]
            self.queue_cluster_ids[: end_ptr % self.queue.size(0)] = combined_ids[first:]

        self.queue_ptr[0] = (ptr + total) % self.queue.size(0)


class MoCoSuperpixelClusterBioptimus(MoCoV2Encoder):
    def __init__(
        self,
        base_encoder="resnet50",
        output_dim=128,
        queue_size=65536,
        momentum=0.999,
        temperature=0.07,
        init_queue=None,
        init_cluster_ids=None,
    ):
        super().__init__(base_encoder, output_dim, queue_size, momentum, temperature)
        # self.device = device
        # self.register_buffer("queue_cluster_ids", torch.zeros(queue_size, 3, dtype=torch.long))
        # Handle optional queue initialization
        if init_queue is not None:
            self.register_buffer("queue", init_queue)
        else:
            self.register_buffer("queue", F.normalize(torch.randn(queue_size, output_dim), dim=1))

        # Initialize queue_cluster_ids
        if init_cluster_ids is not None:
            self.register_buffer("queue_cluster_ids", init_cluster_ids)
        else:
            self.register_buffer("queue_cluster_ids", torch.zeros(queue_size, 3, dtype=torch.long))

    def forward(self, x_q, x_k1, x_k2, q_cluster_ids):  # k1_cluster_ids, k2_cluster_ids
        q = F.normalize(self.encoder_q(x_q), dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k1 = F.normalize(self.encoder_k(x_k1), dim=1)
            k2 = F.normalize(self.encoder_k(x_k2), dim=1)

        false_negatives_em = []
        hard_negatives_em = []
        # false_negatives_in = []
        # hard_negatives_in = []

        if self.queue.shape[0] == self.queue_cluster_ids.shape[0]:
            self.cluster_helper = ClusterNegativeMiner(self.queue, self.queue_cluster_ids)

            negs = self.cluster_helper.get_negatives_by_cluster(q, q_cluster_ids)
            false_negatives_em = negs["false_negative_embeddings"]
            hard_negatives_em = negs["queue_without_fn"]
            # false_negatives_in = negs["false_negative_indices"]
            # hard_negatives_in = negs["hard_negative_indices"]

        return q, k1, k2, false_negatives_em, hard_negatives_em, self.queue

    def update_queue(self, keys, neighbor_keys, keys_cluster_ids, neighbor_keys_cluster_ids):
        """
        Update memory queue with keys and neighbor keys along with their cluster IDs.

        Args:
            keys: Tensor (B, D)
            neighbor_keys: Tensor (B, D)
            keys_cluster_ids: Tensor (B, 3) for [primary, second, third]
            neighbor_keys_cluster_ids: Tensor (B, 3)
        """
        combined_keys = torch.cat([keys, neighbor_keys], dim=0)  # (2B, D)
        combined_cluster_ids = torch.cat(
            [keys_cluster_ids, neighbor_keys_cluster_ids], dim=0
        )  # (2B, 3)

        ptr = int(self.queue_ptr)
        total = combined_keys.size(0)
        end_ptr = ptr + total

        if end_ptr <= self.queue.size(0):
            self.queue[ptr:end_ptr, :] = combined_keys
            self.queue_cluster_ids[ptr:end_ptr, :] = combined_cluster_ids
        else:
            first = self.queue.size(0) - ptr
            self.queue[ptr:, :] = combined_keys[:first]
            self.queue[: end_ptr % self.queue.size(0), :] = combined_keys[first:]

            self.queue_cluster_ids[ptr:, :] = combined_cluster_ids[:first]
            self.queue_cluster_ids[: end_ptr % self.queue.size(0), :] = combined_cluster_ids[
                first:
            ]

        self.queue_ptr[0] = (ptr + total) % self.queue.size(0)
