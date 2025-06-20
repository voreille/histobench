from typing import Dict, List, Tuple

from sklearn.cluster import KMeans
import torch


class ClusterNSMoCo:
    def __init__(self, num_clusters=100, embedding_dim=128, device="cuda"):
        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim
        self.device = device
        self.centroids = None
        self.memory_bank_cluster_ids = None
        self.initialized = False

    def update_centroids(self, memory_bank: torch.Tensor, step: int, update_step: int):
        """
        Cluster the memory bank using cosine similarity (via normalized vectors).
        This will initialize once, and re-cluster every 'update_step' steps after that.
        """
        if memory_bank.shape[0] < self.num_clusters:
            return  # not enough samples to cluster

        if not self.initialized or (step % update_step == 0):
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
            kmeans.fit(memory_bank.cpu().numpy())

            self.centroids = torch.tensor(kmeans.cluster_centers_, device=self.device)
            self.centroids = torch.nn.functional.normalize(self.centroids, dim=1)
            self.memory_bank_cluster_ids = torch.tensor(kmeans.labels_, device=self.device)
            self.memory_bank_embeddings = memory_bank.clone().detach()
            self.initialized = True

    def assign_to_centroids(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign embeddings to the closest centroid and return sorted centroid indices.
        """
        sims = torch.matmul(embeddings, self.centroids.T)
        sorted_indices = torch.argsort(sims, dim=1, descending=True)
        cluster_ids = sorted_indices[:, 0]
        return cluster_ids, sorted_indices

    def get_negatives_by_cluster(
        self, queries: torch.Tensor, memory_bank: torch.Tensor
    ) -> Dict[str, List[torch.Tensor]]:
        """
        For each query:
        - return false negatives (same cluster) [both indices and embeddings]
        - return hard negatives (second closest cluster) [both indices and embeddings]
        - return assigned centroid
        - return all centroids and memory bank cluster assignments
        """
        sims = torch.matmul(queries, self.centroids.T)
        sorted_indices = torch.argsort(sims, dim=1, descending=True)

        results = {
            "false_negative_indices": [],
            "false_negative_embeddings": [],
            "hard_negative_indices": [],
            "hard_negative_embeddings": [],
            "query_centroids": [],
            "centroids": self.centroids,
            "memory_bank_cluster_ids": self.memory_bank_cluster_ids,
        }

        for i in range(queries.shape[0]):
            top1 = sorted_indices[i, 0].item()
            top2 = sorted_indices[i, 1].item()
            top3 = sorted_indices[i, 2].item()

            fn_idx = (self.memory_bank_cluster_ids == top1).nonzero(as_tuple=True)[0]
            hn2_idx = (self.memory_bank_cluster_ids == top2).nonzero(as_tuple=True)[0]
            hn3_idx = (self.memory_bank_cluster_ids == top3).nonzero(as_tuple=True)[0]
            hn_idx = torch.cat([hn2_idx, hn3_idx], dim=0)

            results["false_negative_indices"].append(fn_idx)
            results["false_negative_embeddings"].append(memory_bank[fn_idx])

            results["hard_negative_indices"].append(hn_idx)
            results["hard_negative_embeddings"].append(memory_bank[hn_idx])

            results["query_centroids"].append(self.centroids[top1])

        return results

    def get_memory_bank_clusters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            memory_bank_embeddings: (N, D)
            memory_bank_cluster_ids: (N,)
            centroids: (num_clusters, D)
        """
        return self.memory_bank_embeddings, self.memory_bank_cluster_ids, self.centroids


class ClusterNegativeMiner:
    def __init__(self, memory_bank: torch.Tensor, memory_bank_cluster_ids: torch.Tensor):
        """
        Args:
            memory_bank: Tensor of shape (K, D) with key embeddings.
            memory_bank_cluster_ids: Tensor of shape (K, 3) with cluster IDs:
                                     [:, 0] = primary, [:, 1] = second, [:, 2] = third
        """
        self.memory_bank = memory_bank  # (K, D)
        self.memory_bank_cluster_ids = memory_bank_cluster_ids  # (K, 3)

    def get_negatives_by_cluster(
        self, queries: torch.Tensor, query_cluster_ids: torch.Tensor
    ) -> Dict[str, List[torch.Tensor]]:
        """
        For each query:
        - return false negatives (same primary cluster)
        - return hard negatives (second or third cluster)
        Returns:
            dict with lists of indices and embeddings
        """
        results = {
            "false_negative_indices": [],
            "false_negative_embeddings": [],
            # "hard_negative_indices": [],
            # "hard_negative_embeddings": [],
            "queue_without_fn": [],
        }

        for i in range(queries.shape[0]):
            # q = queries[i]
            q_primary, q_second, q_third = query_cluster_ids[i]  # Scalars

            # Find indices in memory bank that match the query's primary cluster
            fn_idx = (self.memory_bank_cluster_ids[:, 0] == q_primary).nonzero(as_tuple=True)[0]

            full_idx = torch.arange(self.memory_bank.shape[0], device=self.memory_bank.device)
            keep_mask = torch.ones_like(full_idx, dtype=torch.bool)
            keep_mask[fn_idx] = False

            filtered_queue = self.memory_bank[keep_mask]
            results["queue_without_fn"].append(filtered_queue)

            # Find indices that match second OR third cluster
            # hn_idx_2 = (self.memory_bank_cluster_ids[:, 0] == q_second).nonzero(as_tuple=True)[0]
            # hn_idx_3 = (self.memory_bank_cluster_ids[:, 0] == q_third).nonzero(as_tuple=True)[0]
            # hn_idx = torch.cat([hn_idx_2, hn_idx_3], dim=0)

            results["false_negative_indices"].append(fn_idx)
            results["false_negative_embeddings"].append(self.memory_bank[fn_idx])

            # results["hard_negative_indices"].append(hn_idx)
            # results["hard_negative_embeddings"].append(self.memory_bank[hn_idx])

        return results
