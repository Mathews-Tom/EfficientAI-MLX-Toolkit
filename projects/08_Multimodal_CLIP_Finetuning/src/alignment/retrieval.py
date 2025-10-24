#!/usr/bin/env python3
"""
Image-text retrieval utilities.

Provides bidirectional retrieval (image→text and text→image) using CLIP embeddings
and similarity computation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union

import torch
from PIL import Image

from alignment.similarity import SimilarityComputer

if TYPE_CHECKING:
    from model import CLIPFinetuningController

logger = logging.getLogger(__name__)


class ImageTextRetrieval:
    """Bidirectional image-text retrieval.

    Uses CLIP model to encode queries and candidates, then performs
    similarity-based retrieval in both directions (image→text and text→image).

    Attributes:
        model: CLIP fine-tuning controller for encoding
        similarity_computer: Similarity computer for scoring
    """

    def __init__(
        self,
        model: CLIPFinetuningController,
        similarity_computer: SimilarityComputer,
    ) -> None:
        """Initialize retrieval system.

        Args:
            model: CLIP fine-tuning controller for encoding
            similarity_computer: Similarity computer for scoring

        Raises:
            RuntimeError: If model not initialized
        """
        if model.model is None or model.processor is None:
            raise RuntimeError("Model not initialized. Call model.setup() first.")

        self.model = model
        self.similarity_computer = similarity_computer

        logger.info("Initialized ImageTextRetrieval")

    def retrieve_text(
        self,
        image: Union[Image.Image, torch.Tensor],
        texts: list[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Retrieve top-k text matches for an image.

        Args:
            image: Query image (PIL Image or pre-computed embedding)
            texts: Candidate text strings
            top_k: Number of top results to return

        Returns:
            List of (text, similarity_score) tuples, sorted by similarity

        Raises:
            ValueError: If no texts provided or top_k invalid
        """
        if not texts:
            raise ValueError("No texts provided for retrieval")

        if top_k <= 0 or top_k > len(texts):
            raise ValueError(f"top_k must be in range [1, {len(texts)}], got {top_k}")

        # Encode image
        if isinstance(image, Image.Image):
            image_embeds = self.model.encode_image([image])
        else:
            # Assume pre-computed embedding
            if image.dim() == 1:
                image_embeds = image.unsqueeze(0)
            else:
                image_embeds = image

        # Encode texts
        text_embeds = self.model.encode_text(texts)

        # Compute similarity
        similarity = self.similarity_computer.compute(image_embeds, text_embeds)

        # Get top-k
        top_k_values, top_k_indices = self.similarity_computer.top_k(
            similarity, k=top_k, dim=1
        )

        # Convert to list of (text, score) tuples
        results = []
        for idx, score in zip(top_k_indices[0].tolist(), top_k_values[0].tolist()):
            results.append((texts[idx], score))

        return results

    def retrieve_image(
        self,
        text: str,
        images: Union[list[Image.Image], torch.Tensor],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Retrieve top-k image matches for text.

        Args:
            text: Query text string
            images: Candidate images (list of PIL Images or pre-computed embeddings)
            top_k: Number of top results to return

        Returns:
            List of (image_index, similarity_score) tuples, sorted by similarity

        Raises:
            ValueError: If no images provided or top_k invalid
        """
        # Determine number of images
        if isinstance(images, list):
            num_images = len(images)
            if num_images == 0:
                raise ValueError("No images provided for retrieval")
        else:
            num_images = images.size(0)

        if top_k <= 0 or top_k > num_images:
            raise ValueError(f"top_k must be in range [1, {num_images}], got {top_k}")

        # Encode text
        text_embeds = self.model.encode_text([text])

        # Encode images
        if isinstance(images, list):
            image_embeds = self.model.encode_image(images)
        else:
            # Assume pre-computed embeddings
            image_embeds = images

        # Compute similarity
        similarity = self.similarity_computer.compute(image_embeds, text_embeds)

        # Get top-k (across images, dim=0)
        top_k_values, top_k_indices = self.similarity_computer.top_k(
            similarity, k=top_k, dim=0
        )

        # Convert to list of (image_idx, score) tuples
        results = []
        for idx, score in zip(top_k_indices.tolist(), top_k_values.tolist()):
            results.append((idx[0], score[0]))

        return results

    def batch_retrieve(
        self,
        queries: list[Union[str, Image.Image]],
        candidates: list[Union[str, Image.Image]],
        top_k: int = 5,
    ) -> dict[int, list[tuple[int, float]]]:
        """Batch retrieval for efficiency.

        Args:
            queries: List of query items (text strings or images)
            candidates: List of candidate items (text strings or images)
            top_k: Number of top results per query

        Returns:
            Dictionary mapping query_idx -> [(candidate_idx, score), ...]

        Raises:
            ValueError: If queries or candidates are empty or have incompatible types
        """
        if not queries or not candidates:
            raise ValueError("Queries and candidates must not be empty")

        if top_k <= 0 or top_k > len(candidates):
            raise ValueError(
                f"top_k must be in range [1, {len(candidates)}], got {top_k}"
            )

        # Determine query and candidate types
        query_is_text = isinstance(queries[0], str)
        candidate_is_text = isinstance(candidates[0], str)

        # Encode all queries
        if query_is_text:
            query_embeds = self.model.encode_text(queries)
        else:
            # Assume images
            query_embeds = self.model.encode_image(queries)

        # Encode all candidates
        if candidate_is_text:
            candidate_embeds = self.model.encode_text(candidates)
        else:
            # Assume images
            candidate_embeds = self.model.encode_image(candidates)

        # Compute similarity matrix [N_queries, N_candidates]
        similarity = self.similarity_computer.compute(query_embeds, candidate_embeds)

        # Get top-k for each query
        top_k_values, top_k_indices = self.similarity_computer.top_k(
            similarity, k=top_k, dim=1
        )

        # Convert to dictionary
        results = {}
        for query_idx in range(len(queries)):
            query_results = []
            for candidate_idx, score in zip(
                top_k_indices[query_idx].tolist(),
                top_k_values[query_idx].tolist(),
            ):
                query_results.append((candidate_idx, score))
            results[query_idx] = query_results

        return results

    def retrieve_mutual_nn(
        self,
        images: list[Image.Image],
        texts: list[str],
    ) -> list[tuple[int, int, float]]:
        """Find mutual nearest neighbors between images and texts.

        A mutual nearest neighbor is a pair (image_idx, text_idx) where
        the image is the nearest neighbor of the text AND the text is
        the nearest neighbor of the image.

        Args:
            images: List of images
            texts: List of texts

        Returns:
            List of (image_idx, text_idx, avg_similarity) tuples
        """
        if not images or not texts:
            raise ValueError("Images and texts must not be empty")

        # Encode images and texts
        image_embeds = self.model.encode_image(images)
        text_embeds = self.model.encode_text(texts)

        # Compute similarity matrix [N_images, N_texts]
        similarity = self.similarity_computer.compute(image_embeds, text_embeds)

        # Find nearest neighbors in both directions
        i2t_nn = torch.argmax(similarity, dim=1)  # [N_images]
        t2i_nn = torch.argmax(similarity, dim=0)  # [N_texts]

        # Find mutual nearest neighbors
        mutual_nn = []
        for image_idx in range(len(images)):
            text_idx = i2t_nn[image_idx].item()
            if t2i_nn[text_idx].item() == image_idx:
                # Mutual nearest neighbor found
                avg_similarity = (
                    similarity[image_idx, text_idx].item()
                    + similarity[image_idx, text_idx].item()
                ) / 2.0
                mutual_nn.append((image_idx, text_idx, avg_similarity))

        return mutual_nn

    def __repr__(self) -> str:
        """String representation of the retrieval system."""
        return (
            f"ImageTextRetrieval(model={self.model.config.model_name}, "
            f"similarity={self.similarity_computer})"
        )
