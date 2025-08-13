#!/usr/bin/env python3
"""
Text Vectorization with AWS S3 Storage Sample

This script demonstrates how to:
1. Vectorize text input using sentence transformers
2. Store vectors in AWS S3 bucket
3. Retrieve and search vectors from S3

Features:
- Text embedding using pre-trained models
- Vector storage in S3 with metadata
- Basic similarity search functionality
- Error handling and logging
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import numpy as np
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


class VectorTextProcessor:
    """
    Handles text vectorization and S3 storage operations.

    This class provides functionality to convert text to vectors using
    sentence transformers and store/retrieve them from AWS S3.
    """

    def __init__(self):
        """Initialize the processor with AWS and model configurations."""
        # Load environment variables
        load_dotenv()

        # AWS Configuration
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.bucket_name = os.getenv("AWS_S3_BUCKET_NAME")

        # Validate required environment variables
        self._validate_config()

        # Initialize AWS S3 client
        self.s3_client = self._init_s3_client()

        # Vector Configuration
        self.model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.vector_dimension = int(os.getenv("VECTOR_DIMENSION", 384))

        # Initialize sentence transformer model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        logger.info("VectorTextProcessor initialized successfully")

    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required_vars = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_S3_BUCKET_NAME",
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

    def _init_s3_client(self) -> boto3.client:
        """Initialize and return AWS S3 client."""
        try:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region,
            )

            # Test connection by listing buckets
            s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Successfully connected to S3 bucket: {self.bucket_name}")

            return s3_client

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                raise ValueError(f"S3 bucket '{self.bucket_name}' not found")
            elif error_code == "403":
                raise ValueError(f"Access denied to S3 bucket '{self.bucket_name}'")
            else:
                raise ValueError(f"S3 client initialization failed: {e}")
        except BotoCoreError as e:
            raise ValueError(f"AWS configuration error: {e}")

    def vectorize_text(self, text: str) -> np.ndarray:
        """
        Convert text to vector representation.

        Args:
            text: Input text to vectorize

        Returns:
            Vector representation as numpy array

        Raises:
            ValueError: If text is empty or vectorization fails
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        try:
            logger.info(f"Vectorizing text: '{text[:50]}...'")
            vector = self.model.encode([text.strip()])[0]

            # Ensure vector has expected dimension
            if len(vector) != self.vector_dimension:
                logger.warning(
                    f"Vector dimension mismatch: expected {self.vector_dimension}, "
                    f"got {len(vector)}"
                )

            return vector

        except Exception as e:
            raise ValueError(f"Text vectorization failed: {e}")

    def store_vector(
        self, text: str, vector: np.ndarray, document_id: Optional[str] = None
    ) -> str:
        """
        Store vector in S3 with metadata.

        Args:
            text: Original text
            vector: Vector representation
            document_id: Optional document identifier

        Returns:
            S3 object key for the stored vector

        Raises:
            ValueError: If storage operation fails
        """
        # Generate document ID if not provided
        if not document_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            document_id = f"doc_{timestamp}_{hash(text) % 10000:04d}"

        # Prepare vector data with metadata
        vector_data = {
            "document_id": document_id,
            "text": text,
            "vector": vector.tolist(),
            "vector_dimension": len(vector),
            "model_name": self.model_name,
            "created_at": datetime.now().isoformat(),
            "text_length": len(text),
        }

        # S3 object key
        s3_key = f"vectors/{document_id}.json"

        try:
            # Store in S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json.dumps(vector_data, indent=2),
                ContentType="application/json",
                Metadata={
                    "document-id": document_id,
                    "vector-dimension": str(len(vector)),
                    "model-name": self.model_name,
                },
            )

            logger.info(f"Vector stored successfully: s3://{self.bucket_name}/{s3_key}")
            return s3_key

        except ClientError as e:
            raise ValueError(f"Failed to store vector in S3: {e}")

    def retrieve_vector(self, document_id: str) -> Optional[Dict]:
        """
        Retrieve vector and metadata from S3.

        Args:
            document_id: Document identifier

        Returns:
            Dictionary containing vector data and metadata, or None if not found
        """
        s3_key = f"vectors/{document_id}.json"

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)

            vector_data = json.loads(response["Body"].read().decode("utf-8"))

            # Convert vector back to numpy array
            vector_data["vector"] = np.array(vector_data["vector"])

            logger.info(f"Vector retrieved successfully: {document_id}")
            return vector_data

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.warning(f"Vector not found: {document_id}")
                return None
            else:
                raise ValueError(f"Failed to retrieve vector from S3: {e}")

    def list_stored_vectors(self) -> List[str]:
        """
        List all stored vector document IDs.

        Returns:
            List of document IDs
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix="vectors/"
            )

            if "Contents" not in response:
                return []

            document_ids = []
            for obj in response["Contents"]:
                key = obj["Key"]
                if key.endswith(".json"):
                    document_id = Path(key).stem
                    document_ids.append(document_id)

            logger.info(f"Found {len(document_ids)} stored vectors")
            return document_ids

        except ClientError as e:
            raise ValueError(f"Failed to list vectors from S3: {e}")

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if norm_product == 0:
            return 0.0

        return dot_product / norm_product

    def find_similar_texts(
        self, query_text: str, top_k: int = 5
    ) -> List[Tuple[str, float, str]]:
        """
        Find most similar texts to the query.

        Args:
            query_text: Text to find similarities for
            top_k: Number of top results to return

        Returns:
            List of tuples (document_id, similarity_score, original_text)
        """
        # Vectorize query text
        query_vector = self.vectorize_text(query_text)

        # Get all stored vectors
        document_ids = self.list_stored_vectors()

        if not document_ids:
            logger.warning("No stored vectors found for similarity search")
            return []

        similarities = []

        for doc_id in document_ids:
            vector_data = self.retrieve_vector(doc_id)
            if vector_data:
                similarity = self.cosine_similarity(query_vector, vector_data["vector"])
                similarities.append((doc_id, similarity, vector_data["text"]))

        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results
        return similarities[:top_k]


def main():
    """
    Demonstration of the VectorTextProcessor functionality.
    """
    try:
        # Initialize processor
        processor = VectorTextProcessor()

        # Sample texts for demonstration
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "AWS S3 provides scalable object storage in the cloud.",
            "Vector databases enable efficient similarity search.",
            "Python is a popular programming language for data science.",
        ]

        print("=== Text Vectorization and S3 Storage Demo ===\n")

        # Process and store sample texts
        print("1. Vectorizing and storing sample texts:")
        stored_ids = []

        for i, text in enumerate(sample_texts):
            print(f"   Processing: {text}")

            # Vectorize text
            vector = processor.vectorize_text(text)
            print(f"   Vector shape: {vector.shape}")

            # Store in S3
            doc_id = f"sample_{i+1}"
            s3_key = processor.store_vector(text, vector, doc_id)
            stored_ids.append(doc_id)
            print(f"   Stored as: {s3_key}\n")

        # List stored vectors
        print("2. Listing stored vectors:")
        all_vectors = processor.list_stored_vectors()
        for doc_id in all_vectors:
            print(f"   - {doc_id}")
        print()

        # Retrieve a specific vector
        print("3. Retrieving a specific vector:")
        if stored_ids:
            sample_id = stored_ids[0]
            retrieved_data = processor.retrieve_vector(sample_id)
            if retrieved_data:
                print(f"   Document ID: {retrieved_data['document_id']}")
                print(f"   Original text: {retrieved_data['text']}")
                print(f"   Vector dimension: {retrieved_data['vector_dimension']}")
                print(f"   Created at: {retrieved_data['created_at']}")
        print()

        # Similarity search
        print("4. Similarity search:")
        query = "artificial intelligence and machine learning"
        print(f"   Query: {query}")

        similar_texts = processor.find_similar_texts(query, top_k=3)

        if similar_texts:
            print("   Most similar texts:")
            for i, (doc_id, score, text) in enumerate(similar_texts, 1):
                print(f"   {i}. [{doc_id}] Similarity: {score:.4f}")
                print(f"      Text: {text}")
        else:
            print("   No similar texts found.")

        print("\n=== Demo completed successfully! ===")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
