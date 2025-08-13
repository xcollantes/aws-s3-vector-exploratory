#!/usr/bin/env python3
"""
Basic test for the vector text sample functionality.

This test verifies that the VectorTextProcessor can:
1. Initialize properly with mock credentials
2. Vectorize text correctly
3. Handle similarity calculations
"""

import os
import sys
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

# Add the current directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_vectorization_without_aws():
    """
    Test text vectorization functionality without requiring AWS credentials.
    """
    print("Testing text vectorization (offline mode)...")
    
    # Mock environment variables
    test_env = {
        'AWS_ACCESS_KEY_ID': 'test_key',
        'AWS_SECRET_ACCESS_KEY': 'test_secret',
        'AWS_REGION': 'us-east-1',
        'AWS_S3_BUCKET_NAME': 'test-bucket',
        'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
        'VECTOR_DIMENSION': '384'
    }
    
    with patch.dict(os.environ, test_env):
        with patch('boto3.client') as mock_boto3:
            # Mock S3 client
            mock_s3_client = MagicMock()
            mock_s3_client.head_bucket.return_value = None
            mock_boto3.return_value = mock_s3_client
            
            try:
                from vector_text_sample import VectorTextProcessor
                
                # Initialize processor
                processor = VectorTextProcessor()
                
                # Test text vectorization
                test_text = "This is a test sentence for vectorization."
                vector = processor.vectorize_text(test_text)
                
                # Verify vector properties
                assert isinstance(vector, np.ndarray), "Vector should be numpy array"
                assert len(vector) == 384, f"Expected vector dimension 384, got {len(vector)}"
                assert not np.all(vector == 0), "Vector should not be all zeros"
                
                print(f"✓ Text vectorized successfully: {test_text}")
                print(f"✓ Vector shape: {vector.shape}")
                print(f"✓ Vector norm: {np.linalg.norm(vector):.4f}")
                
                # Test similarity calculation
                test_text2 = "This is another test sentence for comparison."
                vector2 = processor.vectorize_text(test_text2)
                
                similarity = processor.cosine_similarity(vector, vector2)
                
                assert 0 <= similarity <= 1, f"Similarity should be between 0 and 1, got {similarity}"
                
                print(f"✓ Similarity calculation works: {similarity:.4f}")
                
                # Test with identical texts
                similarity_identical = processor.cosine_similarity(vector, vector)
                assert abs(similarity_identical - 1.0) < 1e-6, "Identical vectors should have similarity ~1.0"
                
                print(f"✓ Identical text similarity: {similarity_identical:.6f}")
                
                print("\n🎉 All offline tests passed!")
                return True
                
            except ImportError as e:
                print(f"❌ Import error: {e}")
                print("Make sure to install requirements: pip install -r requirements.txt")
                return False
            except Exception as e:
                print(f"❌ Test failed: {e}")
                return False

def test_configuration_validation():
    """
    Test that configuration validation works properly.
    """
    print("\nTesting configuration validation...")
    
    # Test with missing environment variables
    with patch.dict(os.environ, {}, clear=True):
        with patch('boto3.client'):
            try:
                from vector_text_sample import VectorTextProcessor
                processor = VectorTextProcessor()
                print("❌ Should have failed with missing environment variables")
                return False
            except ValueError as e:
                if "Missing required environment variables" in str(e):
                    print("✓ Configuration validation works correctly")
                    return True
                else:
                    print(f"❌ Unexpected error: {e}")
                    return False
            except Exception as e:
                print(f"❌ Unexpected error type: {e}")
                return False

def main():
    """
    Run all tests.
    """
    print("=== Vector Text Sample Tests ===\n")
    
    success_count = 0
    total_tests = 2
    
    # Run tests
    if test_vectorization_without_aws():
        success_count += 1
    
    if test_configuration_validation():
        success_count += 1
    
    # Report results
    print(f"\n=== Test Results ===")
    print(f"Passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)