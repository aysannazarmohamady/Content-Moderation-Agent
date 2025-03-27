"""
Script for training toxicity classifier models.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Optional

import torch

from src.data.loader import DataLoader
from src.models.classifier import TFIDFToxicityClassifier, BERTToxicityClassifier
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a toxicity classifier model")
    
    parser.add_argument(
        "--data-path", 
        type=str, 
        required=True,
        help="Path to the data directory"
    )
    
    parser.add_argument(
        "--model-type", 
        type=str, 
        choices=["tfidf", "bert", "distilbert"], 
        default="tfidf",
        help="Type of model to use"
    )
    
    parser.add_argument(
        "--with-context", 
        action="store_true",
        help="Use context in classification"
    )
    
    parser.add_argument(
        "--model-path", 
        type=str, 
        help="Path to save the trained model"
    )
    
    parser.add_argument(
        "--test-split", 
        type=float, 
        default=0.2,
        help="Fraction of data to use for testing"
    )
    
    parser.add_argument(
        "--val-split", 
        type=float, 
        default=0.1,
        help="Fraction of data to use for validation"
    )
    
    parser.add_argument(
        "--random-state", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def save_metrics(metrics: Dict[str, float], path: str) -> None:
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        path: Path to save metrics
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Metrics saved to {path}")


def main():
    """Main function."""
    args = parse_args()
    
    # Set model path if not provided
    if not args.model_path:
        context_str = "with_context" if args.with_context else "no_context"
        args.model_path = f"models/{args.model_type}_{context_str}_model.pkl"
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    
    # Load data
    try:
        data_loader = DataLoader(args.data_path)
        
        # Get training, validation, and test data
        (
            train_texts, train_labels, train_contexts,
            val_texts, val_labels, val_contexts,
            test_texts, test_labels, test_contexts
        ) = data_loader.get_train_val_test_data(
            with_context=args.with_context,
            test_size=args.test_split,
            validation_size=args.val_split,
            random_state=args.random_state
        )
        
        logger.info(f"Loaded {len(train_texts)} training, {len(val_texts)} validation, and {len(test_texts)} test samples")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Initialize model
    try:
        if args.model_type == "tfidf":
            model = TFIDFToxicityClassifier(context_aware=args.with_context)
        elif args.model_type == "bert":
            model = BERTToxicityClassifier(context_aware=args.with_context)
        elif args.model_type == "distilbert":
            model = BERTToxicityClassifier(
                model_name="distilbert-base-uncased",
                context_aware=args.with_context
            )
        else:
            logger.error(f"Unsupported model type: {args.model_type}")
            sys.exit(1)
        
        logger.info(f"Initialized {args.model_type} model")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        sys.exit(1)
    
    # Train model
    try:
        logger.info("Training model...")
        
        # Check if GPU is available for BERT models
        if args.model_type in ["bert", "distilbert"] and torch.cuda.is_available():
            logger.info("Using GPU for training")
            # Note: You'd need to modify the BERTToxicityClassifier to support GPU
        
        train_metrics = model.train(
            texts=train_texts,
            labels=train_labels,
            contexts=train_contexts,
            validation_split=0.0  # We already have a separate validation set
        )
        
        logger.info(f"Training metrics: {train_metrics}")
        
        # Evaluate on validation set
        val_metrics = model.evaluate(
            texts=val_texts,
            labels=val_labels,
            contexts=val_contexts
        )
        
        logger.info(f"Validation metrics: {val_metrics}")
        
        # Evaluate on test set
        test_metrics = model.evaluate(
            texts=test_texts,
            labels=test_labels,
            contexts=test_contexts
        )
        
        logger.info(f"Test metrics: {test_metrics}")
        
        # Save model
        model.save(args.model_path)
        logger.info(f"Model saved to {args.model_path}")
        
        # Save metrics
        metrics_path = f"{os.path.splitext(args.model_path)[0]}_metrics.json"
        
        metrics = {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
            "config": {
                "model_type": args.model_type,
                "with_context": args.with_context,
                "test_split": args.test_split,
                "val_split": args.val_split,
                "random_state": args.random_state
            }
        }
        
        save_metrics(metrics, metrics_path)
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
