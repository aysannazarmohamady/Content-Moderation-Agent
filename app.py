"""
Main application for running the Content Moderation Agent.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from flask import Flask, jsonify, request

from src.agent.moderator import ContentModerationAgent
from src.data.loader import DataLoader
from src.models.classifier import TFIDFToxicityClassifier, BERTToxicityClassifier
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global agent variable
agent = None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Content Moderation Agent")
    
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="models/tfidf_model.pkl",
        help="Path to the trained model file"
    )
    
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="data/processed",
        help="Path to the data directory"
    )
    
    parser.add_argument(
        "--train", 
        action="store_true",
        help="Train a new model"
    )
    
    parser.add_argument(
        "--model-type", 
        type=str, 
        choices=["tfidf", "bert"], 
        default="tfidf",
        help="Type of model to use"
    )
    
    parser.add_argument(
        "--with-context", 
        action="store_true",
        help="Use context in moderation"
    )
    
    parser.add_argument(
        "--server", 
        action="store_true",
        help="Run as API server"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=5000,
        help="Port for the API server"
    )
    
    return parser.parse_args()


def train_model(args):
    """Train a new model."""
    logger.info(f"Training new {args.model_type} model with{'' if args.with_context else 'out'} context")
    
    # Load data
    data_loader = DataLoader(args.data_path)
    
    # Get training and validation data
    train_texts, train_labels, train_contexts, val_texts, val_labels, val_contexts = data_loader.get_train_val_test_data(
        with_context=args.with_context
    )[0:6]  # Only need train and val data for training
    
    # Initialize model
    if args.model_type == "tfidf":
        model = TFIDFToxicityClassifier(context_aware=args.with_context)
    else:  # bert
        model = BERTToxicityClassifier(context_aware=args.with_context)
    
    # Train model
    metrics = model.train(
        texts=train_texts,
        labels=train_labels,
        contexts=train_contexts
    )
    
    # Evaluate on validation set
    eval_metrics = model.evaluate(
        texts=val_texts,
        labels=val_labels,
        contexts=val_contexts
    )
    
    # Save model
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    model.save(args.model_path)
    
    logger.info(f"Model trained and saved to {args.model_path}")
    logger.info(f"Training metrics: {metrics}")
    logger.info(f"Validation metrics: {eval_metrics}")
    
    return model


def setup_agent(args):
    """Setup the content moderation agent."""
    global agent
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        if args.train:
            # Train a new model
            train_model(args)
        else:
            logger.error(f"Model file {args.model_path} not found. Use --train to train a new model.")
            sys.exit(1)
    
    # Initialize agent
    agent = ContentModerationAgent(
        model_path=args.model_path,
        context_aware=args.with_context,
        suggestion_enabled=True,
        # If you have an OpenAI API key, you can enable suggestions
        # llm_api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    logger.info("Agent initialized successfully")
    return agent


# API endpoints
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route("/moderate", methods=["POST"])
def moderate_text():
    """Moderate text endpoint."""
    data = request.json
    
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data["text"]
    context = data.get("context")
    
    try:
        result = agent.moderate(text, context)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in moderation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/batch_moderate", methods=["POST"])
def batch_moderate():
    """Batch moderate texts endpoint."""
    data = request.json
    
    if not data or "texts" not in data:
        return jsonify({"error": "No texts provided"}), 400
    
    texts = data["texts"]
    contexts = data.get("contexts")
    
    try:
        results = agent.batch_moderate(texts, contexts)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in batch moderation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    """Feedback endpoint for misclassifications."""
    data = request.json
    
    if not data or "text" not in data or "actual_toxic" not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    text = data["text"]
    actual_toxic = data["actual_toxic"]
    
    try:
        agent.feedback(text, actual_toxic)
        return jsonify({"status": "feedback received"})
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return jsonify({"error": str(e)}), 500


def main():
    """Main function."""
    args = parse_args()
    
    # Setup agent
    setup_agent(args)
    
    if args.server:
        # Run as API server
        logger.info(f"Starting API server on port {args.port}")
        app.run(host="0.0.0.0", port=args.port)
    else:
        # Run as CLI
        print("Content Moderation Agent CLI")
        print("Type 'exit' to quit")
        
        while True:
            text = input("\nEnter text to moderate: ")
            
            if text.lower() == "exit":
                break
            
            context = input("Enter context (optional, press Enter to skip): ")
            context = context if context else None
            
            result = agent.moderate(text, context)
            
            print("\nModeration Result:")
            print(f"Is Toxic: {result['is_toxic']}")
            print(f"Confidence: {result['confidence']:.2f}")
            
            if result['is_toxic'] and result['suggestions']:
                print("\nSuggestions for improvement:")
                for i, suggestion in enumerate(result['suggestions'], 1):
                    print(f"{i}. {suggestion}")


if __name__ == "__main__":
    main()
