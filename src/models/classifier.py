import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer

from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)


class ToxicityClassifier:
    """Base class for toxicity classifiers."""
    
    def __init__(self, model_type: str = "bert", context_aware: bool = True):
        """
        Initialize the toxicity classifier.
        
        Args:
            model_type: Type of model to use ("bert", "distilbert", "tf-idf")
            context_aware: Whether to use context in classification
        """
        self.model_type = model_type
        self.context_aware = context_aware
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initializing {model_type} {'with' if context_aware else 'without'} context")
    
    def _prepare_data(
        self, 
        texts: List[str], 
        contexts: Optional[List[str]] = None, 
        labels: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Prepare data for training or prediction.
        
        Args:
            texts: List of text samples
            contexts: Optional list of context samples
            labels: Optional list of labels
            
        Returns:
            DataFrame containing prepared data
        """
        data = {"text": texts}
        
        if contexts:
            data["context"] = contexts
        
        if labels is not None:
            data["label"] = labels
        
        return pd.DataFrame(data)
    
    def train(
        self, 
        texts: List[str], 
        labels: List[int], 
        contexts: Optional[List[str]] = None,
        validation_split: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Train the toxicity classifier.
        
        Args:
            texts: List of text samples
            labels: List of labels (0 for non-toxic, 1 for toxic)
            contexts: Optional list of context samples
            validation_split: Fraction of data to use for validation
            random_state: Random seed
            
        Returns:
            Dictionary containing evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(
        self, 
        text: Union[str, List[str]], 
        context: Optional[Union[str, List[str]]] = None
    ) -> Union[Tuple[int, np.ndarray], Tuple[List[int], List[np.ndarray]]]:
        """
        Predict toxicity for the given text(s).
        
        Args:
            text: Text or list of texts to classify
            context: Optional context or list of contexts
            
        Returns:
            Tuple of predictions and probabilities
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    def evaluate(
        self, 
        texts: List[str], 
        labels: List[int], 
        contexts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the classifier on the given data.
        
        Args:
            texts: List of text samples
            labels: List of labels (0 for non-toxic, 1 for toxic)
            contexts: Optional list of context samples
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Make predictions
        predictions, probabilities = self.predict(texts, contexts)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions),
            "recall": recall_score(labels, predictions),
            "f1": f1_score(labels, predictions),
            "auc_roc": roc_auc_score(labels, [prob[1] for prob in probabilities])
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save(self, path: str) -> None:
        """
        Save the model to the given path.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ToxicityClassifier':
        """
        Load a model from the given path.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model


class BERTToxicityClassifier(ToxicityClassifier):
    """Toxicity classifier using BERT embeddings."""
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased", 
        context_aware: bool = True,
        max_length: int = 128
    ):
        """
        Initialize the BERT toxicity classifier.
        
        Args:
            model_name: Name of the BERT model to use
            context_aware: Whether to use context in classification
            max_length: Maximum sequence length
        """
        super().__init__("bert", context_aware)
        self.model_name = model_name
        self.max_length = max_length
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize BERT model for embeddings
        self.bert_model = AutoModel.from_pretrained(model_name)
        
        # Initialize classifier
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
    
    def _get_embeddings(
        self, 
        texts: List[str], 
        contexts: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Get BERT embeddings for the given texts and contexts.
        
        Args:
            texts: List of text samples
            contexts: Optional list of context samples
            
        Returns:
            numpy array of embeddings
        """
        # Prepare input text with or without context
        if self.context_aware and contexts:
            # Combine text and context
            combined_texts = [f"{text} [SEP] {context}" for text, context in zip(texts, contexts)]
        else:
            combined_texts = texts
        
        # Tokenize
        encoded_inputs = self.tokenizer(
            combined_texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=self.max_length
        )
        
        # Get embeddings from BERT
        with torch.no_grad():
            outputs = self.bert_model(**encoded_inputs)
            # Use the CLS token embedding as the text representation
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embeddings
    
    def train(
        self, 
        texts: List[str], 
        labels: List[int], 
        contexts: Optional[List[str]] = None,
        validation_split: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Train the BERT toxicity classifier.
        
        Args:
            texts: List of text samples
            labels: List of labels (0 for non-toxic, 1 for toxic)
            contexts: Optional list of context samples
            validation_split: Fraction of data to use for validation
            random_state: Random seed
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Training BERT classifier with {len(texts)} samples")
        
        # Split data into train and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=validation_split, random_state=random_state, stratify=labels
        )
        
        if contexts:
            train_contexts, val_contexts = train_test_split(
                contexts, test_size=validation_split, random_state=random_state
            )
        else:
            train_contexts, val_contexts = None, None
        
        # Get embeddings for train set
        train_embeddings = self._get_embeddings(train_texts, train_contexts)
        
        # Train classifier
        self.classifier.fit(train_embeddings, train_labels)
        
        # Evaluate on validation set
        val_embeddings = self._get_embeddings(val_texts, val_contexts)
        val_predictions = self.classifier.predict(val_embeddings)
        val_probabilities = self.classifier.predict_proba(val_embeddings)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(val_labels, val_predictions),
            "precision": precision_score(val_labels, val_predictions),
            "recall": recall_score(val_labels, val_predictions),
            "f1": f1_score(val_labels, val_predictions),
            "auc_roc": roc_auc_score(val_labels, val_probabilities[:, 1])
        }
        
        logger.info(f"Validation metrics: {metrics}")
        return metrics
    
    def predict(
        self, 
        text: Union[str, List[str]], 
        context: Optional[Union[str, List[str]]] = None
    ) -> Union[Tuple[int, np.ndarray], Tuple[List[int], List[np.ndarray]]]:
        """
        Predict toxicity for the given text(s).
        
        Args:
            text: Text or list of texts to classify
            context: Optional context or list of contexts
            
        Returns:
            Tuple of predictions and probabilities
        """
        # Handle single text
        if isinstance(text, str):
            texts = [text]
            contexts = [context] if context else None
            single_input = True
        else:
            texts = text
            contexts = context
            single_input = False
        
        # Get embeddings
        embeddings = self._get_embeddings(texts, contexts)
        
        # Make predictions
        predictions = self.classifier.predict(embeddings)
        probabilities = self.classifier.predict_proba(embeddings)
        
        # Return predictions for single or multiple inputs
        if single_input:
            return predictions[0], probabilities[0]
        else:
            return predictions, probabilities


class TFIDFToxicityClassifier(ToxicityClassifier):
    """Toxicity classifier using TF-IDF features."""
    
    def __init__(self, context_aware: bool = True):
        """
        Initialize the TF-IDF toxicity classifier.
        
        Args:
            context_aware: Whether to use context in classification
        """
        super().__init__("tf-idf", context_aware)
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=5,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
    
    def train(
        self, 
        texts: List[str], 
        labels: List[int], 
        contexts: Optional[List[str]] = None,
        validation_split: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Train the TF-IDF toxicity classifier.
        
        Args:
            texts: List of text samples
            labels: List of labels (0 for non-toxic, 1 for toxic)
            contexts: Optional list of context samples
            validation_split: Fraction of data to use for validation
            random_state: Random seed
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Training TF-IDF classifier with {len(texts)} samples")
        
        # Prepare input text with or without context
        if self.context_aware and contexts:
            # Combine text and context
            combined_texts = [f"{text} {context}" for text, context in zip(texts, contexts)]
        else:
            combined_texts = texts
        
        # Split data into train and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            combined_texts, labels, test_size=validation_split, 
            random_state=random_state, stratify=labels
        )
        
        # Fit TF-IDF vectorizer on training data
        X_train = self.vectorizer.fit_transform(train_texts)
        
        # Train classifier
        self.pipeline.fit(X_train, train_labels)
        
        # Evaluate on validation set
        X_val = self.vectorizer.transform(val_texts)
        val_predictions = self.pipeline.predict(X_val)
        val_probabilities = self.pipeline.predict_proba(X_val)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(val_labels, val_predictions),
            "precision": precision_score(val_labels, val_predictions),
            "recall": recall_score(val_labels, val_predictions),
            "f1": f1_score(val_labels, val_predictions),
            "auc_roc": roc_auc_score(val_labels, val_probabilities[:, 1])
        }
        
        logger.info(f"Validation metrics: {metrics}")
        return metrics
    
    def predict(
        self, 
        text: Union[str, List[str]], 
        context: Optional[Union[str, List[str]]] = None
    ) -> Union[Tuple[int, np.ndarray], Tuple[List[int], List[np.ndarray]]]:
        """
        Predict toxicity for the given text(s).
        
        Args:
            text: Text or list of texts to classify
            context: Optional context or list of contexts
            
        Returns:
            Tuple of predictions and probabilities
        """
        # Handle single text
        if isinstance(text, str):
            texts = [text]
            contexts = [context] if context else None
            single_input = True
        else:
            texts = text
            contexts = context
            single_input = False
        
        # Prepare input text with or without context
        if self.context_aware and contexts:
            # Combine text and context
            combined_texts = [
                f"{text} {ctx}" for text, ctx in zip(texts, contexts)
            ]
        else:
            combined_texts = texts
        
        # Transform with TF-IDF
        X = self.vectorizer.transform(combined_texts)
        
        # Make predictions
        predictions = self.pipeline.predict(X)
        probabilities = self.pipeline.predict_proba(X)
        
        # Return predictions for single or multiple inputs
        if single_input:
            return predictions[0], probabilities[0]
        else:
            return predictions, probabilities
