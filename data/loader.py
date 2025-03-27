"""
This module handles loading and processing data for toxicity detection.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)

class DataLoader:
    """
    Data loader for toxicity detection datasets.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the data directory or file
        """
        self.data_path = data_path
        logger.info(f"Initializing data loader with path: {data_path}")
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load a CSV file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            DataFrame containing the loaded data
        """
        file_path = os.path.join(self.data_path, filename)
        logger.info(f"Loading data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def load_toxicity_data(self, with_context: bool = True) -> pd.DataFrame:
        """
        Load toxicity detection data.
        
        Args:
            with_context: Whether to load data with context
            
        Returns:
            DataFrame containing the loaded data
        """
        # For the Toxicity Detection Context dataset, we have
        # two files: gn.csv (no context) and gc.csv (with context)
        filename = "gc.csv" if with_context else "gn.csv"
        
        df = self.load_csv(filename)
        
        # Ensure the required columns are present
        required_columns = ["text", "label"]
        if with_context:
            required_columns.append("context")
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
    
    def load_combined_dataset(self) -> pd.DataFrame:
        """
        Load and combine both context and non-context datasets.
        
        Returns:
            DataFrame containing the combined dataset
        """
        # Load both datasets
        context_df = self.load_toxicity_data(with_context=True)
        nocontext_df = self.load_toxicity_data(with_context=False)
        
        # Add a flag indicating whether the data has context
        context_df["has_context"] = True
        nocontext_df["has_context"] = False
        
        # If the nocontext_df doesn't have a 'context' column, add an empty one
        if "context" not in nocontext_df.columns:
            nocontext_df["context"] = ""
        
        # Combine the datasets
        combined_df = pd.concat([context_df, nocontext_df], ignore_index=True)
        
        logger.info(f"Combined dataset has {len(combined_df)} rows")
        return combined_df
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2, 
        validation_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: DataFrame to split
            test_size: Fraction of data to use for testing
            validation_size: Fraction of data to use for validation
            random_state: Random seed
            
        Returns:
            Tuple of train, validation, and test DataFrames
        """
        # Calculate the effective validation size
        # If test_size is 0.2 and validation_size is 0.1, then:
        # - 20% of the data goes to test
        # - 10% of the data goes to validation
        # - 70% of the data goes to train
        # So the validation_size relative to the train+validation is:
        # 0.1 / (1 - 0.2) = 0.125
        effective_val_size = validation_size / (1 - test_size)
        
        # First split into train+validation and test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df["label"] if "label" in df.columns else None
        )
        
        # Then split train+validation into train and validation
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=effective_val_size, 
            random_state=random_state,
            stratify=train_val_df["label"] if "label" in train_val_df.columns else None
        )
        
        logger.info(f"Split data into {len(train_df)} train, {len(val_df)} validation, and {len(test_df)} test samples")
        return train_df, val_df, test_df
    
    def get_train_val_test_data(
        self, 
        with_context: bool = True, 
        test_size: float = 0.2,
        validation_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[
        List[str], List[int], Optional[List[str]],
        List[str], List[int], Optional[List[str]],
        List[str], List[int], Optional[List[str]]
    ]:
        """
        Get train, validation, and test data split into texts, labels, and contexts.
        
        Args:
            with_context: Whether to include context
            test_size: Fraction of data to use for testing
            validation_size: Fraction of data to use for validation
            random_state: Random seed
            
        Returns:
            Tuple of (train_texts, train_labels, train_contexts,
                     val_texts, val_labels, val_contexts,
                     test_texts, test_labels, test_contexts)
        """
        # Load data
        df = self.load_toxicity_data(with_context=with_context)
        
        # Split data
        train_df, val_df, test_df = self.split_data(
            df, test_size=test_size, validation_size=validation_size, random_state=random_state
        )
        
        # Extract texts and labels
        train_texts = train_df["text"].tolist()
        train_labels = train_df["label"].tolist()
        
        val_texts = val_df["text"].tolist()
        val_labels = val_df["label"].tolist()
        
        test_texts = test_df["text"].tolist()
        test_labels = test_df["label"].tolist()
        
        # Extract contexts if available
        if with_context and "context" in df.columns:
            train_contexts = train_df["context"].tolist()
            val_contexts = val_df["context"].tolist()
            test_contexts = test_df["context"].tolist()
        else:
            train_contexts = None
            val_contexts = None
            test_contexts = None
        
        return (
            train_texts, train_labels, train_contexts,
            val_texts, val_labels, val_contexts,
            test_texts, test_labels, test_contexts
        )
