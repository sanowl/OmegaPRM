import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import math
import logging
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import wandb
import numpy as np
from torch.utils.data import DataLoader

from ..config import OmegaPRMConfig
from ..models.process_reward_model import ProcessRewardModel
from ..models.lru_cache import LRUCache
from ..data.dataset import PRMDataset
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class OmegaPRM:
    """OmegaPRM implementation with validation, GPU support, and performance optimizations"""
    def __init__(self, config: OmegaPRMConfig, golden_answers: Dict[str, str]):
        """
        Initialize OmegaPRM
        
        Args:
            config (OmegaPRMConfig): Configuration object
            golden_answers (Dict[str, str]): Dictionary of golden answers
        """
        self.config = config
        self.config.validate()
        self.golden_answers = golden_answers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        try:
            self.model = AutoModelForCausalLM.from_pretrained(config.model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            logger.info(f"Successfully loaded model: {config.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
        
        # Initialize components
        self.tree = defaultdict(dict)
        self.visit_counts = defaultdict(int)
        self.Q_values = defaultdict(float)
        self.cache = LRUCache(config.cache_size)
        
        # Initialize PRM
        input_size = self.model.config.hidden_size
        self.prm = ProcessRewardModel(
            input_size=input_size,
            hidden_size=config.hidden_size,
            output_size=1
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.prm.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        self.criterion = nn.BCELoss()
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(project="omega-prm", config=vars(config))
            
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(exist_ok=True)

    def save_checkpoint(self, epoch: int, validation_loss: float):
        """
        Save model checkpoint
        
        Args:
            epoch (int): Current epoch number
            validation_loss (float): Current validation loss
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.prm.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'validation_loss': validation_loss,
            'config': self.config.__dict__
        }
        
        path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> Tuple[int, float]:
        """
        Load model checkpoint
        
        Args:
            path (str): Path to checkpoint file
            
        Returns:
            Tuple[int, float]: Epoch number and validation loss
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.prm.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Loaded checkpoint from {path}")
            return checkpoint['epoch'], checkpoint['validation_loss']
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise

    @torch.no_grad()
    def get_ucb_score(self, state: str, action: str) -> float:
        """
        Calculate Upper Confidence Bound (UCB) score
        
        Args:
            state (str): Current state
            action (str): Proposed action
            
        Returns:
            float: UCB score
        """
        state_action = (state, action)
        Q = self.Q_values[state_action]
        N = self.visit_counts[state_action]
        N_parent = self.visit_counts[state]
        
        exploration_term = self.config.cpuct * math.sqrt(
            math.log(max(N_parent, 1) + 1) / (max(N, 1) + 1)
        )
        return Q + self.config.alpha * exploration_term

    def parallel_monte_carlo_rollout(
        self,
        questions: List[str],
        solution_prefixes: List[str]
    ) -> List[Tuple[List[Tuple[str, float]], float]]:
        """
        Perform Monte Carlo rollouts in parallel
        
        Args:
            questions (List[str]): List of questions
            solution_prefixes (List[str]): List of solution prefixes
            
        Returns:
            List[Tuple[List[Tuple[str, float]], float]]: Results of rollouts
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.num_workers
        ) as executor:
            futures = [
                executor.submit(self.monte_carlo_estimation, q, p)
                for q, p in zip(questions, solution_prefixes)
            ]
            results = [future.result() for future in futures]
        return results

    @torch.no_grad()
    def monte_carlo_estimation(
        self,
        question: str,
        solution_prefix: str
    ) -> Tuple[List[Tuple[str, float]], float]:
        """
        Perform Monte Carlo rollouts with caching and beta discount
        
        Args:
            question (str): Input question
            solution_prefix (str): Solution prefix
            
        Returns:
            Tuple[List[Tuple[str, float]], float]: Rollout results and cumulative reward
        """
        rollouts = []
        cumulative_reward = 0
        
        for step in range(self.config.search_limit):
            # Check cache first
            cache_key = f"{question}_{solution_prefix}_{step}"
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                rollout, reward = cached_result
            else:
                rollout = self.complete_solution(question, solution_prefix)
                correct = self.compare_with_golden_answer(question, rollout)
                reward = float(correct) * (self.config.beta ** step)
                self.cache.put(cache_key, (rollout, reward))
            
            cumulative_reward += reward
            rollouts.append((rollout, reward))
            
            # Update statistics
            state_action = (question, solution_prefix)
            self.visit_counts[state_action] += 1
            self.Q_values[state_action] = (
                self.Q_values[state_action] * (self.visit_counts[state_action] - 1) +
                reward
            ) / self.visit_counts[state_action]
            
        return rollouts, cumulative_reward

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train one epoch of the PRM
        
        Args:
            train_loader (DataLoader): Training data loader
            epoch (int): Current epoch number
            
        Returns:
            float: Average training loss
        """
        self.prm.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
            for batch_idx, (solutions, rewards) in enumerate(pbar):
                solutions = solutions.to(self.device)
                rewards = rewards.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.prm(solutions)
                loss = self.criterion(outputs, rewards)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.prm.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                if self.config.use_wandb:
                    wandb.log({
                        'batch_loss': loss.item(),
                        'epoch': epoch,
                        'batch': batch_idx
                    })
        
        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the PRM
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            float: Average validation loss
        """
        self.prm.eval()
        total_loss = 0
        
        for solutions, rewards in val_loader:
            solutions = solutions.to(self.device)
            rewards = rewards.to(self.device)
            
            outputs = self.prm(solutions)
            loss = self.criterion(outputs, rewards)
            total_loss += loss.item()
        
        return total_loss / len(val_loader)

    def train_prm(
        self,
        train_dataset: List[str],
        val_dataset: List[str],
        num_epochs: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the Process Reward Model
        
        Args:
            train_dataset (List[str]): Training dataset
            val_dataset (List[str]): Validation dataset
            num_epochs (int): Number of training epochs
            
        Returns:
            Dict[str, List[float]]: Training history
        """
        # Prepare datasets
        train_solutions, train_rewards = self.prepare_training_data(train_dataset)
        val_solutions, val_rewards = self.prepare_training_data(val_dataset)
        
        train_data = PRMDataset(
            train_solutions, train_rewards, self.tokenizer, self.config.max_length
        )
        val_data = PRMDataset(
            val_solutions, val_rewards, self.tokenizer, self.config.max_length
        )
        
        train_loader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        val_loader = DataLoader(
            val_data,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        # Training loop
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            logger.info(
                f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
            )
            
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })
            
            # Save checkpoint if best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
        
        return history

    def prepare_training_data(
        self,
        dataset: List[str]
    ) -> Tuple[List[str], List[float]]:
        """
        Prepare training data with parallel processing
        
        Args:
            dataset (List[str]): Raw dataset
            
        Returns:
            Tuple[List[str], List[float]]: Processed solutions and rewards
        """
        solutions = []
        rewards = []
        
        # Process questions in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.num_workers
        ) as executor:
            future_to_question = {
                executor.submit(self.omega_prm, question): question
                for question in dataset
            }
            
            for future in tqdm(
                concurrent.futures.as_completed(future_to_question),
                total=len(dataset),
                desc="Preparing training data"
            ):
                question = future_to_question[future]
                try:
                    tree_data = future.result()
                    for solution, reward in tree_data[question].items():
                        solutions.append(solution)
                        rewards.append(reward)
                except Exception as e:
                    logger.error(f"Error processing question {question}: {str(e)}")
        
        return solutions, rewards

    def complete_solution(self, question: str, solution_prefix: str) -> str:
        """
        Generate solution completion
        
        Args:
            question (str): Input question
            solution_prefix (str): Solution prefix
            
        Returns:
            str: Completed solution
        """
        try:
            inputs = self.tokenizer.encode(
                question + solution_prefix,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True
            ).to(self.device)
            
            outputs = self.model.generate(
                inputs,
                max_length=self.config.max_length,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return completion[len(solution_prefix):]
        except Exception as e:
            logger.error(f"Error in complete_solution: {str(e)}")
            raise

    def compare_with_golden_answer(self, question: str, rollout: str) -> bool:
        """
        Compare generated answer with golden answer
        
        Args:
            question (str): Input question
            rollout (str): Generated answer
            
        Returns:
            bool: Whether the answer matches
        """
        try:
            golden_answer = self.golden_answers.get(question)
            if golden_answer is None:
                raise ValueError(f"No golden answer found for question: {question}")
            return rollout.strip() == golden_answer.strip()
        except Exception as e:
            logger.error(f"Error in compare_with_golden_answer: {str(e)}")
            raise

    def get_metrics(self) -> Dict[str, float]:
        """
        Calculate and return performance metrics
        
        Returns:
            Dict[str, float]: Performance metrics
        """
        metrics = {
            'average_q_value': np.mean(list(self.Q_values.values())),
            'total_visits': sum(self.visit_counts.values()),
            'unique_states': len(self.tree),
            'cache_hit_rate': self.cache.get_hit_rate()
        }
        return metrics

    def export_results(self, output_path: str):
        """
        Export results and metrics to JSON
        
        Args:
            output_path (str): Output file path
        """
        results = {
            'tree': dict(self.tree),
            'metrics': self.get_metrics(),
            'config': vars(self.config)
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results exported to {output_path}")

# Optional: Add CLI interface
def main():
    """Main entry point"""
    # Configuration
    config = OmegaPRMConfig(
        model_name="gpt2",
        use_wandb=True,
        batch_size=32,
        num_workers=4
    )
    
    # Example golden answers
    golden_answers = {
        "What is 2 + 2?": "4",
        "What is the capital of France?": "Paris"
    }
    
    # Initialize OmegaPRM
    omega_prm = OmegaPRM(config, golden_answers)
    
    # Prepare datasets
    train_questions = ["What is 2 + 2?"]
    val_questions = ["What is the capital of France?"]
    
    # Train model
    history = omega_prm.train_prm(
        train_dataset=train_questions,
        val_dataset=val_questions,
        num_epochs=10
    )
    
    # Export results
    omega_prm.export_results("results.json")
    
    # Get and print metrics
    metrics = omega_prm.get_metrics()
    logger.info(f"Final metrics: {metrics}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")