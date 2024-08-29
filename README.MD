# OmegaPRM

This repository contains a test implementation of the OmegaPRM algorithm, as described in the research paper titled *"Improve Mathematical Reasoning in Language Models by Automated Process Supervision."* This code serves as an experimental implementation to demonstrate the concepts and methods discussed in the paper. It is not a full production-level implementation but is intended for educational and testing purposes.

## Overview

The OmegaPRM (Process Reward Model) algorithm is designed to improve the reasoning capabilities of large language models, particularly in tasks that require complex, multi-step reasoning. Traditional models often struggle with such tasks because they are typically trained using outcome supervision, where only the final answer is evaluated. OmegaPRM introduces **process supervision**, where each step of the reasoning process is evaluated, allowing the model to learn more effectively.

### Key Components

1. **Process Reward Model (PRM):**
   - A neural network that evaluates the correctness of each step in the reasoning process.
   - This implementation includes a simple PRM with two fully connected layers.
   
2. **Monte Carlo Tree Search (MCTS):**
   - The algorithm generates multiple possible solutions (rollouts) from a given solution prefix and evaluates each one to determine if it is correct.
   - This helps in identifying where the model might go wrong in its reasoning process.

3. **Binary Search for Error Detection:**
   - To efficiently find the first incorrect step in the solution, the algorithm uses binary search, which significantly reduces the number of checks needed.

4. **State-Action Tree:**
   - The correct parts of the solution are stored in a tree structure, allowing the model to learn from its mistakes and improve its reasoning over time.

## How It Works

### Workflow

1. **Initialize the Model and Tokenizer:**
   - The code uses a pre-trained language model from Hugging Face (e.g., GPT-2) and its corresponding tokenizer.

2. **Run OmegaPRM:**
   - The `omega_prm` function generates an initial solution to the given question and identifies the first error using binary search.
   - The correct portion of the solution is stored in a tree structure.

3. **Train the PRM:**
   - The `train_prm` function trains the Process Reward Model using the data collected from the OmegaPRM algorithm.
   - The PRM is trained to predict the correctness of each step in the reasoning process.

4. **Test the PRM:**
   - After training, you can test the PRM using the `forward` function, which performs a forward pass through the PRM with a new solution prefix to predict its correctness.

### Example Usage

The example in the code initializes the model and tokenizer, runs the OmegaPRM algorithm on a simple math question ("What is 2 + 2?"), and trains the PRM based on the generated data. You can easily modify the `golden_answers` dictionary and the questions to test different scenarios.

### Limitations and Notes

- **Educational Purposes Only:** This implementation is for educational and testing purposes only. It simplifies several aspects of the OmegaPRM algorithm to make it more accessible and easier to understand.
- **Not Production-Ready:** The code is not optimized for production environments. It lacks many of the optimizations and refinements that would be necessary for deploying this in a real-world application.
- **Simplified PRM:** The Process Reward Model in this implementation is a simple neural network. In a full-scale implementation, the PRM might be more complex, depending on the specific requirements and tasks.

## Installation and Setup

To run this code, you need to have Python installed along with the following libraries:

```bash
pip install torch transformers
```

## Conclusion

This test implementation of the OmegaPRM algorithm provides a basic but functional example of how process supervision can be applied to improve the reasoning capabilities of large language models. It is based on concepts from the referenced research paper and serves as an educational tool for understanding these advanced AI techniques.

If you have any questions or need further explanations, feel free to explore the code or reach out!
