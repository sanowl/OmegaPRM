import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

class ProcessRewardModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ProcessRewardModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class OmegaPRM:
    def __init__(self, model, tokenizer, golden_answers, search_limit=100, alpha=0.5, beta=0.9, L=500, cpuct=0.125):
        self.model = model
        self.tokenizer = tokenizer
        self.golden_answers = golden_answers  # Dictionary of correct answers
        self.search_limit = search_limit
        self.alpha = alpha
        self.beta = beta
        self.L = L
        self.cpuct = cpuct
        self.tree = defaultdict(dict)  # To store the state-action tree
        self.visit_counts = defaultdict(int)  # To store visit counts for each state
        
        # Initialize the PRM (input_size, hidden_size, output_size)
        self.prm = ProcessRewardModel(input_size=768, hidden_size=256, output_size=1)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.prm.parameters(), lr=0.001)

    def complete_solution(self, question, solution_prefix):
        """
        Generates the completion of the solution from a given prefix using the model.
        """
        inputs = self.tokenizer.encode(question + solution_prefix, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=512)
        completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return completion[len(solution_prefix):]

    def compare_with_golden_answer(self, question, rollout):
        """
        Compare the generated rollout with the golden (correct) answer.
        """
        golden_answer = self.golden_answers[question]
        return rollout.strip() == golden_answer.strip()

    def monte_carlo_estimation(self, question, solution_prefix):
        """
        Perform Monte Carlo rollouts from the current solution prefix.
        """
        rollouts = []
        for _ in range(self.search_limit):
            rollout = self.complete_solution(question, solution_prefix)
            correct = self.compare_with_golden_answer(question, rollout)
            rollouts.append((rollout, correct))
        return rollouts

    def binary_search(self, question, solution):
        """
        Use binary search to find the first incorrect step in the solution.
        """
        left, right = 0, len(solution)
        while left < right:
            mid = (left + right) // 2
            rollouts = self.monte_carlo_estimation(question, solution[:mid])
            if any(correct for _, correct in rollouts):
                left = mid + 1
            else:
                right = mid
        return left

    def omega_prm(self, question):
        """
        Core function to run the OmegaPRM algorithm and build the state-action tree.
        """
        initial_solution = self.model.generate(self.tokenizer.encode(question, return_tensors="pt"))
        initial_solution = self.tokenizer.decode(initial_solution[0], skip_special_tokens=True)
        first_error = self.binary_search(question, initial_solution)
        self.tree[question] = initial_solution[:first_error]
        return self.tree

    def train_prm(self, dataset):
        """
        Train the Process Reward Model (PRM) using the collected data.
        """
        for question in dataset:
            self.omega_prm(question)
            solution = self.tree[question]
            solution_vector = self.encode_solution(solution)
            correct = torch.tensor([self.compare_with_golden_answer(question, solution)], dtype=torch.float32)
            
            # Forward pass through PRM
            output = self.prm(solution_vector)
            
            # Calculate loss and update PRM
            loss = self.criterion(output, correct)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def encode_solution(self, solution):
        """
        Encode the solution string into a vector using the model's embeddings.
        """
        inputs = self.tokenizer.encode(solution, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.transformer.wte(inputs).mean(dim=1)  # Average of token embeddings
        return outputs

    def forward(self, question, solution_prefix):
        """
        Perform a forward pass with the PRM on the given input.
        """
        solution_vector = self.encode_solution(solution_prefix)
        logits = self.prm(solution_vector)
        return logits

# Example usage:
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example golden answers dictionary for testing
golden_answers = {
    "What is 2 + 2?": "4",
    "What is the capital of France?": "Paris"
}

omega_prm = OmegaPRM(model, tokenizer, golden_answers)

# Example question
question = "What is 2 + 2?"

# Running the OmegaPRM algorithm and training the PRM
omega_prm.train_prm([question])

# Test PRM with a forward pass
test_solution_prefix = "2 + 2 is"
prm_output = omega_prm.forward(question, test_solution_prefix)
print(f"PRM Output: {prm_output.item()}")
