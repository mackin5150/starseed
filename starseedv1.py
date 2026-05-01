import torch
import torch.nn as nn
import time

class LivingMineralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.brain = nn.Linear(10, 3)  # Simplified: 10 features in, 3 minerals out
        self.energy = 1.0              # Starts at 100% energy
        self.decay_rate = 0.05         # Energy lost per task
        self.recovery_rate = 0.1       # Energy gained during rest

    def forward(self, x):
        if self.energy < 0.2:
            print("Model is exhausted... adding entropy (noise).")
            # Adding 'entropy' via random noise because the model is tired
            noise = torch.randn_like(x) * (1 - self.energy)
            x = x + noise
        
        # Identification process
        prediction = self.brain(x)
        
        # Use energy!
        self.energy -= self.decay_rate
        return prediction

    def rest(self, rest_time):
        print(f"Resting for {rest_time} seconds...")
        time.sleep(rest_time)
        self.energy = min(1.0, self.energy + (self.recovery_rate * rest_time))
        print(f"Energy recovered to: {self.energy*100:.1f}%")

# --- Example of usage ---
model = LivingMineralModel()
sample_mineral_data = torch.randn(1, 10)

# Work the model until it gets tired
for i in range(5):
    output = model(sample_mineral_data)
    print(f"Task {i+1} complete. Current Energy: {model.energy:.2f}")

# Give it a rest
model.rest(rest_time=5)
