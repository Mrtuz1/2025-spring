import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import random
from itertools import combinations
import pandas as pd
from tqdm import tqdm
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class PointDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def generate_two_points_data(num_samples):
    """Generate data for problem A: Two points and their Euclidean distance"""
    X = np.zeros((num_samples, 1, 25, 25))
    y = np.zeros(num_samples)
    
    for i in range(num_samples):
        # Generate two random points
        p1 = (random.randint(0, 24), random.randint(0, 24))
        p2 = (random.randint(0, 24), random.randint(0, 24))
        
        # Create the binary matrix
        matrix = np.zeros((25, 25))
        matrix[p1[0], p1[1]] = 1
        matrix[p2[0], p2[1]] = 1
        
        # Calculate Euclidean distance
        distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        X[i, 0] = matrix
        y[i] = distance
    
    return X, y

def generate_n_points_data(num_samples, min_points=3, max_points=10, closest=True):
    """Generate data for problems B and C: N points and distance between closest/farthest points"""
    X = np.zeros((num_samples, 1, 25, 25))
    y = np.zeros(num_samples)
    
    for i in range(num_samples):
        n_points = random.randint(min_points, max_points)
        points = []
        matrix = np.zeros((25, 25))
        
        # Generate n random points
        while len(points) < n_points:
            point = (random.randint(0, 24), random.randint(0, 24))
            if point not in points:
                points.append(point)
                matrix[point[0], point[1]] = 1
        
        # Calculate distances between all pairs
        distances = []
        for p1, p2 in combinations(points, 2):
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            distances.append(dist)
        
        # Get closest or farthest pair distance
        if closest:
            y[i] = min(distances)
        else:
            y[i] = max(distances)
        
        X[i, 0] = matrix
    
    return X, y

def generate_count_points_data(num_samples, min_points=1, max_points=10):
    """Generate data for problem D: Count number of points"""
    X = np.zeros((num_samples, 1, 25, 25))
    y = np.zeros(num_samples)
    
    for i in range(num_samples):
        n_points = random.randint(min_points, max_points)
        points = []
        matrix = np.zeros((25, 25))
        
        while len(points) < n_points:
            point = (random.randint(0, 24), random.randint(0, 24))
            if point not in points:
                points.append(point)
                matrix[point[0], point[1]] = 1
        
        X[i, 0] = matrix
        y[i] = n_points
    
    return X, y

def generate_squares_data(num_samples, min_squares=1, max_squares=10):
    """Generate data for problem E: Count number of squares"""
    X = np.zeros((num_samples, 1, 25, 25))
    y = np.zeros(num_samples)
    
    for i in range(num_samples):
        n_squares = random.randint(min_squares, max_squares)
        matrix = np.zeros((25, 25))
        
        for _ in range(n_squares):
            # Random square size between 2 and 5
            size = random.randint(2, 5)
            # Random position ensuring square fits within matrix
            x = random.randint(0, 25 - size)
            y_pos = random.randint(0, 25 - size)
            
            # Draw the square
            matrix[x:x+size, y_pos:y_pos+size] = 1
        
        X[i, 0] = matrix
        y[i] = n_squares
    
    return X, y

class CNN(nn.Module):
    def __init__(self, output_dim=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    """Train the model and return training history"""
    model = model.to(device)
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test set and return predictions and true values"""
    model.eval()
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.squeeze().cpu().numpy())
            true_values.extend(batch_y.cpu().numpy())
    
    return np.array(predictions), np.array(true_values)

def plot_results(train_losses, val_losses, predictions, true_values, title):
    """Plot training curves and prediction results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training curves
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title(f'{title} - Training Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot predictions vs true values
    ax2.scatter(true_values, predictions, alpha=0.5)
    ax2.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
    ax2.set_title(f'{title} - Predictions vs True Values')
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predictions')
    
    plt.tight_layout()
    plt.show()

def save_datasets(X_train, X_test, y_train, y_test, save_dir='datasets'):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)

def load_datasets(save_dir='datasets'):
    X_train = np.load(os.path.join(save_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(save_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(save_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(save_dir, 'y_test.npy'))
    return X_train, X_test, y_train, y_test

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate datasets
    problems = {
        'A': (generate_two_points_data, 'Two Points Distance'),
        'B': (lambda x: generate_n_points_data(x, closest=True), 'Closest Points Distance'),
        'C': (lambda x: generate_n_points_data(x, closest=False), 'Farthest Points Distance'),
        'D': (generate_count_points_data, 'Count Points'),
        'E': (generate_squares_data, 'Count Squares')
    }
    
    for problem, (generator_func, title) in problems.items():
        print(f"\nTraining model for Problem {problem}: {title}")
        
        # Generate full dataset
        X, y = generator_func(1000)  # 800 train + 200 test
        
        # Create different training set sizes
        train_sizes = [200, 400, 800]  # quarter, half, full
        
        for train_size in train_sizes:
            print(f"\nTraining with {train_size} samples")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=200, random_state=42)
            X_train = X_train[:train_size]
            y_train = y_train[:train_size]
            
            # Save datasets
            save_dir = f'datasets/problem_{problem}_size_{train_size}'
            save_datasets(X_train, X_test, y_train, y_test, save_dir)
            print(f"Datasets saved to {save_dir}")
            
            # Create data loaders
            train_dataset = PointDataset(X_train, y_train)
            test_dataset = PointDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32)
            
            # Initialize model and training components
            model = CNN(output_dim=1)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Train model
            train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50, device=device)
            
            # Evaluate model
            predictions, true_values = evaluate_model(model, test_loader, device)
            
            # Plot results
            plot_results(train_losses, val_losses, predictions, true_values, f"{title} (Train Size: {train_size})")
            
            # Calculate and print metrics
            mse = np.mean((predictions - true_values) ** 2) #(Mean Squared Error)
            mae = np.mean(np.abs(predictions - true_values)) #(Mean Absolute Error)
            print(f"Test MSE: {mse:.4f}")
            print(f"Test MAE: {mae:.4f}")

if __name__ == "__main__":
    main() 