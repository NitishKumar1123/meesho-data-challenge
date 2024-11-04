import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from PIL import Image
import os

# Step 1: Data Preparation - Load and encode attributes
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Use LabelEncoder to encode attributes
label_encoders = {}
for col in [f'attr_{i}' for i in range(1, 11)]:
    label_encoders[col] = LabelEncoder()
    train_df[col] = label_encoders[col].fit_transform(train_df[col].fillna("dummy_value"))

# Step 2: Define Dataset Class
class ProductDataset(Dataset):
    def __init__(self, image_dir, df, transform=None):
        self.image_dir = image_dir
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, f"{self.df.iloc[idx, 0]:06d}.jpg")
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get the label values for each attribute
        attributes = self.df.iloc[idx, 3:].values.astype(np.int64)
        return image, torch.tensor(attributes)

# Step 3: Enhanced Image Transformations
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Step 4: Dataloaders
train_dataset = ProductDataset(image_dir='train_images', df=train_df, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = ProductDataset(image_dir='test_images', df=test_df, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 5: Define Hybrid CNN-RNN Model
class HybridCNNRNNModel(nn.Module):
    def __init__(self, num_attributes=10, num_classes_per_attr=20, hidden_size=512):
        super(HybridCNNRNNModel, self).__init__()
        # Use pre-trained ResNet101 model for feature extraction
        self.cnn = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.cnn.fc = nn.Identity()  # Remove last layer of ResNet101
        cnn_output_size = 2048  # Output size of ResNet101 after removing fc layer
        
        # LSTM for sequential attribute prediction
        self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        
        # Fully connected layer to predict each attribute
        self.fc = nn.ModuleList([nn.Linear(hidden_size, num_classes_per_attr) for _ in range(num_attributes)])

    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract CNN features
        cnn_features = self.cnn(x)  # Output shape: (batch_size, 2048)
        
        # Repeat CNN features across the sequence length for LSTM input
        cnn_features = cnn_features.unsqueeze(1).repeat(1, 10, 1)  # Shape: (batch_size, seq_len, 2048)
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_features)  # Shape: (batch_size, seq_len, hidden_size)
        
        # Predict each attribute from the LSTM output
        outputs = [self.fc[i](lstm_out[:, i, :]) for i in range(10)]
        
        return outputs

# Step 6: Initialize Model, Loss, and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridCNNRNNModel().to(device)

# Use weighted CrossEntropyLoss if class imbalance exists
weights = torch.ones(20)  # Example placeholder; update based on class frequencies
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Step 7: Training Loop with Learning Rate Scheduler
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss for each attribute separately
        loss = 0
        for i in range(len(outputs)):  # len(outputs) == number of attributes
            loss += criterion(outputs[i], labels[:, i])
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Step the scheduler
    scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

# Step 8: Evaluation on the Test Set and Saving Predictions
model.eval()
predictions = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        
        batch_preds = []
        for i in range(len(outputs)):
            _, predicted = torch.max(outputs[i], 1)
            batch_preds.append(predicted.cpu().numpy())
        
        # Transpose to match image-wise predictions
        batch_preds = np.array(batch_preds).T
        predictions.extend(batch_preds)

# Convert predictions list to a numpy array
predictions = np.array(predictions)

# Step 9: Save predictions to submission file
submission_df = test_df.copy()

# Initialize the 'len' column based on the number of attributes for each category
submission_df['len'] = 0

# Fill predictions in the submission file
for i in range(10):
    submission_df[f'attr_{i+1}'] = label_encoders[f'attr_{i+1}'].inverse_transform(predictions[:, i])

# Fill in 'dummy_value' for missing attributes for categories with less than 10 attributes
category_attributes = pd.read_parquet('category_attributes.parquet')
for index, row in submission_df.iterrows():
    category = row['Category']
    no_of_attributes = category_attributes[category_attributes['Category'] == category]['No_of_attribute'].values[0]
    
    # Update the 'len' column
    submission_df.at[index, 'len'] = no_of_attributes

    # Fill with 'dummy_value' for attributes that don't exist for this category
    for i in range(no_of_attributes, 10):
        submission_df.at[index, f'attr_{i+1}'] = 'dummy_value'

# Save the final submission file
submission_df.to_csv('submission5.csv', index=False)

print("Submission file saved as submission5.csv")
