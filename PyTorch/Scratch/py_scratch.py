import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

data_dir = '../../Dataset/Edited_Enhanced/'

# image transformation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7220, 0.6931, 0.6760], std=[0.2378, 0.2450, 0.2467])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7220, 0.6931, 0.6760], std=[0.2378, 0.2450, 0.2467])
])

dataset = ImageFolder(root=data_dir, transform=train_transform)

# data splitting
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# patch embedding class
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  
        x = x.flatten(2)  
        x = x.transpose(1, 2)  
        return x

# position embedding class
class PositionEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
    
    def forward(self, x):
        return x + self.pos_embedding

# transformer encoder block class
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        
  
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.norm2(x)
        return x


# VIT model class
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=46, embed_dim=768, num_heads=12, hidden_dim=3072, num_layers=12, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = PositionEmbedding(self.patch_embed.num_patches, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.transformer = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers) 
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embed(x)
        
        for layer in self.transformer:
            x = layer(x)
        
        x = self.norm(x)
        cls_token_final = x[:, 0]
        logits = self.fc(cls_token_final)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(dataset.classes)
model = VisionTransformer(num_classes=num_classes)  
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# learning rate scheduler for the model
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy

def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_accuracy = 100 * correct / total
    
    unique_labels = sorted(set(all_labels))
    target_names = [dataset.classes[i] for i in unique_labels]

    report = classification_report(all_labels, all_predicted, target_names=target_names, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    
    return epoch_loss, epoch_accuracy, precision, recall, f1



num_epochs = 100
log_file_path = 'py_scratch.log' 

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
val_precisions = []
val_recalls = []
val_f1_scores = []

best_val_loss = float('inf')
epochs_no_improve = 0
best_epoch = 0
best_model = None


# training the model and logging the results
with open(log_file_path, 'a') as log_file:
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = validate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1_scores.append(val_f1)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Epoch {epoch+1}/{num_epochs}")
        print(f"[{current_time}] Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"[{current_time}] Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        print(f"[{current_time}] Validation Precision: {val_precision:.2f}, Recall: {val_recall:.2f}, F1-score: {val_f1:.2f}%")
        
      
        log_file.write("\n")
        log_file.write(f"[{current_time}] Epoch {epoch+1}/{num_epochs}\n")
        log_file.write(f"[{current_time}] Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%\n")
        log_file.write(f"[{current_time}] Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")
        log_file.write(f"[{current_time}] Validation Precision: {val_precision:.2f}, Recall: {val_recall:.2f}, F1-score: {val_f1:.2f}%\n")
        
     
        scheduler.step(val_loss)
        
        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_epoch = epoch
            best_model = model.state_dict()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= 10: 
            print(f"Early stopping at epoch {epoch+1}")
            log_file.write(f"Early stopping at epoch {epoch+1}\n")
            break


torch.save(best_model, 'py_scratch.pth')

plt.figure(figsize=(12, 5))

# plot loss
plt.subplot(1, 2, 1)
plt.plot(range(1, best_epoch + 2), train_losses[:best_epoch + 1], label='Train Loss')
plt.plot(range(1, best_epoch + 2), val_losses[:best_epoch + 1], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# plot accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, best_epoch + 2), train_accuracies[:best_epoch + 1], label='Train Accuracy')
plt.plot(range(1, best_epoch + 2), val_accuracies[:best_epoch + 1], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.show()

test_loss, test_accuracy, test_precision, test_recall, test_f1 = validate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
print(f"Test Precision: {test_precision:.2f}, Recall: {test_recall:.2f}, F1-score: {test_f1:.2f}%")
