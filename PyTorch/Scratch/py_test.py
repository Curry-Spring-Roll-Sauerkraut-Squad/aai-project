import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image

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

class PositionEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
    
    def forward(self, x):
        return x + self.pos_embedding

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


def load_model(model_path, num_classes):
    model = VisionTransformer(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')  
    image = transform(image).unsqueeze(0)  
    return image


def predict_image(model, image_tensor, class_names):
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    class_idx = predicted.item()
    return class_names[class_idx]


model_path = 'py_scratch.pth'
data_dir = '../../Dataset/Edited_Enhanced/'  


dataset = ImageFolder(root=data_dir, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
]))
class_names = dataset.classes


model = load_model(model_path, num_classes=len(class_names))

image_path = '../../Test/image2.jpg'
image_tensor = preprocess_image(image_path)
predicted_class = predict_image(model, image_tensor, class_names)


print(f"Predicted class: {predicted_class}")
