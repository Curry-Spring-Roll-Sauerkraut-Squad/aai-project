import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

data_dir = '/Users/kshitij/Downloads/yoga_poses'
dataset = ImageFolder(root=data_dir, transform=transform)
classes = dataset.classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name)
model.to(device)

state_dict = torch.load('torch_model.pth', map_location=device)


num_labels = state_dict['classifier.weight'].shape[0]
model.classifier = torch.nn.Linear(model.classifier.in_features, num_labels)
model.load_state_dict(state_dict)
model.eval()  #

image_processor = ViTImageProcessor.from_pretrained(model_name)

# to test a given image
def test_single_image(image_path, model, image_processor, device):
    model.eval()
    image = Image.open(image_path).convert('RGB') 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    inputs = transform(image).unsqueeze(0).to(device) 
    
    with torch.no_grad():
        outputs = model(pixel_values=inputs)
    
    logits = outputs.logits
    predicted_class_idx = logits.argmax(1).item()
    
    predicted_class = classes[predicted_class_idx]
    
    return predicted_class

image_path = "/Users/kshitij/Downloads/image3.jpg"
predicted_class = test_single_image(image_path, model, image_processor, device)
print(f"Predicted Class: {predicted_class}")
