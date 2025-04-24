import io
from torchvision import transforms
from PIL import Image
from fastapi import HTTPException


def preprocess_image(image_bytes):
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Apply transforms
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        return image_tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
    

def format_probabilities(probabilities):
    """Format probabilities as percentages with 2 decimal places"""
    return {class_name: f"{float(prob)*100:.2f}%" for class_name, prob in probabilities.items()}