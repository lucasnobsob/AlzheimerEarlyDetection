import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.autograd import Variable

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
    def save_gradients(self, grad):
        self.gradients = grad
    
    def save_activations(self, activations):
        self.activations = activations

    def generate_gradcam(self, image_tensor, target_class):
        """Gera o Grad-CAM para a imagem dada."""
        self.model.eval()
        image_tensor = Variable(image_tensor, requires_grad=True)
        
        # Passa pela rede
        output = self.model(image_tensor)
        
        # Pega a pontuação da classe alvo
        score = output[0, target_class]
        
        # Zero os gradientes
        self.model.zero_grad()
        
        # Faz o backpropagation
        score.backward()

        # Pega os gradientes da camada de ativação alvo
        gradients = self.gradients
        activations = self.activations
        
        # Calcula o Grad-CAM
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        grad_cam = torch.sum(weights * activations, dim=1, keepdim=True)
        grad_cam = grad_cam.squeeze()
        grad_cam = np.maximum(grad_cam, 0)  # ReLU
        grad_cam = cv2.resize(grad_cam, (image_tensor.size(2), image_tensor.size(3)))
        grad_cam = grad_cam / np.max(grad_cam)  # Normaliza
        return grad_cam
    
    def hook_layers(self):
        # Registra o hook para salvar ativações e gradientes
        def forward_hook(module, input, output):
            self.save_activations(output)
        
        def backward_hook(module, grad_input, grad_output):
            self.save_gradients(grad_output[0])
        
        # Hook na camada de interesse
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def preprocess_image(self, image_bytes):
        """Pre-processa a imagem para o modelo."""
        # Aplique as transformações necessárias, por exemplo, redimensionamento e normalização
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))  # Redimensiona para o tamanho esperado pelo modelo
        image = np.transpose(image, (2, 0, 1))  # Converte para formato (C, H, W)
        image = torch.tensor(image).float()
        image = image.unsqueeze(0)  # Adiciona uma dimensão de batch
        return image / 255.0  # Normaliza para [0, 1]

    def create_heatmap_and_class(self, image_tensor):
        """Cria o mapa de calor Grad-CAM e faz a previsão da classe."""
        self.hook_layers()
        
        # Passa a imagem pela rede
        output = self.model(image_tensor)
        
        # Pega a classe com maior probabilidade
        predicted_class = output.argmax(dim=1).item()
        predicted_probability = output[0, predicted_class].item()
        
        # Gera o Grad-CAM
        grad_cam = self.generate_gradcam(image_tensor, predicted_class)
        
        # Converte o Grad-CAM para uma imagem
        grad_cam = np.uint8(255 * grad_cam)  # Escala para valores de 0 a 255
        heatmap = cv2.applyColorMap(grad_cam, cv2.COLORMAP_JET)
        
        # Leva a imagem original (sem a sobreposição) para RGB
        img = image_tensor.squeeze().permute(1, 2, 0).numpy()  # De volta ao formato (H, W, C)
        img = np.uint8(255 * img)  # Converte para valores de 0 a 255
        
        # Cria a sobreposição
        overlayed_image = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
        
        return overlayed_image, predicted_class, predicted_probability
