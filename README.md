# AlzheimerEarlyDetection
This repository was created for the final project of the postgraduate course in Data Science, aiming to provide an API that will return a classification based on a set of clinical data or a list of MRI images.


# Alzheimer Diagnosis API

Esta API utiliza modelos de aprendizado de máquina para diagnosticar Alzheimer com base em imagens de ressonância magnética e dados clínicos.

## 🚀 Como Executar com Docker

### 1️⃣ **Construir a Imagem Docker**
Execute o seguinte comando no terminal, dentro da pasta do projeto:

```bash
docker build -t alzheimer-api .
```

### 2️⃣ **Executar o Contêiner**
Após a construção da imagem, rode o contêiner com:

```bash
docker run -p 8000:8000 alzheimer-api
```

A API estará acessível em `http://localhost:8000`.

### 3️⃣ **Acessar o Contêiner em Modo Interativo (Opcional)**
Se precisar acessar o terminal do contêiner:

```bash
docker run -it --rm -p 8000:8000 alzheimer-api bash
```

### 4️⃣ **Garantir Persistência da Pasta `datasets`**
Caso queira garantir que a pasta `datasets` persista mesmo após a remoção do contêiner:

```bash
docker run -p 8000:8000 -v $(pwd)/datasets:/app/datasets alzheimer-api
```

## 📂 Estrutura do Projeto
```
/project-root
│── datasets/              # Dados para treino e teste
│── models/                # Modelos treinados
│── main.py                # Código principal da API FastAPI
│── AlzheimerCNN.py        # Modelo de CNN para diagnóstico
│── ClinicalData.py        # Processamento de dados clínicos
│── gradcam_utils.py       # Implementação do Grad-CAM
│── requirements.txt       # Dependências do projeto
│── Dockerfile             # Configuração do Docker
│── README.md              # Instruções do projeto
```

## 📜 **Dockerfile**
O arquivo `Dockerfile` está configurado da seguinte maneira:

```dockerfile
# Usar uma imagem oficial do Python
FROM python:3.10

# Definir o diretório de trabalho
WORKDIR /app

# Copiar arquivos do projeto para o contêiner
COPY . /app

# Instalar as dependências
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expor a porta da API
EXPOSE 8000

# Comando para rodar a API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🛠 **Requisitos**
- Docker instalado
- Pasta `datasets/` disponível para armazenamento dos dados

## 📬 **Contato**
Caso tenha dúvidas, sinta-se à vontade para entrar em contato!

