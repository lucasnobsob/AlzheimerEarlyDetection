Aqui está a atualização do seu **README** com todas as instruções necessárias:  

---

# **Alzheimer Early Detection API** 🧠  

Esta API utiliza um modelo de Deep Learning para detecção precoce de Alzheimer a partir de imagens de ressonância magnética e dados clínicos.  

## **1. Instalação das Dependências**  

Antes de executar a aplicação, instale as bibliotecas necessárias:  

```sh
pip install pandas torch scikit-learn joblib matplotlib opencv-python fastapi Pillow torchvision
```

---

## **2. Criando e Executando o Container Docker**  

### **2.1. Criar a Imagem Docker**  

Navegue até o diretório onde está o `Dockerfile` e execute:  

```sh
docker build -t alzheimer-api .
```

### **2.2. Rodar o Container**  

Após a criação da imagem, execute o seguinte comando para rodar a API na porta 8000:  

```sh
docker run -p 8000:8000 alzheimer-api
```

Se precisar rodar o container em segundo plano (modo **detached**), use:  

```sh
docker run -d -p 8000:8000 alzheimer-api
```

---

## **3. Publicando no Docker Hub**  

### **3.1. Login no Docker Hub**  

Antes de publicar a imagem no Docker Hub, faça login:  

```sh
docker login
```

### **3.2. Criar uma Tag para a Imagem**  

Substitua `SEU_USUARIO` pelo seu nome de usuário do Docker Hub:  

```sh
docker tag alzheimer-api SEU_USUARIO/alzheimer-api:v1
```

### **3.3. Enviar a Imagem para o Docker Hub**  

```sh
docker push SEU_USUARIO/alzheimer-api:v1
```

Agora a imagem estará disponível no Docker Hub e poderá ser executada em qualquer máquina.  

---

## **4. Rodando a API Localmente com Docker**  

Caso tenha baixado a imagem do Docker Hub, execute:  

```sh
docker run -p 8000:8000 SEU_USUARIO/alzheimer-api:v1
```

Se precisar visualizar os logs:  

```sh
docker logs -f <CONTAINER_ID>
```

---

## **5. Testando a API no Postman**  

### **5.1. Importar a Collection do Postman**  

1. Abra o **Postman**.  
2. Vá até **File > Import**.  
3. Selecione o arquivo `AlzheimerEarlyDetection.postman_collection`.  
4. Agora os endpoints estarão disponíveis para teste.  

### **5.2. Testar um Endpoint**  

- **URL da API**: `http://localhost:8000`  
- **Exemplo de requisição para prever diagnóstico (usando dados clínicos):**  
  - Endpoint: `POST /predict_clinical_data`  
  - Enviar os dados necessários no **body** da requisição (JSON).  

Se precisar de mais ajustes, só avisar! 🚀