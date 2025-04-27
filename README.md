# Alzheimer Early Detection

Projeto para detecção precoce de Alzheimer utilizando Machine Learning e API em Python.
A aplicação disponibiliza endpoints para realizar inferências baseadas em dados clínicos e demográficos.

---

## 🧀 Sobre o Projeto

Este projeto tem como objetivo auxiliar no diagnóstico precoce da Doença de Alzheimer a partir da análise de variáveis cognitivas e de exames clínicos.
O sistema é dividido em duas partes:

- **Modelo de Machine Learning** treinado para prever o nível de comprometimento cognitivo.
- **API REST** desenvolvida em Python (FastAPI) para disponibilizar o modelo de forma acessível.

---

## 📂 Estrutura do Repositório

```
AlzheimerEarlyDetection/
│
├── api/                  # Código da API (FastAPI)
├── model/                # Modelos treinados e utilitários
├── data/                 # Dataset utilizado
├── Dockerfile            # Dockerfile para criar a imagem da aplicação
├── docker-compose.yml    # Orquestração (não obrigatória para uso final)
├── README.md             # Este arquivo
└── requirements.txt      # Dependências da aplicação
```

---

## 🚀 Como Executar o Projeto

Você pode executar a aplicação rapidamente utilizando o **Docker Hub**, sem precisar clonar o repositório.

### 1. Baixar a imagem Docker

```bash
docker pull lucassobdocker/alzheimer-api
```

### 2. Criar e iniciar o container

```bash
docker run -d -p 8000:8000 --name alzheimer-api lucassobdocker/alzheimer-api
```

- O parâmetro `-p 8000:8000` expõe a API na porta 8000 da sua máquina local.
- A API ficará disponível em: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)

### 3. Parar e remover o container (se necessário)

```bash
docker stop alzheimer-api
docker rm alzheimer-api
```

---

## 📚 Documentação da API

Após o container estar rodando, acesse a documentação automática via Swagger:

🔗 [http://localhost:8000/docs](http://localhost:8000/docs)

Lá você poderá testar os endpoints, enviar dados e visualizar as respostas do modelo.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.11**
- **FastAPI**
- **Scikit-learn**
- **Pandas**
- **Docker**
- **Docker Hub**

---

## 🤝 Contribuições

Contribuições são bem-vindas!
Se você quiser propor melhorias, abrir issues ou enviar pull requests, fique à vontade.

---

## 📄 Licença

Este projeto está licenciado sob a licença MIT.
Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 📬 Contato

Lucas Nóbrega Sobral — [LinkedIn](https://www.linkedin.com/in/lucas-sobrinho/)
lucas.ns.93@hotmail.com
