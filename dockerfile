# Usar uma imagem oficial do Python
FROM python:3.10

# Definir o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copiar os arquivos necessários para o contêiner
COPY . /app

# Instalar as dependências
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expor a porta da API
EXPOSE 8000

# Comando para rodar a API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
