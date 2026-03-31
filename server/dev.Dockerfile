FROM python:3.12

WORKDIR /app

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip config set global.trusted-host mirrors.aliyun.com

# Copy requirements first for better caching
COPY server/requirements.txt .
RUN pip install -r requirements.txt

# Install mem0 in editable mode
WORKDIR /app/packages
COPY pyproject.toml .
COPY poetry.lock .
COPY README.md .
COPY mem0 ./mem0
# 启用图数据库加上[graph]
RUN pip install -e . 

# Return to app directory and copy server code
WORKDIR /app
COPY server .

CMD ["uvicorn", "main_async:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
