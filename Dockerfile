FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs configs
COPY prompts prompts
COPY src src
COPY scripts scripts

ENTRYPOINT ["python", "-m"]
