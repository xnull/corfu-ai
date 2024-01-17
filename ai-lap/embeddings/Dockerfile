FROM pytorch/pytorch

RUN useradd -ms /bin/bash ai
USER ai

WORKDIR /app

# Install dependencies
ADD requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ADD cache.py .
RUN python cache.py

ENV PATH="/home/ai/.local/bin:${PATH}"

ADD embeddings.py .

CMD python embeddings.py
