set -e 
set -x

docker build -t corfudb/ai-log-embeddings:latest .

docker run \
    -ti --rm --user 1000:1000 \
    --name ai-log-embeddings \
    -v /home/ai-lap/embeddings:/logs \
    corfudb/ai-log-embeddings:latest