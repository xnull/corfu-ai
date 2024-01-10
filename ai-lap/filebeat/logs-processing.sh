
rm -rf logs/out/**

docker-compose build
docker run -ti --rm --name ai-log-processing -v $(pwd)/logs:/logs corfudb/ai-log-processing:latest