
logs_docker_build:
	cd ../filebeat && docker compose build

embeddings_docker_build:
	cd ../embeddings && docker build -t corfudb/ai-log-embeddings:latest .
	
run_log_processing: logs_docker_build
	rm -rf logs/out/messages
	docker run -ti --rm --name ai-log-processing -v $(CURDIR)/logs:/logs corfudb/ai-log-processing:latest

run_embeddings: #embeddings_docker_build #run_log_processing
	#rm -rf logs/out/embeddings

	#docker run \
	#	-ti --rm --user 1000:1000 \
	#	--name ai-log-embeddings \
	#	-v $(CURDIR)/logs:/logs \
	#	corfudb/ai-log-embeddings:latest
	
	python3 ../embeddings/embeddings.py

run_ai:
	echo "ai"
	python3 ../ai/ai-app.py

run: run_embeddings