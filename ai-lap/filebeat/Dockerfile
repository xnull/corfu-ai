FROM docker.elastic.co/beats/filebeat:8.11.3

ADD filebeat.yml /usr/share/filebeat/filebeat.yml

#https://github.com/CorfuDB/corfudb-cloud/blob/bdf0a11f50326db7a19db4db6f5491e8f10301bb/cloud/infrastructure/integration-tools/src/main/kotlin/org/corfudb/cloud/infrastructure/integration/loader/LoaderCommand.kt#L43

ENV BASE_DIR=/logs

CMD filebeat -e --strict.perms=false --once run