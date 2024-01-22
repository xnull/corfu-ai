from transformers import pipeline
import json

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

sequence_to_classify = """2023-11-03T20:33:32.747Z | ERROR |                       worker-0 |        o.c.i.NettyServerRouter | Error in handling inbound message
io.netty.handler.codec.DecoderException: javax.net.ssl.SSLHandshakeException: error:14094412:SSL routines:ssl3_read_bytes:sslv3 alert bad certificate
	at io.netty.handler.codec.ByteToMessageDecoder.callDecode(ByteToMessageDecoder.java:480)
	"""

#sequence_to_classify = "alert bad certificate"

candidate_labels = ['error', 'debug', 'handshake error', 'bad certificate', 'successful ssl handshake']
out = classifier(sequence_to_classify, candidate_labels)

print(json.dumps(out, indent=2))

