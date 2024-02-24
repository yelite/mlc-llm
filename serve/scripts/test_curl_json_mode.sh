ENDPOINT=$1
MODEL=$2
curl ${ENDPOINT}/v1/chat/completions -i \
     -H "Authorization: Bearer $OCTOAI_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
         "model" : "'$MODEL'",
         "temperature": 0,
         "messages":[
            {"role": "user", "content": "Hello?"}
        ],
	"response_format": {
	    "type": "json_object",
	    "schema": {"properties": {"answer": {"title": "Answer", "type": "string"}}, "required": ["answer"], "title": "Output", "type": "object"}
        }
     }'
echo "\n"
