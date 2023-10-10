# llmval

## Usage
1. Provide API base and key in .env file. Check out env_sample.txt
2. Test out Anyscale Endpoint with following command by sending 20 requests   
`python llmval.py -r 20 -m "meta-llama/Llama-2-70b-chat-hf"`
3. Control input token numbers by setting min/max lines, and control output token number by setting req-lines and max_tokens  
`python llmval.py -r 20 -f openai -m "gpt-3.5-turbo" --min-lines 8 --max-lines 10`  
`python llmval.py -r 20 -f openai -m "gpt-3.5-turbo" --req-lines 3 --max-tokens 128`
4. Control sleep between rounds to avoid hitting rate limit  
`python llmval.py -r 20 -f fireworks -m "accounts/fireworks/models/llama-v2-70b-chat" --sleep 10`
5. Output will be saved at **framework-timestamp.json** and **framework-timestamp_raw.json**  

