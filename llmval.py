import argparse
import ray, openai
from num2words import num2words 
import time, os, sys, re, json, datetime
import random
from dotenv import load_dotenv
import pandas as pd
from transformers import LlamaTokenizerFast

FRAMEWORKS = ["anyscale","openai","fireworks","vertexai"]

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# TODO(mwk): We use one tokenizer for all models, but we should 
# consider using each framework's tokenizer

#TODO(mwk): too much dependence on args globally. Clean up methods to not directly 
# read from args to facilitate writing scripts. 

tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
sys_prompt = 'You are a helpful assistant that respeonds with the answer in the most concise possible way.'


#NOTE: The defaults are set to mirror our production traffic
def prompt_generator(num_digits=3, min_lines=15, max_lines=1000, file_lines=[]) -> str:
    # Step 1: Generate a random number 
    # Generate the number of digits specified (e.g. if NUM_DIGITS = 3, then
    # any number between 100 and 1000 is OK).
    rnd_num = random.randrange(10**(num_digits-1),10**(num_digits))
    max_lines = max_lines if max_lines < len(file_lines) else len(file_lines)
    rnd_num_lines = random.randrange(min_lines, max_lines)
    rnd_picked_lines = '\n'.join(random.sample(file_lines, rnd_num_lines))

    # Step 2: convert to words. 
    rnd_num_words = num2words(rnd_num)

    # Step 3: convert to a prompt
    user_prompt = f'Convert the following sequence of words into a number: {rnd_num_words}.\nPrint the number first. Then pick {args.req_lines} lines from these poem lines:\n{rnd_picked_lines}'
        
    return user_prompt, rnd_num
  
@ray.remote(num_cpus = 0.001)
def validate(ep_config, sample_lines):
    # The 4 is for the end and start tokens of the messages
    prompt, rnd_num = prompt_generator(args.num_digits, args.min_lines, args.max_lines, sample_lines)
    tokens_in = len(tokenizer.encode(prompt))+len(tokenizer.encode(sys_prompt)) + 4
    messages = [
        {
            'role': 'system',
            'content': sys_prompt
        },
        {
            'role': 'user',
            'content' : prompt
        }
    ]
    
    try:
        st = time.time() 
        response = openai.ChatCompletion.create(
            model = ep_config["model"], 
            messages = messages,
            api_key = ep_config["api_key"], 
            api_base = ep_config["api_base"], 
            max_tokens = args.max_tokens, 
            # Please keep temp at 0. Otherwise increases the number of mismatches. 
            temperature = 0, 
            # Do not set to false. You will get bogus results. 
            stream = True
        )
        words = ''
        ttft = 0
        for tok in response: 
            if tok.choices[0].delta:
                delta = tok.choices[0].delta
                if 'content' in delta:
                    if ttft == 0:
                        ttft = time.time() - st
                    words += delta['content']
        et = time.time() 
        # Get rid of commas. 
        tokens_out = len(tokenizer.encode(words))
        nums = re.findall(r'\d+', words)
        if len(nums) > 0:
            retval = int(nums[0]) 
            valid = 'OK'
            cause = ''
            if retval != rnd_num:
                valid = 'Mismatch'
                cause = f'Input = {rnd_num} output = {retval}\n.Output:\n {words}'
        else:
            valid = 'Mismatch'
            cause = f'Output unparseable. Input = {rnd_num}. Output:\n {words}'
        return (valid, ttft, et-st, tokens_in, tokens_out, cause)
    except Exception as e:
        return ('Exception', -1, -1,-1,-1,str(e))

def endpoint_evaluation(ep_config, sample_lines):
    query_results = []
    num_rounds = int(args.total_requests/args.concur_requests)
    for i in range(num_rounds):
        print(f'Starting round {i}')
        st = time.time()
        futures = [validate.remote(ep_config, sample_lines) for _ in range(args.concur_requests)]
        results = ray.get(futures) 
        query_results.extend(results)
        et = time.time()
        elt = et - st 
        tosleep = args.sleep - elt
        if tosleep > 0:
            print('Sleeping for %.4f seconds'%tosleep)
            time.sleep(tosleep)
        else:
            print(f'No need to sleep for the next round')
        print(f'Round {i} complete')
    return query_results


def results_analysis(query_results, results_dict):
    df = pd.DataFrame(query_results, columns =['valid', 'ttft', 'total_time', 'tokens_in', 'tokens_out', 'cause'])
    ts = int(time.time())
    fn = f'{results_dict["framework"]}-{ts}_raw.json'
    df.to_json(fn)
    print(f'Results saved to: {fn}')
    
    print('Validity results:')
    print(df['valid'].value_counts())
    
    cdf = df[df.valid !='Exception'].copy()
    print(f'Clean DF is: {len(cdf)}')
    cdf['total_tokens_per_s'] = (cdf.tokens_out + cdf.tokens_in)/cdf.total_time
    cdf['out_tokens_per_s'] = cdf.tokens_out/(cdf.total_time-cdf.ttft)
    cdf['inter_tokens_delay'] = (cdf.total_time-cdf.ttft)/cdf.tokens_out
    mean_tokens_in = cdf['tokens_in'].mean() 
    mean_tokens_out = cdf['tokens_out'].mean() 
    mean_ttft = cdf['ttft'].mean()
    gt_3_ttft = len(cdf[cdf['ttft'] > 3])/len(cdf)
    print(f'Mean TTFT: {mean_ttft*1000:.0f} ms (mean tokens in: {mean_tokens_in:.0f}, out: {mean_tokens_out:.0f})')
    print(f'TTFT > 3 s: {gt_3_ttft*100:.2f}%')
    print(f'ITL (out): {cdf.inter_tokens_delay.mean()*1000:.0f} ms/token, mean tokens/s output (out): {cdf.out_tokens_per_s.mean():.2f} token/s')
    
    value_counts = df['valid'].value_counts()
    #Put things in a dictionary and save the results 
    results_dict["end_timestamp"] = datetime.datetime.fromtimestamp(ts).isoformat()  
    results_dict["total_time"] = float(cdf.total_time.mean())
    results_dict["num_valid"] = int(value_counts.get('OK', 0) )
    results_dict["num_exceptions"] = int(value_counts.get('Exception',0))
    results_dict["num_mismatch"] = int(value_counts.get('Mismatch',0))
    results_dict["valid_rate"] = float(results_dict["num_valid"]/results_dict["total_requests"])
    results_dict["mismatch_rate"] = float(results_dict["num_mismatch"]/results_dict["total_requests"])
    results_dict["exception_rate"] = float(results_dict["num_exceptions"]/results_dict["total_requests"])
    results_dict["mean_ttft"] = int(f"{mean_ttft*1000:.0f}")
    results_dict["mean_tokens_in"] = mean_tokens_in
    results_dict["mean_tokens_out"] = mean_tokens_out
    results_dict["total_tokens_per_s"] = float(cdf.total_tokens_per_s.mean())
    results_dict["out_tokens_per_s"] = float(cdf.out_tokens_per_s.mean())
    results_dict["inter_token_delay"] = float(cdf.inter_tokens_delay.mean()*1000)
    results_dict["raw_output"] = fn
    benchmark_result = f"{results_dict['framework']}-{ts}.json"

    with open(benchmark_result, 'w') as fw:
         fw.write(json.dumps(results_dict))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--framework", type=str, default="anyscale", help="Test frame name")
    parser.add_argument("-m","--model", type=str, default="meta-llama/Llama-2-70b-chat-hf", help="model name")
    parser.add_argument("--random-lines-file-name", type=str, default="sonnet.txt", help="Prompt sample file name")
    parser.add_argument("--min-lines", type=int, default=15, help="min number of lines")
    parser.add_argument("--max-lines", type=int, default=50, help="max number of lines")
    parser.add_argument("--num-digits", type=int, default=3, help="number of digits for mismatch search")
    parser.add_argument("--sleep", type=int, default=0, help="sleep between rounds of requests (to deal with rate limiting)")
    parser.add_argument("-c","--concur-requests", type=int, default=10, help="number of concurrent requests")
    parser.add_argument("-r","--total-requests", type=int, default=300, help="number of total requests")
    parser.add_argument("--max-tokens", type=int, default = 384, help="Upper limit on the number of returned tokens to prevent 'runaway LLMs'.")
    parser.add_argument("--req-lines", type=int, default=7, help="Number of lines to request in prompt. Affects tokens out.")
    parser.add_argument("--random-seed", type=int, default=117, help="Random seed to standardize results. By default fully random.")

    args = parser.parse_args()
    load_dotenv()  
    endpoint_config={}
    if args.random_seed >= 0: 
        random.seed(args.random_seed)
    if args.framework not in FRAMEWORKS:
        print(f"Choose a framework in {FRAMEWORKS}")
        sys.exit(0)
    elif args.framework == "anyscale":
        endpoint_config["api_base"]=os.environ['ANYSCALE_API_BASE'] 
        endpoint_config["api_key"]=os.environ['ANYSCALE_API_KEY'] 
    elif args.framework == "openai":
        endpoint_config["api_base"]=os.environ['OPENAI_API_BASE'] 
        endpoint_config["api_key"]=os.environ['OPENAI_API_KEY'] 
    elif args.framework == "fireworks":
        endpoint_config["api_base"]=os.environ['FIREWORKS_API_BASE'] 
        endpoint_config["api_key"]=os.environ['FIREWORKS_API_KEY']     
    
    endpoint_config["framework"] = args.framework
    endpoint_config["model"] = args.model
    
    f = open(args.random_lines_file_name, 'r')
    sample_lines = f.readlines()
    f.close()
    #print(sample_lines)
    
    ## Endpoint evaluation
    query_results = endpoint_evaluation(endpoint_config, sample_lines)
    
    ##Pure debug purpose
    #query_results = pd.read_json("fireworks-1694190875_raw.json")
    
    ## Results Analysis
    args.api_base = endpoint_config["api_base"]
    results_analysis(query_results, vars(args))
    
    
    
