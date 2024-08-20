import base64, io, requests, json
from PIL import Image, PngImagePlugin
from datetime import datetime, date
import os
from concurrent.futures import ThreadPoolExecutor
import threading

server_urls = {
    'http://10.128.0.8:7860': False,
    'http://10.128.0.8:7861': False,
    'http://10.128.0.8:7862': False
}

prompts = []
neg_prompts = []

index = 46

url_num = len(server_urls)

lock = threading.Lock()

def option_img2img(url):
    option_payload = {
    "sd_model_checkpoint": "Deliberate_v6.safetensors" # "cyberrealistic_v50.safetensors" # you can use it also, choose the best
    }

    requests.post(url=f'{url}/sdapi/v1/options', json=option_payload)

def main():
    
    with ThreadPoolExecutor(max_workers=url_num) as executor:
        futures = []
        for idx, prompt in enumerate(prompts):
            futures.append(executor.submit(process, prompts[idx], neg_prompts[idx]))

def process(prompt, neg_prompt):
    global server_urls
    url = None
    while url is None:
        with lock:
            for tmp in server_urls.keys():
                if not server_urls[tmp]:
                    url = tmp
                    server_urls[tmp] = True
                    break

    print(url)
    
    global index
    with lock:
        index = index + 1
        output = f'output/output_{index}_'
    
    print(index)

    # The args for ReActor can be found by 
    # requests.get(url=f'{address}/sdapi/v1/script-info')

    payload = {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "steps": 50,
        "cfg_scale": 7,
        "width": 512,
        "height": 512
    }

    try:
        print('Working... Please wait...')
        result = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload, timeout = 200)
    except Exception as e:
        print(e)
    finally:
        print('Done! Saving file...')

    print(result)

    if result is not None:
        r = result.json()
        n = 0

        for i in r['images']:
            image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

            output_file = output+'_'+str(n)+'.png'
            try:
                image.save(output_file)
            except Exception as e:
                print(e)
            finally:
                print(f'{output_file} is saved\nAll is done!')
            n += 1
    else:
        print('Something went wrong...')

    with lock:
        server_urls[url] = False

if __name__ == "__main__":
    main()