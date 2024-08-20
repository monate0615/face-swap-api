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
url_num = len(server_urls)

lock = threading.Lock()

def option_img2img(url):
    option_payload = {
    "sd_model_checkpoint": "Deliberate_v6.safetensors" # "cyberrealistic_v50.safetensors" # you can use it also, choose the best
    }

    requests.post(url=f'{url}/sdapi/v1/options', json=option_payload)

def main():
    src_imgs = os.listdir('inputs/source-images')
    tar_imgs = os.listdir('inputs/target-images')

    # for url in server_urls:
    #     option_img2img(url)
    
    with ThreadPoolExecutor(max_workers=url_num) as executor:
        futures = []
        for src_img in src_imgs:
            for tar_img in tar_imgs:
                futures.append(executor.submit(process, src_img, tar_img))

def process(source, target):
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
    
    output = f'outputs/api/output_{source}_{target}_'
    try:
        src = Image.open('inputs/source-images/' + source)
        tar = Image.open('inputs/target-images/' + target)
    except Exception as e:
        print(e)

    src_bytes = io.BytesIO()
    tar_bytes = io.BytesIO()
    width = src.width
    height = src.height

    max_size = max(width, height)
    rate = 760. / max_size
    
    height = height * rate
    width = width * rate
        
    src.resize((width, height))
    print(f"{width}, {height}")
    
    src.save(src_bytes, format='PNG')
    tar.save(tar_bytes, format='PNG')
    src_base64 = base64.b64encode(src_bytes.getvalue()).decode('utf-8')
    tar_base64 = base64.b64encode(tar_bytes.getvalue()).decode('utf-8')

    # ReActor arguments
    args = [
        tar_base64,
        True,
        '0',
        '0',
        "inswapper_128.onnx",
        "CodeFormer",
        1,
        True,
        "None",
        1,
        1,
        False,       # swap in source image
        True,      # swap in generated image
        1,
        0,
        0,
        False,
        0.5,
        False,
        False,
        "CUDA",
        False,
        0,
        "None",
        "",
        None,
        False,
        False,
        0.5,
        0
    ]

    # The args for ReActor can be found by 
    # requests.get(url=f'{address}/sdapi/v1/script-info')

    payload = {
        "init_images": [src_base64],
        "steps": 50,
        "cfg_scale": 7,
        "width": width,
        "height": height,
        "batch_size": 1,
        "denoising_strength": 0.0125,
        "alwayson_scripts": {
            "reactor": {
                "name": "reactor",
                "is_alwayson": True,
                "is_img2img": True,
                "args" : args
            }
        }
    }

    try:
        print('Working... Please wait...')
        result = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload, timeout = 200)
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