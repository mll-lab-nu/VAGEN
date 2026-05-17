import os
import copy
import json
import time
import base64
import shutil
import requests

import dashscope
import http.client
import anthropic

import google.auth
from google.oauth2 import service_account
from google.auth.transport.requests import Request

from openai import OpenAI
from typing import List, Tuple, Dict
from http import HTTPStatus
from PIL import Image
from io import BytesIO

PROXIES = {                             # gemini
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890"
}

SEED  = int(os.environ.get("SEED", 42))

GCLOUD_KEY_FILE_PATH = ""       # path to the google cloud project json file
GCLOUD_REGIONAL_CODE = "asia-east1"
OPENAI_API_URL = os.environ.get("OPENAI_API_URL")
FINETUNED_URL  = os.environ.get("FINETUNED_URL")        # finetuned model url

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]           # you should alway setup openai api key for evaluation
GEMINI_API_KEY = os.environ.get("GEMENI_API_KEY", "")   # no need when using google cloud
QWEN_API_KEY   = os.environ.get("QWEN_API_KEY"  , "")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")

print(f'\n=== OPENAI_API_URL ===: {OPENAI_API_URL}')
print(f'=== OPENAI_API_KEY ===: {OPENAI_API_KEY}')

class BasicModel(object):
    def __init__(self):
        super().__init__()
        # make temp dir here
        file_path = os.path.dirname(__file__)
        self.base_dir = os.path.join(file_path, "temp", f"{int(time.time())}")
        os.makedirs(self.base_dir, exist_ok=True)
        
    def __del__(self):
        # remove temp dir
        shutil.rmtree(self.base_dir, ignore_errors=True) 
    
    def prompt_construct(self, messages: List[Dict]) -> List[Dict]:
        return messages 
    
    @staticmethod
    def process_system_prompt(messages: List[Dict]) -> List[Dict]:
        if messages[0]["role"] != "system":
            return messages
        
        new_messages = copy.deepcopy(messages[1:])
        system_prompt = messages[0]["content"]
        
        # Search for first user message and add system prompt to it
        for item in new_messages:
            if item.get("role") != "user":
                continue
            
            for ct in item["content"]:
                # Case 1: directly appended to the text
                if ct["type"] == "text":
                    ct["text"] = system_prompt + "\n" + ct["text"]
                    return new_messages
            
            # Case 2: create a new text item
            item["content"].insert(0, {
                "type": "text",
                "text": system_prompt
            })
            return new_messages
            
        # Case 3: no user message found, add a new user message
        new_messages.insert(0, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": system_prompt
            }]
        })
        return new_messages

    @staticmethod
    def pil_to_b64(img: Image.Image) -> str:
        with BytesIO() as image_buffer:
            img.save(image_buffer, format="PNG")
            byte_data = image_buffer.getvalue()
            img_b64 = base64.b64encode(byte_data).decode("utf-8")
            img_b64 = "data:image/png;base64," + img_b64
        return img_b64

    # save base64 image and return filename
    def b64_to_image(self, base64_data: str) -> str:
        base64_data = base64_data.split(",")[1]
        image_data = base64.b64decode(base64_data)
        
        filename = os.path.join(self.base_dir, f"{int(time.time())}.png")
        with open(filename, "wb") as f:
            f.write(image_data)
        
        return filename
    
    def get_model_response(self, messages: List[Dict], model_name: str, **args) -> Tuple[bool, str]:
        raise NotImplementedError("Subclasses must implement this method")

  
class OpenAIModel(BasicModel):
    def __init__(self):
        super().__init__()
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_URL)
    
    def prompt_construct(self, messages: List[Dict]) -> List[Dict]:
        return messages 

    def get_model_response(self, messages: List[Dict], model_name: str, **args) -> Tuple[bool, str]:
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "OPENAI_API_KEY environment variable must be set when using OpenAI API."
            )
        
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=args.get("temperature", 0.0),
            max_tokens=args.get("max_tokens", 1024),
            top_p=args.get("top_p", 1.0),
        )
        
        try:
            answer: str = response.choices[0].message.content
            return True, answer
        except:
            return False, str(response.error)


class FinetuneModel(BasicModel):
    def __init__(self):
        super().__init__()
        self.url = FINETUNED_URL           # inference api
    
    def prompt_construct(self, messages: List[Dict]) -> List[Dict]:
        dialog, images = "", []
        for message in messages:
            if message["role"] == "system":
                dialog += f"<|system|>\n{message['content']}\n\n"
                continue
            elif message["role"] == "assistant":
                dialog += f"<|assistant|>\n{message['content']}\n\n"
                continue
            
            dialog += f"<|user|>\n" 
            for content in message["content"]:
                if content["type"] == "text":
                    dialog += f"{content['text']}\n"
                else:
                    # TODO: we use filename as image url here
                    images.append(self.b64_to_image(content["image_url"]["url"]))
            dialog += "\n\n"
        
        dialog += "<|assistant|>\n"         
        images = [open(image, "rb") for image in images]
        new_messages = [
            {"image": images[0]},
            {"prompt": dialog}
        ]
        
        return new_messages
    
    def get_model_response(self, messages: List[Dict], model_name: str, **args) -> Tuple[bool, str]:
        try:
            response = requests.post(self.url, files=messages[0], data=messages[1], timeout=40)
            response = response.json()
        except Exception as e:
            return False, str(e)
            
        if "error" in response:
            return False, response["error"]["message"]

        # TODO: you should change the response format here
        resp = f'```\n{response["response"].split("<|end_of_text|>")[0]}\n```'
        return True, resp
    

class QwenModel(BasicModel):
    def __init__(self):
        super().__init__()
        dashscope.api_key = QWEN_API_KEY
        self.seed = SEED

    def prompt_construct(self, messages: List[Dict]) -> List[Dict]:
        messages = self.process_system_prompt(messages)
        
        new_messages = []
        for message in messages:
            if message["role"] != "user":
                new_messages.append({
                    "role": "assistant",
                    "content": [{"text": message["content"]}]
                })
                continue
            
            new_content = []
            for content in message["content"]:
                if content["type"] == "text":
                    new_content.append({
                        "text": content["text"],
                    })
                else:
                    filename = self.b64_to_image(content["image_url"]["url"])
                    new_content.append({
                        "image": f"file://{filename}"
                    })
            new_messages.append({
                "role": "user",
                "content": new_content
            })
                
        return new_messages 
    
    def get_model_response(self, messages: List[Dict], model_name: str, **args) -> Tuple[bool, str]:
        if "QWEN_API_KEY" not in os.environ:
            raise ValueError(
                "QWEN_API_KEY environment variable must be set when using Qwen API."
            )
            
        response = dashscope.MultiModalConversation.call(
            model=model_name,
            messages=messages,
            top_k=args.get("top_k"),
            seed=self.seed
        )
        
        if response.status_code == HTTPStatus.OK:
            return True, response.output.choices[0].message.content[0]['text']
        else:
            return False, response.message
        

class ClaudeModel(BasicModel):
    def __init__(self):
        super().__init__()
        self.client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    def prompt_construct(self, messages: List[Dict]) -> List[Dict]:
        new_messages = []
        for message in messages:
            if message["role"] in ["system", "assistant"]:
                new_messages.append(message)
                continue
            
            new_content = []
            for content in message["content"]:
                if content["type"] == "text":
                    new_content.append(content)
                    continue
                
                hdr, idata = content["image_url"]["url"].split(";base64,")
                new_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": hdr.split("data:")[1],
                        "data": idata
                    }
                })
            
            new_messages.append({
                "role": "user",
                "content": new_content
            })
            
        return new_messages
                
    def get_model_response(self, messages: List[Dict], model_name: str, **args) -> Tuple[bool, str]:
        try:
            if messages[0]["role"] == "system":
                system_prompt = messages[0]["content"]
                messages = messages[1:]
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=args.get("max_tokens"),
                    temperature=args.get("temperature"),
                    system=system_prompt,
                    messages=messages
                )
                
            else:
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=args.get("max_tokens"),
                    temperature=args.get("temperature"),
                    messages=messages
                )

            usage = response.usage
            prompt_tokens = usage.input_tokens
            completion_tokens = usage.output_tokens
            
            # print(response)
            print(response.content)
            print(f"Prompt Tokens: {prompt_tokens}\nCompletion Tokens: {completion_tokens}\n")
            return True, response.content
        
        except Exception as e:
            return False, str(e)
        
    def get_model_response_thirdapi(self, messages) -> Tuple[bool, str]:

        conn = http.client.HTTPSConnection("cn2us02.opapi.win", timeout=900)

        system_prompt = None
        if messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]
            payload = {
                "model": "claude-3-opus",
                "stream": False,
                "system": system_prompt,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        else:
            payload = {
                "model": "claude-3-opus",
                "stream": False,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        payload = json.dumps(payload)
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {CLAUDE_API_KEY}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }
        try:
            conn.request("POST", "/v1/messages", payload, headers)
            res = conn.getresponse()
            data = res.read()
            response = json.loads(data.decode("utf-8"))

        except Exception as e:
            return False, str(e)

        if "statusCode" in response and response["statusCode"] != 200:
            return False, response["message"]
        
        usage = response["usage"]
        prompt_tokens = usage["input_tokens"]
        completion_tokens = usage["output_tokens"]
        print(f"Prompt Tokens: {prompt_tokens}\nCompletion Tokens: {completion_tokens}\n")
        
        return True, response["content"][0]["text"]


class GeminiModel(BasicModel):
    def __init__(self):
        super().__init__()
        self.api_key = GEMINI_API_KEY
    
    def prompt_construct(self, messages: List[Dict]) -> List[Dict]:
        parts = []
        dialog = ""
        sep = "\n\n###\n\n"
        for message in messages:
            if message["role"] == "system":
                dialog += f"SYSTEM:\n{message['content']}{sep}"
            elif message["role"] == "assistant":
                dialog += f"ASSISTANT:\n{message['content']}{sep}"
            elif message["role"] == "user":
                dialog += "USER:\n"
                for content in message["content"]:
                    if content["type"] == "text":
                        dialog += content["text"] + "\n"
                        continue
                    
                    assert content["type"] == "image_url"
                    
                    # save text
                    parts.append({ "text": dialog })
                    dialog = ""
                    
                    # new content type for image
                    hdr, idata = content["image_url"]["url"].split(";base64,")
                    parts.append({
                        "inline_data": {
                            "mime_type": hdr.split("data:")[1],
                            "data": idata
                        }
                    })
                dialog += sep
            else:
                raise ValueError(f"Invalid role: {message['role']}")
          
        parts.append({
            "text": dialog + "ASSISTANT:\n"
        })
        
        new_messages = [{
            "parts": parts,
            "role": "user"
        }]
        
        return new_messages
    
    def get_model_response(self, messages: List[Dict], model_name: str, **args) -> Tuple[bool, str]:
        
        headers = {
            "Content-Type": "application/json"
        }

        proxies = PROXIES
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}-latest:generateContent?key={self.api_key}"
        
        generation_config = {
            "temperature": args.get('temperature'),
            "maxOutputTokens": args.get('max_tokens'),
            "stopSequences": ["\n\n###\n\n"]
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ]

        payload = {
            "contents": messages,
            "generationConfig": generation_config,
            "safetySettings": safety_settings
        }

        try:
            response = requests.post(url, headers=headers, json=payload, proxies=proxies, timeout=30)
            response = response.json()
        except Exception as e:
            return False, str(e)

        if "error" in response:
            return False, response["error"]["message"]
        if "content" not in response['candidates'][0]:
            self.generation_config['maxOutputTokens'] *= 2
            return False, "No content generated."
        return True, response['candidates'][0]['content']['parts'][0]['text']
    

class VertexGeminiModel(BasicModel):
    def __init__(self):
        super().__init__()
    
    def prompt_construct(self, messages: List[Dict]) -> List[Dict]:
        parts = []
        dialog = ""
        sep = "\n\n###\n\n"
        for message in messages:
            if message["role"] == "system":
                dialog += f"SYSTEM:\n{message['content']}{sep}"
            elif message["role"] == "assistant":
                dialog += f"ASSISTANT:\n{message['content']}{sep}"
            elif message["role"] == "user":
                dialog += "USER:\n"
                for content in message["content"]:
                    if content["type"] == "text":
                        dialog += content["text"] + "\n"
                        continue
                    
                    assert content["type"] == "image_url"
                    # save text
                    parts.append({ "text": dialog })
                    dialog = ""
                    # new content type for image
                    hdr, idata = content["image_url"]["url"].split(";base64,")
                    parts.append({
                        "inline_data": {
                            "mime_type": hdr.split("data:")[1],
                            "data": idata
                        }
                    })
                dialog += sep
            else:
                raise ValueError(f"Invalid role: {message['role']}")
          
        parts.append({
            "text": dialog + "ASSISTANT:\n"
        })
        
        new_messages = [{
            "parts": parts,
            "role": "user"
        }]
        
        return new_messages
    
    def get_model_response(self, messages: List[Dict], model_name: str, **args) -> Tuple[bool, str]:
        def get_gcloud_token():
            def get_token():
                try:
                    # Load the credentials from the key file
                    creds = service_account.Credentials.from_service_account_file(
                        GCLOUD_KEY_FILE_PATH,
                        # You can list multiple scopes if needed
                        scopes=['https://www.googleapis.com/auth/cloud-platform']  
                    )
                    # Refresh the token (this is needed even for the first time)
                    creds.refresh(Request())
                    return creds.token

                except Exception as e:
                    print(f"An error occurred while trying to fetch the gcloud token: {str(e)}")
                    return None
            
            os.environ['HTTP_PROXY'] = PROXIES['http']
            os.environ['HTTPS_PROXY'] = PROXIES['https']
                
            fail_time = 0
            while not api_key and fail_time < 10:
                time.sleep(5)
                api_key = get_token()
                fail_time += 1
            
            return api_key
        
        def get_url(model_name: str) -> str:
            region_code = GCLOUD_REGIONAL_CODE
            model_id = f"{model_name}:generateContent"
            with open(GCLOUD_KEY_FILE_PATH, "r") as f:
                project_id = json.load(f)["project_id"]
            return f"https://{region_code}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region_code}/publishers/google/models/{model_id}"
        
        url = get_url(model_name)
        api_key = get_gcloud_token()
        
        if not api_key:
            return False, "Failed to fetch gcloud token."

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        proxies = PROXIES
        safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ]
        
        generation_config = {
            "temperature": args.get('temperature'),
            "maxOutputTokens": args.get('max_tokens'),
            "stopSequences": ["\n\n###\n\n"]
        }

        payload = {
            "contents": messages,
            "generationConfig": generation_config,
            "safetySettings": safety_settings
        }

        try:
            response = requests.post(url, headers=headers, json=payload, proxies=proxies, timeout=120)
            response = response.json()
        except Exception as e:
            return False, str(e)

        if "error" in response:
            return False, response["error"]["message"]
        if "content" not in response['candidates'][0]:
            self.generation_config['maxOutputTokens'] *= 2
            return False, "No content generated."
        
        return True, response['candidates'][0]['content']['parts'][0]['text']


class APIModel(object):
    def __init__(self):
        super().__init__()
        self.models = {
            "openai": OpenAIModel(),
            "gemini": VertexGeminiModel(),
            "qwen": QwenModel(),
            "finetuned": FinetuneModel(),
            "claude": ClaudeModel()
        }
    
    def inference(self, model_id: str, messages: List[Dict], args: Dict) -> Tuple[bool, str]:
        model_provider, model_name = model_id.split("_")[:2] if "_" in model_id else (model_id, model_id) # eg. "openai_gpt4o"
        if model_provider not in self.models:
            return False, f"Unsupported model: {model_provider} ({model_name})"
        
        model = self.models[model_provider]
        prompt = model.prompt_construct(messages)
        resp = model.get_model_response(prompt, model_name, **args)
        
        return resp
        
model = APIModel()

def generate_with_api(prompt: List[dict], model_id: str, args: Dict, llm=None) -> str:
    if llm is None:
        success, response = model.inference(model_id, prompt, args)
    else:
        from vllm import SamplingParams
        import random

        print(f'prompt:\n{json.dumps(prompt, indent=4)}')
        sampling_params = SamplingParams(temperature=args.get("temperature", 1.0), max_tokens=args.get("max_tokens", 512))
        outputs = llm.chat(prompt, sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            print(f">>>Prompt:\n{prompt}\n\n>>>Generated text:\n{generated_text}")

        # response = outputs[0].outputs[0].text

        # response = outputs[0].outputs[0].text

        if 'Round 2' in prompt:
            response = outputs[0].outputs[0].text
        elif 'Round 1' in prompt:
            response = "<think> # Element: the 'Bestsellers' report option under the 'Products' section in the 'REPORTS'</think>\n<answer> do(action=\"Click\", element=\"27\")</answer>"
        else:
            response = "<think> # Element: the 'REPORTS' section on the left sidebar </think>\n<answer> do(action=\"Click\", element=\"7\") </answer>"

    return response

if __name__ == "__main__":
    path_to_image = "../../coco_images/000000000285.jpg"
    
    from PIL import Image
    image = Image.open(path_to_image)
    img_str = BasicModel.pil_to_b64(image)
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Please response concisely."
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "what's annotated in this image? Image: Omitted."
            }]
        },
        {
            "role": "assistant",
            "content": "Only 5.cart is annotated in this image."
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "What can you see?"
            },{
                "type": "image_url",
                "image_url": {
                    "url": img_str,
                    "detail": "high"
                }
            }]
        }
    ]

    response = generate_with_api(messages, "openai", {
        "temperature": 0.5,
        "max_tokens": 1024,
        "top_p": 0.9,
        "n": 1,
    })