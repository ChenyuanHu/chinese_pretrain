import requests
import json
import sys
import os

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/src")
from tokenizer import Tokenizer

def generate_text(prompt, model="qwen2.5:7b"):
    """
    使用本地运行的Ollama生成文本
    
    Args:
        prompt (str): 输入提示词
        model (str): 要使用的模型名称，默认为qwen2.5:14b
    
    Returns:
        str: 生成的文本
    """
    url = "http://localhost:11434/api/chat"
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # 检查是否有错误
        
        # print(f"API 响应状态码: {response.status_code}")
        # print(f"API 响应内容: {response.text}")
        
        try:
            result = response.json()
            # 从message字段中提取content
            if "message" in result and "content" in result["message"]:
                return result["message"]["content"]
            exit(1)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            exit(1)
            
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"错误响应内容: {e.response.text}")
        exit(1)


if __name__ == "__main__":
    # 测试生成
    tokenizer = Tokenizer()

    result = "请给我关于计算机领域的名词"
    with open("result.txt", "w") as f:
        while True:
            result = generate_text("请根据下文，挑选一个不相同的但是有一点点相关话题，总体围绕着计算机运维和开发领域的细节知识，例如可以是某些开源组件的使用。你只需要回答话题名称即可，不用回答别的：" + result)

            print("\nOllama 返回结果:")
            print(result)

            result = generate_text("请这对下面主题，总结它的一些细节知识，甚至操作命令、代码和技巧：" + result)
            
            print("\nOllama 返回结果:")
            print(result)
            text = tokenizer.bos_token + result + tokenizer.eos_token
            f.write(text)
