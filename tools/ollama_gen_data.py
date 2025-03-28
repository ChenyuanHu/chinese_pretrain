import requests
import json
import sys
import os
import random

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/src")
from tokenizer import Tokenizer

def generate_text_openai(prompt, model="qwen25-32b", api_key=None, api_url=None):
    """
    使用OpenAI API生成文本
    
    Args:
        prompt (str): 输入提示词
        model (str): 要使用的模型名称，默认为gpt-3.5-turbo
        api_key (str): OpenAI API密钥
        api_url (str): API端点URL，默认为OpenAI官方API
    
    Returns:
        str: 生成的文本
    """
    api_key = os.getenv("OPENAI_API_KEY")
    api_url = os.getenv("OPENAI_API_URL")

    if not api_key:
        raise ValueError("必须提供API密钥")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    try:
        # 确保URL以/chat/completions结尾
        if not api_url.endswith("/chat/completions"):
            if api_url.endswith("/"):
                api_url = api_url + "chat/completions"
            else:
                api_url = api_url + "/chat/completions"
                
        print(f"请求URL: {api_url}")
        print(f"请求内容: {payload}")
        
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # 检查是否有错误
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        
        try:
            result = response.json()
            # 从choices数组中提取第一个回复的content
            if "choices" in result and len(result["choices"]) > 0:
                if "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                    return result["choices"][0]["message"]["content"]
            print("API返回的响应格式不符合预期")
            exit(1)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            exit(1)
            
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"错误响应内容: {e.response.text}")
        exit(1)


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

topics = [
    # 操作系统
    "Linux Kernel", "FreeBSD", "ReactOS", "Haiku", "Debian", "Ubuntu",
    
    # 编程语言
    "Python", "Ruby", "Go", "Rust", "TypeScript", "PHP", "Perl", "Swift",
    
    # 数据库
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "SQLite", "Cassandra", "CouchDB",
    
    # Web服务器
    "Apache HTTP Server", "Nginx", "Caddy", "Tomcat", "Lighttpd",
    
    # 容器与虚拟化
    "Docker", "Kubernetes", "QEMU", "VirtualBox", "Podman", "LXC",
    
    # 开发工具
    "Git", "VS Code", "Eclipse", "Jenkins", "Ansible", "Gradle", "Maven", "Selenium",
    
    # 大数据
    "Hadoop", "Spark", "Elasticsearch", "Kafka", "Flink", "Hive", "HBase",
    
    # 人工智能
    "TensorFlow", "PyTorch", "OpenCV", "Keras", "Scikit-learn", "MXNet", "Caffe",
    
    # 桌面环境
    "GNOME", "KDE Plasma", "Xfce", "LXQt", "Budgie",
    
    # 网络安全
    "Wireshark", "Nmap", "Metasploit", "Snort", "OpenVAS", "Suricata",
    
    # 区块链
    "Hyperledger Fabric", "Ethereum", "Bitcoin Core", "Corda",
    
    # 工具软件
    "FFmpeg", "GIMP", "Blender", "LibreOffice", "VLC", "Audacity", "Inkscape",
    
    # 浏览器与引擎
    "Chromium", "Firefox", "WebKit", "Gecko",
    
    # 中间件
    "RabbitMQ", "Zookeeper", "Etcd", "Consul",
    
    # 编译器
    "LLVM", "GCC", "Clang", "Roslyn",
    
    # 数学计算
    "Octave", "SageMath", "NumPy", "SciPy",
    
    # 其他
    "WordPress", "Joomla", "Drupal", "Node.js", "React", "Angular", "Vue.js",
    "Raspberry Pi OS", "Arduino IDE", "OpenStack", "Kubernetes", "Prometheus",

    # Linux/Unix
    "ls -l", "cp -r dir", "find / -name '*.log'", "chmod 755 file",
    "top", "free -h", "df -h", "ps aux", "ifconfig", "netstat -tulpn",
    "ssh user@host", "wget url", "grep 'error' logfile", "awk '{print $1}' file",
    "sed 's/old/new/g' file", "tail -f logfile", "tar -xzvf file.tar.gz",
    "scp file user@host:/path", "rsync -avz source/ dest/", "sudo apt update",
    "systemctl start service", "ufw allow 80", "iptables -L", "kill -9 PID",
    "crontab -e", "ln -s source link", "chown user:group file",
    
    # Windows
    "dir /s", "tasklist", "systeminfo", "chkdsk", "sfc /scannow",
    "ipconfig /all", "netstat -ano", "robocopy source dest /MIR",
    "schtasks /create", "diskpart", "taskkill /IM process.exe /F",
    
    # PowerShell
    "Get-Process | Where CPU -gt 10", "Test-NetConnection host -Port 80",
    "Get-ChildItem -Recurse -Include *.log", "Measure-Command { ... }",
    "Start-Service -Name 'service'", "Stop-Process -Name 'process'",
    
    # macOS
    "brew install package", "networksetup -listallhardwareports",
    "diskutil list", "airport -s", "launchctl load /path/to/plist",
    
    # 跨平台
    "curl -I https://example.com", "ping 8.8.8.8", "traceroute example.com",
    "dig example.com", "openssl genrsa -out key.pem 2048", "ssh-keygen -t rsa",
    "docker ps -a", "docker-compose up", "git push origin main",
    "npm install package", "python -m http.server 8000", "java -jar app.jar",
    "node server.js", "php -S localhost:8000", "dotnet run",
    
    # 网络诊断
    "nslookup example.com", "whois example.com", "tcpdump port 80",
    "mtr example.com", "nc -zv host port",
    
    # 系统管理
    "lsof -i :80", "dmesg | grep error", "vmstat 1", "iostat -x 1",
    "sar -u 1 3", "strace -p PID", "htop", "glances",
    
    # 开发调试
    "gdb -ex run ./program", "strace -e trace=open program",
    "valgrind --leak-check=yes ./program", "tcpflow -i eth0 port 80",

# 微服务架构相关关键词
    "服务发现", "配置中心", "API网关", "熔断器模式", "服务网格(Service Mesh)",
    "链路追踪", "分布式事务", "容器化部署", "服务注册", "负载均衡",
    "弹性伸缩", "限流降级", "健康检查", "日志聚合", "指标监控",
    "Istio", "Spring Cloud", "Kubernetes服务", "Consul", "Eureka",
    "Nacos", "Zipkin", "Sleuth", "Hystrix", "Sentinel",
    "Envoy Proxy", "Linkerd", "Kong网关", "Apollo配置中心", "RabbitMQ消息总线",
    "服务拆分策略", "领域驱动设计(DDD)", "CQRS模式", "Event Sourcing", "Sidecar模式",
    "无服务器架构(Serverless)", "服务版本控制", "金丝雀发布", "蓝绿部署", "AB测试路由",

# RPC调用相关关键词
    "gRPC", "Thrift", "Dubbo", "JSON-RPC", "XML-RPC",
    "Protobuf序列化", "Avro序列化", "服务存根(Stub)", "动态代理", "服务注册中心",
    "TCP长连接", "HTTP/2协议", "双向流式通信", "服务端流", "客户端流",
    "负载均衡策略", "集群容错", "超时重试", "心跳检测", "泛化调用",
    "异步RPC", "回调机制", "服务分组", "版本路由", "Mock测试",
    "服务降级", "限流策略", "熔断机制", "权重配置", "跨语言调用",
    "Zookeeper注册中心", "Nacos服务发现", "ETCD协调", "Consul健康检查", "SLA监控",

# 编程语言特性关键词
    "静态类型", "动态类型", "类型推断", "泛型编程", "元编程",
    "函数式编程", "面向切面(AOP)", "反射机制", "内存管理", "垃圾回收(GC)",
    "协程(Coroutine)", "异步/await", "Actor模型", "指针运算", "内存安全",
    "所有权系统(Rust)", "借用检查器", "零成本抽象", "模式匹配", "闭包",
    "高阶函数", "尾递归优化", "运算符重载", "注解/装饰器", "模块系统",
    "包管理", "依赖注入", "接口协议", "多范式编程", "并发原语",
    "不可变数据结构", "类型擦除", "变长参数", "模式解构", "宏系统",
    "JIT编译", "AOT编译", "跨平台字节码", "原生扩展", "FFI接口"
]


if __name__ == "__main__":
    # 测试生成
    tokenizer = Tokenizer()

    prompts = [
        "请这对下面主题，总结它的一些细节知识，甚至操作命令、代码和技巧：",
        "请这对下面主题，总结它的一些历史背景：",
        "请这对下面主题，总结它和其他相关主题是怎么配合的，是什么关系："
    ]

    result = "请给我关于计算机领域的名词"
    with open("result.txt", "a") as f:
        while True:
            prompt = random.choice(prompts)
            topic = random.choice(topics)
            result = generate_text_openai(prompt + topic)
            
            print("\nOllama 返回结果:")
            print(result)
            text = tokenizer.bos_token + result + tokenizer.eos_token
            f.write(text)
