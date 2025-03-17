from flask import Flask, render_template, request, jsonify
import torch
import json
import os
import sys
from tokenizer import Tokenizer
from dataloader import load_dataset_with_cache, MockEnv, TrainDataLoader, text_fn_pretrain, text_fn_sft, TextFnWrapper

app = Flask(__name__)

# 全局变量
env = MockEnv()
tokenizer = Tokenizer()
datasets = {}  # 存储已加载的数据集
dataloaders = {}  # 存储已创建的DataLoader
current_page = 1
items_per_page = 20

# 确保模板目录存在
os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_dataset', methods=['POST'])
def load_dataset_route():
    data = request.json
    dataset_name = data.get('dataset_name', '')
    data_dir = data.get('data_dir', None)
    split = data.get('split', 'train[0%:100%]')
    page = int(data.get('page', 1))
    
    # 生成唯一的数据集ID
    dataset_id = f"{dataset_name}_{data_dir}_{split}"
    
    # 如果数据集尚未加载，则加载
    if dataset_id not in datasets:
        try:
            # 使用load_dataset_with_cache加载数据集
            dataset = load_dataset_with_cache(dataset_name, data_dir=data_dir, split=split)
            datasets[dataset_id] = dataset
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    
    dataset = datasets[dataset_id]
    total_samples = len(dataset)
    total_pages = (total_samples + items_per_page - 1) // items_per_page
    
    # 计算分页
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_samples)
    
    # 获取当前页的数据
    page_data = []
    for i in range(start_idx, end_idx):
        item = dataset[i]
        if isinstance(item, dict):
            # 复制一份，避免修改原始数据
            item_copy = {k: v for k, v in item.items()}
            page_data.append(item_copy)
        else:
            page_data.append(str(item))
    
    return jsonify({
        'status': 'success',
        'dataset_id': dataset_id,
        'total_samples': total_samples,
        'total_pages': total_pages,
        'current_page': page,
        'data': page_data
    })

@app.route('/create_dataloader', methods=['POST'])
def create_dataloader_route():
    data = request.json
    is_sft = data.get('is_sft', False)
    batch_size = int(data.get('batch_size', 1))
    block_size = int(data.get('block_size', 1024))
    use_data_percent = int(data.get('use_data_percent', 100))
    
    # 创建一个唯一ID
    loader_id = f"loader_{is_sft}_{batch_size}_{block_size}_{use_data_percent}"
    
    try:
        # 创建DataLoader
        dataloader = TrainDataLoader(
            env=env,
            batch_size=batch_size,
            block_size=block_size,
            tokenizer=tokenizer,
            use_data_percent=use_data_percent,
            is_sft=is_sft
        )
        dataloaders[loader_id] = dataloader
        
        return jsonify({
            'status': 'success',
            'loader_id': loader_id,
            'message': f"已创建DataLoader (SFT: {is_sft}, Batch Size: {batch_size})"
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/next_batch', methods=['POST'])
def next_batch_route():
    data = request.json
    loader_id = data.get('loader_id', '')
    
    if loader_id not in dataloaders:
        return jsonify({
            'status': 'error',
            'message': '找不到指定的DataLoader'
        })
    
    try:
        # 获取下一批数据
        dataloader = dataloaders[loader_id]
        xs, ys = dataloader.next(device="cpu")
        
        # 解码张量为文本
        decoded_xs = []
        for i in range(min(5, xs.shape[0])):  # 仅显示前5个样本
            tokens = xs[i].tolist()
            text = tokenizer.decode(tokens)
            decoded_xs.append(text)
        
        decoded_ys = []
        for i in range(min(5, ys.shape[0])):
            tokens = ys[i].tolist()
            text = tokenizer.decode(tokens)
            decoded_ys.append(text)
        
        # 将张量转换为可JSON序列化的格式
        xs_json = []
        for i in range(min(5, xs.shape[0])):
            xs_json.append(xs[i].tolist())
        
        ys_json = []
        for i in range(min(5, ys.shape[0])):
            ys_json.append(ys[i].tolist())
        
        return jsonify({
            'status': 'success',
            'batch_size': xs.shape[0],
            'xs_shape': list(xs.shape),
            'ys_shape': list(ys.shape),
            'xs_sample': xs_json,
            'ys_sample': ys_json,
            'decoded_xs': decoded_xs,
            'decoded_ys': decoded_ys
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

# 创建HTML模板
@app.route('/templates/index.html')
def create_template():
    html_content = '''
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>数据集浏览工具</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .container {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            .section {
                border: 1px solid #ddd;
                padding: 20px;
                border-radius: 5px;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input, select {
                width: 100%;
                padding: 8px;
                box-sizing: border-box;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
            #dataset-viewer, #batch-viewer {
                margin-top: 20px;
                overflow: auto;
            }
            .navigation {
                display: flex;
                justify-content: space-between;
                margin-top: 10px;
            }
            .item {
                border: 1px solid #eee;
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 4px;
            }
            .error {
                color: red;
                font-weight: bold;
            }
            pre {
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 4px;
                overflow-x: auto;
            }
            .tab {
                overflow: hidden;
                border: 1px solid #ccc;
                background-color: #f1f1f1;
            }
            .tab button {
                background-color: inherit;
                float: left;
                border: none;
                outline: none;
                cursor: pointer;
                padding: 10px 16px;
                transition: 0.3s;
                color: black;
            }
            .tab button:hover {
                background-color: #ddd;
            }
            .tab button.active {
                background-color: #ccc;
            }
            .tabcontent {
                display: none;
                padding: 6px 12px;
                border: 1px solid #ccc;
                border-top: none;
            }
            .show {
                display: block;
            }
        </style>
    </head>
    <body>
        <h1>数据集浏览工具</h1>
        
        <div class="tab">
            <button class="tablinks active" onclick="openTab(event, 'DatasetTab')">数据集浏览</button>
            <button class="tablinks" onclick="openTab(event, 'BatchTab')">批次查看</button>
        </div>
        
        <div id="DatasetTab" class="tabcontent show">
            <div class="section">
                <h2>加载数据集</h2>
                <div class="form-group">
                    <label for="dataset-name">数据集名称:</label>
                    <input type="text" id="dataset-name" value="opencsg/Fineweb-Edu-Chinese-V2.1" placeholder="例如: opencsg/Fineweb-Edu-Chinese-V2.1">
                </div>
                <div class="form-group">
                    <label for="data-dir">数据目录 (可选):</label>
                    <input type="text" id="data-dir" value="4_5" placeholder="例如: 4_5">
                </div>
                <div class="form-group">
                    <label for="split">数据分片:</label>
                    <input type="text" id="split" value="train[0%:100%]" placeholder="例如: train[0%:100%]">
                </div>
                <button onclick="loadDataset()">加载数据集</button>
                <div id="dataset-status"></div>
                
                <div id="pagination" class="navigation"></div>
                <div id="dataset-viewer"></div>
            </div>
        </div>
        
        <div id="BatchTab" class="tabcontent">
            <div class="section">
                <h2>创建DataLoader并获取批次</h2>
                <div class="form-group">
                    <label for="is-sft">数据类型:</label>
                    <select id="is-sft">
                        <option value="false">预训练数据</option>
                        <option value="true">SFT数据</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="batch-size">批次大小:</label>
                    <input type="number" id="batch-size" value="2" min="1" max="32">
                </div>
                <div class="form-group">
                    <label for="block-size">块大小:</label>
                    <input type="number" id="block-size" value="1024" min="1">
                </div>
                <div class="form-group">
                    <label for="use-data-percent">使用数据百分比:</label>
                    <input type="number" id="use-data-percent" value="100" min="1" max="100">
                </div>
                <button onclick="createDataLoader()">创建DataLoader</button>
                <div id="loader-status"></div>
                
                <div id="batch-controls" style="display: none; margin-top: 20px;">
                    <button onclick="getNextBatch()">获取下一批数据</button>
                </div>
                
                <div id="batch-viewer"></div>
            </div>
        </div>
        
        <script>
            let currentDatasetId = null;
            let currentPage = 1;
            let totalPages = 1;
            let currentLoaderId = null;
            
            function openTab(evt, tabName) {
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }
            
            function loadDataset(page = 1) {
                const datasetName = document.getElementById('dataset-name').value.trim();
                const dataDir = document.getElementById('data-dir').value.trim() || null;
                const split = document.getElementById('split').value.trim();
                
                if (!datasetName) {
                    document.getElementById('dataset-status').innerHTML = '<p class="error">请输入数据集名称</p>';
                    return;
                }
                
                document.getElementById('dataset-status').innerHTML = '<p>正在加载数据集...</p>';
                document.getElementById('dataset-viewer').innerHTML = '';
                document.getElementById('pagination').innerHTML = '';
                
                fetch('/load_dataset', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        dataset_name: datasetName,
                        data_dir: dataDir,
                        split: split,
                        page: page
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'error') {
                        document.getElementById('dataset-status').innerHTML = `<p class="error">错误: ${data.message}</p>`;
                        return;
                    }
                    
                    currentDatasetId = data.dataset_id;
                    currentPage = data.current_page;
                    totalPages = data.total_pages;
                    
                    document.getElementById('dataset-status').innerHTML = `<p>成功加载数据集! 共有 ${data.total_samples} 个样本</p>`;
                    
                    // 显示数据
                    const viewer = document.getElementById('dataset-viewer');
                    viewer.innerHTML = '';
                    
                    data.data.forEach((item, index) => {
                        const itemDiv = document.createElement('div');
                        itemDiv.className = 'item';
                        
                        const itemIndex = (currentPage - 1) * 20 + index;
                        let content = `<h3>样本 #${itemIndex}</h3>`;
                        
                        if (typeof item === 'object') {
                            for (const key in item) {
                                content += `<p><strong>${key}:</strong> <pre>${item[key]}</pre></p>`;
                            }
                        } else {
                            content += `<p>${item}</p>`;
                        }
                        
                        itemDiv.innerHTML = content;
                        viewer.appendChild(itemDiv);
                    });
                    
                    // 分页控件
                    const pagination = document.getElementById('pagination');
                    pagination.innerHTML = '';
                    
                    if (totalPages > 1) {
                        const nav = document.createElement('div');
                        nav.className = 'navigation';
                        
                        // 上一页按钮
                        const prevBtn = document.createElement('button');
                        prevBtn.innerText = '上一页';
                        prevBtn.disabled = currentPage === 1;
                        prevBtn.onclick = () => loadDataset(currentPage - 1);
                        
                        // 页码信息
                        const pageInfo = document.createElement('span');
                        pageInfo.innerText = `第 ${currentPage} 页，共 ${totalPages} 页`;
                        
                        // 下一页按钮
                        const nextBtn = document.createElement('button');
                        nextBtn.innerText = '下一页';
                        nextBtn.disabled = currentPage === totalPages;
                        nextBtn.onclick = () => loadDataset(currentPage + 1);
                        
                        nav.appendChild(prevBtn);
                        nav.appendChild(pageInfo);
                        nav.appendChild(nextBtn);
                        pagination.appendChild(nav);
                    }
                })
                .catch(error => {
                    document.getElementById('dataset-status').innerHTML = `<p class="error">错误: ${error.message}</p>`;
                });
            }
            
            function createDataLoader() {
                const isSft = document.getElementById('is-sft').value === 'true';
                const batchSize = parseInt(document.getElementById('batch-size').value);
                const blockSize = parseInt(document.getElementById('block-size').value);
                const useDataPercent = parseInt(document.getElementById('use-data-percent').value);
                
                document.getElementById('loader-status').innerHTML = '<p>正在创建DataLoader...</p>';
                document.getElementById('batch-controls').style.display = 'none';
                document.getElementById('batch-viewer').innerHTML = '';
                
                fetch('/create_dataloader', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        is_sft: isSft,
                        batch_size: batchSize,
                        block_size: blockSize,
                        use_data_percent: useDataPercent
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'error') {
                        document.getElementById('loader-status').innerHTML = `<p class="error">错误: ${data.message}</p>`;
                        return;
                    }
                    
                    currentLoaderId = data.loader_id;
                    document.getElementById('loader-status').innerHTML = `<p>${data.message}</p>`;
                    document.getElementById('batch-controls').style.display = 'block';
                })
                .catch(error => {
                    document.getElementById('loader-status').innerHTML = `<p class="error">错误: ${error.message}</p>`;
                });
            }
            
            function getNextBatch() {
                if (!currentLoaderId) {
                    document.getElementById('batch-viewer').innerHTML = '<p class="error">请先创建DataLoader</p>';
                    return;
                }
                
                document.getElementById('batch-viewer').innerHTML = '<p>获取下一批数据中...</p>';
                
                fetch('/next_batch', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        loader_id: currentLoaderId
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'error') {
                        document.getElementById('batch-viewer').innerHTML = `<p class="error">错误: ${data.message}</p>`;
                        return;
                    }
                    
                    // 显示批次数据
                    const viewer = document.getElementById('batch-viewer');
                    viewer.innerHTML = `
                        <h3>批次信息</h3>
                        <p>批次大小: ${data.batch_size}</p>
                        <p>输入张量形状: [${data.xs_shape.join(', ')}]</p>
                        <p>输出张量形状: [${data.ys_shape.join(', ')}]</p>
                        
                        <h3>样本预览 (最多5个)</h3>
                    `;
                    
                    // 输入和解码后的文本
                    for (let i = 0; i < data.decoded_xs.length; i++) {
                        const sampleDiv = document.createElement('div');
                        sampleDiv.className = 'item';
                        sampleDiv.innerHTML = `
                            <h4>样本 #${i+1}</h4>
                            <p><strong>输入张量:</strong> <pre>${JSON.stringify(data.xs_sample[i], null, 0)}</pre></p>
                            <p><strong>解码后的输入:</strong> <pre>${data.decoded_xs[i]}</pre></p>
                            <p><strong>输出张量:</strong> <pre>${JSON.stringify(data.ys_sample[i], null, 0)}</pre></p>
                            <p><strong>解码后的输出:</strong> <pre>${data.decoded_ys[i]}</pre></p>
                        `;
                        viewer.appendChild(sampleDiv);
                    }
                })
                .catch(error => {
                    document.getElementById('batch-viewer').innerHTML = `<p class="error">错误: ${error.message}</p>`;
                });
            }
            
            // 初始显示第一个标签页
            document.getElementById('DatasetTab').style.display = 'block';
        </script>
    </body>
    </html>
    '''
    
    with open(os.path.join(os.path.dirname(__file__), 'templates', 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return html_content

if __name__ == "__main__":
    # 确保首次运行时创建模板
    create_template()
    print("数据集浏览工具已启动，请访问 http://127.0.0.1:5000/ 查看")
    app.run(debug=False, port=5000) 