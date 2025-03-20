import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

class EvaluateRunner:
    def __init__(self, data_loader, batch_size):
        # 创建一个验证集，方便模型评估
        val_dataset = []
        for i in range(50):
            x, y = data_loader.next()
            x = torch.tensor(x, dtype=torch.long, device=None)
            y = torch.tensor(y, dtype=torch.long, device=None)
            val_dataset.append((x[0], y[0]))

        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def evaluate(self, model, device, env):
        model.eval()
        total_loss = torch.tensor(0.0, device=device)
        total_tokens = torch.tensor(0, device=device)

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                total_loss += loss.sum().detach()
                total_tokens += y.numel()

        # 同步所有进程的总损失和总token数
        if env.enabled:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

        total_tokens = max(total_tokens, 1)
        avg_loss = (total_loss / total_tokens).item()
        perplexity = torch.exp(total_loss / total_tokens).item()

        return avg_loss, perplexity