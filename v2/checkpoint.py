import os
import time
import glob
import torch
from log import tprint
from config import ModuleConfig

class CheckpointManager:
    def __init__(self, env, save_interval_sec):
        self.env = env
        # 记录上次保存模型的时间
        self.last_save_time = time.time()
        self.checkpoint_dir = "checkpoints"
        self.save_interval_sec = save_interval_sec
        # 创建检查点目录（如果不存在）
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        tprint(f"创建检查点目录: {self.checkpoint_dir}")

    # 检查是否存在checkpoint文件
    def get_latest_checkpoint(self):
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pt"))
        if not checkpoint_files:
            return None
        # 按文件修改时间排序，获取最新的checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        return latest_checkpoint

    def try_load_checkpoint(self, model, optimizer):
        # 添加ModuleConfig到安全globals列表中
        torch.serialization.add_safe_globals([ModuleConfig])

        # 尝试加载最新的checkpoint
        latest_checkpoint = self.get_latest_checkpoint()
        start_epoch = 0
        if latest_checkpoint:
            tprint(f"发现最新的checkpoint: {latest_checkpoint}")
            if self.env.enabled:
                map_location = { "cuda:%d" % 0 : "cuda:%d" % self.env.local_rank }
            else:
                map_location = None
            try:
                state_dict = torch.load(latest_checkpoint, weights_only=True, map_location=map_location)
                # 首先尝试使用weights_only=True加载
                model.load_state_dict(state_dict['model_state_dict'])
                # optimizer.load_state_dict(state_dict['optimizer_state_dict'])
                tprint("跳过optimizer加载")
                start_epoch = state_dict.get('epoch', 0)
                tprint(f"成功加载checkpoint，将从epoch {start_epoch} 继续训练")
            except Exception as e:
                tprint(f"使用weights_only=True模型加载失败: {str(e)}")
                exit()
        else:
            tprint("未找到checkpoint，将从头开始训练")

        return start_epoch

    def check_save_checkpoint(self, model, optimizer, epoch, avg_train_loss, avg_eval_loss):
        current_time = time.time()
        time_since_last_save = current_time - self.last_save_time
        
        if time_since_last_save > self.save_interval_sec:  # 如果超过n秒
            tprint(f"start save checkpoint")
            try:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
                save_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),  FSDP先不保存优化器，简单处理，后续改成torch.distributed.checkpoint
                    'train_loss': avg_train_loss,
                    'val_loss': avg_eval_loss,
                }
                torch.save(save_dict, checkpoint_path)
                tprint(f"检查点已保存到 {checkpoint_path}，距上次保存: {time_since_last_save:.2f}秒")
                self.last_save_time = current_time
                tprint(f"删除旧的checkpoint")
                if os.path.exists(os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")):
                    os.remove(os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))
            except Exception as e:
                tprint(f"保存checkpoint时出错: {str(e)}")
                exit()