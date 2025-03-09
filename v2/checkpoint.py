import os
import time
import glob
import torch
import shutil
from log import tprint
from config import ModuleConfig
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

class NormalCheckpointManager:
    def __init__(self, env, save_interval_sec):
        self.env = env
        # 记录上次保存模型的时间
        self.last_save_time = time.time()
        self.last_save_epoch = 0
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
                start_epoch = state_dict.get('epoch', 0)
                self.last_save_epoch = start_epoch
                tprint(f"成功加载checkpoint，将从epoch {start_epoch} 继续训练")
            except Exception as e:
                tprint(f"使用weights_only=True模型加载失败: {str(e)}, 退出")
                exit()

            try:
                tprint("继续尝试optimizer加载")
                optimizer.load_state_dict(state_dict['optimizer_state_dict'])
                tprint("optimizer加载成功")
            except Exception as e:
                tprint(f"使用optimizer加载失败: {str(e)[:50]}, 跳过optimizer加载")

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
                if os.path.exists(os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{self.last_save_epoch}.pt")):
                    os.remove(os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{self.last_save_epoch}.pt"))
                self.last_save_epoch = epoch + 1
            except Exception as e:
                tprint(f"保存checkpoint时出错: {str(e)}")
                exit()


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model_state_dict"],
            optim_state_dict=state_dict["optimizer_state_dict"]
        )


class DCPCheckpointManager:
    def __init__(self, env, save_interval_sec):
        self.env = env
        # 记录上次保存模型的时间
        self.last_save_time = time.time()
        self.checkpoint_dir = "checkpoints_dcp"
        self.save_interval_sec = save_interval_sec
        # 创建检查点目录（如果不存在）
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        tprint(f"创建检查点目录: {self.checkpoint_dir}")

    def get_latest_checkpoint_dir(self):
        # 获取checkpoints_dcp目录下的所有子目录
        checkpoint_dirs = [d for d in os.listdir(self.checkpoint_dir) 
                         if os.path.isdir(os.path.join(self.checkpoint_dir, d))]
        
        if not checkpoint_dirs:
            return None, 0
            
        # 提取目录名中的epoch数字
        epoch_nums = []
        for d in checkpoint_dirs:
            try:
                epoch_num = int(d.split('_')[-1])
                epoch_nums.append(epoch_num)
            except:
                continue
                
        if not epoch_nums:
            return None, 0
            
        # 找到最大的epoch数字
        latest_epoch = max(epoch_nums)
        latest_dir = os.path.join(self.checkpoint_dir, f"checkpoints_epoch_{latest_epoch}")
        
        return latest_dir if os.path.exists(latest_dir) else None, latest_epoch

    def try_load_checkpoint(self, model, optimizer):
        # 尝试加载最新的checkpoint
        latest_checkpoint, start_epoch = self.get_latest_checkpoint_dir()
        if latest_checkpoint:
            tprint(f"发现最新的checkpoint: {latest_checkpoint}")
            try:
                state_dict = { "app": AppState(model, optimizer)}
                dcp.load(
                    state_dict=state_dict,
                    checkpoint_id=latest_checkpoint,
                )
                tprint(f"成功加载checkpoint，将从epoch {start_epoch} 继续训练")
                self.last_save_epoch = start_epoch
            except Exception as e:
                tprint(f"使用dcp加载失败: {str(e)}, 退出")
                exit()

        else:
            tprint("未找到checkpoint，将从头开始训练")

        return start_epoch
        

    def check_save_checkpoint(self, model, optimizer, epoch):
        current_time = time.time()
        time_since_last_save = current_time - self.last_save_time
        
        if time_since_last_save > self.save_interval_sec:  # 如果超过n秒
            tprint(f"start save dcp checkpoint")
            try:
                checkpoint_id = os.path.join(self.checkpoint_dir, f"checkpoints_epoch_{epoch+1}")
                state_dict = { "app": AppState(model, optimizer) }
                dcp.save(state_dict, checkpoint_id=checkpoint_id)
                tprint(f"检查点已保存到 {checkpoint_id}，距上次保存: {time_since_last_save:.2f}秒")
                self.last_save_time = current_time
                tprint(f"删除旧的checkpoint")
                old_checkpoint = os.path.join(self.checkpoint_dir, f"checkpoints_epoch_{self.last_save_epoch}")
                if os.path.exists(old_checkpoint):
                    shutil.rmtree(old_checkpoint)
                self.last_save_epoch = epoch + 1
            except Exception as e:
                tprint(f"保存checkpoint时出错: {str(e)}")
                exit()


class CheckpointManager:
    def __init__(self, env, save_interval_sec):
        self.normal_manager = NormalCheckpointManager(env, save_interval_sec)
        self.dcp_manager = DCPCheckpointManager(env, save_interval_sec)

    def try_load_checkpoint(self, model, optimizer):
        start_epoch = self.dcp_manager.try_load_checkpoint(model, optimizer)
        if start_epoch == 0:
            start_epoch = self.normal_manager.try_load_checkpoint(model, optimizer)
        return start_epoch

    def check_save_checkpoint(self, model, optimizer, epoch, avg_train_loss, avg_eval_loss):
        self.dcp_manager.check_save_checkpoint(model, optimizer, epoch)