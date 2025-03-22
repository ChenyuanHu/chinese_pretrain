




工具说明

plot_training_log.py
将形如下面的日志转成图表。默认目录是experiments/logs/train_-1.log
[RANK:0][2025-03-22 04:08:22] Epoch [1/10000], 4421.35sec, world 44467.87 tokens/s, 训练损失: 6.5201, 困惑度: 678.6483, LR: 3e-5
[RANK:0][2025-03-22 05:22:23] Epoch [2/10000], 4434.63sec, world 44334.74 tokens/s, 训练损失: 5.2463, 困惑度: 189.8672, LR: 3e-5
[RANK:0][2025-03-22 06:36:09] Epoch [3/10000], 4419.82sec, world 44483.31 tokens/s, 训练损失: 4.7993, 困惑度: 121.4271, LR: 3e-5
[RANK:0][2025-03-22 07:50:04] Epoch [4/10000], 4429.24sec, world 44388.70 tokens/s, 训练损失: 4.5127, 困惑度: 91.1716, LR: 3e-5
[RANK:0][2025-03-22 09:03:58] Epoch [5/10000], 4428.22sec, world 44398.91 tokens/s, 训练损失: 4.2876, 困惑度: 72.7910, LR: 4e-5
[RANK:0][2025-03-22 10:17:55] Epoch [6/10000], 4430.96sec, world 44371.44 tokens/s, 训练损失: 4.1181, 困惑度: 61.4447, LR: 4e-5
[RANK:0][2025-03-22 11:31:58] Epoch [7/10000], 4437.03sec, world 44310.71 tokens/s, 训练损失: 3.9909, 困惑度: 54.1026, LR: 4e-5
[RANK:0][2025-03-22 12:45:59] Epoch [8/10000], 4436.06sec, world 44320.44 tokens/s, 训练损失: 3.8506, 困惑度: 47.0199, LR: 5e-5
[RANK:0][2025-03-22 14:00:01] Epoch [9/10000], 4435.75sec, world 44323.54 tokens/s, 训练损失: 3.7605, 困惑度: 42.9690, LR: 6e-5
[RANK:0][2025-03-22 15:14:04] Epoch [10/10000], 4437.58sec, world 44305.24 tokens/s, 训练损失: 3.6476, 困惑度: 38.3812, LR: 6e-5
[RANK:0][2025-03-22 16:28:14] Epoch [11/10000], 4442.87sec, world 44252.43 tokens/s, 训练损失: 3.5011, 困惑度: 33.1524, LR: 7e-5
[RANK:0][2025-03-22 17:42:27] Epoch [12/10000], 4447.10sec, world 44210.42 tokens/s, 训练损失: 3.3118, 困惑度: 27.4352, LR: 8e-5
[RANK:0][2025-03-22 18:56:35] Epoch [13/10000], 4442.22sec, world 44258.99 tokens/s, 训练损失: 3.1796, 困惑度: 24.0379, LR: 9e-5
[RANK:0][2025-03-22 04:08:22] 数据集使用度: {"fineweb_edu_chinese_2p1": 0.41401291421827474, "codeparrot_clean": 0.28627807571538566, "zh_en_translation_v2": 0.0, "open_r1_math_220k": 0.7875283654892996, "open_math_instruct_2": 0.2626137105826181}
[RANK:0][2025-03-22 05:22:23] 数据集使用度: {"fineweb_edu_chinese_2p1": 0.8443167029444599, "codeparrot_clean": 0.5483209704178286, "zh_en_translation_v2": 0.0, "open_r1_math_220k": 0.7875283654892996, "open_math_instruct_2": 0.4914628012331853}
[RANK:0][2025-03-22 06:36:09] 数据集使用度: {"fineweb_edu_chinese_2p1": 1.2656324229766256, "codeparrot_clean": 0.8255108532533608, "zh_en_translation_v2": 0.0, "open_r1_math_220k": 1.5750567309785992, "open_math_instruct_2": 0.7278151407575416}
[RANK:0][2025-03-22 07:50:04] 数据集使用度: {"fineweb_edu_chinese_2p1": 1.6847011258352862, "codeparrot_clean": 1.1148183265953644, "zh_en_translation_v2": 0.0, "open_r1_math_220k": 2.165703005095574, "open_math_instruct_2": 0.9529126069712143}
[RANK:0][2025-03-22 09:03:58] 数据集使用度: {"fineweb_edu_chinese_2p1": 2.110510880214462, "codeparrot_clean": 1.3950376070575143, "zh_en_translation_v2": 0.0, "open_r1_math_220k": 2.165703005095574, "open_math_instruct_2": 1.1667551998742034}
[RANK:0][2025-03-22 10:17:55] 数据集使用度: {"fineweb_edu_chinese_2p1": 2.5262090573128653, "codeparrot_clean": 1.7161537554790052, "zh_en_translation_v2": 0.0, "open_r1_math_220k": 2.5594671878402235, "open_math_instruct_2": 1.3393299239713525}
[RANK:0][2025-03-22 11:31:58] 数据集使用度: {"fineweb_edu_chinese_2p1": 2.9559510917456744, "codeparrot_clean": 1.9887995418746105, "zh_en_translation_v2": 0.0, "open_r1_math_220k": 3.5438776447018485, "open_math_instruct_2": 1.5269111458160796}
[RANK:0][2025-03-22 12:45:59] 数据集使用度: {"fineweb_edu_chinese_2p1": 3.3885018976453645, "codeparrot_clean": 2.2402395448838908, "zh_en_translation_v2": 0.0, "open_r1_math_220k": 3.740759736074173, "open_math_instruct_2": 1.7632634853404359}
[RANK:0][2025-03-22 14:00:01] 数据集使用度: {"fineweb_edu_chinese_2p1": 3.810379371970906, "codeparrot_clean": 2.5310617170392034, "zh_en_translation_v2": 0.40589680054027083, "open_r1_math_220k": 3.937641827446498, "open_math_instruct_2": 1.969602829369636}
[RANK:0][2025-03-22 15:14:04] 数据集使用度: {"fineweb_edu_chinese_2p1": 4.241244914990467, "codeparrot_clean": 2.81582509394128, "zh_en_translation_v2": 0.40589680054027083, "open_r1_math_220k": 4.1345239188188225, "open_math_instruct_2": 2.134674304592996}
[RANK:0][2025-03-22 16:28:14] 数据集使用度: {"fineweb_edu_chinese_2p1": 4.6771662466504145, "codeparrot_clean": 3.083926783896959, "zh_en_translation_v2": 0.40589680054027083, "open_r1_math_220k": 4.331406010191148, "open_math_instruct_2": 2.307249028690145}
[RANK:0][2025-03-22 17:42:27] 数据集使用度: {"fineweb_edu_chinese_2p1": 5.106346526789848, "codeparrot_clean": 3.3474843774127105, "zh_en_translation_v2": 0.40589680054027083, "open_r1_math_220k": 4.528288101563473, "open_math_instruct_2": 2.536098119340712}
[RANK:0][2025-03-22 18:56:35] 数据集使用度: {"fineweb_edu_chinese_2p1": 5.535526806929281, "codeparrot_clean": 3.6004390792353, "zh_en_translation_v2": 0.40589680054027083, "open_r1_math_220k": 5.709580649797422, "open_math_instruct_2": 2.7724504588650682}


其他说明

dcp checkpoint 转成torch checkpoint
```
python3 -m torch.distributed.checkpoint.format_utils dcp_to_torch checkpoints_epoch_11 checkpoints_epoch_11.pt
```