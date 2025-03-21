from datasets import load_dataset

def fineweb_edu_chinese_2p1_ds_fn():
    return load_dataset("opencsg/Fineweb-Edu-Chinese-V2.1", data_dir="4_5", split="train")

def fineweb_edu_chinese_2p1_text_fn(x):
    return x["text"]

fineweb_edu_chinese_2p1 = {
    "name": "fineweb_edu_chinese_2p1",
    "ds_fn": fineweb_edu_chinese_2p1_ds_fn,
    "text_fn": fineweb_edu_chinese_2p1_text_fn,
}


def fineweb_edu_chinese_2p1_1percent_ds_fn():
    return load_dataset("opencsg/Fineweb-Edu-Chinese-V2.1", data_dir="4_5", split="train[0%:1%]")

def fineweb_edu_chinese_2p1_1percent_text_fn(x):
    return x["text"]

fineweb_edu_chinese_2p1_1percent = {
    "name": "fineweb_edu_chinese_2p1_1percent",
    "ds_fn": fineweb_edu_chinese_2p1_1percent_ds_fn,
    "text_fn": fineweb_edu_chinese_2p1_1percent_text_fn,
}


def codeparrot_clean_ds_fn():
    return load_dataset("codeparrot/codeparrot-clean", split="train")

def codeparrot_clean_text_fn(x):
    return "repo_name: " + x["repo_name"] + "\n" + "path: " + x["path"] + "\n" + "code: " + x["content"]

codeparrot_clean = {
    "name": "codeparrot_clean",
    "ds_fn": codeparrot_clean_ds_fn,
    "text_fn": codeparrot_clean_text_fn,
}


def codeparrot_clean_1percent_ds_fn():
    return load_dataset("codeparrot/codeparrot-clean", split="train[0%:1%]")

def codeparrot_clean_1percent_text_fn(x):
    return "repo_name: " + x["repo_name"] + "\n" + "path: " + x["path"] + "\n" + "code: " + x["content"]

codeparrot_clean_1percent = {
    "name": "codeparrot_clean_1percent",
    "ds_fn": codeparrot_clean_1percent_ds_fn,
    "text_fn": codeparrot_clean_1percent_text_fn,
}


def code_parrot_github_code_ds_fn():
    return load_dataset("macrocosm-os/code-parrot-github-code", split="train")

def code_parrot_github_code_text_fn(x):
    return "repo_name: " + x["repo_name"] + "\n" + "path: " + x["path"] + "\n" + "code: " + x["content"]

code_parrot_github_code = {
    "name": "code_parrot_github_code",
    "ds_fn": code_parrot_github_code_ds_fn,
    "text_fn": code_parrot_github_code_text_fn,
}


def zh_en_translation_ds_fn():
    return load_dataset("Garsa3112/ChineseEnglishTranslationDataset", split="train")

def zh_en_translation_text_fn(x):
    return x["en"] + " => " + x["zh"]

zh_en_translation = {
    "name": "zh_en_translation",
    "ds_fn": zh_en_translation_ds_fn,
    "text_fn": zh_en_translation_text_fn,
}

def zh_en_translation_text_fn_v2(x):
    return x["zh"] + " 上文翻译成英文是：" + x["en"]

zh_en_translation_v2 = {
    "name": "zh_en_translation_v2",
    "ds_fn": zh_en_translation_ds_fn,
    "text_fn": zh_en_translation_text_fn_v2,
}


def sft_r1_distill_ds_fn():
    return load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT", data_dir=None, split="train")

def sft_r1_distill_text_fn(x):
    return "<|im_start|>用户\n" + x["instruction"] + "\n<|im_end|>\n<|im_start|>助手\n" + x["output"] + "\n<|im_end|>"

sft_r1_distill = {
    "name": "sft_r1_distill",
    "ds_fn": sft_r1_distill_ds_fn,
    "text_fn": sft_r1_distill_text_fn,
}


def open_r1_math_220k_ds_fn():
    return load_dataset("open-r1/OpenR1-Math-220k", "all", split="train")

def open_r1_math_220k_text_fn(x):
    return "问题: " + x["problem"] + "\n" + "解题过程: " + x["solution"] + "\n" + "答案: " + x["answer"]

open_r1_math_220k = {
    "name": "open_r1_math_220k",
    "ds_fn": open_r1_math_220k_ds_fn,
    "text_fn": open_r1_math_220k_text_fn,
}


def open_math_instruct_2_ds_fn():
    return load_dataset("nvidia/OpenMathInstruct-2", split="train")

def open_math_instruct_2_text_fn(x):
    return "问题: " + x["problem"] + "\n" + "解题过程: " + x["generated_solution"] + "\n" + "答案: " + x["expected_answer"]

open_math_instruct_2 = {
    "name": "open_math_instruct_2",
    "ds_fn": open_math_instruct_2_ds_fn,
    "text_fn": open_math_instruct_2_text_fn,
}
