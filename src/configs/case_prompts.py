
pretrain_case_prompts = [
    "中华人民共和国2020年的的中共中央总书记是",
    "Today is chinese new year. => ",
    ""
]

sft_case_prompts = [
    "<|im_start|>用户\n请根据规律填充这两个空缺的数字。 4, 3, 4, 3, 4, 3, （），（）\n<|im_end|>\n<|im_start|>助手\n",
    "<|im_start|>用户\n中华人民共和国的2020年的总书记是谁？\n<|im_end|>\n<|im_start|>助手\n",
    "<|im_start|>用户\n你是谁？\n<|im_end|>\n<|im_start|>助手\n",
    "<|im_start|>用户\ntoday这个单词是什么意思？\n<|im_end|>\n<|im_start|>助手\n"
]