
pretrain_case_prompts = [
    "中华人民共和国2020年的的中共中央总书记是",
    "左手拿一个苹果，右手拿一根香蕉。 上文翻译成英文是： ",
    "type struct Person ",
    "def sum(a, b): ",
    "12 + 22 =",
    "x^2 + x - 650 = 0"
]

sft_case_prompts = [
    "<｜User｜>请根据规律填充这两个空缺的数字。 4, 3, 4, 3, 4, 3, （），（）<｜Assistant｜>",
    "<｜User｜>中华人民共和国的2020年的总书记是谁？<｜Assistant｜>",
    "<｜User｜>你是谁？<｜Assistant｜>",
    "<｜User｜>today这个单词是什么意思？<｜Assistant｜>"
]