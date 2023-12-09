# Customized data generator
The code in this repo aims to generate customized chat data using method similar to [self-instruct](https://arxiv.org/abs/2212.10560).
The original paper's method consists of several stages and is rather complex. The code in this repository was developed based on the code from [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca).

The difference between our method and their method lies in our focus on generating costomized chat data. We aim to train the Language Model (LLM) to adapt to specific conversational scenarios, 
such as playing the role of a friend or acting as a translator... In other words, our data is intended for customizing the LLM, as opposed to their method, which is designed to teach the LLM to follow instructions.

## usage
### write costomized prompt
To generate customized data, you should first modify the `customizedGPT.txt`. For example, you want a bot that can help you with your poor English, you can write a prompt like:

> I am a student with limited proficiency in English, and you are a powerful GPT model. I kindly request your detailed assistance in translating my English sentences into Chinese.
> Furthermore, I hope you can understand my needs and aim to improve my English skills by providing explanations and interpretations while translating my sentences.
> Please maintain a friendly tone and ensure the meaningfulness of the translations. If I encounter unfamiliar words, please teach me their meanings, usage, and contexts,
> primarily using Chinese explanations due to my limited English proficiency. My questions may include the translation of a long English passage, a single sentence, or explanations of unfamiliar phrases.
> The English texts I provide may cover various fields and specialties, ranging from colloquial to technical language, and I trust in your strong translation and explanatory abilities.

### create seed data
Secondly, you should provide some seed data.Create a file `tasks.txt`, write the example inputs and outputs like following, note that we use `@@@@` as delimiter
```text:
input: ...
output: ...
@@@@
input: ...
output: ...
@@@@
...
```
you can also number the input and output like：
```text:
1.input: ...
1.output: ...
@@@@
2.input: ...
2.output: ...
@@@@
...
```
After that you can create seed data file via:
```python
python utils.py -f extract_input_output\
        --input input_file\
        --output output_file\
        --similarity_bound 0.7\
        --model_path hfl/chinese-alpaca-2-7b
```
input_file is the file you just created.
output_file contains a json object per line, with keys "input" and "output".
this script will extract the input and output from input_file and write to output_file (similar input will be filtered).
model tokenizer is used to calculate rougeL score to filter out similar input.

### generate data
Then you can generate data simply by using the following commamd:
```bash
python gen_data.py --seed seed_file\ # seed file generated in the last step
          --output output_file\      # genereated data, a json per line.
          --model_path model_path\   # model used to tokenize sentence and generate data
          --similarity_bound similarity\ # if similarity of new data is larger than bound, the data will be filter out
          --num_data num\                # number of data to generate
          --num_tasks n                  # number of example in prompt
```
---
### customize the model used to generate data
In our implementation, you can use model path to specify the model used to generate data.If you want to use other method to generate data (e.g GPT4), you can 
overwrite the `GenerateResponse` class. Modify its `__call__` method, its return type is `List[Dict[str,str]]`. A possible return value is as follows:
```python
[
  {"text": "here is generated sentences", "finish_reason": "stop"},
  {"text": "here is generated ", "finish_reason": "length"}
]
```

### see the prompt
If you wonder what prompt you give to the model to generate data, you could use the following command to generate a prompt:
```bash
python utils.py -f gen_prompts\
        --input input_file\
        --num_prompts 10 \
        --num_tasks 3
```
where input_file contains json object per line, with keys "input" and "output".num_prompts is the number of prompts to generate.num_tasks is the number of tasks used in prompt.

## result
Use comstomized prompt in the section `write costomized prompt`, GPT4 can generate ideal data, here are some examples
```text
input:我在读一篇文章关于气候变化，里面有句话 "The albedo effect refers to the amount of sunlight that is reflected by the surface of the Earth back into space." 我不太明白albedo effect是什么，能帮我解释一下吗？
output:“Albedo effect”指的是“反照率效应”，它是指地球表面反射回太空的阳光量。在这里，“albedo”（反照率）是一个衡量地表或其他物体反射阳光能力的术语，反照率越高，反射的阳光就越多。这个效应对地球的气候有重要影响，因为它决定了地球吸收的热量量。
@@@@
input:刚看了一部电影，有个场景提到 "She has been bitten by the acting bug ever since she was a little girl." 这里的 "bitten by the acting bug" 是什么意思呢？
output:“Bitten by the acting bug”这个短语的意思是“被表演的热情所感染”，形容一个人对表演产生了浓厚的兴趣和热爱。常用来形容某人对某项活动非常热衷，以至于像是被“咬了一口”（即“bug”在这里比喻为一种让人着迷的病毒或者昆虫）就无法停止。
@@@@
input:老师给我们布置了一篇论文，要求我们探讨 "The implications of artificial intelligence in modern warfare." 这个 topic 我有点摸不着头脑，能帮我解释一下这是什么意思吗？
output:这个话题要求你探讨的是“人工智能在现代战争中的影响”。这里，“implications”指的是可能的后果或重要的影响。你需要考虑人工智能如何改变战争的方式，比如无人机的使用、自动化武器系统、以及军事决策中的人工智能应用等。这个话题要求你思考人工智能带来的技术变革如何影响现代战争的战术、战略和伦理。
@@@@
input:在准备GRE考试，碰到一个句子 "The author's prose is imbued with a sense of melancholy." 中的 "imbued" 这个词不懂，能解释一下吗？
output:“Imbued”这个词的含义是“充满的”或“渗透的”。这句话的意思是“作者的散文充满了忧郁的感觉”。这里的“melancholy”意味着一种深沉的悲伤或忧郁。所以如果说一个文本或作品被某种情感“imbued”，就是说那种情感已经深深地渗入到了文本或作品的每一个部分。
```

I also tried to use `hfl/chinese-alpaca-2-7b` to generate data. The result is not satisfactory... Here are some examples
```text
input: 在学习中，我发现有时候很难集中注意力，怎么办？
output: 您可以尝试以下方法来集中注意力：\n\n- 创建一个安静、舒适的工作或学习环境，减少干扰，如关掉手机和社交媒体通知，消除噪音或光线干扰。
- 制定一个详细的学习计划或计划表，并尽量按计划执行，这有助于保持自律和专注。
- 使用一些专注力练习，如冥想、呼吸练习或身体扫描等，这些练习可以帮助您放松身心，减轻焦虑和分散注意力的情况。
- 设定小目标，逐一完成，不要期望一次性完成大量的任务或学习任务，这样可以避免心理疲劳。
- 尝试在学习时使用一些辅助工具，如闹钟、计时器、番茄钟等，帮助您保持专注和按时完成任务。
@@@@
input: Is this article about concurrent programming?
output: 此文章是否关于并发编程？
@@@@
input: Can you help me understand the concept of recursion?
output: 请帮助我理解递归的概念。
@@@@
input: 这句话意思是：“你认为我做得好吗？”（中文）
output: This sentence means: “Do you think I did a good job?”（英文）
@@@@
input: 这句话是关于经济学分析的，翻译一下 "The Benefits of Reading Books in the Digital Age" 的英文标题。
output: 这句话的中文标题是 "数字时代阅读书籍的好处"。
```
