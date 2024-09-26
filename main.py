
from transformers import (GPT2LMHeadModel, GPT2Tokenizer,
                          AutoTokenizer, AutoModelForCausalLM, TextStreamer)
# device = "cuda:1"
# model_name_or_path = "/data0/lizhong/models/gpt/gpt2"
# tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
# model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(device)


device = "cuda:1"
model_name_or_path = "/data0/lizhong/models/gpts/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
model.eval()

# model.config

# 模型的解码策略是在模型的生成配置中定义的 generation_config
# model.generation_config 只会显示与默认生成配置不同的值，而不列出任何默认值。
# http://fancyerii.github.io/2023/12/19/hg-transformer-generate/
# https://huggingface.co/docs/transformers/main_classes/text_generation
# https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/text_generation#transformers.GenerationConfig

prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
print(input_ids)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# generate
# https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/text_generation#transformers.GenerationMixin
# https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/text_generation#transformers.GenerationMixin.generate
output = model.generate(
    input_ids,
    max_length=150,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    # logits_processor=logits_processor_list,  # 修改当前step在词表空间上的概率分布
    # stopping_criteria=stopping_criteria,  # 根据用户所规定的规则来中止生成
    streamer=streamer,
)
# 通过直接将参数及其值传递给generate方法来覆盖任何generation_config, 例如 max_length, max_new_tokens, do_sample, top_k, eos_token_id...
print(output)
# tensor([[7454, 2402,  257,  640,   11,  262,  995,  373,  257, 1295,  286, 1049,
#          8737,  290, 1049, 3514,   13,  383,  995,  373]], device='cuda:1')
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
# Once upon a time, the world was a place of great beauty and great danger. The world was
