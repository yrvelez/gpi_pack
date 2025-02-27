#testing llm modules
from gpi_pack.llm import extract_and_save_hidden_states
from gpi_pack.TarNet import load_hiddens
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

## Specify checkpoint (load LLaMa 3.1-8B)
checkpoint = 'meta-llama/Meta-Llama-3.1-8B-Instruct' #You can replace this if you want to change the model

## Load tokenizer and pretrained model
tokenizer = AutoTokenizer.from_pretrained(checkpoint, token = os.getenv("HUGGINGFACE_TOKEN")) #load tokenizer
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, 
    device_map="auto", 
    torch_dtype=torch.float16,
    cache_dir = "test/development",
)

#last token pooling
extract_and_save_hidden_states(
    prompts = ['This is the test for the package.', 'I hope there is no error.'], #texts or prompts
    output_hidden_dir = "test/data/", #directory to save hidden states
    save_name = 'test/data/test', #path and file name to save generated texts
    tokenizer = tokenizer,
    model = model,
    task_type = "reuse", #'reuse' is when you use LLM to regenerate texts and get hidden states
    # if you want to generate new texts, set task_type == "create"
    # You can specify any task by writing the task prompt and set task_type == <YOUR TASK>
    pooling = "last",
)

# Check hidden states dimension
hiddens = load_hiddens(
    directory = "test/data/", 
    hidden_list = [0,1],
    prefix = "hidden_"
)
if hiddens is None:
    raise ValueError("No hidden states found. Please check the directory and prefix.")
elif hiddens.shape != (2, 4096):
    raise ValueError(f"Hidden states shape is not as expected. Found: {hiddens.shape}")

#mean pooling
extract_and_save_hidden_states(
    prompts = ['This is the test for the package.', 'I hope there is no error.'], #texts or prompts
    output_hidden_dir = "test/data/", #directory to save hidden states
    save_name = 'test/data/test', #path and file name to save generated texts
    tokenizer = tokenizer,
    model = model,
    task_type = "reuse", #'reuse' is when you use LLM to regenerate texts and get hidden states
    # if you want to generate new texts, set task_type == "create"
    # You can specify any task by writing the task prompt and set task_type == <YOUR TASK>
    pooling = "mean",
)

# Check hidden states dimension
hiddens = load_hiddens(
    directory = "test/data/", 
    hidden_list = [0,1],
    prefix = "hidden_"
)
if hiddens is None:
    raise ValueError("No hidden states found. Please check the directory and prefix.")
elif hiddens.shape != (2, 4096):
    raise ValueError(f"Hidden states shape is not as expected. Found: {hiddens.shape}")