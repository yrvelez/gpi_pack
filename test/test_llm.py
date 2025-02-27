import pytest
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from gpi_pack.llm import extract_and_save_hidden_states
from gpi_pack.TarNet import load_hiddens
from dotenv import load_dotenv

# Skip the test if the environment variable is not set
@pytest.fixture(scope="module")
def hf_token():
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        pytest.skip("HUGGINGFACE_TOKEN environment variable not set")
    return token

@pytest.fixture(scope="module")
def model_and_tokenizer(hf_token):
    """Load model and tokenizer once for all tests"""
    checkpoint = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        token=hf_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, 
        device_map="auto", 
        torch_dtype=torch.float16,
        cache_dir="test/development",
    )
    return model, tokenizer

@pytest.fixture
def test_prompts():
    return ['This is the test for the package.', 'I hope there is no error.']

@pytest.fixture
def output_dirs():
    # Create directories if they don't exist
    os.makedirs("test/data/", exist_ok=True)
    return {
        "hidden_dir": "test/data/",
        "save_name": "test/data/test"
    }

def test_last_token_pooling(model_and_tokenizer, test_prompts, output_dirs):
    """Test extraction with last token pooling"""
    model, tokenizer = model_and_tokenizer
    
    extract_and_save_hidden_states(
        prompts=test_prompts,
        output_hidden_dir=output_dirs["hidden_dir"],
        save_name=output_dirs["save_name"],
        tokenizer=tokenizer,
        model=model,
        task_type="reuse",
        pooling="last",
    )
    
    # Check hidden states dimension
    hiddens = load_hiddens(
        directory=output_dirs["hidden_dir"], 
        hidden_list=[0, 1],
        prefix="hidden_"
    )
    
    assert hiddens is not None, "No hidden states found"
    assert hiddens.shape == (2, 4096), f"Unexpected hidden states shape: {hiddens.shape}"

def test_mean_pooling(model_and_tokenizer, test_prompts, output_dirs):
    """Test extraction with mean pooling"""
    model, tokenizer = model_and_tokenizer
    
    extract_and_save_hidden_states(
        prompts=test_prompts,
        output_hidden_dir=output_dirs["hidden_dir"],
        save_name=output_dirs["save_name"],
        tokenizer=tokenizer,
        model=model,
        task_type="reuse",
        pooling="mean",
    )
    
    # Check hidden states dimension
    hiddens = load_hiddens(
        directory=output_dirs["hidden_dir"], 
        hidden_list=[0, 1],
        prefix="hidden_"
    )
    
    assert hiddens is not None, "No hidden states found"
    assert hiddens.shape == (2, 4096), f"Unexpected hidden states shape: {hiddens.shape}"