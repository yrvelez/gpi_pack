from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import os
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_instruction(task_type: str) -> str:
    """
    Get the instruction based on the task type.

    Args:
    - task_type: str, the type of task to be performed (either "create", "repeat", or some user specific task)

    Output:
    - instruction: str, instruction for the model
    """
    if task_type == 'create':
        #Task: create texts
        instruction = "You are a text generator who always produces the texts suggested by the prompts."
    elif task_type == 'repeat':
        #Task: repeat texts
        instruction = "You are a text generator who just repeats the input text."
    else:
        #User specific task
        print(f"Your task is: {task_type}")
        instruction = task_type
    return instruction

def generate_text(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    instruction: str,
    prompts: list,
    max_new_tokens: int,
    save_hidden: str,
    prefix_hidden: str = "hidden_",
    tokenizer_config: dict = {},
    model_config: dict = {},
    pooling: str = "last",
    indices: list = None,
    ) -> list[str, list[int]]:
    """
    Generate text based on the prompts and save the hidden states.

    Args:
    - tokenizer: AutoTokenizer, tokenizer for the model
    - model: AutoModelForCausalLM, model for text generation
    - instruction: str, instruction for the model
    - prompts: list[str], list of prompts
    - max_new_tokens: int, number of tokens to be generated
    - hidden_use: str, how to use hidden states (either "only_first", "except_first", or "all")
        - "only_first": only use the last hidden states of the first token (it contains semantic information of the entire sentences)
        - "except_first": only use the last hidden states of the tokens except the first token (not recommended)
        - "all": use the last hidden states of all tokens
    - save_hidden: str, directory to output the hidden states
    - tokenizer_config: dict, configuration for the tokenizer
    - model_config: dict, configuration for the model
    - pooling: str, pooling strategy for hidden states (default: "last")
        - "last": use the last hidden states of the first token (it contains semantic information of the entire sentences)
        - "mean": use the mean of the last hidden states of all tokens
    - indices: list[int], optional list of original indices for resume functionality

    Output:
    - generated_texts: list[str], list of generated texts
    """
    generated_texts = []
    
    # add parameters to the tokenizer
    tokenizer_config.update({"add_generation_prompt": False})
    tokenizer_config.update({"return_dict": True})
    tokenizer_config.update({"return_tensors": "pt"})
    
    # add parameters to the model
    model_config.update({"max_new_tokens": max_new_tokens})
    model_config.update({"output_hidden_states": True})
    model_config.update({"return_dict_in_generate": True})
    model_config.update({"do_sample": False})
    model_config.update({"pad_token_id": tokenizer.eos_token_id})
    model_config.update({"temperature" : None})
    model_config.update({"top_p" : None})
    
    # Use provided indices or default to sequential numbering
    if indices is None:
        indices = list(range(len(prompts)))
    
    for k, prompt in enumerate(tqdm(prompts, desc="Generating texts")):
        # Use the original index for saving files
        save_index = indices[k]
        
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ]
        input = tokenizer.apply_chat_template(
            messages,
            **tokenizer_config
        ).to(model.device)
        input_ids = input['input_ids'].to(model.device)
        attention_mask = input['attention_mask'].to(model.device)
        
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            **model_config
        )
        
        #extract the last hidden states
        if pooling == "last":
            hidden_all = outputs.hidden_states[-1][-1].view(-1, 4096)
        elif pooling == "mean":
            hidden_all = torch.stack([item[-1] for item in outputs.hidden_states[1:]]).view(-1, 4096)
            hidden_all = hidden_all.mean(dim=0).view(-1, 4096)
        else:
            raise ValueError("Invalid pooling strategy. Choose from 'last', 'mean', or 'all'.")
        #save the hidden states
        torch.save(hidden_all, f"{save_hidden}/{prefix_hidden}{save_index}.pt")
        
        #decode the generated tokens back to texts
        response = outputs.sequences[0][input_ids.shape[-1]:]
        text = tokenizer.decode(response, skip_special_tokens=True)
        generated_texts.append(text)
    return generated_texts

def save_generated_texts(generated_texts: list, prompts: list, save_name: str):
    """
    Save the generated texts to a pickle file.

    Args:
    - generated_texts: list[str], list of generated texts
    - prompts: list[str], list of prompts
    - save_name: str, filename to be saved
    """
    try:
        pd.DataFrame({"X": generated_texts, "P": prompts}).to_pickle(f'{save_name}.pkl')
    except Exception as e:
        logging.error(f"Failed to save generated texts: {e}")
        raise

def extract_and_save_hidden_states(
    prompts: list[str],
    output_hidden_dir: str,
    save_name: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    task_type: str = "create",
    max_new_tokens: int = 1000,
    prefix_hidden: str = "hidden_",
    tokenizer_config: dict = {},
    model_config: dict = {},
    pooling: str = "last",
    resume: bool = False,
):
    """
    High-level function to load prompts, generate texts, extract hidden states from an LLM,
    and save both the hidden states and generated texts.

    Args:
        prompt_file (list): List of prompt strings.
        output_hidden_dir (str): Directory where extracted hidden state files will be saved.
        save_name (str): Base filename (without extension) to save generated texts.
        cache_dir (str, optional): Directory to cache model files.
        task_type (str, optional): Task type or instruction (default: "create").
        max_new_tokens (int, optional): Maximum number of tokens to generate (default: 1000).
        prefix_hidden (str, optional): Prefix for hidden state filenames (default: "hidden_").
        tokenizer_config (dict, optional): Additional config for the tokenizer.
        model_config (dict, optional): Additional config for the model.
        pooling (str, optional): Pooling strategy for hidden states (default: "last").
            - "last": use the last hidden states of the first token (it contains semantic information of the entire sentences)
            - "mean": use the mean of the last hidden states of all tokens
            - "all": use the last hidden states of all tokens (not recommended)
        resume (bool, optional): If True, skip prompts that already have hidden state files (default: False).
    Returns:
        None: The function saves the generated texts and hidden states to specified files.
    """
    # Ensure the output directory exists.
    if not os.path.exists(output_hidden_dir):
        os.makedirs(output_hidden_dir)
        logger.info(f"Created directory: {output_hidden_dir}")

    # Check for existing hidden state files if resume is True
    prompts_to_process = prompts.copy()
    indices_to_process = list(range(len(prompts)))
    
    if resume:
        # Filter out prompts that already have hidden state files
        filtered_prompts = []
        filtered_indices = []
        skipped_count = 0
        
        for i, prompt in enumerate(prompts):
            hidden_file = f"{output_hidden_dir}/{prefix_hidden}{i}.pt"
            if os.path.exists(hidden_file):
                logger.info(f"Skipping prompt {i} (already exists): {hidden_file}")
                skipped_count += 1
            else:
                filtered_prompts.append(prompt)
                filtered_indices.append(i)
        
        prompts_to_process = filtered_prompts
        indices_to_process = filtered_indices
        
        if skipped_count > 0:
            logger.info(f"Resuming: Skipped {skipped_count} already processed prompts")
        
        if not prompts_to_process:
            logger.info("All prompts have already been processed. Nothing to do.")
            return

    instruction = get_instruction(task_type)
    logger.info(f"Using instruction: {instruction}")
    generated_texts = generate_text(
        tokenizer = tokenizer, 
        model = model, 
        instruction = instruction, 
        prompts = prompts_to_process, 
        max_new_tokens = max_new_tokens,
        save_hidden = output_hidden_dir,
        prefix_hidden= prefix_hidden,
        tokenizer_config= tokenizer_config,
        model_config= model_config,
        pooling= pooling,
        indices= indices_to_process if resume else None
    )

    # Load existing results if resuming
    if resume and os.path.exists(f"{save_name}.pkl"):
        import pickle
        with open(f"{save_name}.pkl", 'rb') as f:
            existing_df = pd.read_pickle(f"{save_name}.pkl")
            existing_texts = existing_df["X"].tolist()
            existing_prompts = existing_df["P"].tolist()
        
        # Combine existing and new results
        all_texts = existing_texts.copy()
        for idx, text in zip(indices_to_process, generated_texts):
            # Insert or update at the correct position
            if idx < len(all_texts):
                all_texts[idx] = text
            else:
                all_texts.extend([None] * (idx - len(all_texts) + 1))
                all_texts[idx] = text
        
        save_generated_texts(all_texts, prompts, save_name)
    else:
        save_generated_texts(generated_texts, prompts_to_process if resume else prompts, save_name)
    
    logger.info("Extraction and saving complete.")