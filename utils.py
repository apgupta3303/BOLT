import time
import os
import json
import csv
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import codecs
from typing import List, Dict, Any, Optional

model = 'gpt-4'

def read_prompt_csv(role: str) -> List[Dict[str, Any]]:
    """
    Read and parse prompts from CSV files based on role.
    
    Args:
        role: Either 'therapist' or 'client'
    Returns:
        List of dictionaries containing intent details
    """
    if role not in ['therapist', 'client']:
        raise ValueError("Role must be either 'therapist' or 'client'")
        
    filename = f'prompts/{role}_prompts.csv'
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Could not find prompts file: {filename}")
        
    df = pd.read_csv(filename)
    intent_detail_list = []
    
    for _, row in df.iterrows():
        positive_examples = [
            row['positive example 1'].strip(),
            row['positive example 2'].strip(), 
            row['positive example 3'].strip()
        ]
        negative_examples = [
            row['negative example 1'].strip(),
            row['negative example 2'].strip(),
            row['negative example 3'].strip()
        ]
        intent_detail_list.append({
            'intent': row['intent'].strip(),
            'definition': row['definition'].strip(),
            'positive_examples': positive_examples,
            'negative_examples': negative_examples
        })
    
    return intent_detail_list


def format_intents(intent_map_list, examples = True):
    formatted_string = ""
    for intent in intent_map_list:
        # Add the definition to the string
        formatted_string += f'{intent["intent"]}: {intent["definition"]}\n'

        # Loop through each example and add it to the string, numbered
        if examples:
            formatted_string += f'True Examples of {intent["intent"]} \n'

            for index, example in enumerate(intent["examples"], start=1):
                formatted_string += f'  {index}. {example}\n'

        # Add a newline for spacing between entries
        formatted_string += '\n'

    return formatted_string

def get_multi_label_w_def_and_ex(intent_name_text, intent_detail_list):
    intent_definition_with_examples_map = [
            {
                "intent": detail['intent'],
                "definition": detail['definition'].replace("\\", ""),
                "examples": detail['positive_examples']
            }
            for detail in intent_detail_list
        ]
        
    formatted_intent = format_intents(intent_definition_with_examples_map)

    content = f"""You are a classifier that identifies intents in therapist utterances.
        Here are the possible intents and their definitions:
        
        {formatted_intent}
        
        Set all of these that are exhibited to True [{intent_name_text}].
        
        """
    
    result =  [{
        'role': 'system',
        'content': content
    }]
    print("SYSTEM:")
    print(content)
    print()
    return result

def get_multi_label_w_def(intent_name_text, intent_detail_list):
    intent_definition_with_examples_map = [
            {
                "intent": detail['intent'],
                "definition": detail['definition'].replace("\\", "")
            }
            for detail in intent_detail_list
        ]
        
    formatted_intent = format_intents(intent_definition_with_examples_map, examples=False)
    
    return [{
        'role': 'system',
        'content': f"""You are a classifier that identifies intents in therapist utterances.
        Here are the possible intents and their definitions:
        
        {formatted_intent}
        
        Only choose from this list [{intent_name_text}]"""
    }]

