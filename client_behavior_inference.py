import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from utils import read_prompt_csv, format_intents, get_multi_label_w_def_and_ex, get_multi_label_w_def


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClientIntents(BaseModel):
    changing_unhealthy_behavior: bool
    sustaining_unhealthy_behavior: bool
    sharing_negative_feeling_or_emotion: bool
    sharing_positive_feeling_or_emotion: bool
    gained_insight: bool
    sharing_life_event_or_situation: bool

class ClientBehaviorAnalyzer:
    
    VALID_METHODS = [
        'multi_label_w_def_and_ex',
        'multi_label_w_def'
    ]

    def __init__(self, method: str):
        if method not in self.VALID_METHODS:
            raise ValueError(f"Invalid method. Must be one of: {self.VALID_METHODS}")
            
        self.method = method
        self.intent_detail_list = read_prompt_csv("client")
        self.client = OpenAI(api_key="")
        
        self.context_messages = self._create_context_messages()

    def _create_context_messages(self) -> List[Dict[str, str]]:
        """Create the initial context messages with all intents and examples."""
        intent_name_list = [detail['intent'] for detail in self.intent_detail_list]
        intent_name_text = ', '.join(f'"{word}"' for word in intent_name_list)
        
        if self.method == 'multi_label_w_def_and_ex':
            return get_multi_label_w_def_and_ex(intent_name_text, self.intent_detail_list)
        elif self.method == 'multi_label_w_def':
            return get_multi_label_w_def(intent_name_text, self.intent_detail_list)

    def get_client_intent(self, utterance: str) -> Optional[str]:
        """Analyze therapist intent from an utterance."""
        utterance = utterance.strip()
        messages = self.context_messages + [{
            'role': 'user',
            'content': f"What are all possible intents of this patient utterance: {utterance}?"
        }]
        
        return self._get_completion(messages)

    def _get_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_retries: int = 2
    ) -> Optional[str]:
        """Get completion from OpenAI with retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.client.beta.chat.completions.parse(
                    model="gpt-4o-2024-08-06",
                    messages=messages,
                    temperature=temperature,
                    response_format = ClientIntents
                )
                result = response.choices[0].message.parsed
                return result
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    sleep_time = 3 * (2 ** attempt)
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print("All retry attempts failed")
                    return None

    def process_file(self, input_path: Path, output_path: Path) -> None:
        """Process input file and write results to output file."""
        try:
            print(f'INPUT PATH: {input_path}')
            print(f'OUTPUT PATH: {output_path}')
            input = pd.read_csv(input_path)
            new_data = []
            
            for index, row in input.iterrows():
                utterance = row['utterance']  
                print(utterance)
                print()
                client_behavior = self.get_client_intent(utterance)
                print(f'Client Behavors: {client_behavior}')
                print()
                row_data = {
                    'utterance': utterance,
                    'changing_unhealthy_behavior': 1 if client_behavior.changing_unhealthy_behavior else 0,
					'sustaining_unhealthy_behavior': 1 if client_behavior.sustaining_unhealthy_behavior else 0,
					'sharing_negative_feeling_or_emotion': 1 if client_behavior.sharing_negative_feeling_or_emotion else 0,
					'sharing_positive_feeling_or_emotion': 1 if client_behavior.sharing_positive_feeling_or_emotion else 0,
					'gained_insight': 1 if client_behavior.gained_insight else 0,
					'sharing_life_event_or_situation': 1 if client_behavior.sharing_life_event_or_situation else 0,
                    'turn': row['turn'],
                    'convID': row['convID']
                }
                
                new_data.append(row_data)
                
            new_df = pd.DataFrame(new_data)
            new_df.to_csv(output_path, index=False)
            print("Processed Properly")
            
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_path}")
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
