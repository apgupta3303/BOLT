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
import os


from utils import read_prompt_csv, format_intents, get_multi_label_w_def_and_ex, get_multi_label_w_def


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TherapistsIntents(BaseModel):
    makes_needs_explicit: bool
    makes_emotions_explicit: bool
    makes_values_explicit: bool
    makes_consequences_explicit: bool
    makes_conflict_explicit: bool
    makes_strengths_resources_explicit: bool
    evokes_concrete_elaboration: bool
    evokes_perspective_elaboration: bool
    emotions_check_in: bool
    problem_solving: bool
    planning: bool
    normalizing: bool
    teaching_psychoeducation: bool

class TherapistBehaviorAnalyzer:
    
    VALID_METHODS = [
        'multi_label_w_def_and_ex',
        'multi_label_w_def'
    ]

    def __init__(self, method: str):
        if method not in self.VALID_METHODS:
            raise ValueError(f"Invalid method. Must be one of: {self.VALID_METHODS}")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.method = method
        self.intent_detail_list = read_prompt_csv("therapist")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        self.context_messages = self._create_context_messages()

    def _create_context_messages(self) -> List[Dict[str, str]]:
        """Create the initial context messages with all intents and examples."""
        intent_name_list = [detail['intent'] for detail in self.intent_detail_list]
        intent_name_text = ', '.join(f'"{word}"' for word in intent_name_list)
        
        if self.method == 'multi_label_w_def_and_ex':
            return get_multi_label_w_def_and_ex(intent_name_text, self.intent_detail_list)
        elif self.method == 'multi_label_w_def':
            return get_multi_label_w_def(intent_name_text, self.intent_detail_list)
        

    def get_therapist_intent(self, utterance: str) -> Optional[str]:
        """Analyze therapist intent from an utterance."""
        utterance = utterance.strip()
        messages = self.context_messages + [{
            'role': 'user',
            'content': f"What are all possible intents of this therapist utterance: {utterance}?"
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
                    response_format = TherapistsIntents
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

    def process_file(self, input, output_path: Path) -> None:
        """Process input file and write results to output file."""
        try:
            
            print(f'OUTPUT PATH: {output_path}')
            new_data = []
            print(input.head())
            for index, row in input.iterrows():
                print(index)
                utterance = row['Utterance']  
                therapist_behavior = self.get_therapist_intent(utterance)

                row_data = {
                    'utterance': utterance,
                    'makes_needs_explicit': 1 if therapist_behavior.makes_needs_explicit else 0,
                    'makes_emotions_explicit': 1 if therapist_behavior.makes_emotions_explicit else 0,
                    'makes_values_explicit': 1 if therapist_behavior.makes_values_explicit else 0,
                    'makes_consequences_explicit': 1 if therapist_behavior.makes_consequences_explicit else 0,
                    'makes_conflict_explicit': 1 if therapist_behavior.makes_conflict_explicit else 0,
                    'makes_strengths_resources_explicit': 1 if therapist_behavior.makes_strengths_resources_explicit else 0,
                    'evokes_concrete_elaboration': 1 if therapist_behavior.evokes_concrete_elaboration else 0,
                    'evokes_perspective_elaboration': 1 if therapist_behavior.evokes_perspective_elaboration else 0,
                    'emotions_check_in': 1 if therapist_behavior.emotions_check_in else 0,
                    'problem_solving': 1 if therapist_behavior.problem_solving else 0,
                    'planning': 1 if therapist_behavior.planning else 0,
                    'normalizing': 1 if therapist_behavior.normalizing else 0,
                    'teaching_psychoeducation': 1 if therapist_behavior.teaching_psychoeducation else 0,
                    'urn': row['Turn'],
                    'convID': row['ConvID']

                }
                
                new_data.append(row_data)
                
            new_df = pd.DataFrame(new_data)
            print(new_df)
            new_df.to_csv(output_path, index=False)
            print("Processed Properly")
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")

    def evaluate_behavior_frequencies(self, csv_path, chatbot):
        df = pd.read_csv(csv_path)
        
        behavior_cols = [col for col in df.columns if col not in ['Utterance', 'Turn', 'ConvID']]
        
        stats = []
        for col in behavior_cols:
            print(col)
            messages_with_behavior = df[col].sum() 
            print(f"Messages with Behavior Type: {type(messages_with_behavior)}")
            print(f"Messages with Behavior: {messages_with_behavior}")
            total_messages = len(df)
            print(f"Total Messages Type: {type(total_messages)}")
            percentage = (messages_with_behavior / total_messages) * 100
            
            stats.append({
                'behavior': col,
                'messages_with_behavior': messages_with_behavior,
                'total_messages': total_messages,
                'percentage': round(percentage, 2)
            })
            print(f'{col}: {percentage}')
        
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(f'evaluations/{chatbot}/frequencies.csv')
        return 
    
    def evaluate_temporal_order(self, csv_path, chatbot):
        """
        Analyzes the temporal order of therapist behaviors within conversations.
        
        Args:
            csv_path: Path to the CSV file containing the behavior data
            chatbot: Name of the chatbot for saving results
            
        Returns:
            pandas.DataFrame: DataFrame containing first occurrences of behaviors in each conversation
        """
        
        print("EVALUATING TEMPORAL ORDER")
        df = pd.read_csv(csv_path)
        
        behavior_columns = [col for col in df.columns if col not in ['Utterance', 'Turn', 'ConvID']]
        
        results = []
        
        for conv_id, group in df.groupby('ConvID'):
            group = group.sort_values('Turn')
            
            first_occurrences = {'ConvID': conv_id}
            
            for behavior in behavior_columns:
                first_turn = group.loc[group[behavior] == 1, 'Turn'].min()
                first_occurrences[behavior] = first_turn if pd.notna(first_turn) else None
            
            results.append(first_occurrences)
        
        results_df = pd.DataFrame(results)
        print("\nFirst occurrences per conversation:")
        print(results_df)
        
        averaged_df = pd.DataFrame([results_df.drop('ConvID', axis=1).mean()])
        print("\nAveraged across conversations:")
        print(averaged_df)
        
        output_path = Path(f'evaluations/{chatbot}/temporal_order.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        averaged_df.to_csv(output_path, index=False)
        return averaged_df
    
