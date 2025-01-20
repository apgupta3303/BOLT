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
from therapist_behavior_inference import TherapistBehaviorAnalyzer  
from client_behavior_inference import ClientBehaviorAnalyzer
from DATASET_PATHS import DATASET_PATHS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse and validate command line arguments."""
    parser = ArgumentParser(
        description='Analyze therapist behavior from conversations',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='multi_label_w_def_and_ex',
        choices=TherapistBehaviorAnalyzer.VALID_METHODS,
        help='Classification method to use'
    )
    
    parser.add_argument(
        '--role',
        type=str,
        default="therapist",
        help='Client or Therapist'
    )

    parser.add_argument(
        '--chatbot',
        type=str,
        default="ash",
        help='Chatbot of interest'
    )

    
    return parser.parse_args()


def main():
    """Main entry point of the script."""
    args = parse_arguments()
    
    logger.info(f"Method: {args.method}")
    
    
    role = args.role.lower()
    chatbot = args.chatbot.lower()
    input_path = ""
    output_path = ""

    if role == "therapist":
        input_path = DATASET_PATHS[chatbot]["therapist"]["input"]
        output_path = DATASET_PATHS[chatbot]["therapist"]["output"]
        analyzer = TherapistBehaviorAnalyzer(args.method)
        print(input_path)
    elif role == "client":
        input_path = DATASET_PATHS[chatbot]["client"]["input"]
        output_path = DATASET_PATHS[chatbot]["client"]["output"]
        analyzer = ClientBehaviorAnalyzer(args.method)
    else:
        print("INVALID ARGUMENTS")
        return 
    print(f'INPUT PATH: {input_path}')
    df = pd.read_csv(input_path)
    therapist_df = df[df["Role"] == "therapist"]
    analyzer.process_file(therapist_df, output_path)

    print("EVALAUTING BEHAVIOR FREQUENCY")
    print()
    analyzer.evaluate_behavior_frequencies(output_path, chatbot)
    print("COMPLETED BEHAVIOR FREQUENCY")
    print()

    print("EVALAUTING TEMPORAL ORDER")
    print()
    analyzer.evaluate_temporal_order(output_path, chatbot)
    print("COMPLETED TEMPORAL ORDER")


if __name__ == '__main__':
    main()