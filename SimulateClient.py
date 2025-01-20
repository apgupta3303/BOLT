import pandas as pd
from openai import OpenAI
import os
from datetime import datetime

class ParallelClientSimulator:
    def __init__(self, api_key=None):
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
        self.system_prompt = """You are simulating a therapy client who is similar to one from an existing conversation. 
        You should maintain the same:
        - Conversational style
        - Manner of addressing topics/concerns
        - Life events and emotions being discussed
        
        However, you should simulate this as if it's a parallel universe where this client is talking to a different therapist. Do not state anything about seperate universe
        DO NOT continue the original conversation - instead create a parallel one with similar themes and concerns.
        
        Important: Maintain high consistency in the client's:
        - Background story
        - Emotional patterns
        - Core challenges
        - Communication style
        """
        self.conversation_history = []
        self.original_conversation = None

    def load_conversation(self, csv_path):
        self.original_conversation = pd.read_csv(csv_path)
        
    def format_conversation_history(self, conversation):
        formatted = ""
        for _, row in conversation.iterrows():
            formatted += f"{row['speaker']}: {row['text']}\n"
        return formatted

    def create_simulation_prompt(self):
        current_context = "\n".join([
            f"{'Therapist' if is_therapist else 'Client'}: {text}"
            for is_therapist, text in self.conversation_history
        ])

        prompt = f"""{self.system_prompt}
        
        {self.format_conversation_history(self.original_conversation)}
        
        Current conversation:
        {current_context if self.conversation_history else 'Start of conversation'}
        
        Only write the patient next repsonse.

        Patient's response:"""
        
        return prompt

    def get_llm_response(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return "I'm sorry, I'm having trouble responding right now."

    def save_conversation(self, filename=None):
        """Save the conversation history to a CSV file"""
        if filename is None:
            filename = f"therapy_session.csv"
        filepath = f'data/{filename}'
        
        conversation_data = []
        for i, (is_therapist, text) in enumerate(self.conversation_history, 1):
            row = {
                'utterance': text,
                'role': 'therapist' if is_therapist else 'client',
                'turn': i,
                'convID': 1  
            }
            conversation_data.append(row)
        
        df = pd.DataFrame(conversation_data)
        df = df[['utterance', 'role', 'turn', 'convID']]  
        df.to_csv(filename, index=False)
        print(f"\nConversation saved to {filepath}")
        return filepath

    def run_session(self):
        print("Starting session (type 'quit' to end)\n")
        
        while True:
            therapist_message = input("Therapist: ")
            if therapist_message.lower() == 'quit':
                break
                
            self.conversation_history.append((True, therapist_message))
            
            prompt = self.create_simulation_prompt()
            patient_response = self.get_llm_response(prompt)
            
            self.conversation_history.append((False, patient_response))
            print(f"\n{patient_response}")
        
        print("\nEnter filename to save conversation (press Enter for automatic timestamp):")
        custom_filename = input().strip()

        self.save_conversation(custom_filename if custom_filename else None)


def main():
    csv_index = 206
    file = f"BOLT/HOPE_Data_Examples/patient{csv_index}.csv"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    print(f"Running {file}")
    simulator = ParallelClientSimulator(OPENAI_API_KEY)
    simulator.load_conversation(file)
    simulator.run_session()

if __name__ == "__main__":
    main()


