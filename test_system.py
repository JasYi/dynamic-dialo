from full_system import *
import json
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# run the dialogue state tracking feature on one utterance of multiwoz
def test_multiwoz(conversation, user_response, slot_model):
    # create slot step
    remaining_slots = [key for key, value in slot_model.items() if value == "unfilled"]
    created_slot = check_create_slot(conversation, user_response, slot_model, slot_model.keys())
    
    # fill slot step
    remaining_slots = [key for key, value in slot_model.items() if value == "unfilled"]
    filled_slot = check_fill_slot(conversation, user_response, slot_model, remaining_slots)
    
    return slot_model

def read_dialogue(dialogue):
    # setup tracking variables
    conversation = ""
    slot_model = {}
    results = []
    
    for turn in dialogue:
        if turn['speaker'] == "SYSTEM":
            for frame in turn['frames']:
                for slot in frame['slots']:
                    slot_model[slot['slot']] = slot['value']
        
        elif turn['speaker'] == "USER":
            result_val = {}
            
            # run the dialogue state tracking feature
            slot_model_out = test_multiwoz(conversation, turn['utterance'], slot_model.copy())
            
            result_val['utterance'] = turn['utterance']
            result_val['known_info'] = slot_model.copy()
            result_val['generated'] = slot_model_out.copy()
            
            # update the slot model
            for frame in turn['frames']:
                frame_state = frame['state']
                
                # get requested slots
                for req_slot in frame_state['requested_slots']:
                    slot_model[req_slot] = "unfilled"
                    
                # get filled slots
                slots_to_add = {key: (value[0] if value else None) for key, value in frame_state['slot_values'].items()}
                
                # add the slots to the current slots
                slot_model = slot_model | slots_to_add
            
            result_val['expected'] = slot_model.copy()
            result_val['excess'] = {key: value for key, value in slot_model_out.items() if key not in slot_model}
            result_val['missing'] = {key: value for key, value in slot_model.items() if key not in slot_model_out}
            results.append(result_val)
    return results
    
# read multiwoz file and run the dialogue state tracking feature on each utterance
def read_multiwoz(file_path):
    # open multiwoz json file
    with open(file_path, 'r') as file:
        data = json.load(file)
        
    final_results = []

    # Define a function to process each dialogue
    def process_dialogue(dialogues):
        return {dialogues['dialogue_id']: {"results": read_dialogue(dialogues['turns'])}}

    # Initialize variables
    final_results = []

    # Use ThreadPoolExecutor or ProcessPoolExecutor for parallel processing
    # ThreadPoolExecutor for I/O-bound tasks; ProcessPoolExecutor for CPU-bound tasks
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # Submit tasks to the executor
        results = list(executor.map(process_dialogue, data[10:15]))

    # Collect results
    final_results.extend(results)

    
    
    
    # write the results to results json file
    results_file = open("multiwoz_results_10_15.json", "w")
    json.dump(final_results, results_file)
    results_file.close()
    
if __name__ == "__main__":
    read_multiwoz("../multiwoz/data/MultiWOZ_2.2/train/dialogues_001.json")