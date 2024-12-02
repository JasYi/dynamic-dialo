from openai import OpenAI
import json

# as extension task add the slot type
def create_slot(slot_name, slot_model):
    slot_model[slot_name] = "unfilled"

def fill_slot(slot_name, slot_model, slot_value):
    slot_model[slot_name] = slot_value
    
create_slot_func = {
    "name": "create_slot",
    "description": "function creates a slots based on a user query",
    "parameters": {
        "type": "object",
        "properties": {
            "slots_to_add": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "name of the slot"
                },
                "description": "list of slots to add to accomplish what the user wants"
            }
        },
        "required": ["slot_name"]
    }
}

fill_slot_func = {
    "name": "fill_slot",
    "description": "function fills a slot name with slot value",
    "parameters": {
        "type": "object",
        "properties": {
            "slots_to_fill":{
                "type": "array",
                "items":{
                    "type": "object",
                        "properties": {
                        "slot_name":{
                            "type": "string",
                            "description": "name of the slot"
                        },
                        "slot_value":{
                            "type": "string",
                            "description": "value of the slot"
                        }
                    }
                },
                "description": "list of slots to fill with their values"
            }
        },
        "required": ["slot_name", "slot_value"]
    }
}

create_slot_prompt = '''
    You are an assistant that can determine whether to create a slot.
    You are given a conversation context and a slot model.
    The slot model is a list of slot names that already exist. 
    A slot name corresponds to a piece of information that the user can provide.
    If the user asks a question or requests for information or an action that is not in the slot model then call create_slot_func and pass in a list of slot names designating the information the system would need answered by the user to accomplish what they request.
    If necessary you can add multiple slots representing the different pieces of information the user would need to provide.
    Use the entire conversation context as context but only take an action based on the last user message.
    Do not create a slot if the user is asking for information that is already in the slot model.
    If it is not necessary to create a slot then do nothing and do not call create_slot_func.
'''

fill_slot_prompt = '''
    You are an assistant that can determine whether to fill a slot.
    You are given a conversation context and a slot model.
    The slot model is a list of slot names that need to be filled. 
    A slot name corresponds to a piece of information that the user can provide.
    If the user answers with information answers a topic in the slot model then call fill_slot_func with the slot name and value.
    Use the entire conversation context as context but only take an action based on the last user message.
    Do not fill a slot if the user is not answering a question or providing information that is in the slot model.
    If it is not necessary to fill a slot then do nothing and do not call fill_slot_func.
'''

def check_create_slot(convo_context, slot_model, remaining_slots):
    tools_in = [
        {
            "type": "function",
            "function": create_slot_func,
        }
    ]
    
    user_prompt = f'''
        Here is the conversation context:
        {convo_context}
        Here are the slots that already exist:
        {", ".join(remaining_slots)}
    '''
    
    msgs_in = [
        {"role": "system", "content": create_slot_prompt},
        {"role": "user", "content": user_prompt}
    ]
        
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=msgs_in,
        tools=tools_in,
        temperature=0.3
    )
    
    tool_choice = completion.choices[0].message.tool_calls[0] if completion.choices[0].message.tool_calls else None
    if tool_choice:
        arguments = json.loads(tool_choice.function.arguments)
        for slot in arguments["slots_to_add"]:
            create_slot(slot, slot_model)
        return True
    return False

def check_fill_slot(convo_context, slot_model, remaining_slots):
    tools_in = [
        {
            "type": "function",
            "function": fill_slot_func,
        }
    ]
    
    user_prompt = f'''
        Here is the conversation context:
        {convo_context}
        Here are the slots that need to be filled:
        {", ".join(remaining_slots)}
    '''
    
    msgs_in = [
        {"role": "system", "content": fill_slot_prompt},
        {"role": "user", "content": user_prompt}
    ]
        
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=msgs_in,
        tools=tools_in,
        temperature=0.3
    )
    
    tool_choice = completion.choices[0].message.tool_calls[0] if completion.choices[0].message.tool_calls else None

    if tool_choice:
        arguments = json.loads(tool_choice.function.arguments)
        for elem in arguments["slots_to_fill"]:
            fill_slot(elem["slot_name"], slot_model, elem["slot_value"])
        return True
    return False

def generate_response(task, slot_name="", slot_value="", previous_response=""):
    if task == "regenerate":
        system_prompt = f'''
        '''
        user_prompt = f'''
        '''
        
        
    if task == "find_slot":
        system_prompt = f'''
            You are a system that will generate the next response based on the conversation context and a slot model.
            Generate a response that will prompt the user to provide the value for the slot name.
        '''
        user_prompt = f'''
            Conversation context: 
            {previous_response}
            
            Slot name: {slot_name}
        '''
        
    if task == "confirm":
        system_prompt = f'''
            
        '''
        user_prompt = f'''
        '''
    
    msgs_in = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
        
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=msgs_in,
    )
    
    return completion.choices[0].message.content
    
def generate_summary(conversation, slot_model):
    system_prompt = f'''
        You are a system that will generate a response summarizing the information that has been collected.
        Generate a response that summarizes the information in the slot model.
        The slot model is an object of slot names and their corresponding values.
        You will also be given the conversation context that was used to collect the information.
        This summary message will be sent to the user to confirm the information that has been collected.
        Respond with only the summary message.
    '''
    user_prompt = f'''
        Conversation context: 
        {conversation}
        
        Slot model: 
        {slot_model}
        '''
    
    msgs_in = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
        
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=msgs_in,
    )
    
    return completion.choices[0].message.content

def find_next_slot(conversation, slot_model):
    system_prompt = f'''
        You are an agent that picks the next slot to fill based on the slot model.
        You will be given the remaining slot names to pick from and the conversation context.
        Return only the name of the slot to fill next that makes the most logical sense.
    '''
    user_prompt = f'''
        Conversation context: 
        {conversation}
        
        Remaining slots: 
        {slot_model}
        '''
    
    msgs_in = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
        
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=msgs_in,
    )
    
    return completion.choices[0].message.content
    
def run_dialogue(start_state):
    slot_model = start_state
    remaining_slots = [key for key, value in slot_model.items() if value == "unfilled"]
    conversation = ""
    while remaining_slots:
        slot_name = find_next_slot(conversation, remaining_slots)
        response = generate_response("find_slot", slot_name)
        conversation += f"System: {response}\n"
        print(response)
        user_response = input()
        conversation += f"User: {user_response}\n"
        created_slot = check_create_slot(conversation, slot_model, slot_model.keys())
        remaining_slots = [key for key, value in slot_model.items() if value == "unfilled"]
        filled_slot = check_fill_slot(conversation, slot_model, remaining_slots)
        # response = generate_response("confirm", slot_name, user_response)
        # print(response)
        remaining_slots = [key for key, value in slot_model.items() if value == "unfilled"]
        print(slot_model)
    summary_response = generate_summary(conversation, slot_model)
    print(summary_response)

def run_dialogue_no_input():
    print("Enter a task request:")
    task_request = input()
    tools_in = [
        {
            "type": "function",
            "function": create_slot_func,
        }
    ]
    
    system_prompt = f'''
        Given a request from the user, determine what slots to create.
        A slot is a piece of information necessary to accomplish what the user wants.
        If the user asks a question or requests for information or an action then call create_slot_func and pass in a list of slot names designating the information the system would need answered by the user to accomplish what they request.
    '''
    
    user_prompt = f'''
        Here is the request:
        {task_request}
    '''
    
    msgs_in = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
        
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=msgs_in,
        tools=tools_in,
        temperature=0.3
    )
    
    tool_choice = completion.choices[0].message.tool_calls[0] if completion.choices[0].message.tool_calls else None
    
    slot_model = {}
    
    if tool_choice:
        arguments = json.loads(tool_choice.function.arguments)
        for slot in arguments["slots_to_add"]:
            create_slot(slot, slot_model)
    
    print(slot_model)
    run_dialogue(slot_model)
    

if __name__ == "__main__":
    airport_slots = {
        "destination_city": "unfilled", 
        "departure_city": "unfilled",
        "departure_date": "unfilled", 
        "return_date": "unfilled", 
        "budget": "unfilled", 
        "preferred_airline": "unfilled", 
    }
    
    pizza_slots = {
        "pizza_size": "unfilled",
        "pizza_toppings": "unfilled",
        "delivery_address": "unfilled",
    }
    
    run_dialogue(pizza_slots)
    
    run_dialogue_no_input()