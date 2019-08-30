import numpy as np
import matplotlib.pyplot as plt
import random

DEBUG = False
DEFAULT_REWARD = -0.01 # Negative reward for each non-terminal step
JUNK_CMD_REWARD = -0.1 # Negative reward for invalid commands
QUEST_REWARD = 1 # positive reward for finishing quest
STEP_COUNT = 0  #count the number of steps in current episode
MAX_STEPS = 20

# --Simple quests
quests = ['You are bored.', 'You are getting fat.', 'You are hungry.','You are sleepy.']
quests_map = {}
# --(somewhat) complex quests
# -- quests = {'You are not sleepy but hungry.',
# --                     'You are not hungry but sleepy.',
# --                     'You are not getting fat but bored.',
# --                     'You are not bored but getting fat.'}

quest_actions = ['watch', 'exercise', 'eat', 'sleep'] #aligned to quests above
quest_objects = ['tv', 'bike', 'apple', 'bed'] #aligned to quest actions above

rooms = ['Living', 'Garden', 'Kitchen','Bedroom']
living_desc = ['This room has a couch, chairs and TV.',
          'You have entered the living room. You can watch TV here.',
          'This room has two sofas, chairs and a chandelier.',
          'A huge television that is great for watching games.']
garden_desc = ['This space has a swing, flowers and trees.',
          'You have arrived at the garden. You can exercise here',
          'This area has plants, grass and rabbits.',
          'A nice shiny bike that is fun to ride.',]
kitchen_desc = ['This room has a fridge, oven, and a sink.',
           'You have arrived in the kitchen. You can find food and drinks here.',
           'This living area has pizza, coke, and icecream.',
           'A red juicy fruit.']
bedroom_desc = ['This area has a bed, desk and a dresser.',
           'You have arrived in the bedroom. You can rest here.',
           'You see a wooden cot and a mattress on top of it.',
           'A nice, comfortable bed with pillows and sheets.']
rooms_desc = {'Living': living_desc, 'Garden': garden_desc, 'Kitchen': kitchen_desc, 'Bedroom': bedroom_desc}
rooms_desc_map = {}


actions = ['eat', 'sleep', 'watch', 'exercise', 'go']
objects = ['apple', 'bed', 'tv', 'bike', 'north','south','east','west']

living_valid_act = ['go', 'go', 'watch']
living_valid_obj = ['south', 'west', 'tv']
living_transit = ['Bedroom', 'Garden', 'Living']
garden_valid_act = ['go', 'go', 'exercise']
garden_valid_obj = ['south', 'east', 'bike']
garden_transit = ['Kitchen', 'Living', 'Garden']
kitchen_valid_act = ['go', 'go', 'eat']
kitchen_valid_obj = ['north', 'east', 'apple']
kitchen_transit = ['Garden', 'Bedroom', 'Kitchen']
bedroom_valid_act =['go', 'go', 'sleep']
bedroom_valid_obj =['north', 'west', 'bed']
bedroom_transit = ['Living', 'Kitchen', 'Bedroom']

rooms_valid_acts = {'Living': living_valid_act, 'Garden': garden_valid_act, 'Kitchen': kitchen_valid_act, 'Bedroom': bedroom_valid_act}
rooms_valid_objs = {'Living': living_valid_obj, 'Garden': garden_valid_obj, 'Kitchen': kitchen_valid_obj, 'Bedroom': bedroom_valid_obj}
rooms_transit = {'Living': living_transit, 'Garden': garden_transit, 'Kitchen': kitchen_transit, 'Bedroom': bedroom_transit}

NUM_ROOMS = len(rooms)
NUM_QUESTS = len(quests)
NUM_ACTIONS = len(actions)
NUM_OBJECTS = len(objects)


command_is_valid = np.zeros((NUM_ROOMS,NUM_ACTIONS,NUM_OBJECTS))
transit_matrix = np.zeros((NUM_ROOMS,NUM_ACTIONS,NUM_OBJECTS,NUM_ROOMS))

#build a map rooms_desc_map that maps a room description to the corresponding room index.
# A map quests_map that maps quest text to the quest index
def text_to_hidden_state_mapping():
    for i in range(NUM_ROOMS):
        room_name = rooms[i]
        for room_desc in rooms_desc[room_name]:
            rooms_desc_map[room_desc] = i

    for i in range(NUM_QUESTS):
        quest_text = quests[i]
        quests_map[quest_text] = i


def load_game_data():
    # each state:(room, quest), where "room" is a hidden state
    # observable state: (room description, quest)

    for room_name in rooms_valid_acts:

        room_index = rooms.index(room_name)
        valid_acts = rooms_valid_acts[room_name]
        valid_objs = rooms_valid_objs[room_name]
        transit = rooms_transit[room_name]

        for valid_index, act in enumerate(valid_acts):
            obj = valid_objs[valid_index]
            act_index = actions.index(act)
            obj_index = objects.index(obj)
            # valid commands: A(h,(a,o))=1 if (a,o) is valid for hidden state h.
            command_is_valid[room_index, act_index, obj_index] = 1;

            next_room_name = transit[valid_index]
            next_room_index = rooms.index(next_room_name)
            #deterministic transition
            transit_matrix[room_index, act_index, obj_index, next_room_index] = 1;

    text_to_hidden_state_mapping()


# take a step in the game
def step_game(current_room_desc, current_quest_desc, action_index, object_index):
    global STEP_COUNT
    STEP_COUNT = STEP_COUNT+1
    terminal = (STEP_COUNT >= MAX_STEPS)
    #print('Step=%d' %(STEP_COUNT))
    #print(terminal)

    # room_index: the hidden state.
    current_room_index = rooms_desc_map[current_room_desc]
    quest_index = quests_map[current_quest_desc]

    if (command_is_valid[current_room_index, action_index, object_index]==1):
        # quest has been finished
        if ((actions[action_index]==quest_actions[quest_index]) and (objects[object_index]==quest_objects[quest_index])):
            terminal = True
            reward = QUEST_REWARD

            if DEBUG:
                print('Finish quest: %s at Room %s with command %s %s' %(current_quest_desc, current_room_desc, actions[action_index],objects[object_index]))

        else:
            reward = DEFAULT_REWARD

        # probability distribution of next room.
        next_room_dist = transit_matrix[current_room_index, action_index, object_index, :]
        next_room_index = np.random.choice(NUM_ROOMS, p=next_room_dist)
        next_room_name = rooms[next_room_index]
        next_room_desc_index = np.random.randint(len(rooms_desc[next_room_name]))
        next_room_desc = rooms_desc[next_room_name][next_room_desc_index]
        #if DEBUG:
            #print('Reward: %1.3f' % (reward,))
            #print('Transit to Room %d:%s. %s' %(next_room_index, rooms[next_room_index],rooms_desc[next_room_name][next_room_desc_index]))

    else:
        # penalty for invalid command
        reward = DEFAULT_REWARD + JUNK_CMD_REWARD
        # state remains the same when invalid command executed
        next_room_desc = current_room_desc

        # if DEBUG:
        #     print('Invalid command!')
        #     print('Reward: %1.3f' % (reward,))
        #     print('Remain in Room %d:%s' %(next_room_index, rooms[next_room_index],))

    # quest remains the same during each episode
    next_quest_desc = current_quest_desc
    return (next_room_desc, next_quest_desc, reward, terminal)

# start a new game
def newGame():
    global STEP_COUNT
    STEP_COUNT = 0
    # random initial state: room_index + quest_index
    room_index = np.random.randint(NUM_ROOMS)
    room_name = rooms[room_index]
    room_desc_index = np.random.randint(len(rooms_desc[room_name]))
    room_desc = rooms_desc[room_name][room_desc_index]

    quest_index = np.random.randint(len(quests))
    quest_desc = quests[quest_index]

    terminal = False
    if DEBUG:
        print('Start a new game')
        print('Start Room %d: %s. %s' % (room_index, room_name, room_desc,))
        print('Start quest: %s' % (quest_desc,))

    return (room_desc, quest_desc, terminal)

def get_actions():
    return (actions)

def get_objects():
    return (objects)

def make_all_states_index():
    """
    Returns tow dictionaries:
    1: one for all unique room descriptions occur in the game
    2: one for all unique quests in the game
    """
    dictionary_room_desc = {}
    dictionary_quest_desc = {}
    for room in rooms_desc:
        for desc in rooms_desc[room]:
            if desc not in dictionary_room_desc:
                dictionary_room_desc[desc] = len(dictionary_room_desc)

    for quest in quests:
        if quest not in dictionary_quest_desc:
            dictionary_quest_desc[quest] = len(dictionary_quest_desc)

    return (dictionary_room_desc, dictionary_quest_desc)

# def gameOver(room_index, quest_index, action_index, object_index):
#     if (command_is_valid[room_index, action_index, object_index]==1):
#         # quest has been finished
#         if ((actions[action_index]==quest_actions[quest_index]) and (objects[object_index]==quest_objects[quest_index])):
#             return (True)

#     return (False)

# def output_state(room_index, room_desc_index, quest_index):
#     room_name = rooms[room_index]
#     #print('Room: %s. %s.' %(room_name, rooms_desc[room_name][room_desc_index]))
#     #print('Quest: %s' %(quests[quest_index]))
#     room_desc = rooms_desc[room_name][room_desc_index]
#     quest_desc = quests[quest_index]
#     return (room_desc, quest_desc)

# def output_command(action_index, object_index):
#     print('Command: %s %s' %(actions[action_index], objects[object_index]))


# load_game_data()
# reward_cnt = 0
# step = 0
# max_steps = 300
# game_count = 0

# (current_room_desc, current_quest_desc, terminal) = newGame()


# while step<max_steps:
#     step = step +1

#     # pure random policy
#     action_index = np.random.randint(NUM_ACTIONS)
#     object_index = np.random.randint(NUM_OBJECTS)

#     if DEBUG:
#         print('Step %d: %s %s with Command: %s %s' % (step, current_room_desc, current_quest_desc, actions[action_index], objects[object_index],))

#     (next_room_desc, next_quest_desc, reward, terminal) = step_game(current_room_desc, current_quest_desc, action_index, object_index)
#     reward_cnt = reward_cnt + reward

#     if terminal:
#         (current_room_desc, current_quest_desc, terminal) = newGame()
#         game_count = game_count + 1
#     else:
#         current_room_desc = next_room_desc
#         current_quest_desc = next_quest_desc


# print('Finish %d games. Total reward %6.3f.' % (game_count, reward_cnt,))
