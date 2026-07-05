import random
import numpy as np

# Warehouse targets:
# AISLE 06 2.0 17.0 0.05 
# AISLE 05 -3.0 19.0 0.05
# AISLE 04 -8.2 17.0 0.05
# AISLE 03 -13.0 17.0 0.05
# AISLE 02 -18.0 17.0 0.05
# AISLE 01 -23.0 17.0 0.05
# FORKLIFT 2.5 10.6 0.05
# DANGER -3.0 12.5 0.05
# FIRST AID KIT -26.0 1.1 0.05
# SAFETY SWITCH 0.5 30.0 0.05

def generate_hospital_character_commands(num_characters=3, 
                               num_commands=10, seed=42,
                               filename="hospital_character_command.txt"):
    random.seed(seed)
    hospital_targets = {
        "Sitting Area": (-5.31, -1.36, 0.01),
        "Counter": (1.75, 4.31, 0.01),
        "Hallway": (1.68, -0.78, 0.01),
        "Reception Room": (22.5, -1.01, 0.01),
        "Restroom": (12.5, 3.2, 0.01),
        "Room 1": (24.1, 20.0, 0.01),
        "Room 2": (14.73, 26.65, 0.01),
        "Room 3": (-43.84, 30.28, 0.01),
        "Room 4": (-44.5, 8.5, 0.01),
        "Room 5": (-7.27, 14.02, 0.01),
        "Storage Area": (-29.35, 21.64, 0.01),
        "Vending Machine": (-33.15, 3.12, 0.01)
    }

    actions = ["Idle", "LookAround", "GoTo"]
    commands = []

    for char_id in range(num_characters):
        if char_id == 0:
            character_name = "Character"      
        else:
            character_name = f"Character_{char_id:02d}" 

        for _ in range(num_commands):
            action = random.choice(actions)

            if action == "Idle":
                duration = round(random.uniform(1.0, 5.0), 2)
                commands.append(f"{character_name} Idle {duration}")

            elif action == "LookAround":
                duration = round(random.uniform(1.0, 5.0), 2)
                commands.append(f"{character_name} LookAround {duration}")

            elif action == "GoTo":
                target_name, coords = random.choice(list(hospital_targets.items()))
                x, y, z = coords
                commands.append(f"{character_name} GoTo {x:.2f} {y:.2f} {z:.2f} _")

    # Join into text
    commands_text = "\n".join(commands)

    # Save to file
    with open(filename, "w") as f:
        f.write(commands_text)

    print(f"Commands saved to {filename}")
    return commands_text


def generate_office_character_commands(num_characters=3, 
                             num_commands=10, seed=42,
                             filename="office_character_command.txt"):
    random.seed(seed)
    office_targets = {
        "Lobby": (2.05, 0.02, 0.01),
        "Front Desk": (-6.15, 0.08, 0.01),
        "Sofa Area": (1.12, -8.80, 0.01),
        "Elevator": (-13.83, -2.91, 0.01),
        "Conference Room (Small)": (-19.51, 8.45, 0.01),
        "Male Restroom": (-17.01, 22.01, 0.01),
        "Office Large": (-16.02, 43.03, 0.01),
        "Office Small": (2.47, 29.87, 0.01),
        "Office CEO": (-18.03, 54.02, 0.01),
        "Conference Room (CEO)": (-0.67, 53.02, 0.01),
        "Conference Room (Large)": (-0.75, 41.82, 0.01),
        "Video Conference Room": (0.52, 22.21, 0.01)
    }
    
    actions = ["Idle", "LookAround", "GoTo"]
    commands = []

    for char_id in range(num_characters):
        if char_id == 0:
            character_name = "Character"    
        else:
            character_name = f"Character_{char_id:02d}" 

        for _ in range(num_commands):
            action = random.choice(actions)

            if action == "Idle":
                duration = round(random.uniform(1.0, 5.0), 2)
                commands.append(f"{character_name} Idle {duration}")

            elif action == "LookAround":
                duration = round(random.uniform(1.0, 5.0), 2)
                commands.append(f"{character_name} LookAround {duration}")

            elif action == "GoTo":
                target_name, coords = random.choice(list(office_targets.items()))
                x, y, z = coords
                commands.append(f"{character_name} GoTo {x:.2f} {y:.2f} {z:.2f} _")

    # Join into text
    commands_text = "\n".join(commands)

    # Save to file
    with open(filename, "w") as f:
        f.write(commands_text)

    print(f"Commands saved to {filename}")
    return commands_text


def generate_outdoor_character_commands(num_characters=200, 
                              num_commands=5, seed=42,
                              filename="outdoor_character_command.txt"):
    random.seed(seed)
    outdoor_targets = {
        "Library": (-31.8, 180.0, 6.2),
        "Cafeteria": (-35.0, 135.0, 6.2),
        "Outdoor equipment": (70.5, 117.5, 6.2),
        "Sleevo Thrifty": (82.8, -35.0, 6.2),
        "Apartment": (160.0, 75.0, 6.2),
        "Parking Lot": (86.0, 152.0, 6.2),
        "Coffee": (47.5, 101.5, 6.2),
        "Fitness": (-51.0, 157.5, 5.7),
        "UPSWAY": (80.0, 100.0, 6.2),
        "THRIFTY Burn&Perez": (108.5, 50.8, 6.2),
        "DOMJAN ANTIQUES": (98.0, 68.0, 6.2),
    }
    
    actions = ["Idle", "LookAround", "GoTo"]
    commands = []

    for char_id in range(num_characters):
        if char_id == 0:
            character_name = "Character"    
        else:
            character_name = f"Character_{char_id:02d}" 

        for _ in range(num_commands):
            action = random.choice(actions)

            if action == "Idle":
                duration = round(random.uniform(1.0, 5.0), 2)
                commands.append(f"{character_name} Idle {duration}")

            elif action == "LookAround":
                duration = round(random.uniform(1.0, 5.0), 2)
                commands.append(f"{character_name} LookAround {duration}")

            elif action == "GoTo":
                target_name, coords = random.choice(list(outdoor_targets.items()))
                x, y, z = coords
                commands.append(f"{character_name} GoTo {x:.2f} {y:.2f} {z:.2f} _")

    # Join into text
    commands_text = "\n".join(commands)

    # Save to file
    with open(filename, "w") as f:
        f.write(commands_text)

    print(f"Commands saved to {filename}")
    return commands_text


if __name__ == "__main__":
    print(generate_hospital_character_commands())
    print(generate_office_character_commands())
    print(generate_outdoor_character_commands())