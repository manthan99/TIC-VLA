import random
from typing import Dict, Tuple, Union, Optional, Iterable, List

# -----------------------
# Target dictionaries
# -----------------------

WAREHOUSE_TARGETS: Dict[str, Tuple[float, float, float]] = {
    "AISLE 06": (2.0, 17.0, 0.05),
    "AISLE 05": (-3.0, 19.0, 0.05),
    "AISLE 04": (-8.2, 17.0, 0.05),
    "AISLE 03": (-13.0, 17.0, 0.05),
    "AISLE 02": (-18.0, 17.0, 0.05),
    "AISLE 01": (-23.0, 17.0, 0.05),
    "FORKLIFT": (2.7, 8.6, 0.05),
    "DANGER": (-3.0, 15.3, 0.05),
    "FIRST AID KIT": (-25.0, 1.4, 0.05),
    "SAFETY SWITCH": (0.5, 30.0, 0.05),
}

HOSPITAL_TARGETS: Dict[str, Tuple[float, float, float]] = {
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
    "Vending Machine": (-33.15, 3.12, 0.01),
}

OFFICE_TARGETS: Dict[str, Tuple[float, float, float]] = {
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
    "Video Conference Room": (0.52, 22.21, 0.01),
}

OUTDOOR_TARGETS: Dict[str, Tuple[float, float, float]] = {
    "Library": (-31.8, 180.0, 6.2),
    "Cafeteria": (-35.0, 135.0, 6.2),
    "Outdoor equipment": (70.5, 117.5, 6.2),
    "Sleevo Thrifty": (82.8, -35.0, 6.2),
    "Apartment": (160.0, 75.0, 6.2),
    #"Parking Lot": (86.0, 152.0, 6.2),
    "Coffee": (47.5, 101.5, 6.2),
    "Fitness": (-51.0, 157.5, 5.7),
    "UPSWAY": (80.0, 100.0, 6.2),
    "THRIFTY Burn&Perez": (108.5, 50.8, 6.2),
    "DOMJAN ANTIQUES": (98.0, 68.0, 6.2),
}

TARGET_SETS: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "warehouse": WAREHOUSE_TARGETS,
    "hospital": HOSPITAL_TARGETS,
    "office":   OFFICE_TARGETS,
    "outdoor":  OUTDOOR_TARGETS,
}

# -----------------------
# Unified generator
# -----------------------

def _resolve_targets(targets: Union[str, Dict[str, Tuple[float, float, float]]]) -> tuple[str, Dict[str, Tuple[float, float, float]]]:
    if isinstance(targets, str):
        key = targets.lower()
        if key not in TARGET_SETS:
            raise ValueError(f"Unknown targets key '{targets}'. Available: {', '.join(TARGET_SETS.keys())}")
        return key, TARGET_SETS[key]
    return "custom", targets

def generate_commands(
    targets: Union[str, Dict[str, Tuple[float, float, float]]],
    *,
    actor_kind: str = "character",                # "character" | "robot"
    actor_name: Optional[str] = None,             # default per kind
    num_actors: int = 3,                          # used for characters
    num_commands: int = 10,                       # per actor (characters) or total (robot)
    actions: Optional[Iterable[str]] = None,      # defaults per kind
    seed: Optional[int] = 42,
    filename: Optional[str] = None,
) -> str:
    """
    Generate commands for characters or a robot over a given target set.

    characters:
      - names: Character, Character_01, Character_02, ...
      - default actions: Idle, LookAround, GoTo
      - GoTo format: 'Character GoTo x y z _' (with trailing underscore)

    robot:
      - single actor (actor_name, e.g., 'Nova_Carter')
      - default actions: GoTo only
      - GoTo format: 'Nova_Carter GoTo x y z' (no trailing underscore)
    """
    key, targets_dict = _resolve_targets(targets)

    if seed is not None:
        random.seed(seed)

    # Defaults per kind
    if actor_kind not in ("character", "robot"):
        raise ValueError("actor_kind must be 'character' or 'robot'")

    if actor_kind == "character":
        if actor_name is None:
            actor_name = "Character"
        if actions is None:
            actions = ["Idle", "LookAround", "GoTo"]
        trailing_us = True
        # number of actors applies
        actor_names: List[str] = [actor_name] + [f"{actor_name}_{i:02d}" for i in range(1, num_actors)]
    else:  # robot
        if actor_name is None:
            actor_name = "Nova_Carter"
        if actions is None:
            actions = ["GoTo"]
        trailing_us = False
        # single actor only
        actor_names = [actor_name]

    if filename is None:
        filename = f"{key}_{actor_kind}_command.txt"

    commands: List[str] = []
    for name in actor_names:
        for _ in range(num_commands):
            action = random.choice(list(actions))
            if action in ("Idle", "LookAround"):
                # duration-based actions (characters by default)
                duration = round(random.uniform(1.0, 5.0), 2)
                commands.append(f"{name} {action} {duration}")
            elif action == "GoTo":
                _, (x, y, z) = random.choice(list(targets_dict.items()))
                if trailing_us:
                    commands.append(f"{name} GoTo {x:.2f} {y:.2f} {z:.2f} _")
                else:
                    commands.append(f"{name} GoTo {x:.2f} {y:.2f} {z:.2f}")
            else:
                raise ValueError(f"Unknown action '{action}'")

    commands_text = "\n".join(commands)

    with open(filename, "w") as f:
        f.write(commands_text)

    print(f"Commands saved to {filename}")
    return commands_text

# -----------------------
# Thin convenience wrappers
# -----------------------

def generate_character_commands(
    targets: Union[str, Dict[str, Tuple[float, float, float]]],
    *,
    num_characters: int = 3,
    num_commands: int = 10,
    seed: Optional[int] = 42,
    filename: Optional[str] = None,
    base_name: str = "Character",
) -> str:
    return generate_commands(
        targets,
        actor_kind="character",
        actor_name=base_name,
        num_actors=num_characters,
        num_commands=num_commands,
        seed=seed,
        filename=filename,
    )

def generate_robot_commands(
    targets: Union[str, Dict[str, Tuple[float, float, float]]],
    *,
    robot_type: str = "Nova_Carter",
    num_commands: int = 5,
    seed: Optional[int] = 42,
    filename: Optional[str] = None,
) -> str:
    return generate_commands(
        targets,
        actor_kind="robot",
        actor_name=robot_type,
        num_actors=1,
        num_commands=num_commands,
        seed=seed,
        filename=filename,
    )

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # Characters (same behavior/format as before; trailing '_' on GoTo)
    print(generate_character_commands("hospital", num_characters=3, num_commands=10, seed=42,
                                      filename="hospital_character_command.txt"))
    print(generate_character_commands("office", num_characters=3, num_commands=10, seed=42,
                                      filename="office_character_command.txt"))
    print(generate_character_commands("outdoor", num_characters=200, num_commands=5, seed=42,
                                      filename="outdoor_character_command.txt"))
    print(generate_character_commands("warehouse", num_characters=3, num_commands=10, seed=42,
                                      filename="warehouse_character_command.txt"))

    # Robots (GoTo only; no trailing '_')
    print(generate_robot_commands("hospital", robot_type="Nova_Carter", num_commands=5, seed=42,
                                  filename="hospital_robot_command.txt"))
    print(generate_robot_commands("office", robot_type="Nova_Carter", num_commands=5, seed=42,
                                  filename="office_robot_command.txt"))
    print(generate_robot_commands("outdoor", robot_type="Nova_Carter", num_commands=5, seed=42,
                                  filename="outdoor_robot_command.txt"))
    print(generate_robot_commands("warehouse", robot_type="Nova_Carter", num_commands=5, seed=42,
                                  filename="warehouse_robot_command.txt"))
