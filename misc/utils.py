import pandas as pd

def load_pokedex(pokedex_file="data/pokedex.csv"):
    df = pd.read_csv(pokedex_file, header=None, names=["number", "name"])
    # Create a mapping: name (lowercase) -> number
    return dict(zip(df["name"].str.strip().str.lower(), df["number"]))

def load_abilities(abilities_file="data/abilities.csv"):
    df = pd.read_csv(abilities_file, header=None, names=["number", "name"])
    # Remove spaces and lowercase for the key
    return dict(zip(df["name"].str.strip().str.lower().str.replace(" ", "", regex=False), df["number"]))

def load_moves(moves_file="data/moves.csv"):
    df = pd.read_csv(moves_file, header=None, names=["number", "name"])
    # Remove spaces and lowercase for the key
    return dict(zip(df["name"].str.strip().str.lower().str.replace(" ", "", regex=False), df["number"]))

def load_items(items_file="data/items.csv"):
    df = pd.read_csv(items_file, header=None, names=["number", "name"])
    # Remove spaces and lowercase for the key
    return dict(zip(df["name"].str.strip().str.lower().str.replace(" ", "", regex=False), df["number"]))

if __name__ == "__main__":
    # Load data once
    pokedex = load_pokedex()
    abilities = load_abilities()
    moves = load_moves()
    items = load_items()

    # Example usage
    pokemon_name = "Blaziken"
    ability_name = "Static"
    move_name = "Thunderbolt"
    item_name = "blackglasses"

    pokedex_number = pokedex.get(pokemon_name.lower())
    ability_number = abilities.get(ability_name.lower().replace(" ", ""))
    move_number = moves.get(move_name.lower().replace(" ", ""))
    item_number = items.get(item_name.lower().replace(" ", ""))

    print(f"Pokédex number of {pokemon_name}: {pokedex_number}")
    print(f"Ability number of {ability_name}: {ability_number}")
    print(f"Move number of {move_name}: {move_number}")
    print(f"Item number of {item_name}: {item_number}")

