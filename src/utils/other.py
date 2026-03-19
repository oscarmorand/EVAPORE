
def pretty_dict_print(d: dict, level: int = 1, indent: int = 4):
    if not isinstance(d, dict):
        print(f"{d}")
        return
    print("{")
    for key, value in d.items():
        print(f"{' ' * level * indent}{key}: ", end='')
        pretty_dict_print(value, level + 1, indent)
    print(f"{' ' * (level - 1) * indent}", end='')
    print("}")