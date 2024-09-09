

def parse_namespace(file_path):
    # Open and read the file
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Remove the 'Namespace(' prefix and the closing ')'
    content = content.replace('Namespace(', '').rstrip(')')
    
    # Split the content into key-value pairs
    pairs = content.split(', ')
    
    # Parse the key-value pairs and store them in a dictionary
    parsed_data = {}
    for pair in pairs:
        key, value = pair.split('=')
        # Attempt to evaluate the value as Python literal if possible
        try:
            parsed_data[key] = eval(value)
        except:
            parsed_data[key] = value
    
    return parsed_data