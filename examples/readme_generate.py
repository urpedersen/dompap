import os


def extract_description(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('"""'):
                # Remove the triple quotes and return the description
                return line.strip('"""').strip()
    return None


def generate(directory):
    readme_content = "# List of Examples\n\n"
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.py') and filename != 'readme_generate.py':
            description = extract_description(os.path.join(directory, filename))
            if description:
                readme_content += f"* [{filename}]({filename}): {description}\n"

    print(readme_content, file=open('readme.md', 'w'))


if __name__ == '__main__':
    generate('./')
