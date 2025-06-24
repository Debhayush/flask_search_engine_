import os
import re
import csv

def extract_problem_statement(text):
    lines = text.splitlines()
    problem_lines = []

    for line in lines:
        if line.strip().lower().startswith("example"):
            break
        problem_lines.append(line.strip())

    return '\n'.join([line for line in problem_lines if line])

def get_problem_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

def create_problemdata_files(
    problems_folder='problems',
    titles_file='problemtitles.txt',
    output_folder='problemdata',
    csv_file='problemdata.csv'
):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read titles
    with open(titles_file, 'r', encoding='utf-8') as f:
        titles = [line.strip() for line in f.readlines() if line.strip()]

    # Get and sort problem files
    problem_files = [f for f in os.listdir(problems_folder) if f.startswith('problemtext')]
    problem_files.sort(key=get_problem_number)

    all_data = []

    for i, filename in enumerate(problem_files):
        number = get_problem_number(filename)
        input_path = os.path.join(problems_folder, filename)

        if os.path.isfile(input_path):
            try:
                with open(input_path, 'r', encoding='utf-8') as infile:
                    text = infile.read()
            except UnicodeDecodeError:
                with open(input_path, 'r', encoding='ISO-8859-1') as infile:
                    text = infile.read()

            title = titles[i] if i < len(titles) else f"Untitled Problem {number}"
            statement = extract_problem_statement(text)

            full_content = f"{title}\n\n{statement}"

            # Save individual text file
            output_txt = os.path.join(output_folder, f'problemdata{number}.txt')
            with open(output_txt, 'w', encoding='utf-8') as outfile:
                outfile.write(full_content)

            # Collect for single-column CSV
            all_data.append([full_content])

            print(f"✓ Created: problemdata{number}.txt")

    # Write CSV version with one column: 'Problem'
    with open(csv_file, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Problem'])  # Column header
        writer.writerows(all_data)

    print(f"\n✓ CSV file created: {csv_file}")

# Run the process
create_problemdata_files()
