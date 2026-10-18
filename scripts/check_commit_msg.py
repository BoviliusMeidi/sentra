import sys
import re

def main():
    commit_msg_filepath = sys.argv[1]

    with open(commit_msg_filepath, 'r', encoding='utf-8') as file:
        commit_msg = file.readline().strip()

    pattern = r'^(feat|chore|fix|refactor)\([a-zA-Z0-9_-]+\): .+'

    if not re.match(pattern, commit_msg):
        print("\n[X] COMMIT REJECTED: Invalid SENTRA commit format!")
        print("-" * 55)
        print("Format  : type(scope): your commit message")
        print("Allowed : feat, chore, fix, refactor")
        print("-" * 55)
        print("Examples:")
        print("feat(ml-engine): add abstract base class")
        print("refactor(ui): update streamlit slider layout")
        print("fix(gee-masking): resolve qa60 bitmask error\n")
        sys.exit(1)

if __name__ == "__main__":
    main()