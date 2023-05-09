from prep_data import prep_data
import sys

def main():
    prep_data()

if __name__ == "__main__":
    sys.exit(int(main() or 0))