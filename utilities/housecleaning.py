import os
import shutil
import pathlib

####### NEEDS WORK !!!!!!!!!!! #########
# ACCESS DENIED when trying to delete folders, empty or otherwise

def clean_directory(dir_path):
    """Clean all files and subdirectories in the given directory"""
    if os.path.exists(dir_path):
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            try:
                if os.path.isdir(item_path):
                    for sub_item in os.listdir(item_path):
                        sub_item_path = os.path.join(item_path, sub_item)
                        os.remove(sub_item_path)
                    pathlib.Path.rmdir(item_path) 
                else:
                    os.remove(item_path)
            except PermissionError:
                print(f"Permission denied: {item_path}")
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")
        print(f"Cleaned {dir_path}")
    else:
        print(f"Directory {dir_path} does not exist")

def main():
    # Directories to clean
    directories = [
        "../raw_llm_responses",
        "../testing/test_results",
        "../recovery"
    ]
    
    print("Housecleaning Utility")
    print("====================")
    print("This will delete ALL files and subdirectories in the following folders:")
    for dir_path in directories:
        print(f"- {os.path.abspath(dir_path)}")
    
    confirmation = input("\nDo you want to proceed? (yes/no): ").lower()
    
    if confirmation in ["yes", "y"]:
        for dir_path in directories:
            clean_directory(dir_path)
        print("\nHousecleaning completed successfully!")
    else:
        print("\nOperation cancelled.")

if __name__ == "__main__":
    main()