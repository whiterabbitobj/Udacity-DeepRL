import os.path

files = [str(f) for f in os.listdir('.') if os.path.isfile(f) and os.path.splitext(f)[1] == '.pth']
files = sorted(files, key=lambda x: os.path.getmtime(x))
message = '\n'.join(["{}. {}".format(len(files)-i, file) for i, file in enumerate(files)]) + " (LATEST)\nPlease choose a saved Agent training file: "
save_file = input(message)
try:
    file_index = len(files) - int(save_file)
    if file_index < 0:
        raise Exception()
    save_file = files[len(files) - int(save_file)]
    print("\nProceeding with file: {}".format(save_file))
except:
    print("Invalid choice.")
