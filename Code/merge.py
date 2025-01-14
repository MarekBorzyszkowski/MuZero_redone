import os
import sys
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

def extract_events(file_path, step_offset=0):
    ea = event_accumulator.EventAccumulator(file_path)
    ea.Reload()
    
    events = []
    for tag in ea.Tags()['scalars']:
        scalar_events = ea.Scalars(tag)
        for event in scalar_events:
            events.append((event.step + step_offset, tag, event.value))
    return events

def merge_tfevents(input_folders, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    writer = SummaryWriter(output_folder)
    step_offset = 0

    for folder in input_folders:
        for file_name in sorted(os.listdir(folder)):
            if file_name.startswith("events.out.tfevents"):
                file_path = os.path.join(folder, file_name)
                events = extract_events(file_path, step_offset)
                for step, tag, value in events:
                    writer.add_scalar(tag, value, step)
                step_offset += max(event[0] for event in events) + 1

    writer.close()
    print(f"Scalony plik zapisano w folderze: {output_folder}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge.py <input_folder1> <input_folder2> ... <output_folder>")
        sys.exit(1)

    *input_folders, output_folder = sys.argv[1:]
    merge_tfevents(input_folders, output_folder)
