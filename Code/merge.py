import os
import sys
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator, tag_types

SIZE_GUIDANCE = {
    tag_types.COMPRESSED_HISTOGRAMS: 500,
    tag_types.IMAGES: 4,
    tag_types.AUDIO: 4,
    tag_types.SCALARS: 0,
    tag_types.HISTOGRAMS: 1,
    tag_types.TENSORS: 10,
}


def extract_events(file_path, step_offset=0):
    ea = event_accumulator.EventAccumulator(file_path, size_guidance=SIZE_GUIDANCE)
    ea.Reload()

    events = []
    number_of_events = 0
    for tag in ea.Tags()['scalars']:
        scalar_events = ea.Scalars(tag)
        number_of_events = len(scalar_events)
        for event in scalar_events:
            events.append((tag, event.step + step_offset, event.value, event.wall_time))
    return events, number_of_events


def merge_tfevents(input_folders, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    writer = SummaryWriter(output_folder)
    step_offset = 0

    for folder in input_folders:
        for file_name in sorted(os.listdir(folder)):
            if file_name.startswith("events.out.tfevents"):
                file_path = os.path.join(folder, file_name)
                print(f"Processing {file_path}")
                events, number_of_events = extract_events(file_path, step_offset)
                print(f"Extracted {number_of_events} events")
                for tag, step, value, walltime in events:
                    writer.add_scalar(tag, value, step, walltime=walltime)
                print(f"Written {number_of_events} events")
                step_offset += number_of_events

    writer.close()
    print(f"Scalony plik zapisano w folderze: {output_folder}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge.py <input_folder1> <input_folder2> ... <output_folder>")
        sys.exit(1)

    *input_folders, output_folder = sys.argv[1:]
    merge_tfevents(input_folders, output_folder)
