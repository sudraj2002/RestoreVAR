import os
import cv2

def split_and_save(root, write_dir, size=None):

    def make_dirs(path):
        os.makedirs(path, exist_ok=True)

    write_path = write_dir
    restored_root = os.path.join(write_path)

    make_dirs(write_path)
    make_dirs(restored_root)

    for dataset in sorted(os.listdir(root)):
        dataset_dir = os.path.join(root, dataset)

        if not os.path.isdir(dataset_dir):
            continue
        if dataset == write_dir:
            continue

        print(f"Splitting {dataset}")
        restored_dir = os.path.join(restored_root, dataset)

        make_dirs(restored_dir)

        for frame in sorted(os.listdir(dataset_dir)):
            frame_path = os.path.join(dataset_dir, frame)
            if not os.path.isfile(frame_path):
                continue

            frame_im = cv2.imread(frame_path)
            if frame_im is None:
                print(f"[WARN] Could not read image: {frame_path}")
                continue

            h, w = frame_im.shape[:2]

            cur_size = h if size is None else size

            restored = frame_im[:, cur_size:2 * cur_size, :]

            restored = cv2.resize(restored, (cur_size, cur_size))

            cv2.imwrite(os.path.join(restored_dir, frame), restored)

root = 'results_gen'
write_dir = 'results_gen_split'
split_and_save(root, write_dir, size=512)