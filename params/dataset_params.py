FFmpeg = "cache/ffmpeg/"
FFmpeg_old = "cache/ffmpeg_old/"
Qemu = "cache/qemu/"
Qemu_old = "cache/qemu_old/"

REVEAL = "cache/reveal_data2/"
REVEAL3 = "cache/reveal_data3/"
LIBXML2 = "cache/libxml2/"

PATCHDB_DIR = "cache/patchdb"

W2V_PARAMS = {
    "type": "w2v",
    "vector_size": 100,
    "window": 3,
    "training_files": [
        (FFmpeg_old, True),
        (Qemu, False),
        (REVEAL, False),
        (LIBXML2, False),
        (PATCHDB_DIR, True)
    ],
    "cache_dir": "W2V_main"
}

FASTTEXT_PARAMS = {
    "type": "fastText",
    "vector_size": 100,
    "window": 3,
    "training_files": [
        (FFmpeg_old, True),
        (Qemu, False),
        (REVEAL, False),
        (LIBXML2, False),
        (PATCHDB_DIR, True)
    ],
    "cache_dir": "fastText_main"
}

TRANSFORMER_PARAMS = {
    "type": "transformer",
    "vector_size": 128,
    "model_file": "cache/cpg_reconstruction/models/model_9.chkpt",
    "training_files": [
        (FFmpeg_old, True),
        (Qemu, False),
        (REVEAL, False),
        (LIBXML2, False),
        (PATCHDB_DIR, True)
    ]
}

QEMU_FASTTEXT = {
    "dataset_dir": Qemu_old,
    "cache_dir": "QEMU_FASTTEXT",
    "encoding_params": FASTTEXT_PARAMS,
    "c_in_dot": True,
    "name": "QEMU"
}

FFMPEG_PARAMS = {
    "dataset_dir": FFmpeg_old,
    "cache_dir": "FFMPEG",
    "c_in_dot": True,
    "encoding_params": W2V_PARAMS,
    "name": "FFMPEG"
}

QEMU_PARAMS = {
    "dataset_dir": Qemu_old,
    "cache_dir": "QEMU",
    "encoding_params": W2V_PARAMS,
    "c_in_dot": True,
    "name": "QEMU"
}

LIBXML2_PARAMS = {
    "dataset_dir": LIBXML2,
    "cache_dir": "LIBXML2",
    "cpg_file_filter": "*.cpg",
    "overwrite_cache": False,
    "encoding": "w2v",
    "encoding_params": W2V_PARAMS,
    "downsample": True,
    "split": "random",
    "name": "LIBXML2"
}

REVEAL_DATASET_PARAMS = {
    "dataset_dir": REVEAL,
    "cache_dir": "CHROMIUM",
    "c_in_dot": False,
    "encoding_params": W2V_PARAMS,
    "name": "CHROMIUM"
}

REVEAL_TRANSFORMER = {
    "dataset_dir": REVEAL,
    "cache_dir": "REVEAL_TRANSFORMER",
    "c_in_dot": False,
    "encoding_params": TRANSFORMER_PARAMS,
    "max_seq_len": 4096,
    "name": "REVEAL_TRANSFORMER"
}

REVEAL_TRANSFORMER_DOWNSAMPLED = {
    "type": "Downsampled",
    "dataset": REVEAL_TRANSFORMER,
    "name": "REVEAL_TRANSFORMER_DOWNSAMPLED"
}

PATCHDB_PARAMS = {
    "dataset_dir": "cache/patchdb/",
    "cache_dir": "PATCHDB",
    "c_in_dot": True,
    "encoding_params": W2V_PARAMS,
    "name": "PATCHDB"
}

REVEAL3_PARAMS = {
    "dataset_dir": REVEAL3,
    "cache_dir": "REVEAL3",
    "c_in_dot": True,
    "encoding_params": W2V_PARAMS,
    "name": "REVEAL3"
}

REVEAL3_TRANSFORMER = {
    "dataset_dir": REVEAL3_PARAMS["dataset_dir"],
    "cache_dir": "REVEAL3_TRANSFORMER",
    "c_in_dot": True,
    "encoding_params": TRANSFORMER_PARAMS,
    "max_seq_len": 4096,
    "name": "REVEAL3_TRANSFORMER"
}

REVEAL3_NODEDROP = {
    "type": "NodeDrop",
    "dataset": REVEAL3_PARAMS,
    "name": "REVEAL3_NODEDROP",
}

REVEAL3_EDGEDROP = {
    "type": "EdgeDrop",
    "dataset": REVEAL3_PARAMS,
    "name": "REVEAL3_EDGEDROP",
}

SARD_PARAMS = {
    "dataset_dir": "cache/sard/",
    "cache_dir": "SARD",
    "c_in_dot": True,
    "encoding_params": W2V_PARAMS,
    "name": "SARD"
}

REVEAL3_FILLUP_SARD = {
    "type": "Fillup",
    "dataset": REVEAL3_PARAMS,
    "fill_from": SARD_PARAMS,
    "name": "REVEAL3_FILLED_SARD",
}

PATCHDB_TRANSFORMER = {
    "dataset_dir": "cache/patchdb/",
    "cache_dir": "PATCHDB_TRANSFORMER",
    "c_in_dot": True,
    "encoding_params": TRANSFORMER_PARAMS,
    "name": "PATCHDB_TRANSFORMER"
}

QEMU_FFMPEG_PARAMS = {
    "type": "Combined",
    "datasets": [QEMU_PARAMS, FFMPEG_PARAMS],
    "name": "QEMU_FFMPEG"
}

REVEAL_DEVIGN_PARAMS = {
    "type": "Combined",
    "datasets": [REVEAL3_PARAMS, QEMU_PARAMS, FFMPEG_PARAMS],
    "name": "REVEAL_DEVIGN"
}

QEMU_FFMPEG_DOWNSAMPLED = {
    "type": "Downsampled",
    "dataset": QEMU_FFMPEG_PARAMS,
    "name": "QEMU_FFMPEG_DOWNSAMPLED"
}

FFMPEG_DOWNSAMPLED = {
    "type": "Downsampled",
    "dataset": FFMPEG_PARAMS,
    "name": "FFMPEG_DOWNSAMPLED"
}

REVEAL_DOWNSAMPLED = {
    "type": "Downsampled",
    "dataset": REVEAL_DATASET_PARAMS,
    "name": "REVEAL_DOWNSAMPLED"
}

PATCHDB_DOWNSAMPLED = {
    "type": "Downsampled",
    "dataset": PATCHDB_PARAMS,
    "name": "PATCHDB_DOWNSAMPLED"
}

REVEAL3_DOWNSAMPLED = {
    "type": "Downsampled",
    "dataset": REVEAL3_TRANSFORMER,
    "name": "REVEAL3_DOWNSAMPLED"
}

QEMU_FFMPEG_FILLUP_SARD = {
    "type": "Fillup",
    "dataset": QEMU_FFMPEG_PARAMS,
    "fill_from": SARD_PARAMS,
    "name": "QEMU_FFMPEG_FILLED_SARD",
}

QEMU_FFMPEG_NODEDROP = {
    "type": "NodeDrop",
    "dataset": QEMU_FFMPEG_PARAMS,
    "name": "QEMU_FFMPEG_NODEDROP",
}

PATCHDB_NODEDROP = {
    "type": "NodeDrop",
    "dataset": PATCHDB_PARAMS,
    "name": "PATCHDB_NODEDROP"
}

PATCHDB_FILLUP_SARD = {
    "type": "Fillup",
    "dataset": PATCHDB_PARAMS,
    "fill_from": SARD_PARAMS,
    "name": "PATCHDB_FILLED_SARD",
}