import glob
import os
from typing import Final, List, Union
import warnings


AUDIO_EXTS: Final = [".wav", ".mp3"]
MEL_EXTS: Final = [".npy", ".pt"]
OUTPUT_EXTS: Final = [".wav", ".pt"]
INPUT_EXTS: Final = AUDIO_EXTS + MEL_EXTS


def device_type(device: Union[str, int]):
    if device == "cpu":
        return device

    assert device.isnumeric()
    return f"cuda:{device}"


def check_paths(input_filelist: List[str], output_filelist: List[str]):
    """Check errors and return filelist if no error detected.

    This methode checks 3 types of errors:
        (1) Extension
            Check if filelists have proper extensions. The allowed extensions
             are defined as constants.
        (2) Lengths
            Check input and output filelist have same length
        (3) Undefined behaviours
            If there is 'mel -> mel' match in input and output filelists, it
             raises error.
    """

    def _check_extension(filelist: List[str], key: str):
        """Check if filelist"""
        allowed_exts = INPUT_EXTS if key == "input" else OUTPUT_EXTS

        if any([os.path.splitext(x)[1] not in allowed_exts for x in filelist]):
            raise ValueError(
                f"Unsupported format for `{key}_files`. Only {allowed_exts} are"
                " supported."
            )

    def _check_lengths():
        if len(input_filelist) != len(output_filelist):
            raise ValueError(
                f"Mistmatched input / output length. Input length:"
                f" {len(input_filelist)}, Output length: {len(output_filelist)}"
            )

    def _check_mel_to_mel():
        for in_file, out_file in zip(input_filelist, output_filelist):
            _, in_ext = os.path.splitext(in_file)
            _, out_ext = os.path.splitext(out_file)
            if in_ext == out_ext in MEL_EXTS:
                raise ValueError(
                    "Unsupported behaviour. Behaviour for melspectrogram to"
                    f" melspectrogram is not defined. got '-i {in_file} -o "
                    f"{out_file}'."
                )

    _check_extension(input_filelist, key="input")
    _check_extension(output_filelist, key="output")
    _check_lengths()
    _check_mel_to_mel()
    return input_filelist, output_filelist


def preprocess_inout_files(input_files: List[str], output_files: List[str]):
    """Process input and output filelists.

    For various types of input / output path arguments, it preprocess paths. For
     input paths, it reads text files containing audio file paths or search
     directories using `glob.glob`. For output paths, it automatically generates
     output file paths corresponding to input file paths.

    For more informations or examples, please refer `tests/testcases_path.yaml`.

    Params:
        input_files: file paths for inputs.
        output_files: file paths for outputs.

    Notes:
        `input_files` and `output_files` support various formats. However only
         naive list of paths is supported for multiple item, i.e, multiple items
         for directory path, text file or paths including wild card will result
         in unexpected outcome.

    Supported formats for arguments include:
        single or multiple audio / mel file paths:
            - ["foo/bar/input2.wav"]
            - ["foo/bar/input1.wav", "bar/input2.pt", "foo/input3.mp3"]
        (single) text file path
            - ["foo/bar/input.txt"]
        (single) directory
            - ["foo/"]
            - ["bar/"]
        (single) file or directory path with wild card
            - ["foo/*.wav"]
            - ["foo/**/*]
            - ["bar/**/input*.wav"]

    """

    def _common_process(filelist, key):
        # Only "input" and "output" are allowed for key
        assert key in ["input", "output"], f"Unknown key: {key}"

        file_base, file_ext = os.path.splitext(filelist[0])

        if file_ext == ".txt":
            return [line.strip() for line in open(filelist[0])]
        elif "*" in file_base:
            return filelist[0]
        elif file_ext == "":
            # Directory path case
            return os.path.join(file_base, "**/*")

        return filelist

    def _add_file(input_filelist: List[str], output_filelist: List[str], new_item):
        """Check file name confliction and add file path to the list.

        If `new_item` exists in `input_filelist` or `output_filelist`, rename it
         by add extension (e.g., ".wav") of input file name right before its
         real extension.
        """
        while new_item in input_filelist + output_filelist:
            new_item_base, new_item_ext = os.path.splitext(new_item)
            _, input_ext = os.path.splitext(input_filelist[len(output_filelist)])
            new_item = new_item_base + input_ext + new_item_ext
        output_filelist.append(new_item)

    # Read text files or add wild card to the path prpoperly
    input_files = _common_process(input_files, "input")
    output_files = _common_process(output_files, "output")

    if len(input_files) == 0:
        # If `input_files` is empty, make a warning and exit immediately
        warnings.warn("No inputs.", UserWarning)
        return [], []
    elif "*" not in input_files and "*" not in output_files:
        # If there are no wild card in both lists, return them
        return check_paths(input_files, output_files)

    # If there is wild card in `input_files`, `output_files` should have it too
    assert "*" in output_files

    # Search `input_files` using `glob.glob`
    if "*" in input_files:
        in_base, in_ext = os.path.splitext(input_files)

        input_files = []
        for ext in INPUT_EXTS if in_ext in ["", ".*"] else [in_ext]:
            input_files.extend(glob.glob(f"{in_base}{ext}", recursive=True))

    # Aliases for further processing
    input_root = os.path.commonpath([os.path.split(f)[0] for f in input_files])
    input_prefix = os.path.commonprefix([os.path.split(f)[1] for f in input_files])

    keep_subdirs = "**" in output_files
    output_root, output_tail = (x.split("*", 1)[0] for x in os.path.split(output_files))

    _, output_ext = os.path.splitext(output_files)
    if not output_ext:
        output_ext = ".wav"

    output_files = []
    for in_file in input_files:
        subdir, filename = os.path.split(in_file)
        subdir = subdir.removeprefix(input_root).strip("/")

        # Drop sub-directory path if needed
        if not keep_subdirs:
            subdir = ""

        # If output files have certain naming convention
        # e.g.) "foo/bar/output*.wav"
        if output_tail:
            filename = output_tail + filename.removeprefix(input_prefix)

        file_base, _ = os.path.splitext(filename)
        filepath = os.path.join(output_root, subdir, f"{file_base}{output_ext}")

        # Check if `filepath` conflicts and add it to `output_files`
        _add_file(input_files, output_files, filepath)

    return check_paths(input_files, output_files)
