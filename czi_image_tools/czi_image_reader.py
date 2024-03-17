"""
czi_image_reader.py
Open a czi file and read the slides.
"""

# pylint: disable=invalid-name, line-too-long
# noqa: E501

# from .plotting_utils import plot_czi_slices

import os
import platform
import re

import cv2
import psutil
from pylibCZIrw import czi as pyczi

from .cell_detector import contains_cells
from .plotting_utils import plot_single_czi_image_with_legend


def clear() -> None:
    """
    Limpa a tela do terminal independente do sistema operacional
    """
    if platform.system() == "Windows":
        os.system(command="cls")
    else:
        os.system(command="clear")


def find_values_in_parentheses(input_string: str) -> list:
    """
    Find all values within parentheses in the given string.

    Parameters:
    - input_string (str): The input string containing values within parentheses.  # noqa: E501

    Returns:
    - list: A list of values found within parentheses.
    """
    pattern = r'\((.*?)\)'
    return re.findall(pattern, input_string)


def is_notebook() -> bool:
    """
    Verifica se o código está sendo executado em um notebook

    Returns
    -------
    bool
        True se estiver em um notebook, False caso contrário
    """
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return False
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except Exception:
        return False
    return True


def slice_czi_image_info(file_path, output_dim=(1600, 1200), plot=False) -> dict:  # noqa: E501
    """
    Slice a CZI image into smaller images

    Parameters
    ----------
    file_path : str
        Path to the CZI file
    output_dim : tuple
        Dimensions of the output images

    Returns
    -------
    dict
        Dictionary containing the slices infile_typeion
    """
    with pyczi.open_czi(filepath=file_path) as czidoc:
        # Dimensões da imagem e das cenas
        scenes_bounding_rectangle = czidoc.scenes_bounding_rectangle

        slices_info = {}
        scene_index = 0

        for scene, bounding_rect in scenes_bounding_rectangle.items():
            scene_slices = []
            x_start, y_start = bounding_rect.x, bounding_rect.y
            x_end, y_end = x_start + bounding_rect.w, y_start + bounding_rect.h

            num_slices_x = (x_end - x_start) // output_dim[0]
            num_slices_y = (y_end - y_start) // output_dim[1]

            for i in range(num_slices_x):
                for j in range(num_slices_y):
                    # Calculamos as coordenadas da ROI para cada slice
                    x = x_start + i * output_dim[0]
                    y = y_start + j * output_dim[1]
                    roi = (x, y, output_dim[0], output_dim[1])
                    scene_slices.append({'roi': roi, 'scene': scene})

            slices_info[f'scene_{scene_index}'] = scene_slices
            scene_index += 1

    return slices_info


def process_and_save_single_image_to_file_type(slices_info, file_path, slice_index, scene_index, output_dir, file_type, plot=True) -> None:  # noqa: E501
    """
    Process and save a single image

    Parameters
    ----------
    slices_info : dict
        Dictionary containing the slices infile_typeion
    file_path : str
        Path to the CZI file
    slice_index : int
        Index of the slice to be processed
    scene_index : int
        Index of the scene to be processed
    output_dir : str
        Output directory
    """
    current_slice = slices_info[f'scene_{scene_index}'][slice_index]
    roi = current_slice['roi']
    filename = os.path.basename(file_path).split('.')[0]
    new_filename = f'{filename}_scene_{scene_index}_slice_{slice_index+1}.{file_type}'  # noqa: E501
    with pyczi.open_czi(filepath=file_path) as czidoc:
        slice_img = czidoc.read(roi=roi, plane={'C': 0}, pixel_type='Bgr24')
    output_path = os.path.join(output_dir, new_filename)
    cv2.imwrite(filename=output_path,  # pylint: disable=no-member
                img=slice_img)
    # print(f'Processing: {new_filename}')
    if plot:
        plot_single_czi_image_with_legend(file_path=file_path, slice_index=slice_index, scene_index=scene_index, slices_info=slices_info)  # noqa: E501


def find_max_slice_size_to_memory(file_path, start_dim=(100, 100), step=100, max_mem_usage=80):  # noqa: E501
    """
    Find the maximum slice size that can be used without exceeding the maximum memory usage.  # noqa: E501

    Parameters
    ----------
    file_path : str
        Path to the CZI file.
    start_dim : tuple
        Initial dimensions of the output images.
    step : int
        Step to increase the dimensions.
    max_mem_usage : int
        Maximum memory usage percentage.

    Returns
    -------
    tuple
        Maximum slice size.
    """
    max_slice_size = start_dim
    mem = psutil.virtual_memory()
    total_mem = mem.total
    max_mem_bytes = (max_mem_usage / 100) * total_mem

    with pyczi.open_czi(filepath=file_path) as czidoc:
        scenes_bounding_rectangle = czidoc.scenes_bounding_rectangle

        for scene, bounding_rect in scenes_bounding_rectangle.items():
            width, height = bounding_rect.w, bounding_rect.h

            # Calculate the initial number of slices and memory usage
            num_slices_width = width // max_slice_size[0]
            num_slices_height = height // max_slice_size[1]
            num_slices = num_slices_width * num_slices_height
            mem_usage = num_slices * \
                max_slice_size[0] * max_slice_size[1] * 3  # 3 bytes per pixel

            # Adjust the slice size until the memory usage is within the limit
            while mem_usage > max_mem_bytes:
                max_slice_size = (max_slice_size[0] + step, max_slice_size[1] + step)  # noqa: E501
                num_slices_width = width // max_slice_size[0]
                num_slices_height = height // max_slice_size[1]
                num_slices = num_slices_width * num_slices_height
                mem_usage = num_slices * \
                    max_slice_size[0] * max_slice_size[1] * \
                    3  # 3 bytes per pixel

    return max_slice_size


def list_files_with_czi_in_name(folder_path: str) -> list:
    """
    List all files with .czi in the name in a folder.

    Parameters:
    - folder_path (str): The path to the folder.

    Returns:
    - list: List of files with .czi in the name.
    """
    return [os.path.join(folder_path, file) for file in os.listdir(path=folder_path) if '.czi' in file]  # noqa: E501


def check_last_index_processed(output_dir: str) -> int:
    """
    Check the last index processed.

    Parameters:
    - output_dir (str): The path to the output directory.

    Returns:
    - int: The last index processed.

    """
    files = os.listdir(output_dir)
    indices = [int(file.split('_')[-1].split('.')[0]) for file in files]
    return max(indices)


def detect_cell_in_czi_slice(file_path, file_name, slice_index, scene_index, slices_info, output='images_output', plot=True, file_type='png', threshold_ratio=0.1) -> None:  # noqa: E501
    """
    Detect cells in a single slice

    Parameters
    ----------
    file_path : str
        Path to the CZI file
    slice_index : int
        Index of the slice to be processed
    scene_index : int
        Index of the scene to be processed
    slices_info : dict
        Dictionary containing the slices infile_typeion
    plot : bool
        Whether to plot the results
    file_type : str
        Output file_type (e.g., 'png', 'jpg')
    create : bool
        Whether to create a separate folder for each czi file

    Returns
    -------
    None
    """
    current_slice = slices_info[f'scene_{scene_index}'][slice_index]
    roi = current_slice['roi']
    new_filename = f'{file_name}_scene_{scene_index}_slice_{slice_index+1}.{file_type}'  # noqa: E501

    output_path = os.path.join(output, new_filename)
    if not os.path.isfile(path=output_path):
        with pyczi.open_czi(filepath=file_path) as czidoc:
            slice_img = czidoc.read(
                roi=roi, plane={'C': 0}, pixel_type='Bgr24')
        cells = contains_cells(
            image=slice_img,
            display=False,
            threshold_ratio=threshold_ratio
        )
        # print(f'slice {slice_index+1} - {"contains cells" if cells else "does not contain cells"}')  # noqa: E501
        if plot and cells:
            plot_single_czi_image_with_legend(file_path=file_path, slice_index=slice_index, scene_index=scene_index, slices_info=slices_info)  # noqa: E501
        if cells:
            print('saving image')
            print(f'output_path: {output_path}')
            cv2.imwrite(filename=output_path,  # pylint: disable=no-member
                        img=slice_img)
    else:
        print(f'File {output_path} already exists')


def process_czi_folder(folder_path, output_dir, output_dim=(2000, 2000), file_type='png', thr=0.2) -> None:  # noqa: E501
    """
    Process all czi files in a folder.

    Parameters:
    - folder_path (str): Path to the folder containing the czi files.
    - output_dir (str): Path to the output directory.
    - output_dim (tuple): Dimensions of the output images (width, height).
    - file_type (str): Output file_type (e.g., 'png', 'jpg').
    - create_folder (bool): Whether to create a separate folder for each czi file.  # noqa: E501
    """
    czi_files = list_files_with_czi_in_name(folder_path=folder_path)

    create_folder = input('Create a folder for each patient? y/n: ')
    create_folder = string_input_to_boolean(string=create_folder)

    for czi_file in czi_files:
        print(f'Processing file: {czi_file}')
        slices_info = slice_czi_image_info(
            file_path=czi_file, output_dim=output_dim)

        number = find_values_in_parentheses(os.path.basename(czi_file))[0]
        file_name = os.path.basename(czi_file).split('.')[0]
        file_output_dir = os.path.join(
            output_dir, f'{file_name} ({number})') if create_folder else os.path.join(  # noqa: E501
            output_dir, f'{file_name}')
        if create_folder:
            if not os.path.isdir(file_output_dir):
                os.makedirs(file_output_dir, exist_ok=True)

        for scene, slices in slices_info.items():
            for slice_index, slice_info in enumerate(iterable=slices):
                detect_cell_in_czi_slice(
                    file_path=czi_file,
                    file_name=file_name,
                    slice_index=slice_index,
                    scene_index=int(scene.split('_')[-1]),
                    slices_info=slices_info,
                    output=file_output_dir,
                    plot=False,
                    file_type=file_type,
                    threshold_ratio=thr
                )


def string_input_to_boolean(string: str) -> bool:
    """
    Convert a string input to a boolean value.

    Parameters:
    - string (str): The input string.

    Returns:
    - bool: The boolean value.
    """
    if string.lower() in ['y', 'yes', 'true']:
        return True
    elif string.lower() in ['n', 'no', 'false']:
        return False
    else:
        raise ValueError(f'Invalid input: {string}')


if __name__ == "__main__":
    file_path = "/home/luis/Documents/Doutorado/01 - REVISÃO - CITOLOGIA/10 - DATASETS/FEULGEN/NEW/2019_12_09__09_49__0001.czi (193).czi"  # noqa: E501
    print(f'file size (GB): {os.path.getsize(filename=file_path) / 1e9}')
    max_slice_size = find_max_slice_size_to_memory(file_path=file_path)
    print(f"Maximum slice size: {max_slice_size}")
    # slices_info = slice_czi_image(file_path=file_path, output_dim=(4096, 4096))  # funciona  # noqa: E501

    # max = 152700
    x_dim = 1600
    y_dim = 1200

    print(f'x_dim: {x_dim}, y_dim: {y_dim} chosen manually')

    slices_info = slice_czi_image_info(file_path=file_path, output_dim=(x_dim, y_dim), plot=True)  # noqa: E501
    n_samples = sum([len(slices) for slices in slices_info.values()])
    print(f'Number of samples: {n_samples}')
    output = 'images_output'
    print(f'Output directory: {output}')
    print(f'{"Exists" if os.path.exists(path=output) else "Does not exist"}')
    quit()
    for sample_index in range(n_samples):
        detect_cell_in_czi_slice(file_path=file_path, slice_index=sample_index, scene_index=0, slices_info=slices_info, output=output, plot=True)  # noqa: E501
        if sample_index > 150:
            quit()
    # detect_cell_in_czi_slice(file_path=file_path, slice_index=0, scene_index=0, slices_info=slices_info, plot=False)  # noqa: E501
    # plot_single_czi_image_with_legend(file_path=file_path, slice_index=0, scene_index=0, slices_info=slices_info)  # noqa: E501

    # file_type = 'png'

    # process_and_save_single_image_to_file_type(
    #     slices_info=slices_info,
    #     file_path=file_path,
    #     slice_index=0,
    #     scene_index=0,
    #     output_dir='images_output/2019_12_09__09_49__0001 (193)',
    #     file_type=file_type,
    #     plot=False
    #     )
    # plot_czi_slices(file_path=file_path, slices_info=slices_info, scene_index=0, start_index=0)  # noqa: E501
