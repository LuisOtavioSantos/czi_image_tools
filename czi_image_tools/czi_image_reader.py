"""
czi_image_reader.py
Open a czi file and read the slides.
"""

# pylint: disable=invalid-name, line-too-long
# noqa: E501

# from .plotting_utils import plot_czi_slices

import glob
import os

import cv2
import psutil
from cell_detector import contains_cells
from plotting_utils import plot_single_czi_image_with_legend
from pylibCZIrw import czi as pyczi


def slice_czi_image_info(file_path, output_dim=(1200, 1600), plot=False) -> dict:  # noqa: E501
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
        Dictionary containing the slices information
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


def process_and_save_single_image_to_format(slices_info, file_path, slice_index, scene_index, output_dir, format, plot=True) -> None:  # noqa: E501
    """
    Process and save a single image

    Parameters
    ----------
    slices_info : dict
        Dictionary containing the slices information
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
    new_filename = f'{filename}_scene_{scene_index}_slice_{slice_index+1}.{format}'  # noqa: E501
    with pyczi.open_czi(filepath=file_path) as czidoc:
        slice_img = czidoc.read(roi=roi, plane={'C': 0}, pixel_type='Bgr24')
    output_path = os.path.join(output_dir, new_filename)
    cv2.imwrite(filename=output_path, img=slice_img)
    print(f'Processing: {new_filename}')
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


def list_files_by_extension(directory: str, extension: str) -> tuple:
    """
    List all files with a given extension in a directory.

    Parameters:
    - directory (str): The directory path.
    - extension (str): The file extension to look for.

    Returns:
    - tuple: A tuple containing the list of files and the count of files.
    """
    if not os.path.exists(path=directory):
        raise ValueError(f"The directory {directory} does not exist.")

    files_list = glob.glob(pathname=os.path.join(directory, "*." + extension))
    files_list.sort()

    file_count = len(files_list)

    return files_list, file_count


def detect_cell_in_czi_slice(file_path, slice_index, scene_index, slices_info, output='images_output', plot=True, format='png', create=True) -> None:  # noqa: E501
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
        Dictionary containing the slices information
    plot : bool
        Whether to plot the results
    format : str
        Output format (e.g., 'png', 'jpg')
    create : bool
        Whether to create a separate folder for each czi file

    Returns
    -------
    list
        List containing the detected cells
    """
    current_slice = slices_info[f'scene_{scene_index}'][slice_index]
    roi = current_slice['roi']
    with pyczi.open_czi(filepath=file_path) as czidoc:
        slice_img = czidoc.read(roi=roi, plane={'C': 0}, pixel_type='Bgr24')
    cells = contains_cells(image=slice_img, display=False)
    print(f'{"contains cells" if cells else "does not contain cells"}')  # noqa: E501
    if plot and cells:
        plot_single_czi_image_with_legend(file_path=file_path, slice_index=slice_index, scene_index=scene_index, slices_info=slices_info)  # noqa: E501
    if cells:
        print('saving image')
        file_name = os.path.basename(file_path).split('.')[0]
        new_filename = f'{file_name}_scene_{scene_index}_slice_{slice_index+1}.{format}'  # noqa: E501
        if create:
            output_folder = os.path.join(
                output, new_filename.replace('.png', ''))
            output_path = os.path.join(output_folder, new_filename)
            create_folder_if_not_exists(folder_path=output_folder)
        else:
            output_path = os.path.join(output, new_filename)
        print(f'output_path: {output_path}')
        cv2.imwrite(filename=output_path, img=slice_img)

    return cells


def create_folder_if_not_exists(folder_path: str) -> None:
    """
    Create a folder if it does not exist.

    Parameters:
    - folder_path (str): The path to the folder.

    Returns:
    - None
    """
    if not os.path.exists(path=folder_path):
        os.makedirs(name=folder_path)


def process_czi_folder(folder_path, output_dir, output_dim=(2000, 2000), format='png', create_folder=True) -> None:  # noqa: E501
    """
    Process all czi files in a folder.

    Parameters:
    - folder_path (str): Path to the folder containing the czi files.
    - output_dir (str): Path to the output directory.
    - output_dim (tuple): Dimensions of the output images (width, height).
    - format (str): Output format (e.g., 'png', 'jpg').
    - create_folder (bool): Whether to create a separate folder for each czi file.  # noqa: E501
    """
    czi_files, _ = list_files_by_extension(
        directory=folder_path, extension='czi')

    create = input(__prompt='Create a folder for each patient? y/n: ')
    create = string_input_to_boolean(string=create)

    for czi_file in czi_files:
        print(f'Processing file: {czi_file}')
        slices_info = slice_czi_image_info(
            file_path=czi_file, output_dim=output_dim)

        file_name = os.path.basename(czi_file).split('.')[0]
        file_output_dir = os.path.join(
            output_dir, file_name) if create_folder else output_dir

        if create_folder:
            os.makedirs(file_output_dir, exist_ok=True)

        for scene, slices in slices_info.items():
            for slice_index, slice_info in enumerate(iterable=slices):
                detect_cell_in_czi_slice(
                    file_path=czi_file,
                    slice_index=slice_index,
                    scene_index=int(scene.split('_')[-1]),
                    slices_info=slices_info,
                    output=file_output_dir,
                    format=format,
                    create=create
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
    x_dim = 2000
    y_dim = 2000

    print(f'x_dim: {x_dim}, y_dim: {y_dim} chosen manually')

    slices_info = slice_czi_image_info(file_path=file_path, output_dim=(x_dim, y_dim), plot=True)  # noqa: E501
    n_samples = sum([len(slices) for slices in slices_info.values()])
    print(f'Number of samples: {n_samples}')
    output = 'images_output'
    print(f'Output directory: {output}')
    print(f'{"Exists" if os.path.exists(path=output) else "Does not exist"}')
    # quit()
    for sample_index in range(n_samples):
        detect_cell_in_czi_slice(file_path=file_path, slice_index=sample_index, scene_index=0, slices_info=slices_info, output=output, plot=True)  # noqa: E501
    # detect_cell_in_czi_slice(file_path=file_path, slice_index=0, scene_index=0, slices_info=slices_info, plot=False)  # noqa: E501
    # plot_single_czi_image_with_legend(file_path=file_path, slice_index=0, scene_index=0, slices_info=slices_info)  # noqa: E501

    # format = 'png'

    # process_and_save_single_image_to_format(
    #     slices_info=slices_info,
    #     file_path=file_path,
    #     slice_index=0,
    #     scene_index=0,
    #     output_dir='images_output/2019_12_09__09_49__0001 (193)',
    #     format=format,
    #     plot=False
    #     )
    # plot_czi_slices(file_path=file_path, slices_info=slices_info, scene_index=0, start_index=0)  # noqa: E501
