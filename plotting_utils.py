"""
plotting_utils.py
Plotting utilities for CZI files

"""

import matplotlib.pyplot as plt
from pylibCZIrw import czi as pyczi


def plot_single_czi_image_with_legend(file_path, slice_index, scene_index, slices_info) -> None:  # noqa: E501
    """
    Plot a single image with a legend

    Parameters
    ----------
    file_path : str
        Path to the CZI file
    slice_index : int
        Index of the slice to be plotted
    scene_index : int
        Index of the scene to be plotted
    """
    with pyczi.open_czi(filepath=file_path) as czidoc:
        current_slice = slices_info[f'scene_{scene_index}'][slice_index]
        roi = current_slice['roi']
        slice_img = czidoc.read(roi=roi, plane={'C': 0}, pixel_type='Bgr24')
        plt.imshow(slice_img)
        plt.colorbar()
        plt.show()


def plot_czi_slices(file_path, slices_info, scene_index=0, start_index=0) -> None:  # noqa: E501
    """
    Plot the slices of a CZI file

    Parameters
    ----------
    file_path : str
        Path to the CZI file
    slices_info : dict
        Dictionary containing the slices information
    scene_index : int
        Index of the scene to be plotted
    start_index : int
        Index of the first slice to be plotted

    Returns
    -------
    None
    """
    with pyczi.open_czi(filepath=file_path) as czidoc:
        # Calculamos o número de linhas necessário para plotar 9 imagens em 3 colunas  # noqa: E501
        num_rows = min(3, len(slices_info[f'scene_{scene_index}']) // 3 + (1 if len(slices_info[f'scene_{scene_index}']) % 3 > 0 else 0))  # noqa: E501
        fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(20, 15))

        axes = axes.flatten()

        for i in range(9):
            slice_index = start_index + i
            if slice_index < len(slices_info[f'scene_{scene_index}']):
                slice_info = slices_info[f'scene_{scene_index}'][slice_index]
                roi = slice_info['roi']
                ch0_slice = czidoc.read(roi=roi, plane={'C': 0}, pixel_type='Bgr24')  # noqa: E501

                # Convertendo de BGR para RGB
                # ch0_slice_rgb = ch0_slice[..., ::-1]

                # axes[i].imshow(ch0_slice[..., 0])  # single channel
                axes[i].imshow(ch0_slice)  # BGR
                # axes[i].imshow(ch0_slice_rgb)  # RGB
                axes[i].set_title(f"Slice {slice_index} of scene {scene_index}")  # noqa: E501
            else:
                axes[i].axis('off')

        print(ch0_slice.shape)
        plt.tight_layout()
        plt.show()