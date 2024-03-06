import os

from czi_image_tools.czi_image_reader import (detect_cell_in_czi_slice,
                                              find_max_slice_size_to_memory,
                                              process_czi_folder,
                                              slice_czi_image_info)

if __name__ == "__main__":
    file_path = "/home/luis/Documents/Doutorado/01 - REVISÃO - CITOLOGIA/10 - DATASETS/FEULGEN/NEW/2019_12_09__09_49__0001.czi (193).czi"  # noqa: E501
    input_folder = "/home/luis/Documents/Doutorado/01 - REVISÃO - CITOLOGIA/10 - DATASETS/FEULGEN/NEW"  # noqa: E501
    print(os.path.isdir(input_folder))
    output_folder = "/home/luis/Documents/GitHub/DoutoradoRepos/CZI_IMAGE_TOOLS/images_output"  # noqa: E501
    process_czi_folder(
        folder_path=input_folder,
        output_dir=output_folder,
        output_dim=(1600, 1200),
        file_type='png',
    )
    # print(f'file size (GB): {os.path.getsize(filename=file_path) / 1e9}')
    # max_slice_size = find_max_slice_size_to_memory(file_path=file_path)
    # print(f"Maximum slice size: {max_slice_size}")
    # # slices_info = slice_czi_image(file_path=file_path, output_dim=(4096, 4096))  # funciona  # noqa: E501

    # # max = 152700
    # x_dim = 2000
    # y_dim = 2000

    # print(f'x_dim: {x_dim}, y_dim: {y_dim} chosen manually')

    # slices_info = slice_czi_image_info(file_path=file_path, output_dim=(x_dim, y_dim), plot=True)  # noqa: E501
    # n_samples = sum([len(slices) for slices in slices_info.values()])
    # print(f'Number of samples: {n_samples}')
    # output = 'images_output'
    # print(f'Output directory: {output}')
    # print(f'{"Exists" if os.path.exists(path=output) else "Does not exist"}')
    # # quit()
    # for sample_index in range(n_samples):
    #     detect_cell_in_czi_slice(file_path=file_path, slice_index=sample_index, scene_index=0, slices_info=slices_info, output=output, plot=True)  # noqa: E501
    #     if sample_index > 150:
    #         quit()
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
