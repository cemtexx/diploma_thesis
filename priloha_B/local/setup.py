import cx_Freeze

executables = [cx_Freeze.Executable("main.py")]

cx_Freeze.setup(
    name='ScalesCounter',
    options={'build_exe': {'packages': [
        'cv2',
        'tqdm',
        'yaml',
        'numpy',
        'psd_tools',
        'rawpy',
        'imageio',
        'skimage',
        'tensorflow'
    ],
        'include_files': [
            'scales_counter.py',
            'yolov4-custom.cfg',
            'README.md',
            'config.yaml',
            'models/unet_model_one.h5',
            'models/unet_model_two.h5',
            'models/unet_model_three.h5',
            'models/yolov4_model_name.weights'
    ]}},
    executables=executables

)

# just type into a conslole 'python setup.py build' in a main directory
# and make new package with executable file
