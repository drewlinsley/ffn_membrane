"""Project config file."""
import os
import numpy as np
from ops import py_utils3 as py_utils


class Config:
    """Config class with global project variables."""

    def __init__(self, **kwargs):
        """Global config file for normalization experiments."""
        self.project_directory = '/media/data_cifs/connectomics/'
        self.local_data_path = self.project_directory  # noqa '/media/data/connectomics/'
        self.log_dir = os.path.join(
            self.project_directory,
            'segmentation_logs')
        self.errors = 'error_logs'
        self.coord_path = os.path.join('db', 'coordinates.npy')
        self.synapse_coord_path = os.path.join('db', 'synapse_coordinates.npy')
        self.synapse_vols = os.path.join(
            self.project_directory,
            'synapse_vols_v2')
        self.tf_records = os.path.join(
            self.project_directory,
            'tf_records')
        self.berson_path = os.path.join(
            self.project_directory,
            'from_berson')

        # Segmentation properties
        # self.path_str = os.path.join(self.project_directory, 'mag1_images/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.raw')  # nopep8
        self.path_str = os.path.join(self.local_data_path, 'mag1/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.raw')  # nopep8
        self.nii_path_str = os.path.join(self.project_directory, 'mag1_segs/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.nii')  # nopep8
        self.nii_merge_path_str = os.path.join(self.project_directory, 'mag1_merge_segs/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.nii')  # nopep8
        self.mem_str = os.path.join(self.project_directory, 'mag1_membranes/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.raw')  # nopep8
        self.nii_mem_str = os.path.join(self.project_directory, 'mag1_membranes_nii/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.nii')  # nopep8
        self.merge_str = os.path.join(self.project_directory, 'merge_segs/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.npy')
        # self.ffn_ckpt = os.path.join(self.project_directory, 'ffn_ckpts/64_fov/feedback_hgru_v5_3l_notemp_f_v4_berson4x_w_inf_memb_r0/model.ckpt-225915')  # nopep8
        # self.ffn_ckpt = os.path.join(self.project_directory, 'ffn_ckpts/64_fov/ts_1/model.ckpt-629727')  # nopep8
        self.ffn_ckpt = os.path.join(self.project_directory, 'ffn_ckpts/64_fov/ts_1/model.ckpt-1632105')  # nopep8
        self.membrane_ckpt = os.path.join(self.project_directory, 'checkpoints/l3_fgru_constr_berson_0_berson_0_2019_02_16_22_32_22_290193/fixed_model_137000.ckpt-137000')  # nopep8
        self.ffn_formatted_output = os.path.join(self.project_directory, 'ding_segmentations/x%s/y%s/z%s/v%s/')  # noqa
        self.ffn_merge_formatted_output = os.path.join(self.project_directory, 'ding_segmentations_merge/x%s/y%s/z%s/v%s/')  # noqa
        self.test_segmentation_path = os.path.join(self.project_directory, 'datasets/berson_0.npz')  # noqa
        self.ffn_model = 'feedback_hgru_v5_3l_notemp_f_v5_ts_1'  # 2382.443995
        self.shape = np.array([128, 128, 128])  # Shape of an EM image volume

        # DB
        self.db_ssh_forward = False
        machine_name = os.uname()[1]
        if (machine_name != 'serrep7'):
            # Docker container or master p-node
            self.db_ssh_forward = True
        else:
            self.db_ssh_forward = False
        # self.db_ssh_forward = False

        # Create directories if they do not exist
        check_dirs = [
            self.tf_records,
            self.log_dir,
            self.errors
        ]
        [py_utils.make_dir(x) for x in check_dirs]

    def __getitem__(self, name):
        """Get item from class."""
        return getattr(self, name)

    def __contains__(self, name):
        """Check if class contains field."""
        return hasattr(self, name)

