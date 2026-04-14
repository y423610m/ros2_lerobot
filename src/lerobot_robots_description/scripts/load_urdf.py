import os
from ament_index_python.packages import get_package_share_directory
from urdf_parser_py.urdf import URDF


def main():
    urdf_path = os.path.join(
        get_package_share_directory('my_robot_description'),
        'urdf',
        'SO101',
        'so101_new_calib.urdf'
    )
    with open(urdf_path, 'r') as infp:
        robot_description_content = infp.read()

    model = URDF.from_xml_string(robot_description_content)

    import IPython; IPython.embed()

main()