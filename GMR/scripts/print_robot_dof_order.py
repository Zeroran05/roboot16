import argparse
import pathlib
import sys
import xml.etree.ElementTree as ET

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting.params import ROBOT_XML_DICT


def get_dof_names_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError(f"No <actuator> section found in {xml_path}")

    dof_names = []
    for elem in actuator:
        joint_name = elem.attrib.get("joint")
        if joint_name:
            dof_names.append(joint_name)

    if not dof_names:
        raise ValueError(f"No actuated joints found in {xml_path}")
    return dof_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", required=True, choices=sorted(ROBOT_XML_DICT.keys()))
    args = parser.parse_args()

    xml_path = pathlib.Path(ROBOT_XML_DICT[args.robot])
    dof_names = get_dof_names_from_xml(xml_path)

    print(f"robot: {args.robot}")
    print(f"xml: {xml_path}")
    print(f"num_dofs: {len(dof_names)}")
    for idx, name in enumerate(dof_names):
        print(f"{idx}: {name}")


if __name__ == "__main__":
    main()
