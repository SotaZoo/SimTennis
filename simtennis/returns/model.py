"""Derive the reduced embodiment; preserve the shared ball/contact/court model."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


def model_xml(face_angle=.30):
    tree = ET.fromstring((ROOT / 'models/car_match.xml').read_text())
    tree.set('model', 'single return: planar force and swing torque, uncalibrated contacts')
    world = tree.find('worldbody')
    world.remove(world.find("body[@name='player1_chassis']"))
    boom = world.find("body[@name='chassis']/body[@name='boom']")
    boom.remove(boom.find("joint[@name='lift']"))
    grip = boom.find("body[@name='grip']")
    grip.remove(grip.find("joint[@name='tilt']"))
    grip.set('quat', f'{math.cos(face_angle/2)} {math.sin(face_angle/2)} 0 0')
    for node in list(tree.find('contact')):
        if node.get('geom2', '').startswith('player1_'):
            tree.find('contact').remove(node)
    for node in list(tree.find('actuator')):
        if node.get('name', '').startswith('player1_') or node.get('joint') in ('lift', 'tilt'):
            tree.find('actuator').remove(node)
    custom = tree.find('custom')
    custom.find("text[@name='controller_joints']").set('data', 'swing')
    for name, value in [('controller_kp', '180'), ('controller_kd', '12'), ('controller_rates', '16')]:
        custom.find(f"numeric[@name='{name}']").set('data', value)
    tree.find("keyframe/key[@name='ready']").set('qpos', '0 0 -.7 0 .1 17 1 0 0 0')
    return ET.tostring(tree, encoding='unicode')
