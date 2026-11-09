"""Two cars with planar force, swing and restored lift, sharing the native ball."""
import math
import xml.etree.ElementTree as ET

from simtennis.returns.model import ROOT


def model_xml(face_angle=.30):
    tree = ET.fromstring((ROOT / 'models/car_match.xml').read_text())
    tree.set('model', 'two-player reduced car tennis: uncalibrated contacts')
    for prefix in ('', 'player1_'):
        boom = tree.find(f".//body[@name='{prefix}boom']")
        grip = boom.find(f"body[@name='{prefix}grip']")
        grip.remove(grip.find(f"joint[@name='{prefix}tilt']"))
        grip.set('quat', f'{math.cos(face_angle/2)} {math.sin(face_angle/2)} 0 0')
        # Restored height motion can reach the floor. Let the same native
        # contact solver enforce racket-ground contact rather than pass through.
        for name in ('stringbed', 'racket_handle', 'racket_frame_top', 'racket_frame_bottom',
                     'racket_frame_tip', 'racket_frame_throat'):
            ET.SubElement(tree.find('contact'), 'pair', geom1=prefix+name, geom2='court',
                          condim='3', friction='.6 .6 .001 .0001 .0001',
                          solref='-80000 -70', solimp='.95 .95 .001')
    actuator = tree.find('actuator')
    for node in list(actuator):
        if node.get('joint') in ('tilt', 'player1_tilt'):
            actuator.remove(node)
    custom = tree.find('custom')
    custom.find("text[@name='controller_joints']").set('data', 'swing lift')
    for name, value in [('controller_kp', '180 180'), ('controller_kd', '12 12'), ('controller_rates', '16 16')]:
        custom.find(f"numeric[@name='{name}']").set('data', value)
    tree.find("keyframe/key[@name='ready']").set('qpos', '0 0 0 -.7 0 0 0 -.7 0 .1 17 1 0 0 0')
    return ET.tostring(tree, encoding='unicode')
