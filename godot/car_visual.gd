extends Node3D
## Presentation for the explicit car mechanism. World poses come from physics;
## wheels are decorative because the base currently uses planar support.

var parts: Dictionary = {}
var paint: StandardMaterial3D


func build(court: Node3D, color: Color) -> void:
	var body = _part("chassis")
	paint = court._material("car_" + color.to_html(), color)
	paint.metallic = 0.25
	paint.roughness = 0.28
	var rubber: Material = court._material("rubber", Color("151b22"))
	var metal: Material = court._material("car_metal", Color("b4c4ce"))
	metal.metallic = 0.75
	metal.roughness = 0.32
	court._box(Vector3(0.72, 0.28, 0.96), Vector3.ZERO, paint, body)
	court._box(Vector3(0.55, 0.12, 0.54), Vector3(0, 0.18, 0.06), paint, body)
	court._box(Vector3(0.48, 0.05, 0.17), Vector3(0, 0.22, -0.10), rubber, body)
	for x in [-0.36, 0.36]:
		for z in [-0.30, 0.30]:
			court._rod(Vector3(x - 0.045, -0.10, z), Vector3(x + 0.045, -0.10, z), 0.14, rubber, body)
			court._rod(Vector3(x - 0.047, -0.10, z), Vector3(x + 0.047, -0.10, z), 0.07, metal, body)
	for x in [-0.23, 0.23]:
		court._box(Vector3(0.12, 0.06, 0.015), Vector3(x, 0.06, -0.487), court._material("headlight", Color("ffecc3"), true), body)
	court._rod(Vector3(0, 0.14, 0), Vector3(0, 0.65, 0), 0.04, metal, body)
	var boom = _part("boom")
	court._sphere(0.06, rubber, boom)
	court._rod(Vector3.ZERO, Vector3(0.45, 0, 0), 0.018, metal, boom)
	var grip = _part("grip")
	court._sphere(0.028, rubber, grip)
	var racket = _part("racket")
	court._racket(racket, "res://models/car_match.xml")


func update_poses(bodies: Dictionary) -> void:
	for name in parts:
		parts[name].transform = bodies[name]


func set_color(color: Color) -> void:
	paint.albedo_color = color


func _part(name: String) -> Node3D:
	var node := Node3D.new()
	node.name = name.capitalize()
	add_child(node)
	parts[name] = node
	return node
