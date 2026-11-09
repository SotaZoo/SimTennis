extends Node3D
## A regulation-size mesh with a screen-size cue for distant court views.
## This node only reads the camera; its pose is supplied by the simulation.

const RADIUS := 0.033
const CUE_DIAMETER := 14.0
var _cue: Sprite3D


func _init() -> void:
	var sphere := SphereMesh.new()
	sphere.radius = RADIUS
	sphere.height = RADIUS * 2.0
	sphere.radial_segments = 24
	sphere.rings = 12
	var material := StandardMaterial3D.new()
	material.albedo_color = Color("d9ed47")
	material.roughness = 0.95
	var ball := MeshInstance3D.new()
	ball.mesh = sphere
	ball.material_override = material
	add_child(ball)
	# A soft cue is easier to follow than enlarging the physical ball. Keep
	# depth testing so it cannot reveal the ball through an opaque player/net.
	var image := Image.create(32, 32, false, Image.FORMAT_RGBA8)
	for y in range(32):
		for x in range(32):
			var radius := Vector2(x - 15.5, y - 15.5).length() / 15.5
			image.set_pixel(x, y, Color(0.86, 0.97, 0.25, 1.0 - smoothstep(0.5, 1.0, radius)))
	_cue = Sprite3D.new()
	_cue.texture = ImageTexture.create_from_image(image)
	_cue.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	_cue.fixed_size = true
	_cue.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(_cue)


func _process(_delta: float) -> void:
	var camera := get_viewport().get_camera_3d()
	if camera == null: return
	var depth := -(camera.global_transform.affine_inverse() * global_position).z
	_cue.visible = depth > camera.near
	if not _cue.visible: return
	var height := get_viewport().get_visible_rect().size.y
	var pixels_per_unit := height / (2.0 * tan(deg_to_rad(camera.fov) * 0.5))
	# Godot's fixed_size sprite is projected as if it were one metre away.
	_cue.pixel_size = CUE_DIAMETER / (32.0 * pixels_per_unit)
	var ball_diameter := 2.0 * RADIUS * pixels_per_unit / depth
	_cue.modulate.a = clampf(1.0 - ball_diameter / CUE_DIAMETER, 0.0, 1.0)
