extends Node3D
## Presentation only. Every moving collider is drawn from an owned native pose.

const CHARACTER_PATH := "res://assets/character/tennis_player.glb"
const CarVisual = preload("res://car_visual.gd")
const PlayerAppearance = preload("res://player_appearance.gd")
const BallVisual = preload("res://ball_visual.gd")
const CourtMaterials = preload("res://court_materials.gd")
const TARGET := Vector3(0.8, 0.0006, -2.0)
const TARGET_RADIUS := 1.5
const ASSISTED_CROUCH := 0.045

var camera: Camera3D
var inspection := false
var debug_geometry := false
var actors: Array[Dictionary] = []
var character_loaded := false
var debug_material: StandardMaterial3D
var landing_marker: MeshInstance3D
var _landing_fade: Tween
var shared_ball: Node3D
var drill_target: MeshInstance3D
var _materials: Dictionary = {}
var _court_materials = CourtMaterials.new()
var _match_mode := false
var _match_camera := 0
var _near_torso := Transform3D.IDENTITY
var _ball_trail: ImmediateMesh
var _trail_points: Array[Dictionary] = []
var _trail_revision := -1
var view_player := 0
var _orbit := false
var _orbit_yaw := 0.0
var _orbit_pitch := 0.5
var _orbit_distance := 26.0
var _camera_focus := Vector3(0, 0.4, 0)


func build(initial_state: Dictionary, player_count: int = 1) -> void:
	_build_stage()
	for index in range(player_count):
		actors.append(_actor(initial_state, Vector3(-3.6 * index, 0.0, 0.0)))
	set_inspection(false)


func build_match(snapshot: Dictionary) -> void:
	_match_mode = true
	_build_stage()
	drill_target.visible = false
	for index in range(snapshot.players.size()):
		var player: Dictionary = snapshot.players[index]
		if player.bodies.has("chassis"):
			var car = CarVisual.new()
			add_child(car)
			car.build(self, PlayerAppearance.COLORS[index])
			actors.append({"car": car, "skeleton": null})
		else:
			actors.append(_actor(player, Vector3.ZERO, false))
	shared_ball = BallVisual.new()
	shared_ball.name = "Ball"
	add_child(shared_ball)
	_ball_trail = ImmediateMesh.new()
	var trail_material := _material("ball_trail", Color.WHITE, true)
	trail_material.vertex_color_use_as_albedo = true
	trail_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	_mesh(_ball_trail, trail_material, self)
	update_match(snapshot)
	set_debug_geometry(false)
	set_match_camera(0)


func _build_stage() -> void:
	_lighting()
	_ground()
	_net()
	debug_material = _material("support", Color("e7a84b"))
	var ring := TorusMesh.new()
	ring.inner_radius = 0.12
	ring.outer_radius = 0.19
	ring.rings = 32
	ring.ring_segments = 8
	var landing_material := _material("landing", Color("b1ddeb"), true)
	landing_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	landing_marker = _mesh(ring, landing_material, self)
	landing_marker.scale.y = 0.05
	landing_marker.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	landing_marker.visible = false
	camera = Camera3D.new()
	camera.fov = 52.0
	camera.near = 0.05
	camera.far = 180.0
	add_child(camera)
	camera.make_current()


func update_match(snapshot: Dictionary, motion_states: Array = [], ends: int = 0) -> void:
	for player_index in range(mini(actors.size(), snapshot.players.size())):
		var player: Dictionary = snapshot.players[player_index]
		if actors[player_index].has("car"):
			actors[player_index].car.set_color(PlayerAppearance.COLORS[player_index ^ ends])
		update_actor(player, player_index)
		var motion: Dictionary = motion_states[player_index] if player_index < motion_states.size() else {}
		_animate_assisted(actors[player_index], player, float(snapshot.time), motion)
	shared_ball.transform = snapshot.bodies.ball
	_update_ball_trail(shared_ball.position, snapshot.time, snapshot.get("ball_revision", 0))
	var player_bodies: Dictionary = snapshot.players[view_player].bodies
	_near_torso = player_bodies.chassis if player_bodies.has("chassis") else player_bodies.torso
	_update_match_camera()


func _update_ball_trail(position: Vector3, time: float, revision: int) -> void:
	# A short history makes the regulation-size ball readable in the court view.
	# It draws measured positions only, with no collision or trajectory prediction.
	if revision != _trail_revision:
		_trail_points.clear()
		_trail_revision = revision
	if not _trail_points.is_empty():
		if time < float(_trail_points.back().time) or position.distance_to(_trail_points.back().position) > 4.0:
			_trail_points.clear()
		elif time == float(_trail_points.back().time):
			return
	_trail_points.append({"time": time, "position": position})
	while not _trail_points.is_empty() and time - float(_trail_points[0].time) > 0.12:
		_trail_points.pop_front()
	_ball_trail.clear_surfaces()
	if _trail_points.size() < 2:
		return
	_ball_trail.surface_begin(Mesh.PRIMITIVE_LINES)
	for index in range(1, _trail_points.size()):
		var strength := 1.0 - (time - float(_trail_points[index].time)) / 0.12
		_ball_trail.surface_set_color(Color(0.85, 0.94, 0.32, strength * 0.7))
		_ball_trail.surface_add_vertex(_trail_points[index - 1].position)
		_ball_trail.surface_add_vertex(_trail_points[index].position)
	_ball_trail.surface_end()


func update_actor(snapshot: Dictionary, index: int = 0) -> void:
	if index >= actors.size():
		return
	var actor: Dictionary = actors[index]
	var bodies: Dictionary = snapshot.bodies
	if actor.has("car"):
		actor.car.update_poses(bodies)
		return
	if actor.character != null:
		actor.character.transform = bodies.torso * actor.character_from_torso
	actor.support.transform = bodies.torso * actor.support_from_torso
	for body_name in actor.visuals:
		if bodies.has(body_name):
			actor.visuals[body_name].transform = bodies[body_name]
	if actor.skeleton != null:
		var skeleton: Skeleton3D = actor.skeleton
		for binding in actor.bindings:
			var physical_pose: Transform3D = bodies[binding.body]
			physical_pose.origin += actor.offset
			var skin_pose := Transform3D(physical_pose.basis * binding.rotation, physical_pose.origin)
			skeleton.set_bone_global_pose_override(binding.index, skeleton.global_transform.affine_inverse() * skin_pose, 1.0, true)


func mark_landing(point: Vector3) -> void:
	clear_landing()
	landing_marker.position = Vector3(point.x, 0.004, point.z)
	_materials.landing.albedo_color.a = 0.85
	landing_marker.visible = true
	_landing_fade = create_tween()
	_landing_fade.tween_property(_materials.landing, "albedo_color:a", 0.0, 1.2)
	_landing_fade.tween_callback(landing_marker.hide)


func clear_landing() -> void:
	if _landing_fade != null:
		_landing_fade.kill()
		_landing_fade = null
	landing_marker.visible = false


func set_inspection(enabled: bool) -> void:
	if _match_mode:
		set_match_camera(2 if enabled else 0)
		return
	inspection = enabled
	set_debug_geometry(debug_geometry)
	if camera != null:
		camera.position = Vector3(3.5, 2.65, 10.4) if enabled else Vector3(5.0, 5.7, 16.2)
		camera.look_at(Vector3(0.8, 1.28, 7.8) if enabled else Vector3(0.0, 0.5, -0.8))


func set_match_camera(mode: int) -> void:
	_match_camera = clampi(mode, 0, 2)
	_orbit = false
	inspection = _match_camera == 2
	_update_match_camera()


func _update_match_camera() -> void:
	if camera == null:
		return
	if _match_camera == 2:
		_camera_focus = _near_torso * Vector3(0.5, -0.05, 0)
	elif not _orbit:
		_camera_focus = Vector3(0, 0.4, 0)
	if _orbit:
		camera.position = _camera_focus + _orbit_distance * Vector3(sin(_orbit_yaw) * cos(_orbit_pitch), sin(_orbit_pitch), cos(_orbit_yaw) * cos(_orbit_pitch))
		camera.look_at(_camera_focus)
		return
	match _match_camera:
		0:
			_fit_court_view(Vector3(13.0, 10.5, 19.0))
		1:
			var side := 1.0 if view_player == 0 else -1.0
			_fit_court_view(Vector3(0, 15.6, 22.0 * side))
		2:
			camera.position = _near_torso * Vector3(3.0, 1.2, 2.5)
			camera.look_at(_near_torso * Vector3(0.5, -0.05, 0))


func _fit_court_view(direction: Vector3) -> void:
	# Fit both players' movement bounds, including the baseline service position,
	# inside the play area. The compact HUD no longer needs a wide bottom strip.
	# Derive distance from the frustum instead of one window size.
	direction = direction.normalized()
	camera.position = _camera_focus + direction
	camera.look_at(_camera_focus)
	var viewport_size := get_viewport().get_visible_rect().size
	var tangent := tan(deg_to_rad(camera.fov) * 0.5)
	var vertical := tangent * .80
	var horizontal := tangent * viewport_size.x / maxf(1.0, viewport_size.y) * .90
	var horizontal_bounds := Vector2(INF, -INF)
	var vertical_bounds := Vector2(INF, -INF)
	var inverse := camera.basis.inverse()
	var distance := 0.0
	for x in [-7.0, 7.0]:
		for y in [0.0, 2.0]:
			for z in [-15.0, 15.0]:
				var local: Vector3 = inverse * (Vector3(x, y, z) - _camera_focus)
				horizontal_bounds.x = minf(horizontal_bounds.x, local.x - horizontal * local.z)
				horizontal_bounds.y = maxf(horizontal_bounds.y, local.x + horizontal * local.z)
				vertical_bounds.x = minf(vertical_bounds.x, local.y - vertical * local.z)
				vertical_bounds.y = maxf(vertical_bounds.y, local.y + vertical * local.z)
				distance = maxf(distance, local.z + camera.near)
	distance = maxf(distance, (horizontal_bounds.y - horizontal_bounds.x) / (2.0 * horizontal))
	distance = maxf(distance, (vertical_bounds.y - vertical_bounds.x) / (2.0 * vertical))
	# Center the projected bounds, which are asymmetric in a perspective view.
	_camera_focus += camera.basis.x * (horizontal_bounds.x + horizontal_bounds.y) * .5
	_camera_focus += camera.basis.y * (vertical_bounds.x + vertical_bounds.y) * .5
	camera.position = _camera_focus + direction * distance
	# Keep the same view ray but orbit around court height, so close zoom cannot
	# place the camera underground when the fitted target lies below the floor.
	_camera_focus = camera.position - direction * ((camera.position.y - .4) / direction.y)


func orbit_camera(delta: Vector2) -> void:
	_begin_orbit()
	_orbit_yaw -= delta.x * 0.004
	_orbit_pitch = clampf(_orbit_pitch + delta.y * 0.004, deg_to_rad(8), deg_to_rad(85))
	_update_match_camera()


func zoom_camera(steps: float) -> void:
	_begin_orbit()
	_orbit_distance = clampf(_orbit_distance * pow(0.88, steps), 3.0, 50.0)
	_update_match_camera()


func _begin_orbit() -> void:
	if _orbit or camera == null: return
	var offset := camera.position - _camera_focus
	_orbit_distance = clampf(offset.length(), 3.0, 50.0)
	_orbit_yaw = atan2(offset.x, offset.z)
	_orbit_pitch = clampf(asin(offset.normalized().y), deg_to_rad(8), deg_to_rad(85))
	_orbit = true


func set_debug_geometry(enabled: bool) -> void:
	debug_geometry = enabled
	for actor in actors:
		if actor.has("car"):
			continue
		actor.support.visible = enabled
		actor.arm.visible = enabled or not actor.has_character
		if actor.character != null:
			actor.character.visible = not enabled


func _lighting() -> void:
	var sky_material := ProceduralSkyMaterial.new()
	sky_material.sky_top_color = Color("526f8f")
	sky_material.sky_horizon_color = Color("b5c5c5")
	sky_material.ground_bottom_color = Color("273d36")
	sky_material.ground_horizon_color = Color("a9b6b0")
	var sky := Sky.new()
	sky.sky_material = sky_material
	var environment := Environment.new()
	environment.background_mode = Environment.BG_SKY
	environment.sky = sky
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.ambient_light_color = Color("c6dded")
	environment.ambient_light_energy = 0.35
	environment.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	var world := WorldEnvironment.new()
	world.environment = environment
	add_child(world)
	var sun := DirectionalLight3D.new()
	sun.light_color = Color("ffe8cb")
	sun.light_energy = 0.85
	sun.rotation_degrees = Vector3(-42, -34, 0)
	sun.shadow_enabled = true
	sun.directional_shadow_max_distance = 45.0
	add_child(sun)


func _ground() -> void:
	_box(Vector3(100.0, 0.15, 100.0), Vector3(0, -0.17, 0), _material("grass", Color("253d34")), self)
	_surface(Vector2(19.0, 36.6), Vector3(0, -0.001, 0), Color("173a2e"))
	for x in [-4.8, 4.8]:
		_surface(Vector2(1.37, 23.77), Vector3(x, 0, 0), Color("183347"))
	_surface(Vector2(8.23, 23.77), Vector3.ZERO, Color("20465e"))
	var paint := _material("paint", Color("edf0d9"))
	for x in [-5.485, -4.115, 4.115, 5.485]:
		_box(Vector3(0.04, 0.0004, 23.77), Vector3(x, 0.0002, 0), paint, self)
	for z in [-11.885, 11.885]:
		_box(Vector3(10.97, 0.0004, 0.05), Vector3(0, 0.0002, z), paint, self)
		_box(Vector3(0.05, 0.0004, 0.16), Vector3(0, 0.0002, z - signf(z) * 0.08), paint, self)
	for z in [-6.4, 6.4]:
		_box(Vector3(8.23, 0.0004, 0.04), Vector3(0, 0.0002, z), paint, self)
	_box(Vector3(0.04, 0.0004, 12.8), Vector3(0, 0.0002, 0), paint, self)
	var target_lines := ImmediateMesh.new()
	target_lines.surface_begin(Mesh.PRIMITIVE_LINES, _material("target", Color("c0dc9e"), true))
	for segment in range(64):
		var angle := TAU * segment / 64.0
		var next := TAU * (segment + 1) / 64.0
		target_lines.surface_add_vertex(TARGET + Vector3(cos(angle), 0, sin(angle)) * TARGET_RADIUS)
		target_lines.surface_add_vertex(TARGET + Vector3(cos(next), 0, sin(next)) * TARGET_RADIUS)
	target_lines.surface_end()
	drill_target = _mesh(target_lines, null, self)
	var fence := _material("fence", Color("233730"))
	for x in [-9.4, 9.4]:
		_box(Vector3(0.12, 0.65, 36.6), Vector3(x, 0.325, 0), fence, self)
	for z in [-18.2, 18.2]:
		_box(Vector3(18.8, 0.65, 0.12), Vector3(0, 0.325, z), fence, self)


func _net() -> void:
	# Keep the current flat native net's width, top and thickness visible.
	# Sag, compliant strands and physical posts remain simulation work.
	var mesh := QuadMesh.new()
	mesh.size = Vector2(10.97, 0.914)
	var net := _mesh(mesh, _court_materials.net(mesh.size), self)
	net.position.y = 0.457
	net.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_box(Vector3(10.97, 0.05, 0.03), Vector3(0, 0.889, 0), _material("tape", Color("f1f0dc")), self)
	for x in [-5.585, 5.585]:
		_rod(Vector3(x, 0, 0), Vector3(x, 1.07, 0), 0.035, _material("posts", Color("d7dbcd")), self)
	_box(Vector3(0.03, 0.914, 0.04), Vector3(0, 0.457, 0), _materials.tape, self)


func _actor(initial_state: Dictionary, offset: Vector3, include_ball := true) -> Dictionary:
	var root := Node3D.new()
	root.position = offset
	add_child(root)
	var visuals: Dictionary = {}
	var arm := Node3D.new()
	root.add_child(arm)
	for specification in [["upper_arm", 0.30, 0.045], ["forearm", 0.28, 0.035]]:
		var part := Node3D.new()
		arm.add_child(part)
		_rod(Vector3.ZERO, Vector3(specification[1], 0, 0), specification[2], debug_material, part)
		_sphere(specification[2] * 1.2, _material("joint", Color("153341")), part)
		visuals[specification[0]] = part
	var hand := Node3D.new()
	arm.add_child(hand)
	_sphere(0.025, debug_material, hand)
	visuals.hand = hand
	var racket := Node3D.new()
	root.add_child(racket)
	_racket(racket)
	visuals.racket = racket
	if include_ball:
		var ball := Node3D.new()
		ball.name = "Ball"
		root.add_child(ball)
		_sphere(0.033, _material("ball", Color("d9ed47")), ball)
		visuals.ball = ball
	var support := Node3D.new()
	root.add_child(support)
	var shoulder: Vector3 = initial_state.bodies.upper_arm.origin
	_rod(Vector3(shoulder.x, 0.04, shoulder.z), shoulder, 0.065, debug_material, support)
	_box(Vector3(0.65, 0.06, 0.65), Vector3(shoulder.x, 0.03, shoulder.z), debug_material, support)
	var torso: Transform3D = initial_state.bodies.torso
	var actor := {"root": root, "visuals": visuals, "arm": arm, "support": support, "offset": offset,
		"character": null, "skeleton": null, "bindings": [], "has_character": false,
		"character_from_torso": Transform3D.IDENTITY, "support_from_torso": torso.affine_inverse(),
		"assisted_bones": {}, "gait_distance": 0.0, "last_time": -1.0, "last_position": torso.origin}
	if ResourceLoader.exists(CHARACTER_PATH):
		var scene := load(CHARACTER_PATH) as PackedScene
		if scene != null:
			var character := scene.instantiate() as Node3D
			root.add_child(character)
			character.basis = torso.basis * Basis(Vector3.UP, PI)
			var skeleton := _find_skeleton(character)
			if skeleton != null:
				var shoulder_bone := skeleton.find_bone("upperarm_r")
				if shoulder_bone >= 0:
					var rest_shoulder: Transform3D = skeleton.global_transform * skeleton.get_bone_global_rest(shoulder_bone)
					character.position += shoulder + offset - rest_shoulder.origin
					actor.character_from_torso = torso.affine_inverse() * character.transform
					for mapping in [["upperarm_r", "upper_arm"], ["lowerarm_r", "forearm"], ["hand_r", "hand"]]:
						var bone := skeleton.find_bone(mapping[0])
						if bone >= 0:
							var rest: Transform3D = skeleton.global_transform * skeleton.get_bone_global_rest(bone)
							var physical: Transform3D = initial_state.bodies[mapping[1]]
							actor.bindings.append({"index": bone, "body": mapping[1], "rotation": physical.basis.inverse() * rest.basis})
					actor.skeleton = skeleton
					actor.has_character = actor.bindings.size() == 3
					for name in ["pelvis", "spine_01", "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r", "upperarm_l", "lowerarm_l", "hand_l"]:
						var bone := skeleton.find_bone(name)
						if bone >= 0:
							var rest: Transform3D = skeleton.global_transform * skeleton.get_bone_global_rest(bone)
							rest.origin -= offset
							actor.assisted_bones[name] = {"index": bone, "rest": torso.affine_inverse() * rest}
					# The unused arm has a static resting pose. It is presentation,
					# not another controlled or balanced physical limb.
					var left_arm := skeleton.find_bone("upperarm_l")
					if left_arm >= 0:
						var left_rest: Transform3D = skeleton.global_transform * skeleton.get_bone_global_rest(left_arm)
						left_rest.basis = Basis(Vector3.BACK, deg_to_rad(70.0)) * left_rest.basis
						skeleton.set_bone_global_pose_override(left_arm, skeleton.global_transform.affine_inverse() * left_rest, 1.0, true)
			actor.character = character
	character_loaded = character_loaded or actor.has_character
	return actor


func _animate_assisted(actor: Dictionary, player: Dictionary, time: float, motion: Dictionary) -> void:
	# These leg/off-arm poses are animation assistance. They never contribute
	# contact forces, move the native torso, or override the hitting arm.
	if actor.skeleton == null:
		return
	var torso: Transform3D = player.bodies.torso
	var velocity: Vector3 = player.torso_velocity
	velocity.y = 0.0
	var speed := velocity.length()
	if time < float(actor.last_time):
		actor.gait_distance = 0.0
	elif time > float(actor.last_time) and float(actor.last_time) >= 0.0:
		actor.gait_distance += Vector2(torso.origin.x - actor.last_position.x, torso.origin.z - actor.last_position.z).length()
	actor.last_time = time
	actor.last_position = torso.origin
	var amount := clampf(speed / 3.0, 0.0, 1.0)
	var direction := velocity.normalized() if speed > 0.01 else -torso.basis.z
	var rests: Dictionary = {}
	for name in actor.assisted_bones:
		rests[name] = torso * actor.assisted_bones[name].rest
	# The supported torso stays at its physical height. A small visual pelvis
	# crouch provides knee bend; the spine remains anchored to that torso.
	var pelvis: Transform3D = rests.pelvis
	pelvis.origin.y -= ASSISTED_CROUCH
	_set_assisted_bone(actor, "pelvis", pelvis)
	_set_assisted_bone(actor, "spine_01", rests.spine_01)
	for side in ["l", "r"]:
		var phase := float(actor.gait_distance) * TAU / 1.25 + (PI if side == "r" else 0.0)
		var hip: Transform3D = rests["thigh_" + side]
		var knee: Transform3D = rests["calf_" + side]
		var ankle: Transform3D = rests["foot_" + side]
		var goal := ankle.origin + direction * cos(phase) * 0.20 * amount
		goal.y += maxf(0.0, -sin(phase)) * 0.09 * amount
		_pose_leg(actor, side, hip, knee, ankle, goal, -torso.basis.z)
	var progress := clampf(float(motion.get("progress", 0.0)), 0.0, 1.0)
	var stroke := 0.0
	match str(motion.get("phase", "ready")):
		"prepare": stroke = progress
		"swing": stroke = 1.0 - 2.0 * progress
		"recover": stroke = progress - 1.0
	var upper: Transform3D = rests.upperarm_l
	var forearm: Transform3D = rests.lowerarm_l
	var hand: Transform3D = rests.hand_l
	var upper_rotation := Basis(torso.basis.z, deg_to_rad(65.0 - 15.0 * stroke))
	var lower_rotation := Basis(torso.basis.x, -0.8 - 0.2 * stroke) * upper_rotation
	var elbow := upper.origin + upper_rotation * (forearm.origin - upper.origin)
	var wrist := elbow + lower_rotation * (hand.origin - forearm.origin)
	_set_assisted_bone(actor, "upperarm_l", Transform3D(upper_rotation * upper.basis, upper.origin))
	_set_assisted_bone(actor, "lowerarm_l", Transform3D(lower_rotation * forearm.basis, elbow))
	_set_assisted_bone(actor, "hand_l", Transform3D(lower_rotation * hand.basis, wrist))


func _pose_leg(actor: Dictionary, side: String, hip: Transform3D, knee: Transform3D, ankle: Transform3D, goal: Vector3, forward: Vector3) -> void:
	var upper_length := hip.origin.distance_to(knee.origin)
	var lower_length := knee.origin.distance_to(ankle.origin)
	var hip_position := hip.origin - Vector3.UP * ASSISTED_CROUCH
	var leg := goal - hip_position
	var distance := clampf(leg.length(), absf(upper_length - lower_length) + 0.001, upper_length + lower_length - 0.001)
	var direction := leg.normalized()
	var bend := (forward - direction * forward.dot(direction)).normalized()
	var along := (upper_length * upper_length - lower_length * lower_length + distance * distance) / (2.0 * distance)
	var height := sqrt(maxf(0.0, upper_length * upper_length - along * along))
	var knee_position := hip_position + direction * along + bend * height
	var ankle_position := hip_position + direction * distance
	var upper_rest := (knee.origin - hip.origin).normalized()
	var lower_rest := (ankle.origin - knee.origin).normalized()
	_set_assisted_bone(actor, "thigh_" + side, Transform3D(Basis(Quaternion(upper_rest, (knee_position - hip_position).normalized())) * hip.basis, hip_position))
	_set_assisted_bone(actor, "calf_" + side, Transform3D(Basis(Quaternion(lower_rest, (ankle_position - knee_position).normalized())) * knee.basis, knee_position))
	_set_assisted_bone(actor, "foot_" + side, Transform3D(ankle.basis, ankle_position))


func _set_assisted_bone(actor: Dictionary, name: String, world_pose: Transform3D) -> void:
	var skeleton: Skeleton3D = actor.skeleton
	skeleton.set_bone_global_pose_override(actor.assisted_bones[name].index, skeleton.global_transform.affine_inverse() * world_pose, 1.0, true)


func _racket(parent: Node3D, model_path := "res://models/supported_arm.xml") -> void:
	# Draw the same named shapes used by MuJoCo, so changing racket dimensions
	# cannot leave a small visual over a larger, invisible contact area.
	var xml := XMLParser.new()
	if xml.open(model_path) != OK:
		push_error("Cannot draw racket from " + model_path)
		return
	var grip := _material("grip", Color("1a2530"))
	var frame := _material("frame", Color("eef2e5"))
	var center := Vector3.ZERO
	var half_size := Vector3.ZERO
	while xml.read() == OK:
		if xml.get_node_type() != XMLParser.NODE_ELEMENT or xml.get_node_name() != "geom": continue
		var name := xml.get_named_attribute_value_safe("name")
		if name == "stringbed":
			var position := xml.get_named_attribute_value("pos").split_floats(" ", false)
			var size := xml.get_named_attribute_value("size").split_floats(" ", false)
			center = Vector3(position[0], position[1], position[2])
			half_size = Vector3(size[0], size[1], size[2])
		elif name == "racket_handle" or name.begins_with("racket_frame_"):
			var ends := xml.get_named_attribute_value("fromto").split_floats(" ", false)
			var radius := float(xml.get_named_attribute_value("size"))
			_rod(Vector3(ends[0], ends[1], ends[2]), Vector3(ends[3], ends[4], ends[5]), radius, grip if name == "racket_handle" else frame, parent)
	if half_size.x <= 0 or half_size.y <= 0:
		push_error("Missing stringbed geometry in " + model_path)
		return
	var strings := ImmediateMesh.new()
	strings.surface_begin(Mesh.PRIMITIVE_LINES, _material("strings", Color("e2e7df"), true))
	for row in range(11):
		var y := lerpf(-half_size.y, half_size.y, row / 10.0)
		strings.surface_add_vertex(center + Vector3(-half_size.x, y, 0))
		strings.surface_add_vertex(center + Vector3(half_size.x, y, 0))
	for column in range(15):
		var x := lerpf(-half_size.x, half_size.x, column / 14.0)
		strings.surface_add_vertex(center + Vector3(x, -half_size.y, 0))
		strings.surface_add_vertex(center + Vector3(x, half_size.y, 0))
	strings.surface_end()
	var mesh := _mesh(strings, null, parent)
	mesh.name = "Strings"


func _find_skeleton(node: Node) -> Skeleton3D:
	if node is Skeleton3D:
		return node as Skeleton3D
	for child in node.get_children():
		var found := _find_skeleton(child)
		if found != null:
			return found
	return null


func _material(key: String, color: Color, unshaded: bool = false) -> StandardMaterial3D:
	if _materials.has(key):
		return _materials[key]
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	material.roughness = 0.85
	if unshaded:
		material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	_materials[key] = material
	return material


func _mesh(shape: Mesh, material: Material, parent: Node3D) -> MeshInstance3D:
	var instance := MeshInstance3D.new()
	instance.mesh = shape
	instance.material_override = material
	parent.add_child(instance)
	return instance


func _box(size: Vector3, center: Vector3, material: Material, parent: Node3D) -> MeshInstance3D:
	var shape := BoxMesh.new()
	shape.size = size
	var instance := _mesh(shape, material, parent)
	instance.position = center
	return instance


func _surface(size: Vector2, center: Vector3, color: Color) -> void:
	var shape := PlaneMesh.new()
	shape.size = size
	var instance := _mesh(shape, _court_materials.hard_court(color, size), self)
	instance.position = center


func _sphere(radius: float, material: Material, parent: Node3D) -> MeshInstance3D:
	var shape := SphereMesh.new()
	shape.radius = radius
	shape.height = radius * 2.0
	shape.radial_segments = 24
	shape.rings = 12
	return _mesh(shape, material, parent)


func _rod(from: Vector3, to: Vector3, radius: float, material: Material, parent: Node3D) -> MeshInstance3D:
	var shape := CylinderMesh.new()
	shape.top_radius = radius
	shape.bottom_radius = radius
	shape.height = from.distance_to(to)
	shape.radial_segments = 12
	var instance := _mesh(shape, material, parent)
	instance.position = (from + to) * 0.5
	instance.quaternion = Quaternion(Vector3.UP, (to - from).normalized())
	return instance
