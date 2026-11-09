extends SceneTree
## Recorded native poses only; this viewer neither predicts nor repairs motion.

const Court = preload("res://court.gd")
var recording: Dictionary
var court: Node3D
var label: Label
var attempt_index := 0
var frame_index := 0
var elapsed := 0.0
var playback_paused := false
var camera_index := 0
var capture_path := ""

class Controls extends Node:
	var handler: Callable
	func _input(event: InputEvent) -> void:
		handler.call(event)


func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	if args.is_empty():
		push_error("Supply the experiment replay.json after --")
		quit(1)
		return
	var parsed = JSON.parse_string(FileAccess.get_file_as_string(args[0]))
	if not parsed is Dictionary or parsed.get("attempts", []).is_empty():
		push_error("No recorded return attempts")
		quit(1)
		return
	recording = parsed
	for arg in args:
		if arg.begins_with("--capture="):
			capture_path = arg.trim_prefix("--capture=")
	call_deferred("_start")


func _start() -> void:
	root.title = "Tennis return policy — recorded evaluation"
	court = Court.new()
	root.add_child(court)
	court.build_match(_snapshot(_attempt().frames[0].state))
	elapsed = float(_attempt().frames[0].state.time)
	var layer := CanvasLayer.new()
	root.add_child(layer)
	label = Label.new()
	label.position = Vector2(20, 16)
	label.add_theme_font_size_override("font_size", 22)
	label.add_theme_color_override("font_outline_color", Color("182c34"))
	label.add_theme_constant_override("outline_size", 4)
	layer.add_child(label)
	var controls := Controls.new()
	controls.handler = _handle_input
	root.add_child(controls)
	await process_frame
	_show_frame()


func _attempt() -> Dictionary:
	return recording.attempts[attempt_index]


func _snapshot(state: Dictionary) -> Dictionary:
	var converted := state.duplicate(true)
	converted.bodies = _poses(state.bodies)
	for index in range(converted.players.size()):
		converted.players[index].bodies = _poses(state.players[index].bodies)
	converted.ball_revision = attempt_index
	return converted


func _poses(source: Dictionary) -> Dictionary:
	var converted := {}
	for body in source:
		var pose: Dictionary = source[body]
		var p: Array = pose.position
		var q: Array = pose.orientation
		converted[body] = Transform3D(Basis(Quaternion(q[1], q[2], q[3], q[0])), Vector3(p[0], p[1], p[2]))
	return converted


func _show_frame() -> void:
	var frame: Dictionary = _attempt().frames[frame_index]
	court.update_match(_snapshot(frame.state))
	var status: String = _attempt().outcome if frame_index == _attempt().frames.size()-1 else "in flight"
	label.text = "Attempt %d/%d · seed %d · %s\nSpace pause · N next · C camera · scroll zoom" % [
		attempt_index+1, recording.attempts.size(), _attempt().seed, status]


func _process(delta: float) -> bool:
	if court == null or playback_paused:
		return false
	elapsed += delta
	while frame_index+1 < _attempt().frames.size() and float(_attempt().frames[frame_index+1].state.time) <= elapsed:
		frame_index += 1
	_show_frame()
	if not capture_path.is_empty() and elapsed-float(_attempt().frames[0].state.time) >= .36:
		playback_paused = true
		_capture.call_deferred()
	elif elapsed > float(_attempt().frames.back().state.time)+1.0:
		_next()
	return false


func _capture() -> void:
	await RenderingServer.frame_post_draw
	var error := root.get_texture().get_image().save_png(capture_path)
	print("Replay capture: ", error)
	quit(0 if error == OK else 1)


func _next() -> void:
	attempt_index = (attempt_index+1) % recording.attempts.size()
	frame_index = 0
	elapsed = float(_attempt().frames[0].state.time)
	_show_frame()


func _handle_input(event: InputEvent) -> void:
	if court == null:
		return
	if event is InputEventKey and event.pressed and not event.echo:
		match event.physical_keycode:
			KEY_SPACE: playback_paused = not playback_paused
			KEY_N: _next()
			KEY_C:
				camera_index = (camera_index+1) % 3
				court.set_match_camera(camera_index)
			KEY_ESCAPE: quit()
	elif event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			court.zoom_camera(-1)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			court.zoom_camera(1)
	elif event is InputEventMouseMotion and event.button_mask & MOUSE_BUTTON_MASK_RIGHT:
		court.orbit_camera(event.relative)
