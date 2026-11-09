extends "res://return_replay.gd"
## A scored recording, using the same native-pose renderer as single returns.

const SPEEDS := [0.5, 1.0, 2.0, 4.0]
var playback_speed := 1.0
var speed_index := 1
var playback_finished := false
var game_labels: Array[Label] = []
var point_labels: Array[Label] = []
var player_labels: Array[Label] = []
var status_label: Label
var help_panel: PanelContainer
var diagnostics: Label


func _initialize() -> void:
	super._initialize()
	if not recording.is_empty() and (recording.get("schema") != "rally-v1" or recording.get("players", []).size() != 2):
		push_error("Expected a rally-v1 recording with two player identities")
		quit(1)


func _start() -> void:
	await super._start()
	root.title = recording.get("title", "Tennis — recorded set")
	label.position = Vector2(18, 14)
	label.add_theme_font_size_override("font_size", 18)
	label.text = root.title
	var panel := _panel(Vector2(18, 46))
	var grid := GridContainer.new()
	grid.columns = 3
	grid.add_theme_constant_override("h_separation", 18)
	grid.add_theme_constant_override("v_separation", 6)
	panel.add_child(grid)
	for title in ["PLAYER", "" if recording.get("format") == "first-to-three" else "GAMES", "POINTS"]:
		grid.add_child(_text(title, 12, Color("b9c9d4")))
	for identity in range(2):
		var name_label := _text(recording.players[identity], 21, Court.PlayerAppearance.COLORS[identity].lightened(0.2))
		name_label.custom_minimum_size.x = 172
		name_label.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
		grid.add_child(name_label)
		player_labels.append(name_label)
		var game := _text("0", 23)
		grid.add_child(game)
		game_labels.append(game)
		var point := _text("0", 23)
		grid.add_child(point)
		point_labels.append(point)
	status_label = _text("", 16)
	status_label.position = Vector2(18, 170)
	status_label.add_theme_color_override("font_outline_color", Color("182c34"))
	status_label.add_theme_constant_override("outline_size", 4)
	label.get_parent().add_child(status_label)
	help_panel = _panel(Vector2(18, 230))
	var help := VBoxContainer.new()
	help_panel.add_child(help)
	help.add_child(_text("Space  Pause / resume    N / P  Next / previous point\n[ / ]  Playback speed    C  Camera    Scroll  Zoom\nRight-drag  Orbit    H  Help / diagnostics    Esc  Close", 16))
	help.add_child(_text("Learned rally · scripted physical serve / recovery\nAll recorded attempts; 1 s pauses between attempts.", 15, Color("b9c9d4")))
	diagnostics = _text("", 14, Color("b9c9d4"))
	help.add_child(diagnostics)
	help_panel.hide()
	_show_frame()


func _panel(position: Vector2) -> PanelContainer:
	var panel := PanelContainer.new()
	panel.position = position
	panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.055, 0.09, 0.12, 0.94)
	style.set_corner_radius_all(9)
	style.content_margin_left = 14
	style.content_margin_right = 14
	style.content_margin_top = 11
	style.content_margin_bottom = 11
	panel.add_theme_stylebox_override("panel", style)
	label.get_parent().add_child(panel)
	return panel


func _text(value: String, size: int, color: Color = Color("edf3f5")) -> Label:
	var text := Label.new()
	text.text = value
	text.mouse_filter = Control.MOUSE_FILTER_IGNORE
	text.add_theme_font_size_override("font_size", size)
	text.add_theme_color_override("font_color", color)
	return text


func _show_frame() -> void:
	var attempt := _attempt()
	var state: Dictionary = attempt.frames[frame_index].state
	# The physical end changes on the next setup, not on the scoring frame.
	var ends := int(attempt.score_before.ends)
	court.update_match(_snapshot(state), [], ends)
	if status_label == null:
		return
	var resolved: bool = frame_index == attempt.frames.size()-1
	var score: Dictionary = attempt.score_after if resolved else attempt.score_before
	for identity in range(2):
		player_labels[identity].text = ("• " if identity == int(attempt.score_before.server) else "  ") + str(recording.players[identity])
		game_labels[identity].text = "" if recording.get("format") == "first-to-three" else str(int(score.games[identity]))
		point_labels[identity].text = str(score.point_text[identity])
	var mode := "Paused" if playback_paused else "Playing"
	var detail := "Serve %d" % int(attempt.attempt) if attempt.setup.kind == "serve" else "Training feed"
	if resolved:
		detail = str(attempt.outcome).replace("_", " ")
		if int(score.get("winner", -1)) >= 0:
			detail = "%s wins %s" % [recording.players[int(score.winner)], "the contest" if recording.get("format") == "first-to-three" else "the set"]
		elif int(attempt.get("winner", -1)) >= 0:
			detail = "%s wins point · %s" % [recording.players[int(attempt.winner) ^ ends], detail]
	status_label.text = "Point %d · %s\n%s · %s× · H help" % [int(attempt.point_index)+1, detail, mode, str(playback_speed)]
	if help_panel.visible:
		var velocity: Array = state.ball_velocity
		var speed := Vector3(velocity[0], velocity[1], velocity[2]).length()
		diagnostics.text = "Seed %d · attempt %d/%d · t %.2f s · ball %.1f m/s\nTick %d · frame %d/%d" % [
			int(attempt.seed), attempt_index+1, recording.attempts.size(), float(state.time), speed,
			int(state.tick), frame_index+1, attempt.frames.size()]


func _process(delta: float) -> bool:
	if court == null or playback_paused:
		return false
	elapsed += delta * playback_speed
	while frame_index+1 < _attempt().frames.size() and float(_attempt().frames[frame_index+1].state.time) <= elapsed:
		frame_index += 1
	_show_frame()
	if not capture_path.is_empty() and elapsed-float(_attempt().frames[0].state.time) >= .36:
		playback_paused = true
		_capture.call_deferred()
	elif elapsed > float(_attempt().frames.back().state.time)+1.0:
		_next()
	return false


func _next() -> void:
	if attempt_index+1 == recording.attempts.size():
		playback_finished = true
		playback_paused = true
		frame_index = _attempt().frames.size()-1
		_show_frame()
	else:
		_jump_to(attempt_index+1)


func _jump_to(index: int) -> void:
	attempt_index = index
	frame_index = 0
	elapsed = float(_attempt().frames[0].state.time)
	playback_finished = false
	_show_frame()


func _point(direction: int) -> void:
	var point_index := int(_attempt().point_index)
	for distance in range(1, recording.attempts.size()+1):
		var index := wrapi(attempt_index+direction*distance, 0, recording.attempts.size())
		if int(recording.attempts[index].point_index) != point_index:
			while index > 0 and recording.attempts[index-1].point_index == recording.attempts[index].point_index:
				index -= 1
			_jump_to(index)
			return
	_jump_to(0)


func _handle_input(event: InputEvent) -> void:
	if court == null:
		return
	if event is InputEventKey and event.pressed and not event.echo:
		match event.physical_keycode:
			KEY_N: _point(1)
			KEY_P: _point(-1)
			KEY_H: help_panel.visible = not help_panel.visible
			KEY_BRACKETLEFT, KEY_BRACKETRIGHT:
				speed_index = clampi(speed_index + (1 if event.physical_keycode == KEY_BRACKETRIGHT else -1), 0, SPEEDS.size()-1)
				playback_speed = SPEEDS[speed_index]
			KEY_SPACE:
				if playback_finished:
					_jump_to(0)
					playback_paused = false
				else:
					super._handle_input(event)
			_: super._handle_input(event)
	else:
		super._handle_input(event)
	_show_frame()
