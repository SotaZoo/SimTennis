extends "res://rally_replay.gd"

func _process(_delta: float) -> bool:
	return false


func _expect(condition: bool, message: String) -> bool:
	if not condition:
		push_error(message)
		quit(1)
	return condition


func _key(code: int) -> void:
	var event := InputEventKey.new()
	event.pressed = true
	event.physical_keycode = code
	Input.parse_input_event(event)
	Input.flush_buffered_events()
	await process_frame


func _start() -> void:
	await super._start()
	if not _expect(root.title == recording.title, "Replay must identify the recorded experiment"):
		return
	for index in range(recording.attempts.size()):
		attempt_index = index
		for frame_number in range(_attempt().frames.size()):
			frame_index = frame_number
			_show_frame()
			var snapshot := _snapshot(_attempt().frames[frame_index].state)
			if not _expect(court.shared_ball.transform.is_equal_approx(snapshot.bodies.ball), "Ball pose must come from recorded physics"):
				return
			for slot in range(2):
				for body in court.actors[slot].car.parts:
					if not _expect(court.actors[slot].car.parts[body].transform.is_equal_approx(snapshot.players[slot].bodies[body]), "Both car/racket poses must come from recorded physics"):
						return
				var identity := slot ^ int(_attempt().score_before.ends)
				if not _expect(court.actors[slot].car.paint.albedo_color.is_equal_approx(court.PlayerAppearance.COLORS[identity]), "Player color must keep its identity through a change of ends"):
					return
			var score: Dictionary = _attempt().score_after if frame_index == _attempt().frames.size()-1 else _attempt().score_before
			for player in range(2):
				var expected_game := "" if recording.get("format") == "first-to-three" else str(int(score.games[player]))
				if not _expect(get("game_labels")[player].text == expected_game, "Replay game score must match the recorded decision"):
					return
				if not _expect(get("point_labels")[player].text == str(score.point_text[player]), "Replay point score must match the recorded decision"):
					return
	print("PASS: both recorded players, identity colors and score transitions across all attempts")
	call("_jump_to", 0)
	await _key(KEY_SPACE)
	if not _expect(playback_paused, "Space must pause the recording"):
		return
	await _key(KEY_N)
	var expected := 0
	for index in range(1, recording.attempts.size()):
		if recording.attempts[index].point_index != recording.attempts[0].point_index:
			expected = index
			break
	if not _expect(attempt_index == expected, "Next point must skip service retries, not playback frames"):
		return
	await _key(KEY_P)
	if not _expect(attempt_index == 0, "Previous point must return to its first service attempt"):
		return
	await _key(KEY_BRACKETRIGHT)
	if not _expect(get("playback_speed") == 2.0, "Playback speed must change independently of recorded time"):
		return
	await _key(KEY_H)
	if not _expect(get("help_panel").visible, "Help and diagnostics must be available on demand"):
		return
	await _key(KEY_C)
	if not _expect(camera_index == 1, "Camera input must remain available"):
		return
	var before: Vector3 = court.camera.position
	var wheel := InputEventMouseButton.new()
	wheel.pressed = true
	wheel.button_index = MOUSE_BUTTON_WHEEL_UP
	Input.parse_input_event(wheel)
	Input.flush_buffered_events()
	await process_frame
	if not _expect(not court.camera.position.is_equal_approx(before), "Scroll must zoom the recorded court"):
		return
	var stopped_time := elapsed
	super._process(0.1)
	if not _expect(is_equal_approx(elapsed, stopped_time), "Paused playback must not advance recorded time"):
		return
	await _key(KEY_SPACE)
	super._process(0.005)
	if not _expect(is_equal_approx(elapsed, stopped_time+0.01), "Twice-speed playback must advance twice the elapsed display time"):
		return
	call("_jump_to", recording.attempts.size()-1)
	frame_index = _attempt().frames.size()-1
	_show_frame()
	call("_next")
	if not _expect(attempt_index == recording.attempts.size()-1 and playback_paused, "End of set must retain the final view and score"):
		return
	print("PASS: replay pause, point navigation, speed, help, camera, zoom and final-frame hold")
	quit()
