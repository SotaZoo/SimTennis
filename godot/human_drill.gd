extends SceneTree

class Controls extends Node:
    var handler: Callable
    func _input(event: InputEvent) -> void:
        handler.call(event)

var simulation
var court
var state: Dictionary
var accumulator := 0.0
var frames := 0
var label: Label
var check := false

func _initialize() -> void:
    call_deferred("setup")

func setup() -> void:
    check = OS.get_cmdline_user_args().has("--check")
    simulation = ClassDB.instantiate("TennisBridge")
    assert(simulation.configure(FileAccess.get_file_as_string("res://models/supported_arm.xml")))
    state = simulation.snapshot()
    var controls := Controls.new()
    controls.handler = handle_input
    root.add_child(controls)
    court = load("res://court.gd").new()
    root.add_child(court)
    court.build(state)
    court.update_actor(state)
    court.set_inspection(true)
    assert(court.character_loaded)
    var layer := CanvasLayer.new()
    root.add_child(layer)
    label = Label.new()
    label.position = Vector2(24, 24)
    label.text = "SimTennis | Supported-arm drill\nScripted joint targets | C: camera | D: physical mechanism | Esc: quit"
    label.add_theme_font_size_override("font_size", 20)
    layer.add_child(label)

func _process(delta: float) -> bool:
    if simulation == null:
        return false
    accumulator += 1.0 / 30.0 if check else delta
    var ticks := mini(int(accumulator / float(state.timestep)), 1000)
    if ticks == 0:
        return false
    var phase := fmod(float(state.time), 3.0)
    var targets := PackedFloat64Array([0.35 * sin(2.0 * PI * phase / 3.0), 0.0, 0.20 * (1.0 - cos(2.0 * PI * phase / 3.0)), 0.10 * sin(2.0 * PI * phase / 3.0), 0.0, 0.0])
    assert(simulation.submit_intent(state.time, targets, 1.0))
    assert(simulation.advance(ticks))
    accumulator -= ticks * float(state.timestep)
    state = simulation.snapshot()
    court.update_actor(state)
    frames += 1
    if check and frames == 60:
        assert(float(state.time) > 1.9)
        assert(simulation.last_error().is_empty())
        print("PASS: human character loaded; supported arm advanced through two seconds of native simulation")
        quit()
    return false

func handle_input(event: InputEvent) -> void:
    if event is InputEventKey and event.pressed and not event.echo:
        if event.keycode == KEY_ESCAPE:
            quit()
        elif event.keycode == KEY_D:
            court.set_debug_geometry(not court.debug_geometry)
        elif event.keycode == KEY_C:
            court.set_inspection(not court.inspection)
