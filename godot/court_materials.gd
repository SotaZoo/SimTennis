extends RefCounted
## Original procedural materials. Mesh dimensions in metres set repeat scale;
## physics stays in the native model. Mipmaps keep distant surfaces stable.

var _grain: NoiseTexture2D
var _normal: NoiseTexture2D
var _net: ImageTexture


func hard_court(color: Color, size: Vector2) -> StandardMaterial3D:
	if _grain == null:
		var noise := FastNoiseLite.new()
		noise.seed = 19
		noise.frequency = 0.2
		var ramp := Gradient.new()
		ramp.colors = PackedColorArray([Color(0.85, 0.85, 0.85), Color.WHITE])
		_grain = NoiseTexture2D.new()
		_grain.noise = noise
		_grain.seamless = true
		_grain.color_ramp = ramp
		_normal = NoiseTexture2D.new()
		_normal.noise = noise
		_normal.seamless = true
		_normal.as_normal_map = true
		_normal.bump_strength = 1.0
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	material.albedo_texture = _grain
	material.roughness = 0.94
	material.normal_enabled = true
	material.normal_texture = _normal
	material.normal_scale = 0.35
	material.uv1_scale = Vector3(size.x * 2.0, size.y * 2.0, 1.0)
	material.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
	return material


func net(size: Vector2) -> StandardMaterial3D:
	if _net == null:
		# 40 mm grid spacing with 1.25 mm strands. Unlike fixed screen-width lines,
		# filtered coverage remains thin when the net is distant or oblique.
		var image := Image.create(64, 64, false, Image.FORMAT_RGBA8)
		for y in range(64):
			for x in range(64):
				image.set_pixel(x, y, Color(1, 1, 1, 1.0 if x < 2 or y < 2 else 0.0))
		image.generate_mipmaps()
		_net = ImageTexture.create_from_image(image)
	var material := StandardMaterial3D.new()
	material.albedo_color = Color("24312d")
	material.albedo_texture = _net
	material.roughness = 0.95
	material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	material.cull_mode = BaseMaterial3D.CULL_DISABLED
	material.uv1_scale = Vector3(size.x / 0.04, size.y / 0.04, 1.0)
	material.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
	return material
