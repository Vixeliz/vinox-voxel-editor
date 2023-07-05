use ggez::graphics::{
    Camera3d, Canvas3d, DrawParam, DrawParam3d, ImageFormat, Mesh3d, Mesh3dBuilder, Rect, Sampler,
    Vertex3d,
};
use ggez::graphics::{Image, Shader};
use ggez::input::keyboard::KeyCode;
use ggez::{
    event,
    glam::*,
    graphics::{self, Color},
    Context, GameResult,
};
use ggez_atlas::atlas::{TextureAtlas, TextureAtlasBuilder};
use itertools::izip;
use std::collections::HashMap;
use std::{env, path};
use vinox_voxel::prelude::*;

struct MainState {
    camera: Camera3d,
    meshes: Vec<(Mesh3d, Vec3, Vec3)>,
    psx: bool,
    psx_shader: Shader,
    custom_shader: Shader,
    texture_atlas: TextureAtlas<String>,
}

impl MainState {
    fn new(ctx: &mut Context) -> GameResult<Self> {
        let mut camera = Camera3d::default();
        let mut registry = BlockRegistry::default();
        let mut tb = TextureAtlasBuilder::default();
        let grass = Image::from_path(ctx, "/grass.png")?;
        let stone = Image::from_path(ctx, "/stone.png")?;
        tb.add_texture("vinox:test".to_string(), grass);
        tb.add_texture("vinox:slab".to_string(), stone);

        let texture_atlas = tb.build(ctx)?;

        registry.0.insert(
            "vinox:test".to_string(),
            Block {
                identifier: "vinox:test".to_string(),
                textures: None,
                geometry: Some(BlockGeometry::Block),
                auto_geo: None,
                visibility: Some(VoxelVisibility::Opaque),
                has_item: None,
            },
        );
        registry.0.insert(
            "vinox:slab".to_string(),
            Block {
                identifier: "vinox:slab".to_string(),
                textures: None,
                geometry: Some(BlockGeometry::Slab),
                auto_geo: None,
                visibility: Some(VoxelVisibility::Opaque),
                has_item: None,
            },
        );

        let mut chunk = ChunkData::<BlockData, BlockRegistry>::default();
        for y in 0..3 {
            for x in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    if y == 0 {
                        chunk.set(
                            RelativeVoxelPos::new(x as u32, y + 1, z as u32),
                            BlockData::new("vinox".to_string(), "test".to_string()),
                        );
                    }
                    if y == 2 && x == CHUNK_SIZE - 2 || z == CHUNK_SIZE - 2 || x == 1 || z == 1 {
                        chunk.set(
                            RelativeVoxelPos::new(x as u32, y + 1, z as u32),
                            BlockData::new("vinox".to_string(), "test".to_string()),
                        );
                        continue;
                    }
                    if y == 1 && x < CHUNK_SIZE - 2 && z < CHUNK_SIZE - 2 && x > 1 && z > 1 {
                        chunk.set(
                            RelativeVoxelPos::new(x as u32, y + 1, z as u32),
                            BlockData::new("vinox".to_string(), "slab".to_string()),
                        );
                        continue;
                    }
                    if x == CHUNK_SIZE - 2 || z == CHUNK_SIZE - 2 || x == 1 || z == 1 {
                        chunk.set(
                            RelativeVoxelPos::new(x as u32, y + 1, z as u32),
                            BlockData::new("vinox".to_string(), "test".to_string()),
                        );
                    }
                }
            }
        }

        let mut geo_table = GeometryRegistry(HashMap::default());
        geo_table.insert("vinox:block".to_string(), Geometry::default());
        geo_table.insert(
            "vinox:slab".to_string(),
            Geometry {
                namespace: "vinox".to_string(),
                name: "slab".to_string(),
                blocks: [false, false, true, false, false, false],
                blocks_self: Some([true, true, false, false, true, true]),
                element: BlockGeo {
                    pivot: (0, 0, 0),
                    rotation: (0, 0, 0),
                    cubes: vec![FaceDescript {
                        uv: [
                            ((0, 0), (16, 8)),
                            ((0, 0), (16, 8)),
                            ((16, 16), (-16, -16)),
                            ((16, 16), (-16, -16)),
                            ((0, 0), (16, 8)),
                            ((0, 0), (16, 8)),
                        ],
                        discard: [false, false, false, false, false, false],
                        texture_variance: [false, false, false, false, false, false],
                        cull: [true, true, true, false, true, true],
                        origin: (0, 0, 0),
                        end: (16, 8, 16),
                        rotation: (0, 0, 0),
                        pivot: (8, 8, 8),
                    }],
                },
            },
        );

        let mut texture_uvs = ahash::HashMap::default();
        texture_uvs.insert(
            "vinox:test".to_string(),
            [
                rect_to_uv_rect(
                    *texture_atlas
                        .textures
                        .get(&"vinox:test".to_string())
                        .unwrap(),
                ),
                rect_to_uv_rect(
                    *texture_atlas
                        .textures
                        .get(&"vinox:test".to_string())
                        .unwrap(),
                ),
                rect_to_uv_rect(
                    *texture_atlas
                        .textures
                        .get(&"vinox:test".to_string())
                        .unwrap(),
                ),
                rect_to_uv_rect(
                    *texture_atlas
                        .textures
                        .get(&"vinox:test".to_string())
                        .unwrap(),
                ),
                rect_to_uv_rect(
                    *texture_atlas
                        .textures
                        .get(&"vinox:test".to_string())
                        .unwrap(),
                ),
                rect_to_uv_rect(
                    *texture_atlas
                        .textures
                        .get(&"vinox:test".to_string())
                        .unwrap(),
                ),
            ],
        );
        texture_uvs.insert(
            "vinox:slab".to_string(),
            [
                rect_to_uv_rect(
                    *texture_atlas
                        .textures
                        .get(&"vinox:slab".to_string())
                        .unwrap(),
                ),
                rect_to_uv_rect(
                    *texture_atlas
                        .textures
                        .get(&"vinox:slab".to_string())
                        .unwrap(),
                ),
                rect_to_uv_rect(
                    *texture_atlas
                        .textures
                        .get(&"vinox:slab".to_string())
                        .unwrap(),
                ),
                rect_to_uv_rect(
                    *texture_atlas
                        .textures
                        .get(&"vinox:slab".to_string())
                        .unwrap(),
                ),
                rect_to_uv_rect(
                    *texture_atlas
                        .textures
                        .get(&"vinox:slab".to_string())
                        .unwrap(),
                ),
                rect_to_uv_rect(
                    *texture_atlas
                        .textures
                        .get(&"vinox:slab".to_string())
                        .unwrap(),
                ),
            ],
        );

        let asset_registry = AssetRegistry {
            texture_uvs,
            texture_size: mint::Point2 {
                x: texture_atlas.size.x as f32,
                y: texture_atlas.size.y as f32,
            },
        };

        // asset_registry.texture_uvs;

        let mesh = full_mesh(
            &asset_registry,
            &ChunkBoundary::<BlockData, BlockRegistry>::new(
                chunk,
                Box::default(),
                &registry,
                &geo_table,
                &asset_registry,
            ),
            IVec3::new(0, 0, 0).into(),
        );

        let vertices = izip!(
            mesh.chunk_mesh.vertices,
            mesh.chunk_mesh.colors.unwrap(),
            mesh.chunk_mesh.uvs.unwrap(),
            mesh.chunk_mesh.normals
        )
        .map(|(pos, colors, uvs, normals)| {
            Vertex3d::new(
                mint::Vector3::from(pos),
                Vec2::from(uvs),
                Color::new(colors[0], colors[1], colors[2], colors[3]),
                mint::Vector3::from(normals),
            )
        })
        .collect();

        let mesh = Mesh3dBuilder::new()
            .from_data(
                vertices,
                mesh.chunk_mesh.indices,
                Some(texture_atlas.image.clone()),
            )
            .build(ctx);

        camera.transform.yaw = 0.0;
        camera.transform.pitch = 0.0;
        camera.projection.zfar = 1000.0;
        ggez::input::mouse::set_cursor_hidden(ctx, true);
        ggez::input::mouse::set_cursor_grabbed(ctx, true)?;

        Ok(MainState {
            camera,
            meshes: vec![(mesh, Vec3::new(1.0, 1.0, 1.0), Vec3::new(0.0, 0.0, 0.0))],
            custom_shader: graphics::ShaderBuilder::from_path("/fancy.wgsl")
                .build(&ctx.gfx)
                .unwrap(),
            psx_shader: graphics::ShaderBuilder::from_path("/psx.wgsl")
                .build(&ctx.gfx)
                .unwrap(),
            psx: true,
            texture_atlas,
        })
    }
}

impl event::EventHandler for MainState {
    fn resize_event(&mut self, _: &mut Context, width: f32, height: f32) -> GameResult {
        self.camera.projection.resize(width as u32, height as u32);
        Ok(())
    }
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        // set_cursor_hidden(ctx, true);
        // set_cursor_grabbed(ctx, true)?;
        let k_ctx = &ctx.keyboard.clone();
        let (yaw_sin, yaw_cos) = self.camera.transform.yaw.sin_cos();
        let dt = ctx.time.delta().as_secs_f32();
        let forward = Vec3::new(yaw_cos, 0.0, yaw_sin).normalize() * 25.0 * dt;
        let right = Vec3::new(-yaw_sin, 0.0, yaw_cos).normalize() * 25.0 * dt;

        if k_ctx.is_key_pressed(KeyCode::Q) {
            self.meshes[0].1 += 1.0 * dt;
        }
        if k_ctx.is_key_pressed(KeyCode::E) {
            self.meshes[0].1 -= 1.0 * dt;
        }
        if k_ctx.is_key_pressed(KeyCode::Space) {
            self.camera.transform.position.y += 25.0 * dt;
        }
        if k_ctx.is_key_pressed(KeyCode::C) {
            self.camera.transform.position.y -= 25.0 * dt;
        }
        if k_ctx.is_key_pressed(KeyCode::W) {
            self.camera.transform = self.camera.transform.translate(forward);
        }
        if k_ctx.is_key_just_pressed(KeyCode::K) {
            self.psx = !self.psx;
        }
        if k_ctx.is_key_pressed(KeyCode::S) {
            self.camera.transform = self.camera.transform.translate(-forward);
        }
        if k_ctx.is_key_pressed(KeyCode::D) {
            self.camera.transform = self.camera.transform.translate(right);
        }
        if k_ctx.is_key_pressed(KeyCode::A) {
            self.camera.transform = self.camera.transform.translate(-right);
        }
        if k_ctx.is_key_pressed(KeyCode::Right) {
            self.camera.transform.yaw += 1.0_f32.to_radians() * dt * 75.0;
        }
        if k_ctx.is_key_pressed(KeyCode::Left) {
            self.camera.transform.yaw -= 1.0_f32.to_radians() * dt * 75.0;
        }
        if k_ctx.is_key_pressed(KeyCode::Up) {
            self.camera.transform.pitch += 1.0_f32.to_radians() * dt * 75.0;
        }
        if k_ctx.is_key_pressed(KeyCode::Down) {
            self.camera.transform.pitch -= 1.0_f32.to_radians() * dt * 75.0;
        }

        let mouse_delta = ctx.mouse.raw_delta();
        let speed = 0.5;
        let mouse_delta_y = mouse_delta.y as f32 * speed * dt * -1.0;
        let mouse_delta_x = mouse_delta.x as f32 * speed * dt;
        self.camera.transform.yaw += mouse_delta_x;
        self.camera.transform.pitch += mouse_delta_y;

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let canvas_image = Image::new_canvas_image(ctx, ImageFormat::Bgra8UnormSrgb, 320, 240, 1);
        let mut canvas3d = if self.psx {
            let mut canvas3d = Canvas3d::from_image(ctx, canvas_image.clone(), Color::BLACK);
            canvas3d.set_shader(&self.psx_shader);
            canvas3d
        } else {
            let mut canvas3d = Canvas3d::from_frame(ctx, Color::BLACK);
            canvas3d.set_shader(&self.custom_shader);
            canvas3d
        };
        canvas3d.set_projection(self.camera.to_matrix());
        canvas3d.set_sampler(Sampler::nearest_clamp());
        for mesh in self.meshes.iter() {
            canvas3d.draw(&mesh.0, DrawParam3d::default().scale(mesh.1));
        }
        canvas3d.finish(ctx)?;
        let mut canvas = graphics::Canvas::from_frame(ctx, None);

        // Do 2d drawing
        if self.psx {
            canvas.set_sampler(Sampler::nearest_clamp());
            let params = DrawParam::new().dest(Vec2::new(0.0, 0.0)).scale(Vec2::new(
                ctx.gfx.drawable_size().0 / 320.0,
                ctx.gfx.drawable_size().1 / 240.0,
            ));
            canvas.draw(&canvas_image, params);
        }
        let dest_point1 = Vec2::new(10.0, 210.0);
        let dest_point2 = Vec2::new(10.0, 250.0);
        canvas.draw(
            &graphics::Text::new("You can mix 3d and 2d drawing;"),
            dest_point1,
        );
        canvas.draw(&self.texture_atlas.image, dest_point2);

        canvas.finish(ctx)?;

        Ok(())
    }
}

pub fn main() -> GameResult {
    let resource_dir = if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        let mut path = path::PathBuf::from(manifest_dir);
        path.push("resources");
        path
    } else {
        path::PathBuf::from("./resources")
    };

    let cb = ggez::ContextBuilder::new("cube", "ggez")
        .window_mode(ggez::conf::WindowMode::default().resizable(true))
        .add_resource_path(resource_dir);

    let (mut ctx, events_loop) = cb.build()?;
    let state = MainState::new(&mut ctx)?;
    event::run(ctx, events_loop, state)
}

fn rect_to_uv_rect(rect: Rect) -> UVRect {
    UVRect {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
    }
}