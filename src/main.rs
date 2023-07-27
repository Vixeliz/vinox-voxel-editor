use ggez::conf::NumSamples;
use ggez::graphics::{
    Aabb, Camera3d, Canvas3d, DrawParam, DrawParam3d, ImageFormat, Mesh3d, Mesh3dBuilder, Rect,
    Sampler, Vertex3d,
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
use vinox_voxel_formats::level::VoxelLevel;

/// A plane defined by a normal and distance value along the normal
/// Any point p is in the plane if n.p = d
/// For planes defining half-spaces such as for frusta, if n.p > d then p is on the positive side of the plane.
#[derive(Clone, Copy, Debug, Default)]
pub struct Plane {
    pub normal_d: Vec4,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Frustum {
    pub planes: [Plane; 6],
}

impl Frustum {
    // NOTE: This approach of extracting the frustum planes from the view
    // projection matrix is from Foundations of Game Engine Development 2
    // Rendering by Lengyel. Slight modification has been made for when
    // the far plane is infinite but we still want to cull to a far plane.
    pub fn from_view_projection(
        view_projection: &Mat4,
        view_translation: &Vec3,
        view_backward: &Vec3,
        far: f32,
    ) -> Self {
        let row3 = view_projection.row(3);
        let mut planes = [Plane::default(); 6];
        for (i, plane) in planes.iter_mut().enumerate().take(5) {
            let row = view_projection.row(i / 2);
            plane.normal_d = if (i & 1) == 0 && i != 4 {
                row3 + row
            } else {
                row3 - row
            }
            .normalize();
        }
        let far_center = *view_translation - far * *view_backward;
        planes[5].normal_d = view_backward
            .extend(-view_backward.dot(far_center))
            .normalize();
        Self { planes }
    }

    // pub fn intersects_sphere(&self, sphere: &Sphere) -> bool {
    //     for plane in &self.planes {
    //         if plane.normal_d.dot(sphere.center.extend(1.0)) + sphere.radius <= 0.0 {
    //             return false;
    //         }
    //     }
    //     true
    // }

    pub fn intersects_obb(&self, aabb: &Aabb, model_to_world: &Mat4) -> bool {
        let aabb_center_world = *model_to_world * Vec3::from(aabb.center).extend(1.0);
        let axes = [
            Vec3A::from(model_to_world.x_axis),
            Vec3A::from(model_to_world.y_axis),
            Vec3A::from(model_to_world.z_axis),
        ];
        for plane in &self.planes {
            let p_normal = Vec3A::from(plane.normal_d);
            let half_extents = Vec3A::from(aabb.half_extents);
            let relative_radius = Vec3A::new(
                p_normal.dot(axes[0]),
                p_normal.dot(axes[1]),
                p_normal.dot(axes[2]),
            )
            .abs()
            .dot(half_extents);
            // let relative_radius = aabb.relative_radius(&p_normal, &axes);
            if plane.normal_d.dot(aabb_center_world) + relative_radius <= 0.0 {
                return false;
            }
        }
        true
    }
}

struct MainState {
    camera: Camera3d,
    psx: bool,
    psx_shader: Shader,
    custom_shader: Shader,
    texture_atlas: TextureAtlas<String>,
    level: VoxelLevel,
    chunk_meshes: Vec<(Mesh3d, UVec3)>,
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

        // let mut chunk = ChunkData::<BlockData, BlockRegistry>::default();

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

        let mut level = VoxelLevel::new(UVec3::new(8, 6, 8));
        let mut chunk_meshes = Vec::new();
        // // asset_registry.texture_uvs;
        for chunk in level.loaded_chunks.as_mut().unwrap() {
            for y in 0..3 {
                for x in 0..CHUNK_SIZE {
                    for z in 0..CHUNK_SIZE {
                        if y == 0 {
                            chunk.set(
                                RelativeVoxelPos::new(x as u32, y + 1, z as u32),
                                BlockData::new("vinox".to_string(), "test".to_string()),
                            );
                        }
                        if y == 2 && x == CHUNK_SIZE - 2 || z == CHUNK_SIZE - 2 || x == 1 || z == 1
                        {
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
        }
        for (idx, chunk) in level.loaded_chunks.as_ref().unwrap().iter().enumerate() {
            let mesh = full_mesh(
                &asset_registry,
                &ChunkBoundary::<BlockData, BlockRegistry>::new(
                    chunk.clone(),
                    Default::default(),
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

            chunk_meshes.push((
                Mesh3dBuilder::new()
                    .from_data(
                        vertices,
                        mesh.chunk_mesh.indices,
                        Some(texture_atlas.image.clone()),
                    )
                    .build(ctx),
                level.delinearize(idx).into(),
            ));
        }

        level.block_registry = registry;
        level.geometry_registry = geo_table;
        level.asset_registry = asset_registry;

        camera.transform.yaw = 0.0;
        camera.transform.pitch = 0.0;
        camera.projection.zfar = 1000.0;
        ggez::input::mouse::set_cursor_hidden(ctx, true);
        ggez::input::mouse::set_cursor_grabbed(ctx, true)?;

        Ok(MainState {
            camera,
            // meshes: vec![(mesh, Vec3::new(1.0, 1.0, 1.0), Vec3::new(0.0, 0.0, 0.0))],
            custom_shader: graphics::ShaderBuilder::from_path("/fancy.wgsl")
                .build(&ctx.gfx)
                .unwrap(),
            psx_shader: graphics::ShaderBuilder::from_path("/psx.wgsl")
                .build(&ctx.gfx)
                .unwrap(),
            psx: false,
            texture_atlas,
            chunk_meshes,
            level,
        })
    }
}

impl event::EventHandler for MainState {
    fn resize_event(&mut self, _: &mut Context, width: f32, height: f32) -> GameResult {
        self.camera.projection.resize(width as u32, height as u32);
        Ok(())
    }
    // fn quit_event(&mut self, _ctx: &mut Context) -> Result<bool, ggez::GameError> {
    //     Ok(true)
    // }
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        // set_cursor_hidden(ctx, true);
        // set_cursor_grabbed(ctx, true)?;
        let k_ctx = &ctx.keyboard.clone();
        let (yaw_sin, yaw_cos) = self.camera.transform.yaw.sin_cos();
        let (pitch_sin, pitch_cos) = self.camera.transform.pitch.sin_cos();
        let dt = ctx.time.delta().as_secs_f32();
        let forward = Vec3::new(yaw_cos, 0.0, yaw_sin).normalize() * 25.0 * dt;
        let right = Vec3::new(-yaw_sin, 0.0, yaw_cos).normalize() * 25.0 * dt;
        if k_ctx.is_key_just_pressed(KeyCode::R) {
            if let Some((voxel_pos, _, _)) = self.level.raycast(
                self.camera.transform.position,
                Vec3::new(yaw_cos, pitch_sin, yaw_sin),
                16.0,
            ) {
                let vox_pos = Vec3::from(mint::Vector3::<f32>::from(voxel_pos)).as_uvec3();
                self.level.set_voxel(vox_pos, BlockData::default());
                let chunk_pos = UVec3::new(
                    (vox_pos.x as f32 / (CHUNK_SIZE as f32)).floor() as u32,
                    (vox_pos.y as f32 / (CHUNK_SIZE as f32)).floor() as u32,
                    (vox_pos.z as f32 / (CHUNK_SIZE as f32)).floor() as u32,
                );

                if let Some(chunk) = self.level.get_chunk(chunk_pos) {
                    let mesh = full_mesh(
                        &self.level.asset_registry,
                        &ChunkBoundary::<BlockData, BlockRegistry>::new(
                            chunk.clone(),
                            Default::default(),
                            &self.level.block_registry,
                            &self.level.geometry_registry,
                            &self.level.asset_registry,
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

                    let idx = self.level.linearize(chunk_pos);

                    self.chunk_meshes[idx as usize] = (
                        Mesh3dBuilder::new()
                            .from_data(
                                vertices,
                                mesh.chunk_mesh.indices,
                                Some(self.texture_atlas.image.clone()),
                            )
                            .build(ctx),
                        chunk_pos.into(),
                    );
                }
            }
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

        if k_ctx.is_key_just_pressed(KeyCode::Escape) {
            ggez::input::mouse::set_cursor_hidden(ctx, !ctx.mouse.cursor_hidden());
            ggez::input::mouse::set_cursor_grabbed(ctx, !ggez::input::mouse::cursor_grabbed(ctx))?;
        }

        if ctx.mouse.cursor_hidden() {
            let mouse_delta = ctx.mouse.raw_delta();
            let speed = 0.5;
            let mouse_delta_y = mouse_delta.y as f32 * speed * dt * -1.0;
            let mouse_delta_x = mouse_delta.x as f32 * speed * dt;
            self.camera.transform.yaw += mouse_delta_x;
            self.camera.transform.pitch += mouse_delta_y;
        }

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
        let frustum = Frustum::from_view_projection(
            &Mat4::from(self.camera.to_matrix()),
            &Vec3::from(self.camera.transform.position),
            &(Quat::from_euler(
                EulerRot::XYZ,
                self.camera.transform.pitch,
                self.camera.transform.yaw,
                0.0,
            ) * Vec3::Z),
            self.camera.projection.zfar,
        );
        for chunk_mesh in self.chunk_meshes.iter() {
            let param = DrawParam3d::default().position(chunk_mesh.1.as_vec3() * CHUNK_SIZE as f32);
            if chunk_mesh.0.aabb.is_some_and(|x| {
                frustum.intersects_obb(
                    &Aabb {
                        center: (Vec3::from(x.center)).into(),
                        // half_extents: x.half_extents,
                        half_extents: Vec3::ONE.into(),
                    },
                    &Mat4::from(param.transform.to_bare_matrix()),
                )
            }) {
                canvas3d.draw(&chunk_mesh.0, param);
            }
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
            &graphics::Text::new(format!("{}", ctx.time.fps())),
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
        .window_setup(ggez::conf::WindowSetup {
            title: "Vinox Editor".to_owned(),
            samples: NumSamples::One,
            vsync: false,
            icon: "".to_owned(),
            srgb: true,
        })
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
