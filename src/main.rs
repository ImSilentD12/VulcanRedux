//! Vulkan 3D Cube Renderer - Main Entry Point
//! 
//! A foundational Vulkan application optimized for AMD APU (Ryzen 3500U/Vega 8)
//! with fork-ready architecture for future Vector-based Culling and Smart LOD logic.
//!
//! # Architecture Overview
//! 
//! - **VulkanCore**: Handles Instance, Physical Device selection (prefers iGPU), Logical Device
//! - **MemoryManager**: UMA-optimized memory allocation (HOST_VISIBLE | DEVICE_LOCAL)
//! - **FrameData**: Double buffering to minimize input lag
//! - **Compute Queue**: Separate queue for future physics/culling tasks
//! - **Push Constants**: Fast MVP matrix updates for maximum Vega 8 performance

mod renderer;

use glam::{Mat4, Vec3};
use log::{error, info};
use raw_window_handle::HasWindowHandle;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use renderer::{PushConstants, Renderer};

/// Application state
struct App {
    window: Option<Window>,
    renderer: Option<Renderer>,
    rotation_angle: f32,
    last_time: Instant,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            rotation_angle: 0.0,
            last_time: Instant::now(),
        }
    }

    /// Calculate MVP matrix for the rotating cube
    fn calculate_mvp(&self, aspect_ratio: f32) -> PushConstants {
        // Model matrix: Rotate cube
        let model = Mat4::from_rotation_y(self.rotation_angle)
            * Mat4::from_rotation_x(self.rotation_angle * 0.5);

        // View matrix: Camera position
        let eye = Vec3::new(0.0, 0.0, 2.0);
        let target = Vec3::new(0.0, 0.0, 0.0);
        let up = Vec3::Y;
        let view = Mat4::look_at_rh(eye, target, up);

        // Projection matrix: Perspective
        let fov = 45.0_f32.to_radians();
        let near = 0.1;
        let far = 100.0;
        let proj = Mat4::perspective_rh(fov, aspect_ratio, near, far);

        PushConstants { model, view, proj }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        info!("Creating window...");
        
        let window_attributes = Window::default_attributes()
            .with_title("Vulkan Cube Layer - AMD APU Optimized")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            .with_resizable(true);

        match event_loop.create_window(window_attributes) {
            Ok(window) => {
                self.window = Some(window);
            }
            Err(e) => {
                error!("Failed to create window: {}", e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                info!("Close requested, shutting down...");
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                if physical_size.width > 0 && physical_size.height > 0 {
                    if let Some(ref mut renderer) = self.renderer {
                        if let Err(e) = renderer.resize(physical_size.width, physical_size.height) {
                            error!("Failed to resize swapchain: {}", e);
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                // Calculate MVP before borrowing renderer mutably (fixes E0502)
                let window_size = self.window.as_ref().unwrap().inner_size();
                let aspect_ratio = window_size.width as f32 / window_size.height as f32;
                let mvp = self.calculate_mvp(aspect_ratio);

                if let (Some(ref window), Some(ref mut renderer)) = (&self.window, &mut self.renderer) {
                    // Update rotation
                    let now = Instant::now();
                    let delta_time = now.duration_since(self.last_time).as_secs_f32();
                    self.last_time = now;
                    self.rotation_angle += delta_time * 0.5; // Rotate at 0.5 rad/s

                    // Render frame
                    if let Err(e) = renderer.render(mvp) {
                        error!("Render error: {}", e);
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Request redraw to animate
        if let Some(ref window) = self.window {
            window.request_redraw();
            event_loop.set_control_flow(ControlFlow::Poll);
        }
    }

    fn new_events(&mut self, event_loop: &ActiveEventLoop, _cause: winit::event::StartCause) {
        // Initialize renderer once window is available
        if self.renderer.is_none() {
            if let Some(ref window) = self.window {
                info!("Initializing renderer...");
                
                match window.window_handle() {
                    Ok(handle) => {
                        match Renderer::new(window, handle) {
                            Ok(renderer) => {
                                info!("Renderer initialized successfully");
                                self.renderer = Some(renderer);
                            }
                            Err(e) => {
                                error!("Failed to initialize renderer: {}", e);
                                event_loop.exit();
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to get window handle: {}", e);
                        event_loop.exit();
                    }
                }
            }
        }
    }
}

fn main() {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    info!("=== Vulkan Cube Layer ===");
    info!("Optimized for AMD APU (Ryzen 3500U/Vega 8)");
    info!("Features:");
    info!("  - Dynamic Rendering (Vulkan 1.3+)");
    info!("  - UMA Memory Management");
    info!("  - Double Buffering");
    info!("  - Compute Queue Support");
    info!("  - Push Constants for MVP");

    let event_loop = match EventLoop::new() {
        Ok(el) => el,
        Err(e) => {
            error!("Failed to create event loop: {}", e);
            return;
        }
    };

    let mut app = App::new();
    
    match event_loop.run_app(&mut app) {
        Ok(_) => info!("Application exited cleanly"),
        Err(e) => error!("Application error: {}", e),
    }
}
