# Vulkan Cube Layer

A foundational Vulkan 3D Cube renderer optimized for AMD APU (Ryzen 3500U/Vega 8) with fork-ready architecture.

## Features

- **Dynamic Rendering (Vulkan 1.3+)**: Avoids complex RenderPass/Framebuffer boilerplate
- **UMA Memory Management**: Uses HOST_VISIBLE | DEVICE_LOCAL flags for zero-copy buffer uploads on integrated GPUs
- **Double Buffering**: FrameData structure minimizes input lag
- **Compute Queue Support**: Separate queue ready for physics/culling tasks
- **Push Constants**: Fast MVP matrix updates optimized for Vega 8

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Application                          │
├─────────────────────────────────────────────────────────────┤
│  Renderer                                                   │
│  ├── VulkanCore (Instance, PhysicalDevice, LogicalDevice)   │
│  ├── MemoryManager (UMA-optimized allocation)               │
│  ├── FrameData (Double buffering)                           │
│  ├── Pipeline (Graphics + Compute)                          │
│  └── Swapchain                                              │
├─────────────────────────────────────────────────────────────┤
│  Shaders (GLSL → SPIR-V via shaderc)                        │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- Vulkan SDK 1.3 or higher
- Rust 1.70+
- libshaderc (for shader compilation)

### Linux Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install libshaderc-dev

# Fedora
sudo dnf install shaderc-devel

# Arch
sudo pacman -S shaderc
```

## Building

```bash
cargo build --release
```

## Running

```bash
cargo run --release
```

## Project Structure

```
.
├── Cargo.toml          # Dependencies and build configuration
├── build.rs            # Shader compilation script
├── src/
│   ├── main.rs         # Application entry point
│   ├── renderer.rs     # Vulkan pipeline setup
│   └── shaders/
│       ├── vertex.glsl # Vertex shader (MVP transform)
│       └── fragment.glsl # Fragment shader (color output)
```

## Extending

This project is designed as a "Vulkan Layer/Hook" for easy extension:

### Adding Vector-based Culling

```rust
// In renderer.rs, use the compute queue:
if let Some(compute_queue) = self.get_compute_queue() {
    // Submit culling compute commands
}
```

### Adding Smart LOD

```rust
// Modify PushConstants to include LOD parameters
pub struct PushConstants {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
    pub lod_params: Vec4, // Add LOD control
}
```

## License

MIT
