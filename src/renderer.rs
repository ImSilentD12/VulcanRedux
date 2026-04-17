//! Renderer module containing Vulkan pipeline setup
//! Optimized for AMD APU (Ryzen 3500U/Vega 8) with UMA architecture

use ash::vk;
use ash::{self, Device};
use glam::{Mat4, Vec3};
use log::{debug, error, info, warn};
use std::ffi::CStr;
use std::mem;

/// Macro for creating C strings - must be defined at the top for availability
macro_rules! cstr {
    ($s:expr) => {{
        const fn check(s: &str) -> &str {
            s
        }
        unsafe { CStr::from_bytes_with_nul_unchecked(check(concat!($s, "\0")).as_bytes()) }
    }};
}

/// Maximum number of frames in flight for double buffering
const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Vertex structure matching the shader input layout
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    /// Get the vertex attribute descriptions for pipeline creation
    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            // Position attribute (location = 0)
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            // Color attribute (location = 1)
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(mem::size_of::<[f32; 3]>() as u32),
        ]
    }

    /// Get the vertex binding description
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }
}

/// Push constants structure for MVP matrix
/// Optimized for fast updates on Vega 8
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PushConstants {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
}

impl Default for PushConstants {
    fn default() -> Self {
        Self {
            model: Mat4::IDENTITY,
            view: Mat4::IDENTITY,
            proj: Mat4::IDENTITY,
        }
    }
}

/// Frame data for double buffering
/// Contains per-frame resources to minimize input lag
pub struct FrameData {
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub fence: vk::Fence,
}

impl FrameData {
    pub fn new(device: &Device) -> Result<Self, vk::Result> {
        let command_pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(0); // Graphics queue family

        unsafe {
            let command_pool = device.create_command_pool(&command_pool_create_info, None)?;

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let command_buffer =
                unsafe { device.allocate_command_buffers(&command_buffer_allocate_info)?[0] };

            let semaphore_create_info = vk::SemaphoreCreateInfo::default();
            let image_available_semaphore =
                unsafe { device.create_semaphore(&semaphore_create_info, None)? };
            let render_finished_semaphore =
                unsafe { device.create_semaphore(&semaphore_create_info, None)? };

            let fence_create_info =
                vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            let fence = unsafe { device.create_fence(&fence_create_info, None)? };

            Ok(Self {
                command_pool,
                command_buffer,
                image_available_semaphore,
                render_finished_semaphore,
                fence,
            })
        }
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_fence(self.fence, None);
            device.destroy_semaphore(self.render_finished_semaphore, None);
            device.destroy_semaphore(self.image_available_semaphore, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}

/// Memory manager using UMA principles for AMD APU
/// Uses HOST_VISIBLE | DEVICE_LOCAL flags to avoid staging copies
pub struct MemoryManager {
    device: Device,
    is_uma: bool,
}

impl MemoryManager {
    pub fn new(device: Device, physical_device: vk::PhysicalDevice, instance: &ash::Instance) -> Self {
        // Check if device supports unified memory architecture
        let mut memory_properties = vk::PhysicalDeviceMemoryProperties::default();
        unsafe {
            instance.get_physical_device_memory_properties(physical_device, &mut memory_properties);
        }

        // Detect UMA by checking for DEVICE_LOCAL + HOST_VISIBLE memory types
        let is_uma = (0..memory_properties.memory_type_count).any(|i| {
            let flags = memory_properties.memory_types[i as usize].property_flags;
            flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                && flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        });

        info!("Memory Manager initialized - UMA detected: {}", is_uma);

        Self { device, is_uma }
    }

    /// Allocate memory optimized for UMA (AMD APU)
    /// For UMA: Uses DEVICE_LOCAL | HOST_VISIBLE (zero-copy)
    /// For discrete GPU: Falls back to HOST_VISIBLE (with potential staging)
    pub fn allocate_buffer_memory(
        &self,
        buffer: vk::Buffer,
        _usage: vk::BufferUsageFlags,
    ) -> Result<vk::DeviceMemory, vk::Result> {
        let mem_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let properties = if self.is_uma {
            // UMA optimization: Single copy from host to device-local
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE
        } else {
            // Discrete GPU fallback
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        };

        let memory_type_index =
            self.find_memory_type(mem_requirements.memory_type_bits, properties)?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);

        unsafe { self.device.allocate_memory(&alloc_info, None) }
    }

    /// Find a suitable memory type with required properties
    fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, &'static str> {
        let mut mem_properties = vk::PhysicalDeviceMemoryProperties::default();
        unsafe {
            self.device
                .get_physical_device()
                .get_physical_device_memory_properties(&mut mem_properties);
        }

        for i in 0..mem_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && mem_properties.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
            {
                return Ok(i);
            }
        }

        Err("Failed to find suitable memory type")
    }

    /// Map memory for writing (UMA optimized)
    pub unsafe fn map_memory<T>(
        &self,
        memory: vk::DeviceMemory,
        size: usize,
    ) -> Result<*mut T, vk::Result> {
        self.device
            .map_memory(memory, 0, size as u64, vk::MemoryMapFlags::empty())
            .map(|ptr| ptr.as_ptr() as *mut T)
    }

    pub fn unmap_memory(&self, memory: vk::DeviceMemory) {
        unsafe {
            self.device.unmap_memory(memory);
        }
    }

    pub fn destroy_memory(&self, memory: vk::DeviceMemory) {
        unsafe {
            self.device.free_memory(memory, None);
        }
    }
}

/// Main renderer struct handling Vulkan pipeline
pub struct Renderer {
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,
    graphics_queue: vk::Queue,
    compute_queue: Option<vk::Queue>,
    surface: vk::SurfaceKHR,
    surface_format: vk::SurfaceFormatKHR,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    memory_manager: MemoryManager,
    frame_data: Vec<FrameData>,
    current_frame: usize,
    extent: vk::Extent2D,
    graphics_queue_family: u32,
}

impl Renderer {
    /// Create a new renderer instance
    pub fn new(
        window: &winit::window::Window,
        raw_handle: raw_window_handle::WindowHandle,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        info!("Initializing Vulkan Renderer...");

        // Load Vulkan entry point
        let entry = unsafe { ash::Entry::linked() };

        // Create Vulkan instance
        let instance = Self::create_instance(&entry)?;

        // Create surface
        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, raw_handle, None)?
        };

        // Select physical device (prefer integrated GPU for AMD APU)
        let (physical_device, graphics_queue_family, compute_queue_family) =
            Self::select_physical_device(&instance, surface)?;

        info!(
            "Selected physical device: {:?}",
            unsafe {
                instance
                    .get_physical_device_properties(physical_device)
                    .device_name
            }
        );

        // Create logical device with graphics and compute queues
        let (device, graphics_queue, compute_queue) = Self::create_logical_device(
            physical_device,
            graphics_queue_family,
            compute_queue_family,
        )?;

        // Create memory manager (UMA optimized)
        let memory_manager = MemoryManager::new(device.clone(), physical_device);

        // Create swapchain
        let (surface_format, extent, swapchain) =
            Self::create_swapchain(&device, surface, physical_device, graphics_queue_family)?;

        // Create image views for swapchain
        let swapchain_images = Self::create_swapchain_image_views(&device, swapchain, surface_format.format)?;

        // Create render pass using Dynamic Rendering (Vulkan 1.3+)
        let render_pass = Self::create_render_pass(&device, surface_format.format)?;

        // Create shaders from build-generated SPIR-V
        let vert_spv_path = std::path::PathBuf::from(std::env::var("OUT_DIR")?)
            .join("shaders")
            .join("vertex.spv");
        let frag_spv_path = std::path::PathBuf::from(std::env::var("OUT_DIR")?)
            .join("shaders")
            .join("fragment.spv");

        let vert_spv = std::fs::read(&vert_spv_path)
            .unwrap_or_else(|_| include_bytes!(concat!(env!("OUT_DIR"), "/shaders/vertex.spv")).to_vec());
        let frag_spv = std::fs::read(&frag_spv_path)
            .unwrap_or_else(|_| include_bytes!(concat!(env!("OUT_DIR"), "/shaders/fragment.spv")).to_vec());

        let vert_shader = Self::create_shader_module(&device, &vert_spv)?;
        let frag_shader = Self::create_shader_module(&device, &frag_spv)?;

        // Create pipeline layout with push constants
        let pipeline_layout = Self::create_pipeline_layout(&device)?;

        // Create graphics pipeline
        let pipeline = Self::create_pipeline(
            &device,
            render_pass,
            pipeline_layout,
            extent,
            vert_shader,
            frag_shader,
        )?;

        // Cleanup shader modules
        unsafe {
            device.destroy_shader_module(vert_shader, None);
            device.destroy_shader_module(frag_shader, None);
        }

        // Create vertex and index buffers (UMA optimized)
        let (vertex_buffer, vertex_buffer_memory) =
            Self::create_vertex_buffer(&device, &memory_manager)?;
        let (index_buffer, index_buffer_memory) =
            Self::create_index_buffer(&device, &memory_manager)?;

        // Create frame data for double buffering
        let mut frame_data = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            frame_data.push(FrameData::new(&device)?);
        }

        info!("Vulkan Renderer initialized successfully");

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            graphics_queue,
            compute_queue,
            surface,
            surface_format,
            swapchain,
            swapchain_images,
            render_pass,
            pipeline,
            pipeline_layout,
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            memory_manager,
            frame_data,
            current_frame: 0,
            extent,
            graphics_queue_family,
        })
    }

    /// Create Vulkan instance with required extensions
    fn create_instance(entry: &ash::Entry) -> Result<ash::Instance, vk::Result> {
        let app_info = vk::ApplicationInfo::default()
            .application_name(cstr!("Vulkan Cube Layer"))
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_3); // Use Vulkan 1.3 for dynamic rendering

        // Get required extensions for windowing
        let mut extensions = ash_window::enumerate_required_extensions(None)?.to_vec();
        extensions.push(vk::ExtDebugUtilsFn::name());
        extensions.push(vk::KhrGetPhysicalDeviceProperties2Fn::name());

        // Enable validation layers in debug builds
        #[cfg(debug_assertions)]
        let layer_names = vec![cstr!("VK_LAYER_KHRONOS_validation")];
        #[cfg(not(debug_assertions))]
        let layer_names: Vec<&CStr> = vec![];

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layer_names);

        unsafe { entry.create_instance(&instance_create_info, None) }
    }

    /// Select physical device, preferring integrated GPU (AMD APU)
    fn select_physical_device(
        instance: &ash::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<(vk::PhysicalDevice, u32, Option<u32>), Box<dyn std::error::Error>> {
        let devices = unsafe { instance.enumerate_physical_devices()? };

        if devices.is_empty() {
            return Err("No Vulkan physical devices found".into());
        }

        info!("Found {} physical device(s)", devices.len());

        // Score devices and select the best one
        let mut best_device = None;
        let mut best_score = 0i32;
        let mut best_graphics_queue = 0u32;
        let mut best_compute_queue = None;

        for device in devices {
            let props = unsafe { instance.get_physical_device_properties(device) };
            let score = match props.device_type {
                vk::PhysicalDeviceType::INTEGRATED_GPU => {
                    info!("Found integrated GPU (preferred for AMD APU): {:?}", unsafe {
                        props.device_name
                    });
                    1000 // Prefer integrated GPU
                }
                vk::PhysicalDeviceType::DISCRETE_GPU => 500,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 100,
                vk::PhysicalDeviceType::CPU => 10,
                _ => 0,
            };

            // Check queue families
            let queue_families = unsafe { device.get_physical_device_queue_family_properties() };
            let mut graphics_queue = None;
            let mut compute_queue = None;

            for (i, family) in queue_families.iter().enumerate() {
                let supports_graphics = family
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS);
                let supports_compute = family
                    .queue_flags
                    .contains(vk::QueueFlags::COMPUTE);
                let supports_present = unsafe {
                    instance.get_physical_device_surface_support(device, i as u32, surface)?
                };

                if supports_graphics && supports_present && graphics_queue.is_none() {
                    graphics_queue = Some(i as u32);
                }

                // Look for separate compute queue (for physics/culling)
                if supports_compute && !supports_graphics && compute_queue.is_none() {
                    compute_queue = Some(i as u32);
                }
            }

            if let Some(gq) = graphics_queue {
                let total_score = score + if compute_queue.is_some() { 100 } else { 0 };
                if total_score > best_score {
                    best_score = total_score;
                    best_device = Some(device);
                    best_graphics_queue = gq;
                    best_compute_queue = compute_queue;
                }
            }
        }

        best_device
            .map(|d| (d, best_graphics_queue, best_compute_queue))
            .ok_or_else(|| "No suitable physical device found".into())
    }

    /// Create logical device with graphics and compute queues
    fn create_logical_device(
        physical_device: vk::PhysicalDevice,
        graphics_queue_family: u32,
        compute_queue_family: Option<u32>,
    ) -> Result<(Device, vk::Queue, Option<vk::Queue>), vk::Result> {
        let mut queue_create_infos = Vec::new();

        // Graphics queue
        let graphics_priority = [1.0f32];
        let graphics_queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics_queue_family)
            .queue_priorities(&graphics_priority);
        queue_create_infos.push(graphics_queue_info);

        // Separate compute queue (if available)
        if let Some(compute_family) = compute_queue_family {
            if compute_family != graphics_queue_family {
                let compute_priority = [1.0f32];
                let compute_queue_info = vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(compute_family)
                    .queue_priorities(&compute_priority);
                queue_create_infos.push(compute_queue_info);
            }
        }

        // Enable features needed for dynamic rendering
        let features = vk::PhysicalDeviceFeatures::default()
            .fill_mode_non_solid(true);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&features);

        let device = unsafe { physical_device.create_device(&device_create_info, None)? };

        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family, 0) };

        let compute_queue = compute_queue_family.and_then(|compute_family| {
            if compute_family != graphics_queue_family {
                Some(unsafe { device.get_device_queue(compute_family, 0) })
            } else {
                None
            }
        });

        Ok((device, graphics_queue, compute_queue))
    }

    /// Create swapchain
    fn create_swapchain(
        device: &Device,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
        graphics_queue_family: u32,
    ) -> Result<(vk::SurfaceFormatKHR, vk::Extent2D, vk::SwapchainKHR), vk::Result>
    {
        let surface_formats =
            unsafe { device.get_physical_device_surface_formats(physical_device, surface)? };
        let surface_capabilities =
            unsafe { device.get_physical_device_surface_capabilities(physical_device, surface)? };

        // Choose surface format
        let surface_format = surface_formats
            .iter()
            .find(|fmt| {
                fmt.format == vk::Format::B8G8R8A8_SRGB
                    && fmt.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .copied()
            .unwrap_or(surface_formats[0]);

        // Choose present mode (prefer mailbox for low latency)
        let present_modes =
            unsafe { device.get_physical_device_surface_present_modes(physical_device, surface)? };
        let present_mode = present_modes
            .iter()
            .find(|mode| *mode == vk::PresentModeKHR::MAILBOX)
            .copied()
            .unwrap_or(vk::PresentModeKHR::FIFO);

        // Determine extent
        let extent = if surface_capabilities.current_extent.width != u32::MAX {
            surface_capabilities.current_extent
        } else {
            vk::Extent2D {
                width: surface_capabilities.min_image_extent.width.clamp(
                    surface_capabilities.min_image_extent.width,
                    surface_capabilities.max_image_extent.width,
                ),
                height: surface_capabilities.min_image_extent.height.clamp(
                    surface_capabilities.min_image_extent.height,
                    surface_capabilities.max_image_extent.height,
                ),
            }
        };

        // Determine image count
        let image_count = surface_capabilities.min_image_count.clamp(
            surface_capabilities.min_image_count,
            surface_capabilities.max_image_count,
        );

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .queue_family_indices(&[graphics_queue_family]);

        let swapchain = unsafe { device.create_swapchain(&create_info, None)? };

        Ok((surface_format, extent, swapchain))
    }

    /// Create image views for swapchain images
    fn create_swapchain_image_views(
        device: &Device,
        swapchain: vk::SwapchainKHR,
        format: vk::Format,
    ) -> Result<Vec<vk::ImageView>, vk::Result> {
        let images = unsafe { device.get_swapchain_images(swapchain)? };
        let mut image_views = Vec::with_capacity(images.len());

        for image in images {
            let create_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::_2D)
                .format(format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            let view = unsafe { device.create_image_view(&create_info, None)? };
            image_views.push(view);
        }

        Ok(image_views)
    }

    /// Create render pass using Dynamic Rendering (Vulkan 1.3+)
    fn create_render_pass(
        device: &Device,
        format: vk::Format,
    ) -> Result<vk::RenderPass, vk::Result> {
        // Dynamic rendering doesn't use traditional render passes
        // We return VK_NULL_HANDLE and use DynamicRenderingInfo instead
        // But for API compatibility, we create a minimal render pass
        let attachment_desc = vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&[color_attachment_ref]);

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&[attachment_desc])
            .subpasses(&[subpass])
            .dependencies(&[dependency]);

        unsafe { device.create_render_pass(&render_pass_info, None) }
    }

    /// Create shader module from SPIR-V bytes
    fn create_shader_module(
        device: &Device,
        spirv: &[u8],
    ) -> Result<vk::ShaderModule, vk::Result> {
        let create_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        unsafe { device.create_shader_module(&create_info, None) }
    }

    /// Create pipeline layout with push constants for MVP
    fn create_pipeline_layout(
        device: &Device,
    ) -> Result<vk::PipelineLayout, vk::Result> {
        // Push constant range for MVP matrix (3 x 16 floats = 192 bytes)
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(mem::size_of::<PushConstants>() as u32);

        let create_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&[push_constant_range]);

        unsafe { device.create_pipeline_layout(&create_info, None) }
    }

    /// Create graphics pipeline
    #[allow(clippy::too_many_arguments)]
    fn create_pipeline(
        device: &Device,
        render_pass: vk::RenderPass,
        pipeline_layout: vk::PipelineLayout,
        extent: vk::Extent2D,
        vert_shader: vk::ShaderModule,
        frag_shader: vk::ShaderModule,
    ) -> Result<vk::Pipeline, vk::Result> {
        let vert_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader)
            .name(cstr!("main"));

        let frag_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader)
            .name(cstr!("main"));

        let stages = [vert_stage.build(), frag_stage.build()];

        let binding_description = Vertex::get_binding_description();
        let attribute_descriptions = Vertex::get_attribute_descriptions();

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding_description))
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(extent);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(false);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .depth_stencil_state(&depth_stencil)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info.build()], None)?
                .0[0]
        };

        Ok(pipeline)
    }

    /// Create vertex buffer with cube geometry (UMA optimized)
    fn create_vertex_buffer(
        device: &Device,
        memory_manager: &MemoryManager,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn std::error::Error>> {
        // Define cube vertices with colors
        let vertices: [Vertex; 8] = [
            // Front face (red)
            Vertex { position: [-0.5, -0.5,  0.5], color: [1.0, 0.0, 0.0] },
            Vertex { position: [ 0.5, -0.5,  0.5], color: [1.0, 0.0, 0.0] },
            Vertex { position: [ 0.5,  0.5,  0.5], color: [1.0, 0.0, 0.0] },
            Vertex { position: [-0.5,  0.5,  0.5], color: [1.0, 0.0, 0.0] },
            // Back face (green)
            Vertex { position: [-0.5, -0.5, -0.5], color: [0.0, 1.0, 0.0] },
            Vertex { position: [ 0.5, -0.5, -0.5], color: [0.0, 1.0, 0.0] },
            Vertex { position: [ 0.5,  0.5, -0.5], color: [0.0, 1.0, 0.0] },
            Vertex { position: [-0.5,  0.5, -0.5], color: [0.0, 1.0, 0.0] },
        ];

        let buffer_size = mem::size_of_val(&vertices) as u64;

        // Create buffer
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_create_info, None)? };

        // Allocate UMA-optimized memory
        let memory = memory_manager.allocate_buffer_memory(
            buffer,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;

        // Upload vertex data (UMA: direct mapping, no staging)
        unsafe {
            let mapped_ptr = memory_manager.map_memory::<Vertex>(memory, buffer_size as usize)?;
            std::ptr::copy_nonoverlapping(vertices.as_ptr(), mapped_ptr, vertices.len());
            memory_manager.unmap_memory(memory);
        }

        Ok((buffer, memory))
    }

    /// Create index buffer for cube (UMA optimized)
    fn create_index_buffer(
        device: &Device,
        memory_manager: &MemoryManager,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn std::error::Error>> {
        // Define indices for cube triangles (36 indices for 12 triangles)
        let indices: [u16; 36] = [
            // Front face
            0, 1, 2, 2, 3, 0,
            // Right face
            1, 5, 6, 6, 2, 1,
            // Back face
            7, 6, 5, 5, 4, 7,
            // Left face
            4, 0, 3, 3, 7, 4,
            // Top face
            3, 2, 6, 6, 7, 3,
            // Bottom face
            4, 5, 1, 1, 0, 4,
        ];

        let buffer_size = mem::size_of_val(&indices) as u64;

        // Create buffer
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::INDEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_create_info, None)? };

        // Allocate UMA-optimized memory
        let memory = memory_manager.allocate_buffer_memory(
            buffer,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;

        // Upload index data (UMA: direct mapping, no staging)
        unsafe {
            let mapped_ptr = memory_manager.map_memory::<u16>(memory, buffer_size as usize)?;
            std::ptr::copy_nonoverlapping(indices.as_ptr(), mapped_ptr, indices.len());
            memory_manager.unmap_memory(memory);
        }

        Ok((buffer, memory))
    }

    /// Render a frame
    pub fn render(&mut self, mvp: PushConstants) -> Result<(), Box<dyn std::error::Error>> {
        let frame_data = &self.frame_data[self.current_frame];

        // Wait for previous frame to complete
        unsafe {
            self.device.wait_for_fences(&[frame_data.fence], true, u64::MAX)?;
            self.device.reset_fences(&[frame_data.fence])?;
        }

        // Acquire next swapchain image
        let (image_index, _) = unsafe {
            self.device.acquire_next_image(
                self.swapchain,
                u64::MAX,
                frame_data.image_available_semaphore,
                vk::Fence::null(),
            )?
        };

        // Record command buffer
        let cmd_begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(frame_data.command_buffer, &cmd_begin)?;

            // Set up dynamic rendering
            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(self.swapchain_images[image_index as usize])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.1, 0.1, 0.1, 1.0],
                    },
                });

            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.extent,
                })
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment));

            self.device.cmd_begin_rendering(frame_data.command_buffer, &rendering_info);

            // Bind pipeline
            self.device.cmd_bind_pipeline(
                frame_data.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            // Bind vertex and index buffers
            let vertex_offset = 0u64;
            self.device.cmd_bind_vertex_buffers(
                frame_data.command_buffer,
                0,
                &[self.vertex_buffer],
                &[vertex_offset],
            );
            self.device.cmd_bind_index_buffer(
                frame_data.command_buffer,
                self.index_buffer,
                0,
                vk::IndexType::UINT16,
            );

            // Push MVP constants (fast update for Vega 8)
            self.device.cmd_push_constants(
                frame_data.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&mvp),
            );

            // Draw indexed
            self.device.cmd_draw_indexed(frame_data.command_buffer, 36, 1, 0, 0, 0);

            self.device.cmd_end_rendering(frame_data.command_buffer);
            self.device.end_command_buffer(frame_data.command_buffer)?;
        }

        // Submit command buffer
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&[frame_data.image_available_semaphore])
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&[frame_data.command_buffer])
            .signal_semaphores(&[frame_data.render_finished_semaphore]);

        unsafe {
            self.device
                .queue_submit(self.graphics_queue, &[submit_info.build()], frame_data.fence)?;
        }

        // Present
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&[frame_data.render_finished_semaphore])
            .swapchains(&[self.swapchain])
            .image_indices(&[image_index]);

        unsafe {
            let khr = ash::khr::swapchain::Swapchain::new(&self.instance, &self.device);
            khr.queue_present(self.graphics_queue, &present_info)?;
        }

        // Advance frame counter
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    /// Get compute queue for future physics/culling tasks
    pub fn get_compute_queue(&self) -> Option<vk::Queue> {
        self.compute_queue
    }

    /// Resize swapchain on window resize
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), Box<dyn std::error::Error>> {
        // Wait for device idle
        unsafe {
            self.device.device_wait_idle()?;
        }

        // Cleanup old swapchain resources
        for view in self.swapchain_images.drain(..) {
            unsafe {
                self.device.destroy_image_view(view, None);
            }
        }
        unsafe {
            self.device.destroy_swapchain(self.swapchain, None);
        }

        // Recreate swapchain
        let (surface_format, extent, swapchain) =
            Self::create_swapchain(&self.device, self.surface, self.physical_device, self.graphics_queue_family)?;

        self.surface_format = surface_format;
        self.extent = extent;
        self.swapchain = swapchain;
        self.swapchain_images =
            Self::create_swapchain_image_views(&self.device, swapchain, surface_format.format)?;

        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();

            // Destroy frame resources
            for frame in self.frame_data.drain(..) {
                frame.destroy(&self.device);
            }

            // Destroy buffers and memory
            self.device.destroy_buffer(self.index_buffer, None);
            self.memory_manager.destroy_memory(self.index_buffer_memory);
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.memory_manager.destroy_memory(self.vertex_buffer_memory);

            // Destroy pipeline resources
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);

            // Destroy swapchain resources
            for view in self.swapchain_images.drain(..) {
                self.device.destroy_image_view(view, None);
            }
            self.device.destroy_swapchain(self.swapchain, None);

            // Destroy device and instance
            self.device.destroy_device(None);
            self.instance.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}
