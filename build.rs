//! Build script to compile GLSL shaders to SPIR-V

use std::path::Path;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = std::env::var("OUT_DIR").unwrap();
    
    let shader_dir = Path::new(&manifest_dir).join("src/shaders");
    let output_dir = Path::new(&out_dir).join("shaders");
    
    // Create output directory
    std::fs::create_dir_all(&output_dir).expect("Failed to create shader output directory");
    
    // Compile vertex shader
    compile_shader(
        &shader_dir.join("vertex.glsl"),
        &output_dir.join("vertex.spv"),
        shaderc::ShaderKind::Vertex,
    );
    
    // Compile fragment shader
    compile_shader(
        &shader_dir.join("fragment.glsl"),
        &output_dir.join("fragment.spv"),
        shaderc::ShaderKind::Fragment,
    );
    
    // Tell Cargo to rerun if shaders change
    println!("cargo:rerun-if-changed=src/shaders/vertex.glsl");
    println!("cargo:rerun-if-changed=src/shaders/fragment.glsl");
}

fn compile_shader(input: &Path, output: &Path, kind: shaderc::ShaderKind) {
    let compiler = shaderc::Compiler::new().expect("Failed to create shaderc compiler");
    
    let source = std::fs::read_to_string(input)
        .unwrap_or_else(|e| panic!("Failed to read shader {:?}: {}", input, e));
    
    let compilation_result = compiler.compile_into_spirv(
        &source,
        kind,
        input.to_str().unwrap(),
        "main",
        None,
    ).unwrap_or_else(|e| panic!("Failed to compile shader {:?}: {}", input, e));
    
    if compilation_result.get_num_warnings() > 0 {
        println!("cargo:warning={}", compilation_result.get_warning_messages());
    }
    
    let spirv = compilation_result.as_binary_u8();
    
    std::fs::write(output, spirv)
        .unwrap_or_else(|e| panic!("Failed to write SPIR-V to {:?}: {}", output, e));
    
    println!("Compiled {:?} -> {:?}", input, output);
}
