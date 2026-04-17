// Fragment Shader for 3D Cube
#version 450

// Input from vertex shader
layout(location = 0) in vec3 frag_color;

// Output color
layout(location = 0) out vec4 out_color;

void main() {
    // Output the interpolated vertex color
    out_color = vec4(frag_color, 1.0);
}
