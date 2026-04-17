// Vertex Shader for 3D Cube
// Uses Push Constants for MVP matrix (optimized for AMD Vega 8)
#version 450

// Push constant block for MVP matrix
layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 view;
    mat4 proj;
} push_constants;

// Input attributes
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

// Output to fragment shader
layout(location = 0) out vec3 frag_color;

void main() {
    // Transform vertex position through MVP
    gl_Position = push_constants.proj * push_constants.view * push_constants.model * vec4(in_position, 1.0);
    
    // Pass color to fragment shader
    frag_color = in_color;
}
