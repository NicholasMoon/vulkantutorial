#version 450

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;



void main() {
    vec3 diffuseColor = vec3(texture(texSampler, fragTexCoord));
	float diffuseTerm = dot(normalize(fragNormal), normalize(vec3(1,-.5,-.5)));
	diffuseTerm = clamp(diffuseTerm, 0, 1);
	float ambientTerm = 0.01;
	float lightIntensity = diffuseTerm + ambientTerm;
	outColor = vec4(diffuseColor.rgb * lightIntensity,1);
}