#define STB_IMAGE_IMPLEMENTATION
#define NOMINMAX
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "stb_image.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <windows.h>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <sstream>
#include <algorithm>
#include <deque>

constexpr unsigned SCR_WIDTH = 1920, SCR_HEIGHT = 1080;
constexpr unsigned SHADOW_WIDTH = 2048, SHADOW_HEIGHT = 2048;

glm::vec3 cameraPos(0, 3, 8), cameraFront(0, -0.3f, -1), cameraUp(0, 1, 0);
float deltaTime = 0, lastFrame = 0, yaw = -90, pitch = -10, lastX = SCR_WIDTH / 2.f, lastY = SCR_HEIGHT / 2.f;
bool firstMouse = true, uiMode = false;

constexpr int FPS_HISTORY = 120;
std::deque<float> fpsHistory;

std::string openFileDialog() {
	char buf[512] = {};
	OPENFILENAMEA ofn{};
	ofn.lStructSize = sizeof(ofn);
	ofn.lpstrFilter = "Images\0*.png;*.jpg;*.jpeg;*.tga;*.bmp\0All\0*.*\0";
	ofn.lpstrFile = buf;
	ofn.nMaxFile = sizeof(buf);
	ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
	return GetOpenFileNameA(&ofn) ? std::string(buf) : "";
}

const char* pbrVertSrc = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aTexCoord;
layout(location=3) in vec3 aTangent;
out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out vec4 FragPosLightSpace;
out mat3 TBN;
uniform mat4 model, view, projection, lightSpaceMatrix;
void main(){
    vec4 world = model * vec4(aPos, 1.0);
    FragPos = world.xyz;
    mat3 normalMat = mat3(transpose(inverse(model)));
    Normal = normalMat * aNormal;
    TexCoord = aTexCoord;
    FragPosLightSpace = lightSpaceMatrix * world;
    vec3 T = normalize(normalMat * aTangent);
    vec3 N = normalize(Normal);
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);
    TBN = mat3(T, B, N);
    gl_Position = projection * view * world;
})";

const char* pbrFragSrc = R"(
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
in vec4 FragPosLightSpace;
in mat3 TBN;
out vec4 FragColor;

struct DirLight   { vec3 direction, color; float intensity; };
struct PointLight { vec3 position, color; float intensity, constant, linear, quadratic; };
struct SpotLight  { vec3 position, direction, color; float intensity, cutOff, outerCutOff, constant, linear, quadratic; };

uniform DirLight   dirLight;
uniform PointLight pointLights[4];
uniform int        numPointLights;
uniform SpotLight  spotLight;
uniform bool       useSpotLight;
uniform bool       useShadows;
uniform bool       usePBRTextures;
uniform vec3       objectColor;
uniform vec3       pbrAlbedoColor;
uniform vec3       viewPos;
uniform float      specularStrength;
uniform int        shininess;
uniform float      metallic;
uniform float      roughness;
uniform sampler2D  shadowMap;
uniform sampler2D  texAlbedo;
uniform sampler2D  texNormal;
uniform sampler2D  texMetallic;
uniform sampler2D  texRoughness;
uniform sampler2D  texAO;
uniform bool       hasAlbedo, hasNormal, hasMetallic, hasRoughness, hasAO;

const float PI = 3.14159265359;

float shadowCalc(vec4 fragPosLightSpace, vec3 norm, vec3 lightDir) {
    if (!useShadows) return 0.0;
    vec3 proj = fragPosLightSpace.xyz / fragPosLightSpace.w * 0.5 + 0.5;
    if(proj.z > 1.0) return 0.0;
    float bias = max(0.005 * (1.0 - dot(norm, lightDir)), 0.001);
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; x++)
        for(int y = -1; y <= 1; y++){
            float pcf = texture(shadowMap, proj.xy + vec2(x,y) * texelSize).r;
            shadow += proj.z - bias > pcf ? 1.0 : 0.0;
        }
    return shadow / 9.0;
}

float DistributionGGX(vec3 N, vec3 H, float r) {
    float a = r * r, a2 = a * a;
    float NdH = max(dot(N, H), 0.0);
    float d = NdH * NdH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}
float GeometrySchlick(float NdV, float r) {
    float k = (r + 1.0); k = k * k / 8.0;
    return NdV / (NdV * (1.0 - k) + k);
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float r) {
    return GeometrySchlick(max(dot(N,V),0.0), r) * GeometrySchlick(max(dot(N,L),0.0), r);
}
vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 calcPBR(vec3 N, vec3 V, vec3 L, vec3 lightColor, float lightIntensity,
             vec3 albedo, float met, float rough, vec3 F0) {
    vec3 H = normalize(V + L);
    float NDF = DistributionGGX(N, H, rough);
    float G   = GeometrySmith(N, V, L, rough);
    vec3  F   = FresnelSchlick(max(dot(H, V), 0.0), F0);
    vec3  num = NDF * G * F;
    float den = 4.0 * max(dot(N,V),0.0) * max(dot(N,L),0.0) + 0.0001;
    vec3 spec = num / den;
    vec3 kD = (vec3(1.0) - F) * (1.0 - met);
    return (kD * albedo / PI + spec) * lightColor * lightIntensity * max(dot(N,L),0.0);
}

vec3 calcDirLight(vec3 norm, vec3 viewDir, vec3 albedo, float met, float rough, vec3 F0) {
    vec3 ld = normalize(-dirLight.direction);
    float shadow = shadowCalc(FragPosLightSpace, norm, ld);
    float ao = (usePBRTextures && hasAO) ? texture(texAO, TexCoord).r : 1.0;
    vec3 ambient = 0.15 * albedo * ao + 0.1 * F0;
    if (!usePBRTextures) {
        float diff = max(dot(norm, ld), 0.0);
        vec3 ref = reflect(-ld, norm);
        float spec = pow(max(dot(viewDir, ref), 0.0), shininess);
        vec3 a2 = 0.15 * dirLight.color * albedo;
        vec3 d2 = diff * dirLight.color * albedo * dirLight.intensity;
        vec3 s2 = specularStrength * spec * dirLight.color;
        return a2 + (1.0 - shadow) * (d2 + s2);
    }
    return ambient + (1.0 - shadow) * calcPBR(norm, viewDir, ld, dirLight.color, dirLight.intensity, albedo, met, rough, F0);
}

vec3 calcPointLight(PointLight light, vec3 norm, vec3 viewDir, vec3 albedo, float met, float rough, vec3 F0) {
    vec3 ld = normalize(light.position - FragPos);
    float dist = length(light.position - FragPos);
    float att = 1.0 / (light.constant + light.linear * dist + light.quadratic * dist * dist);
    if (!usePBRTextures) {
        float diff = max(dot(norm, ld), 0.0);
        vec3 ref = reflect(-ld, norm);
        float spec = pow(max(dot(viewDir, ref), 0.0), shininess);
        return att * light.intensity * (diff * light.color * albedo + specularStrength * spec * light.color);
    }
    return att * calcPBR(norm, viewDir, ld, light.color, light.intensity, albedo, met, rough, F0);
}

vec3 calcSpotLight(vec3 norm, vec3 viewDir, vec3 albedo, float met, float rough, vec3 F0) {
    vec3 ld = normalize(spotLight.position - FragPos);
    float theta = dot(ld, normalize(-spotLight.direction));
    float eps = spotLight.cutOff - spotLight.outerCutOff;
    float intensity = clamp((theta - spotLight.outerCutOff) / eps, 0.0, 1.0);
    float dist = length(spotLight.position - FragPos);
    float att = 1.0 / (spotLight.constant + spotLight.linear * dist + spotLight.quadratic * dist * dist);
    if (!usePBRTextures) {
        float diff = max(dot(norm, ld), 0.0);
        vec3 ref = reflect(-ld, norm);
        float spec = pow(max(dot(viewDir, ref), 0.0), shininess);
        return att * intensity * spotLight.intensity * (diff * spotLight.color * albedo + specularStrength * spec * spotLight.color);
    }
    return att * intensity * calcPBR(norm, viewDir, ld, spotLight.color, spotLight.intensity, albedo, met, rough, F0);
}

void main(){
    vec3 albedo = usePBRTextures && hasAlbedo
        ? pow(texture(texAlbedo, TexCoord).rgb, vec3(2.2))
        : (usePBRTextures ? pow(pbrAlbedoColor, vec3(2.2)) : objectColor);
    float met   = usePBRTextures && hasMetallic  ? texture(texMetallic,  TexCoord).r : metallic;
    float rough = usePBRTextures && hasRoughness ? texture(texRoughness, TexCoord).r : roughness;

    vec3 norm;
    if (usePBRTextures && hasNormal) {
        norm = texture(texNormal, TexCoord).rgb * 2.0 - 1.0;
        norm = normalize(TBN * norm);
    } else {
        norm = normalize(Normal);
    }

    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 F0 = mix(vec3(0.04), albedo, met);

    vec3 result = calcDirLight(norm, viewDir, albedo, met, rough, F0);
    for(int i = 0; i < numPointLights; i++)
        result += calcPointLight(pointLights[i], norm, viewDir, albedo, met, rough, F0);
    if(useSpotLight)
        result += calcSpotLight(norm, viewDir, albedo, met, rough, F0);

    if (usePBRTextures) {
        vec3 kS_ambient = FresnelSchlickRoughness(max(dot(norm, viewDir), 0.0), F0, rough);
        result += kS_ambient * 0.3;
        result = result / (result + vec3(1.0));
        result = pow(result, vec3(1.0/2.2));
    }
    FragColor = vec4(result, 1.0);
})";

const char* shadowVertSrc = R"(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 lightSpaceMatrix, model;
void main(){ gl_Position = lightSpaceMatrix * model * vec4(aPos, 1.0); })";

const char* shadowFragSrc = R"(
#version 330 core
void main(){})";

const char* outlineVertSrc = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 model, view, projection;
uniform float outlineScale;
void main(){
    gl_Position = projection * view * model * vec4(aPos + aNormal * outlineScale, 1.0);
})";

const char* outlineFragSrc = R"(
#version 330 core
out vec4 FragColor;
uniform vec3 outlineColor;
void main(){ FragColor = vec4(outlineColor, 1.0); })";

const char* gridVertSrc = R"(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 view, projection;
void main(){ gl_Position = projection * view * vec4(aPos, 1.0); })";

const char* gridFragSrc = R"(
#version 330 core
out vec4 FragColor;
uniform vec4 gridColor;
void main(){ FragColor = gridColor; })";

unsigned int compileShader(GLenum type, const char* src) {
	unsigned int s = glCreateShader(type);
	glShaderSource(s, 1, &src, nullptr);
	glCompileShader(s);
	int ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
	if (!ok) { char log[512]; glGetShaderInfoLog(s, 512, nullptr, log); std::cerr << log; }
	return s;
}

unsigned int linkProgram(const char* vs, const char* fs) {
	unsigned int v = compileShader(GL_VERTEX_SHADER, vs);
	unsigned int f = compileShader(GL_FRAGMENT_SHADER, fs);
	unsigned int p = glCreateProgram();
	glAttachShader(p, v); glAttachShader(p, f);
	glLinkProgram(p);
	glDeleteShader(v); glDeleteShader(f);
	return p;
}

struct UniformCache {
	unsigned int prog;
	std::unordered_map<std::string, GLint> locs;
	GLint get(const std::string& name) {
		auto it = locs.find(name);
		if (it != locs.end()) return it->second;
		GLint loc = glGetUniformLocation(prog, name.c_str());
		locs[name] = loc;
		return loc;
	}
	void set1i(const std::string& n, int v) { glUniform1i(get(n), v); }
	void set1f(const std::string& n, float v) { glUniform1f(get(n), v); }
	void set3fv(const std::string& n, const float* v) { glUniform3fv(get(n), 1, v); }
	void setMat4(const std::string& n, const float* v) { glUniformMatrix4fv(get(n), 1, GL_FALSE, v); }
};

void framebuffer_size_callback(GLFWwindow*, int w, int h) { glViewport(0, 0, w, h); }

void mouse_callback(GLFWwindow*, double xpos, double ypos) {
	if (uiMode) return;
	if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
	float xo = (xpos - lastX) * 0.1f, yo = (lastY - ypos) * 0.1f;
	lastX = xpos; lastY = ypos;
	yaw += xo; pitch += yo;
	pitch = glm::clamp(pitch, -89.f, 89.f);
	cameraFront = glm::normalize(glm::vec3(
		cos(glm::radians(yaw)) * cos(glm::radians(pitch)),
		sin(glm::radians(pitch)),
		sin(glm::radians(yaw)) * cos(glm::radians(pitch))
	));
}

struct Vertex { float x, y, z, nx, ny, nz, u, v, tx, ty, tz; };

Vertex cubeVerts[] = {
	{-0.5f,-0.5f,-0.5f, 0,0,-1, 0,0, 1,0,0}, { 0.5f,-0.5f,-0.5f, 0,0,-1, 1,0, 1,0,0}, { 0.5f, 0.5f,-0.5f, 0,0,-1, 1,1, 1,0,0},
	{ 0.5f, 0.5f,-0.5f, 0,0,-1, 1,1, 1,0,0}, {-0.5f, 0.5f,-0.5f, 0,0,-1, 0,1, 1,0,0}, {-0.5f,-0.5f,-0.5f, 0,0,-1, 0,0, 1,0,0},
	{-0.5f,-0.5f, 0.5f, 0,0, 1, 0,0,-1,0,0}, { 0.5f,-0.5f, 0.5f, 0,0, 1, 1,0,-1,0,0}, { 0.5f, 0.5f, 0.5f, 0,0, 1, 1,1,-1,0,0},
	{ 0.5f, 0.5f, 0.5f, 0,0, 1, 1,1,-1,0,0}, {-0.5f, 0.5f, 0.5f, 0,0, 1, 0,1,-1,0,0}, {-0.5f,-0.5f, 0.5f, 0,0, 1, 0,0,-1,0,0},
	{-0.5f, 0.5f, 0.5f,-1,0, 0, 1,0, 0,0,1}, {-0.5f, 0.5f,-0.5f,-1,0, 0, 1,1, 0,0,1}, {-0.5f,-0.5f,-0.5f,-1,0, 0, 0,1, 0,0,1},
	{-0.5f,-0.5f,-0.5f,-1,0, 0, 0,1, 0,0,1}, {-0.5f,-0.5f, 0.5f,-1,0, 0, 0,0, 0,0,1}, {-0.5f, 0.5f, 0.5f,-1,0, 0, 1,0, 0,0,1},
	{ 0.5f, 0.5f, 0.5f, 1,0, 0, 1,0, 0,0,-1},{ 0.5f, 0.5f,-0.5f, 1,0, 0, 1,1, 0,0,-1},{ 0.5f,-0.5f,-0.5f, 1,0, 0, 0,1, 0,0,-1},
	{ 0.5f,-0.5f,-0.5f, 1,0, 0, 0,1, 0,0,-1},{ 0.5f,-0.5f, 0.5f, 1,0, 0, 0,0, 0,0,-1},{ 0.5f, 0.5f, 0.5f, 1,0, 0, 1,0, 0,0,-1},
	{-0.5f,-0.5f,-0.5f, 0,-1,0, 0,1, 1,0,0}, { 0.5f,-0.5f,-0.5f, 0,-1,0, 1,1, 1,0,0}, { 0.5f,-0.5f, 0.5f, 0,-1,0, 1,0, 1,0,0},
	{ 0.5f,-0.5f, 0.5f, 0,-1,0, 1,0, 1,0,0}, {-0.5f,-0.5f, 0.5f, 0,-1,0, 0,0, 1,0,0}, {-0.5f,-0.5f,-0.5f, 0,-1,0, 0,1, 1,0,0},
	{-0.5f, 0.5f,-0.5f, 0, 1,0, 0,1, 1,0,0}, { 0.5f, 0.5f,-0.5f, 0, 1,0, 1,1, 1,0,0}, { 0.5f, 0.5f, 0.5f, 0, 1,0, 1,0, 1,0,0},
	{ 0.5f, 0.5f, 0.5f, 0, 1,0, 1,0, 1,0,0}, {-0.5f, 0.5f, 0.5f, 0, 1,0, 0,0, 1,0,0}, {-0.5f, 0.5f,-0.5f, 0, 1,0, 0,1, 1,0,0},
};

struct ShadowMap { unsigned int fbo, texture; };

ShadowMap createShadowMap() {
	ShadowMap sm;
	glGenFramebuffers(1, &sm.fbo);
	glGenTextures(1, &sm.texture);
	glBindTexture(GL_TEXTURE_2D, sm.texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	float border[] = { 1,1,1,1 };
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border);
	glBindFramebuffer(GL_FRAMEBUFFER, sm.fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, sm.texture, 0);
	glDrawBuffer(GL_NONE); glReadBuffer(GL_NONE);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	return sm;
}

struct ViewportFBO { unsigned int fbo, colorTex, rbo; int width, height; };

ViewportFBO createViewportFBO(int w, int h) {
	ViewportFBO vf{ 0, 0, 0, w, h };
	glGenFramebuffers(1, &vf.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, vf.fbo);
	glGenTextures(1, &vf.colorTex);
	glBindTexture(GL_TEXTURE_2D, vf.colorTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, vf.colorTex, 0);
	glGenRenderbuffers(1, &vf.rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, vf.rbo);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, vf.rbo);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	return vf;
}

void resizeViewportFBO(ViewportFBO& vf, int w, int h) {
	if (vf.width == w && vf.height == h) return;
	vf.width = w; vf.height = h;
	glBindTexture(GL_TEXTURE_2D, vf.colorTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
	glBindRenderbuffer(GL_RENDERBUFFER, vf.rbo);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
}

unsigned int loadTexture(const char* path) {
	unsigned int id;
	glGenTextures(1, &id);
	int w, h, ch;
	stbi_set_flip_vertically_on_load(true);
	unsigned char* data = stbi_load(path, &w, &h, &ch, 0);
	if (data) {
		GLenum internalFmt = ch == 1 ? GL_R8 : ch == 3 ? GL_RGB : GL_RGBA;
		GLenum fmt = ch == 1 ? GL_RED : ch == 3 ? GL_RGB : GL_RGBA;
		glBindTexture(GL_TEXTURE_2D, id);
		glTexImage2D(GL_TEXTURE_2D, 0, internalFmt, w, h, 0, fmt, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}
	else {
		std::cerr << "Failed to load texture: " << path << "\n";
	}
	stbi_image_free(data);
	return id;
}

struct PBRMaterial {
	unsigned int albedo = 0, normal = 0, metallic = 0, roughness = 0, ao = 0;
	bool hasAlbedo = false, hasNormal = false, hasMetallic = false, hasRoughness = false, hasAO = false;
	char pathAlbedo[512] = "", pathNormal[512] = "", pathMetallic[512] = "", pathRoughness[512] = "", pathAO[512] = "";

	void loadMap(unsigned int& tex, bool& has, const char* path) {
		if (tex) glDeleteTextures(1, &tex);
		tex = loadTexture(path);
		has = true;
	}
	void clearMap(unsigned int& tex, bool& has, char* pathBuf) {
		if (tex) { glDeleteTextures(1, &tex); tex = 0; }
		has = false; pathBuf[0] = '\0';
	}
};

struct SceneObject {
	std::string name;
	glm::vec3 position{ 0 }, scale{ 1 };
	glm::vec3 color{ 0.8f, 0.3f, 0.02f };
	glm::vec3 pbrColor{ 0.5f, 0.5f, 0.5f };
	bool visible = true;
	bool showOutline = true;
	float outlineScale = 0.04f;
	glm::vec3 outlineColor{ 1.0f, 0.6f, 0.0f };
	std::string folder = "";
	bool usePBR = false;
	float metallic = 0.0f, roughness = 0.5f;
	PBRMaterial pbr;
	bool locked = false;
};

struct SceneFolder { std::string name; bool open = true; };

struct PointLightData {
	glm::vec3 position{ 0, 2, 0 };
	glm::vec3 color{ 1, 1, 1 };
	float intensity = 1.0f;
	float constant = 1.0f, linear = 0.09f, quadratic = 0.032f;
	bool enabled = true;
};

struct TerminalLine { std::string text; ImVec4 color; };

struct Command {
	int objectIndex;
	glm::vec3 prevPos, nextPos;
	glm::vec3 prevScale, nextScale;
};

enum class TransformMode { Translate, Rotate, Scale };
TransformMode transformMode = TransformMode::Translate;

enum class CameraSnap { Free, Top, Front, Right, Iso };

std::vector<SceneObject> objects;
std::vector<Command> undoStack, redoStack;

void applyUndo() {
	if (undoStack.empty()) return;
	auto& cmd = undoStack.back();
	objects[cmd.objectIndex].position = cmd.prevPos;
	objects[cmd.objectIndex].scale = cmd.prevScale;
	redoStack.push_back(cmd); undoStack.pop_back();
}

void applyRedo() {
	if (redoStack.empty()) return;
	auto& cmd = redoStack.back();
	objects[cmd.objectIndex].position = cmd.nextPos;
	objects[cmd.objectIndex].scale = cmd.nextScale;
	undoStack.push_back(cmd); redoStack.pop_back();
}

unsigned int buildVAO(const void* data, size_t size) {
	unsigned int VAO, VBO;
	glGenVertexArrays(1, &VAO); glGenBuffers(1, &VBO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, x));  glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, nx)); glEnableVertexAttribArray(1);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, u));  glEnableVertexAttribArray(2);
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tx)); glEnableVertexAttribArray(3);
	return VAO;
}

unsigned int buildGridVAO(std::vector<float>& out) {
	const int G = 20;
	for (int i = -G; i <= G; i++) {
		float f = (float)i;
		out.insert(out.end(), { f, 0.f, -(float)G, f, 0.f, (float)G });
		out.insert(out.end(), { -(float)G, 0.f, f, (float)G, 0.f, f });
	}
	unsigned int VAO, VBO;
	glGenVertexArrays(1, &VAO); glGenBuffers(1, &VBO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, out.size() * sizeof(float), out.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);
	return VAO;
}

float rayAABB(glm::vec3 ro, glm::vec3 rd, glm::vec3 boxMin, glm::vec3 boxMax) {
	glm::vec3 invD = 1.0f / rd;
	glm::vec3 t0 = (boxMin - ro) * invD, t1 = (boxMax - ro) * invD;
	glm::vec3 tMin = glm::min(t0, t1), tMax = glm::max(t0, t1);
	float tNear = glm::max(glm::max(tMin.x, tMin.y), tMin.z);
	float tFar = glm::min(glm::min(tMax.x, tMax.y), tMax.z);
	if (tFar < 0 || tNear > tFar) return -1.f;
	return tNear;
}

int pickObject(const std::vector<SceneObject>& objs, glm::vec2 mouseVP, int vpW, int vpH,
	glm::mat4 view, glm::mat4 proj) {
	float ndcX = (2.f * mouseVP.x) / vpW - 1.f;
	float ndcY = -(2.f * mouseVP.y) / vpH + 1.f;
	glm::vec4 eye = glm::inverse(proj) * glm::vec4(ndcX, ndcY, -1.f, 1.f);
	eye = glm::vec4(eye.x / eye.w, eye.y / eye.w, -1.f, 0.f);
	glm::vec3 worldDir = glm::normalize(glm::vec3(glm::inverse(view) * eye));
	int best = -1; float bestT = 1e9f;
	for (int i = 0; i < (int)objs.size(); i++) {
		if (!objs[i].visible) continue;
		glm::vec3 half = objs[i].scale * 0.5f;
		float t = rayAABB(cameraPos, worldDir, objs[i].position - half, objs[i].position + half);
		if (t > 0.f && t < bestT) { bestT = t; best = i; }
	}
	return best;
}

void applyCameraSnap(CameraSnap snap) {
	switch (snap) {
	case CameraSnap::Top:
		cameraPos = { 0, 12, 0 };
		cameraFront = { 0, -1, -0.001f };
		pitch = -89.f; yaw = -90.f;
		break;
	case CameraSnap::Front:
		cameraPos = { 0, 3, 10 };
		cameraFront = { 0, 0, -1 };
		pitch = 0.f; yaw = -90.f;
		break;
	case CameraSnap::Right:
		cameraPos = { 10, 3, 0 };
		cameraFront = { -1, 0, 0 };
		pitch = 0.f; yaw = 180.f;
		break;
	case CameraSnap::Iso:
		cameraPos = { 7, 7, 7 };
		cameraFront = glm::normalize(-cameraPos);
		pitch = -35.f; yaw = -135.f;
		break;
	default: break;
	}
	firstMouse = true;
}

void processInput(GLFWwindow* w, int& selectedObject) {
	static bool tabPressed = false, zPressed = false, yPressed = false;
	static bool delPressed = false;
	static bool f1 = false, f2 = false, f3 = false;

	if (glfwGetKey(w, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(w, true);
	if (glfwGetKey(w, GLFW_KEY_TAB) == GLFW_PRESS && !tabPressed) {
		uiMode = !uiMode;
		glfwSetInputMode(w, GLFW_CURSOR, uiMode ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
		firstMouse = true; tabPressed = true;
	}
	if (glfwGetKey(w, GLFW_KEY_TAB) == GLFW_RELEASE) tabPressed = false;

	bool ctrl = glfwGetKey(w, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(w, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
	if (ctrl && glfwGetKey(w, GLFW_KEY_Z) == GLFW_PRESS && !zPressed) { applyUndo(); zPressed = true; }
	if (glfwGetKey(w, GLFW_KEY_Z) == GLFW_RELEASE) zPressed = false;
	if (ctrl && glfwGetKey(w, GLFW_KEY_Y) == GLFW_PRESS && !yPressed) { applyRedo(); yPressed = true; }
	if (glfwGetKey(w, GLFW_KEY_Y) == GLFW_RELEASE) yPressed = false;

	if (glfwGetKey(w, GLFW_KEY_G) == GLFW_PRESS && !f1) { transformMode = TransformMode::Translate; f1 = true; }
	if (glfwGetKey(w, GLFW_KEY_G) == GLFW_RELEASE) f1 = false;
	if (glfwGetKey(w, GLFW_KEY_R) == GLFW_PRESS && !f2) { transformMode = TransformMode::Rotate; f2 = true; }
	if (glfwGetKey(w, GLFW_KEY_R) == GLFW_RELEASE) f2 = false;
	if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS && !f3) { transformMode = TransformMode::Scale; f3 = true; }
	if (glfwGetKey(w, GLFW_KEY_S) == GLFW_RELEASE) f3 = false;

	if (glfwGetKey(w, GLFW_KEY_DELETE) == GLFW_PRESS && !delPressed && !uiMode) {
		if (selectedObject >= 0 && selectedObject < (int)objects.size() && !objects[selectedObject].locked) {
			objects.erase(objects.begin() + selectedObject);
			selectedObject = std::max(0, selectedObject - 1);
		}
		delPressed = true;
	}
	if (glfwGetKey(w, GLFW_KEY_DELETE) == GLFW_RELEASE) delPressed = false;

	if (uiMode) return;
	float s = 5.0f * deltaTime;
	if (glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) s *= 3.f;
	if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) cameraPos += s * cameraFront;
	if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) cameraPos -= s * cameraFront;
	if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * s;
	if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * s;
	if (glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS) cameraPos -= cameraUp * s;
	if (glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS) cameraPos += cameraUp * s;
}

void drawTextureSlot(const char* label, unsigned int tex, bool has,
	char* pathBuf, size_t bufSize,
	std::function<void(const char*)> onLoad,
	std::function<void()> onClear)
{
	ImGui::PushID(label);
	if (has) {
		ImGui::Image((ImTextureID)(intptr_t)tex, { 40, 40 });
		ImGui::SameLine();
		ImGui::BeginGroup();
		ImGui::TextUnformatted(label);
		if (ImGui::SmallButton("x")) onClear();
		ImGui::EndGroup();
	}
	else {
		ImGui::Text("%s", label); ImGui::SameLine(); ImGui::TextDisabled("(none)");
	}
	float btnW = 26.f;
	ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - btnW - 4);
	if (ImGui::InputText("##path", pathBuf, bufSize, ImGuiInputTextFlags_EnterReturnsTrue))
		onLoad(pathBuf);
	ImGui::SameLine();
	if (ImGui::Button("...")) {
		std::string picked = openFileDialog();
		if (!picked.empty()) {
			strncpy_s(pathBuf, bufSize, picked.c_str(), bufSize - 1);
			onLoad(pathBuf);
		}
	}
	ImGui::PopID();
}

void drawTransformModeBar() {
	auto modeBtn = [](const char* label, TransformMode mode, ImVec4 activeCol) {
		bool active = (transformMode == mode);
		if (active) ImGui::PushStyleColor(ImGuiCol_Button, activeCol);
		if (ImGui::Button(label, { 58, 22 })) transformMode = mode;
		if (active) ImGui::PopStyleColor();
		ImGui::SameLine(0, 2);
		};
	modeBtn("  G Move", TransformMode::Translate, { 0.2f, 0.6f, 0.95f, 1.f });
	modeBtn("  R Rot", TransformMode::Rotate, { 0.9f, 0.55f, 0.1f, 1.f });
	modeBtn("  S Scale", TransformMode::Scale, { 0.2f, 0.85f, 0.4f, 1.f });
}

void drawViewportOverlay(float fps, int objCount, int vpW, int vpH) {
	ImGui::SetNextWindowBgAlpha(0.55f);
	ImGui::SetNextWindowPos({ ImGui::GetWindowPos().x + 8, ImGui::GetWindowPos().y + 36 }, ImGuiCond_Always);
	ImGui::SetNextWindowSize({ 180, 0 }, ImGuiCond_Always);
	ImGui::Begin("##vpstats", nullptr,
		ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
		ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove |
		ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
	ImGui::TextColored({ 0.4f, 1.f, 0.5f, 1.f }, "%.1f FPS", fps);
	ImGui::Text("Objects : %d", objCount);
	ImGui::Text("Viewport: %dx%d", vpW, vpH);
	ImGui::Text("Cam: %.1f %.1f %.1f", cameraPos.x, cameraPos.y, cameraPos.z);
	ImGui::End();
}

void drawStatsPanel(float fps) {
	fpsHistory.push_back(fps);
	while ((int)fpsHistory.size() > FPS_HISTORY) fpsHistory.pop_front();

	std::vector<float> buf(fpsHistory.begin(), fpsHistory.end());
	float maxFPS = *std::max_element(buf.begin(), buf.end());
	char overlay[32]; snprintf(overlay, sizeof(overlay), "%.0f fps", fps);
	ImGui::PlotLines("##fps", buf.data(), (int)buf.size(), 0, overlay, 0.f, maxFPS + 10.f, { ImGui::GetContentRegionAvail().x, 55 });
	ImGui::Text("Min: %.0f  Max: %.0f  Avg: %.0f",
		*std::min_element(buf.begin(), buf.end()), maxFPS,
		[&] { float s = 0; for (auto v : buf) s += v; return s / buf.size(); }());
}

std::string executeCommand(const std::string& raw, int& selectedObject,
	std::vector<TerminalLine>& lines, bool& scroll)
{
	std::istringstream ss(raw);
	std::string cmd; ss >> cmd;
	std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::tolower);

	auto ok = [&](const std::string& msg) { lines.push_back({ "  " + msg, {0.4f,1.f,0.5f,1.f} }); scroll = true; };
	auto err = [&](const std::string& msg) { lines.push_back({ "  ERROR: " + msg, {1.f,0.35f,0.35f,1.f} }); scroll = true; };

	if (cmd == "help") {
		ok("Commands: add, remove, select, rename, list, move, scale, color, reset, clear, hide, show, lock, unlock, focus");
		return "";
	}
	if (cmd == "list") {
		for (int i = 0; i < (int)objects.size(); i++)
			ok("[" + std::to_string(i) + "] " + objects[i].name + (objects[i].locked ? " (locked)" : ""));
		return "";
	}
	if (cmd == "add") {
		std::string name; ss >> name;
		if (name.empty()) name = "Cube_" + std::to_string(objects.size());
		objects.push_back({ name, {0, 0.5f, 0}, {1,1,1}, {0.5f,0.5f,0.8f} });
		selectedObject = (int)objects.size() - 1;
		ok("Added: " + name);
		return "";
	}
	if (cmd == "remove") {
		int idx = selectedObject;
		ss >> idx;
		if (idx < 0 || idx >= (int)objects.size()) { err("Invalid index"); return ""; }
		if (objects[idx].locked) { err("Object is locked"); return ""; }
		ok("Removed: " + objects[idx].name);
		objects.erase(objects.begin() + idx);
		selectedObject = std::max(0, idx - 1);
		return "";
	}
	if (cmd == "select") {
		int idx; if (!(ss >> idx)) { err("Usage: select <index>"); return ""; }
		if (idx < 0 || idx >= (int)objects.size()) { err("Invalid index"); return ""; }
		selectedObject = idx;
		ok("Selected: " + objects[idx].name);
		return "";
	}
	if (cmd == "rename") {
		std::string name; ss >> name;
		if (name.empty()) { err("Usage: rename <n>"); return ""; }
		objects[selectedObject].name = name;
		ok("Renamed to: " + name);
		return "";
	}
	if (cmd == "move") {
		float x, y, z;
		if (!(ss >> x >> y >> z)) { err("Usage: move <x> <y> <z>"); return ""; }
		objects[selectedObject].position = { x, y, z };
		ok("Moved to " + std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(z));
		return "";
	}
	if (cmd == "scale") {
		float x, y, z;
		if (!(ss >> x >> y >> z)) { err("Usage: scale <x> <y> <z>"); return ""; }
		objects[selectedObject].scale = { x, y, z };
		ok("Scaled to " + std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(z));
		return "";
	}
	if (cmd == "color") {
		float r, g, b;
		if (!(ss >> r >> g >> b)) { err("Usage: color <r> <g> <b>  (0..1)"); return ""; }
		objects[selectedObject].color = { r, g, b };
		ok("Color set");
		return "";
	}
	if (cmd == "reset") {
		objects[selectedObject].position = { 0, 0.5f, 0 };
		objects[selectedObject].scale = { 1, 1, 1 };
		ok("Transform reset");
		return "";
	}
	if (cmd == "clear") {
		lines.clear(); scroll = true;
		return "";
	}
	if (cmd == "hide") {
		objects[selectedObject].visible = false;
		ok("Hidden: " + objects[selectedObject].name);
		return "";
	}
	if (cmd == "show") {
		objects[selectedObject].visible = true;
		ok("Shown: " + objects[selectedObject].name);
		return "";
	}
	if (cmd == "lock") {
		objects[selectedObject].locked = true;
		ok("Locked: " + objects[selectedObject].name);
		return "";
	}
	if (cmd == "unlock") {
		objects[selectedObject].locked = false;
		ok("Unlocked: " + objects[selectedObject].name);
		return "";
	}
	if (cmd == "focus") {
		if (!objects.empty()) {
			cameraPos = objects[selectedObject].position + glm::vec3(3, 2, 3);
			cameraFront = glm::normalize(objects[selectedObject].position - cameraPos);
		}
		ok("Focused on: " + objects[selectedObject].name);
		return "";
	}
	err("Unknown command '" + cmd + "'. Type 'help' for a list.");
	return "";
}

void applyDarkStyle() {
	ImGuiStyle& s = ImGui::GetStyle();
	s.WindowRounding = 4.f;
	s.FrameRounding = 3.f;
	s.GrabRounding = 3.f;
	s.ChildRounding = 3.f;
	s.PopupRounding = 3.f;
	s.TabRounding = 3.f;
	s.FramePadding = { 5, 4 };
	s.ItemSpacing = { 6, 4 };
	s.WindowPadding = { 8, 8 };
	s.ScrollbarSize = 10.f;
	s.GrabMinSize = 8.f;
	s.WindowBorderSize = 0.f;

	ImVec4* c = s.Colors;
	c[ImGuiCol_WindowBg] = { 0.10f, 0.11f, 0.14f, 1.f };
	c[ImGuiCol_ChildBg] = { 0.08f, 0.09f, 0.11f, 1.f };
	c[ImGuiCol_PopupBg] = { 0.10f, 0.11f, 0.14f, 0.98f };
	c[ImGuiCol_Header] = { 0.18f, 0.20f, 0.27f, 1.f };
	c[ImGuiCol_HeaderHovered] = { 0.26f, 0.29f, 0.40f, 1.f };
	c[ImGuiCol_HeaderActive] = { 0.22f, 0.25f, 0.36f, 1.f };
	c[ImGuiCol_FrameBg] = { 0.14f, 0.15f, 0.19f, 1.f };
	c[ImGuiCol_FrameBgHovered] = { 0.18f, 0.20f, 0.27f, 1.f };
	c[ImGuiCol_FrameBgActive] = { 0.20f, 0.23f, 0.31f, 1.f };
	c[ImGuiCol_Button] = { 0.18f, 0.20f, 0.28f, 1.f };
	c[ImGuiCol_ButtonHovered] = { 0.26f, 0.30f, 0.45f, 1.f };
	c[ImGuiCol_ButtonActive] = { 0.22f, 0.26f, 0.40f, 1.f };
	c[ImGuiCol_TitleBg] = { 0.07f, 0.08f, 0.10f, 1.f };
	c[ImGuiCol_TitleBgActive] = { 0.08f, 0.09f, 0.12f, 1.f };
	c[ImGuiCol_Tab] = { 0.10f, 0.11f, 0.14f, 1.f };
	c[ImGuiCol_TabHovered] = { 0.26f, 0.30f, 0.45f, 1.f };
	c[ImGuiCol_TabActive] = { 0.18f, 0.22f, 0.36f, 1.f };
	c[ImGuiCol_SliderGrab] = { 0.35f, 0.50f, 0.95f, 1.f };
	c[ImGuiCol_SliderGrabActive] = { 0.50f, 0.65f, 1.00f, 1.f };
	c[ImGuiCol_CheckMark] = { 0.45f, 0.75f, 1.00f, 1.f };
	c[ImGuiCol_Separator] = { 0.20f, 0.22f, 0.30f, 1.f };
	c[ImGuiCol_ScrollbarBg] = { 0.06f, 0.07f, 0.09f, 1.f };
	c[ImGuiCol_ScrollbarGrab] = { 0.20f, 0.22f, 0.30f, 1.f };
	c[ImGuiCol_Text] = { 0.88f, 0.89f, 0.92f, 1.f };
	c[ImGuiCol_TextDisabled] = { 0.40f, 0.42f, 0.50f, 1.f };
}

int main() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Valgut Engine", nullptr, nullptr);
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_STENCIL_TEST);

	unsigned int pbrProg = linkProgram(pbrVertSrc, pbrFragSrc);
	unsigned int shadowProg = linkProgram(shadowVertSrc, shadowFragSrc);
	unsigned int outlineProg = linkProgram(outlineVertSrc, outlineFragSrc);
	unsigned int gridProg = linkProgram(gridVertSrc, gridFragSrc);

	UniformCache pbrU{ pbrProg };

	unsigned int cubeVAO = buildVAO(cubeVerts, sizeof(cubeVerts));
	std::vector<float> gridData;
	unsigned int gridVAO = buildGridVAO(gridData);

	ShadowMap   shadowMap = createShadowMap();
	ViewportFBO viewport = createViewportFBO(1280, 720);

	objects = {
		{"Cube_0", {0,  0.5f, 0}, {1,1,1}, {0.8f, 0.3f, 0.02f}},
		{"Cube_1", {3,  0.5f, 0}, {1,1,1}, {0.2f, 0.5f, 0.9f}},
		{"Cube_2", {-3, 0.5f, 1}, {1,1,1}, {0.1f, 0.8f, 0.3f}},
	};
	int selectedObject = 0;

	std::vector<SceneFolder> folders;
	bool showNewFolderPopup = false;
	char newFolderName[64] = "";

	std::vector<TerminalLine> terminalLines;
	terminalLines.push_back({ "> Valgut Engine started. Type 'help' for commands.", {0.4f, 1.0f, 0.4f, 1.0f} });
	char terminalInput[256] = "";
	bool terminalScrollToBottom = false;
	float terminalHeight = 200.f;

	std::vector<PointLightData> pointLights = {
		{{2,  3,  2}, {1.0f, 0.8f, 0.6f}, 2.0f},
		{{-3, 2, -1}, {0.4f, 0.6f, 1.0f}, 1.5f},
	};

	glm::vec3 dirLightDir(-0.5f, -1.0f, -0.5f);
	glm::vec3 dirLightColor(1.0f, 0.98f, 0.9f);
	float     dirLightIntensity = 0.8f;

	bool      useSpotLight = false;
	float     spotCutoff = 12.5f, spotOuterCutoff = 17.5f;
	glm::vec3 spotColor(1.0f, 1.0f, 0.8f);
	float     spotIntensity = 2.0f;

	float specularStrength = 0.5f;
	int   shininess = 32;
	bool  useShadows = true;
	bool  showStatsPanel = true;

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::GetStyle().ScaleAllSizes(1.15f);
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init();
	applyDarkStyle();

	glm::vec3 dragStartPos(0), dragStartScale(1);

	std::vector<std::string> cmdHistory;
	int cmdHistoryIdx = -1;

	while (!glfwWindowShouldClose(window)) {
		float t = (float)glfwGetTime();
		deltaTime = t - lastFrame; lastFrame = t;
		float fps = deltaTime > 0.f ? 1.f / deltaTime : 0.f;

		processInput(window, selectedObject);

		glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
		glm::mat4 proj = glm::perspective(glm::radians(45.f), (float)viewport.width / viewport.height, 0.1f, 200.f);

		glm::vec3 lightOrigin(-5, 10, -5);
		glm::mat4 lightProj = glm::ortho(-12.f, 12.f, -12.f, 12.f, 1.f, 40.f);
		glm::mat4 lightView = glm::lookAt(lightOrigin, glm::vec3(0), glm::vec3(0, 1, 0));
		glm::mat4 lightSpace = lightProj * lightView;

		glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
		glBindFramebuffer(GL_FRAMEBUFFER, shadowMap.fbo);
		glClear(GL_DEPTH_BUFFER_BIT);
		glUseProgram(shadowProg);
		glUniformMatrix4fv(glGetUniformLocation(shadowProg, "lightSpaceMatrix"), 1, GL_FALSE, glm::value_ptr(lightSpace));
		for (auto& obj : objects) {
			if (!obj.visible) continue;
			glm::mat4 model = glm::scale(glm::translate(glm::mat4(1), obj.position), obj.scale);
			glUniformMatrix4fv(glGetUniformLocation(shadowProg, "model"), 1, GL_FALSE, glm::value_ptr(model));
			glBindVertexArray(cubeVAO);
			glDrawArrays(GL_TRIANGLES, 0, 36);
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		resizeViewportFBO(viewport, viewport.width, viewport.height);
		glViewport(0, 0, viewport.width, viewport.height);
		glBindFramebuffer(GL_FRAMEBUFFER, viewport.fbo);
		glClearColor(0.08f, 0.10f, 0.14f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		glUseProgram(gridProg);
		glUniformMatrix4fv(glGetUniformLocation(gridProg, "view"), 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(glGetUniformLocation(gridProg, "projection"), 1, GL_FALSE, glm::value_ptr(proj));
		glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glUniform4f(glGetUniformLocation(gridProg, "gridColor"), 0.4f, 0.4f, 0.5f, 0.5f);
		glBindVertexArray(gridVAO);
		glDrawArrays(GL_LINES, 0, (int)(gridData.size() / 3));
		glDisable(GL_BLEND);

		glUseProgram(pbrProg);
		pbrU.setMat4("view", glm::value_ptr(view));
		pbrU.setMat4("projection", glm::value_ptr(proj));
		pbrU.setMat4("lightSpaceMatrix", glm::value_ptr(lightSpace));
		pbrU.set3fv("viewPos", glm::value_ptr(cameraPos));
		pbrU.set1i("useShadows", useShadows);
		pbrU.set3fv("dirLight.direction", glm::value_ptr(dirLightDir));
		pbrU.set3fv("dirLight.color", glm::value_ptr(dirLightColor));
		pbrU.set1f("dirLight.intensity", dirLightIntensity);
		pbrU.set1f("specularStrength", specularStrength);
		pbrU.set1i("shininess", shininess);

		int activePLights = 0;
		for (int i = 0; i < (int)pointLights.size() && i < 4; i++) {
			if (!pointLights[i].enabled) continue;
			std::string b = "pointLights[" + std::to_string(activePLights) + "]";
			pbrU.set3fv(b + ".position", glm::value_ptr(pointLights[i].position));
			pbrU.set3fv(b + ".color", glm::value_ptr(pointLights[i].color));
			pbrU.set1f(b + ".intensity", pointLights[i].intensity);
			pbrU.set1f(b + ".constant", pointLights[i].constant);
			pbrU.set1f(b + ".linear", pointLights[i].linear);
			pbrU.set1f(b + ".quadratic", pointLights[i].quadratic);
			activePLights++;
		}
		pbrU.set1i("numPointLights", activePLights);
		pbrU.set1i("useSpotLight", useSpotLight);
		if (useSpotLight) {
			pbrU.set3fv("spotLight.position", glm::value_ptr(cameraPos));
			pbrU.set3fv("spotLight.direction", glm::value_ptr(cameraFront));
			pbrU.set3fv("spotLight.color", glm::value_ptr(spotColor));
			pbrU.set1f("spotLight.intensity", spotIntensity);
			pbrU.set1f("spotLight.cutOff", glm::cos(glm::radians(spotCutoff)));
			pbrU.set1f("spotLight.outerCutOff", glm::cos(glm::radians(spotOuterCutoff)));
			pbrU.set1f("spotLight.constant", 1.0f);
			pbrU.set1f("spotLight.linear", 0.09f);
			pbrU.set1f("spotLight.quadratic", 0.032f);
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, shadowMap.texture);
		pbrU.set1i("shadowMap", 0);
		pbrU.set1i("texAlbedo", 1);
		pbrU.set1i("texNormal", 2);
		pbrU.set1i("texMetallic", 3);
		pbrU.set1i("texRoughness", 4);
		pbrU.set1i("texAO", 5);

		for (int i = 0; i < (int)objects.size(); i++) {
			auto& obj = objects[i];
			if (!obj.visible) continue;
			glm::mat4 model = glm::scale(glm::translate(glm::mat4(1), obj.position), obj.scale);

			pbrU.set1i("usePBRTextures", obj.usePBR);
			pbrU.set3fv("objectColor", glm::value_ptr(obj.color));
			pbrU.set3fv("pbrAlbedoColor", glm::value_ptr(obj.pbrColor));
			pbrU.set1f("metallic", obj.metallic);
			pbrU.set1f("roughness", obj.roughness);
			pbrU.set1i("hasAlbedo", obj.usePBR && obj.pbr.hasAlbedo);
			pbrU.set1i("hasNormal", obj.usePBR && obj.pbr.hasNormal);
			pbrU.set1i("hasMetallic", obj.usePBR && obj.pbr.hasMetallic);
			pbrU.set1i("hasRoughness", obj.usePBR && obj.pbr.hasRoughness);
			pbrU.set1i("hasAO", obj.usePBR && obj.pbr.hasAO);

			if (obj.usePBR) {
				auto bindSlot = [](unsigned int tex, bool has, int slot) {
					glActiveTexture(GL_TEXTURE0 + slot);
					if (has) glBindTexture(GL_TEXTURE_2D, tex);
					};
				bindSlot(obj.pbr.albedo, obj.pbr.hasAlbedo, 1);
				bindSlot(obj.pbr.normal, obj.pbr.hasNormal, 2);
				bindSlot(obj.pbr.metallic, obj.pbr.hasMetallic, 3);
				bindSlot(obj.pbr.roughness, obj.pbr.hasRoughness, 4);
				bindSlot(obj.pbr.ao, obj.pbr.hasAO, 5);
			}

			glStencilFunc(GL_ALWAYS, 1, 0xFF);
			glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
			glStencilMask(0xFF);
			pbrU.setMat4("model", glm::value_ptr(model));
			glBindVertexArray(cubeVAO);
			glDrawArrays(GL_TRIANGLES, 0, 36);

			if (obj.showOutline && i == selectedObject) {
				glStencilFunc(GL_NOTEQUAL, 1, 0xFF);
				glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
				glStencilMask(0x00);
				glDisable(GL_DEPTH_TEST);
				glUseProgram(outlineProg);
				glUniformMatrix4fv(glGetUniformLocation(outlineProg, "model"), 1, GL_FALSE, glm::value_ptr(model));
				glUniformMatrix4fv(glGetUniformLocation(outlineProg, "view"), 1, GL_FALSE, glm::value_ptr(view));
				glUniformMatrix4fv(glGetUniformLocation(outlineProg, "projection"), 1, GL_FALSE, glm::value_ptr(proj));
				glUniform3fv(glGetUniformLocation(outlineProg, "outlineColor"), 1, glm::value_ptr(obj.outlineColor));
				glUniform1f(glGetUniformLocation(outlineProg, "outlineScale"), obj.outlineScale);
				glDrawArrays(GL_TRIANGLES, 0, 36);
				glStencilMask(0xFF);
				glStencilFunc(GL_ALWAYS, 0, 0xFF);
				glEnable(GL_DEPTH_TEST);
				glUseProgram(pbrProg);
			}
		}
		glStencilMask(0xFF);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
		glClearColor(0.06f, 0.06f, 0.08f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		int folderToDelete = -1;

		ImGui::SetNextWindowPos({ 0, 0 }, ImGuiCond_Always);
		ImGui::SetNextWindowSize({ 270, (float)SCR_HEIGHT - terminalHeight }, ImGuiCond_Always);
		ImGui::SetNextWindowBgAlpha(0.97f);
		ImGui::Begin("Explorer", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

		if (ImGui::SmallButton("+ Cube")) {
			objects.push_back({ "Cube_" + std::to_string(objects.size()), {0, 0.5f, 0}, {1,1,1}, {0.6f,0.6f,0.6f} });
			selectedObject = (int)objects.size() - 1;
			terminalLines.push_back({ "> Added: " + objects.back().name, {0.4f,1.f,0.5f,1.f} });
			terminalScrollToBottom = true;
		}
		ImGui::SameLine();
		if (ImGui::SmallButton("+ Folder")) { showNewFolderPopup = true; memset(newFolderName, 0, sizeof(newFolderName)); }
		ImGui::SameLine();
		ImGui::TextDisabled("|");
		ImGui::SameLine();
		if (ImGui::SmallButton("Stats")) showStatsPanel = !showStatsPanel;

		ImGui::Separator();

		if (showStatsPanel) {
			ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.06f, 0.07f, 0.10f, 1.f));
			ImGui::BeginChild("StatsInner", { 0, 90 }, true);
			ImGui::TextColored({ 0.4f, 1.f, 0.5f, 1.f }, "Performance");
			drawStatsPanel(fps);
			ImGui::EndChild();
			ImGui::PopStyleColor();
			ImGui::Spacing();
		}

		ImGui::Text("Scene");
		ImGui::Separator();
		float treeH = (float)(SCR_HEIGHT - (int)terminalHeight) * (showStatsPanel ? 0.28f : 0.42f);
		ImGui::BeginChild("SceneTree", { 0, treeH }, false);
		if (ImGui::TreeNodeEx("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
			for (int fi = 0; fi < (int)folders.size(); fi++) {
				ImGui::PushID(fi);
				bool folderOpen = ImGui::TreeNodeEx(folders[fi].name.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
				ImGui::SameLine();
				if (ImGui::SmallButton("x")) folderToDelete = fi;
				if (folderOpen) {
					for (int i = 0; i < (int)objects.size(); i++) {
						if (objects[i].folder != folders[fi].name) continue;
						bool sel = (selectedObject == i);
						std::string label = objects[i].name;
						if (objects[i].locked) label += " [L]";
						if (!objects[i].visible) label += " [H]";
						if (ImGui::Selectable(label.c_str(), sel)) selectedObject = i;
					}
					ImGui::TreePop();
				}
				ImGui::PopID();
			}
			for (int i = 0; i < (int)objects.size(); i++) {
				if (!objects[i].folder.empty()) continue;
				bool sel = (selectedObject == i);
				std::string label = objects[i].name;
				if (objects[i].locked) label += " [L]";
				if (!objects[i].visible) label += " [H]";
				if (ImGui::Selectable(label.c_str(), sel)) selectedObject = i;
			}
			ImGui::TreePop();
		}
		ImGui::EndChild();

		if (folderToDelete >= 0) {
			for (auto& obj : objects)
				if (obj.folder == folders[folderToDelete].name) obj.folder = "";
			terminalLines.push_back({ "> Folder deleted: " + folders[folderToDelete].name, {1.0f, 0.6f, 0.4f, 1.0f} });
			terminalScrollToBottom = true;
			folders.erase(folders.begin() + folderToDelete);
		}

		ImGui::Separator();
		ImGui::Text("Assets");
		ImGui::Separator();
		ImGui::BeginChild("Assets", { 0, 0 }, false);
		const char* assets[] = { "Cube.mesh", "default.mat", "phong.vert", "phong.frag" };
		for (auto& a : assets) ImGui::Selectable(a);
		ImGui::EndChild();
		ImGui::End();

		if (showNewFolderPopup) { ImGui::OpenPopup("New Folder"); showNewFolderPopup = false; }
		if (ImGui::BeginPopupModal("New Folder", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
			ImGui::Text("Folder name:");
			ImGui::SetNextItemWidth(200);
			ImGui::InputText("##fn", newFolderName, sizeof(newFolderName));
			if (ImGui::Button("Create") && newFolderName[0] != '\0') {
				folders.push_back({ std::string(newFolderName) });
				terminalLines.push_back({ std::string("> Folder created: ") + newFolderName, {0.7f, 0.9f, 1.0f, 1.0f} });
				terminalScrollToBottom = true;
				ImGui::CloseCurrentPopup();
			}
			ImGui::SameLine();
			if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
			ImGui::EndPopup();
		}

		ImGui::SetNextWindowPos({ 270, 0 }, ImGuiCond_Always);
		ImGui::SetNextWindowSize({ (float)SCR_WIDTH - 270 - 330, (float)SCR_HEIGHT }, ImGuiCond_Always);
		ImGui::SetNextWindowBgAlpha(0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 0, 0 });
		ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
			ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoTitleBar);
		ImVec2 vpSize = ImGui::GetContentRegionAvail();
		resizeViewportFBO(viewport, (int)vpSize.x, (int)vpSize.y);
		ImGui::Image((ImTextureID)(intptr_t)viewport.colorTex, vpSize, { 0,1 }, { 1,0 });

		if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
			ImVec2 mp = ImGui::GetMousePos();
			ImVec2 imgMin = ImGui::GetItemRectMin();
			int hit = pickObject(objects, { mp.x - imgMin.x, mp.y - imgMin.y }, viewport.width, viewport.height, view, proj);
			if (hit >= 0) selectedObject = hit;
		}

		ImGui::SetCursorPos({ 10, 10 });
		ImGui::BeginGroup();
		drawTransformModeBar();
		ImGui::SameLine(0, 16);
		ImGui::TextDisabled("|");
		ImGui::SameLine(0, 6);
		if (ImGui::SmallButton("Top"))   applyCameraSnap(CameraSnap::Top);
		ImGui::SameLine(0, 2);
		if (ImGui::SmallButton("Front")) applyCameraSnap(CameraSnap::Front);
		ImGui::SameLine(0, 2);
		if (ImGui::SmallButton("Right")) applyCameraSnap(CameraSnap::Right);
		ImGui::SameLine(0, 2);
		if (ImGui::SmallButton("Iso"))   applyCameraSnap(CameraSnap::Iso);
		ImGui::SameLine(0, 16);
		ImGui::TextDisabled("|");
		ImGui::SameLine(0, 6);
		if (ImGui::SmallButton(useShadows ? "Shad ON" : "Shad OFF")) useShadows = !useShadows;
		ImGui::SameLine(0, 6);
		if (ImGui::SmallButton(useSpotLight ? "Flash ON" : "Flash OFF")) useSpotLight = !useSpotLight;
		ImGui::EndGroup();

		ImGui::SetCursorPos({ 10, vpSize.y - 22 });
		ImGui::TextDisabled("[TAB] UI  |  WASD+QE Move  |  Shift: Sprint  |  G/R/S: Mode  |  Del: Delete  |  Ctrl+Z/Y: Undo/Redo");

		drawViewportOverlay(fps, (int)objects.size(), viewport.width, viewport.height);

		ImGui::End();
		ImGui::PopStyleVar();

		ImGui::SetNextWindowPos({ (float)SCR_WIDTH - 330, 0 }, ImGuiCond_Always);
		ImGui::SetNextWindowSize({ 330, (float)SCR_HEIGHT }, ImGuiCond_Always);
		ImGui::SetNextWindowBgAlpha(0.97f);
		ImGui::Begin("Inspector", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

		if (selectedObject >= 0 && selectedObject < (int)objects.size()) {
			auto& obj = objects[selectedObject];

			ImGui::PushStyleColor(ImGuiCol_Text, obj.locked ? ImVec4(1.f, 0.6f, 0.2f, 1.f) : ImVec4(0.88f, 0.89f, 0.92f, 1.f));
			ImGui::Text("%s  %s", obj.name.c_str(), obj.locked ? "[LOCKED]" : "");
			ImGui::PopStyleColor();

			ImGui::SameLine(ImGui::GetContentRegionAvail().x - 80);
			if (ImGui::SmallButton(obj.visible ? "Hide" : "Show"))
				obj.visible = !obj.visible;
			ImGui::SameLine(0, 2);
			if (ImGui::SmallButton(obj.locked ? "Unlock" : "Lock"))
				obj.locked = !obj.locked;

			ImGui::Separator();

			if (obj.locked) {
				ImGui::TextDisabled("Pos:   %.2f  %.2f  %.2f", obj.position.x, obj.position.y, obj.position.z);
				ImGui::TextDisabled("Scale: %.2f  %.2f  %.2f", obj.scale.x, obj.scale.y, obj.scale.z);
			}
			else {
				ImGui::Text("Transform");

				ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.30f, 0.08f, 0.08f, 1.f));
				ImGui::DragFloat("##px", &obj.position.x, 0.05f); ImGui::SameLine(0, 1);
				ImGui::PopStyleColor();
				ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.08f, 0.28f, 0.08f, 1.f));
				ImGui::DragFloat("##py", &obj.position.y, 0.05f); ImGui::SameLine(0, 1);
				ImGui::PopStyleColor();
				ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.08f, 0.10f, 0.32f, 1.f));
				ImGui::DragFloat("##pz", &obj.position.z, 0.05f);
				ImGui::PopStyleColor();
				ImGui::SameLine(); ImGui::TextDisabled("Pos");

				if (ImGui::IsItemActivated()) dragStartPos = obj.position;
				if (ImGui::IsItemDeactivatedAfterEdit() && dragStartPos != obj.position) {
					undoStack.push_back({ selectedObject, dragStartPos, obj.position, obj.scale, obj.scale });
					redoStack.clear();
				}

				ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.30f, 0.08f, 0.08f, 1.f));
				ImGui::DragFloat("##sx", &obj.scale.x, 0.05f, 0.01f, 20.f); ImGui::SameLine(0, 1);
				ImGui::PopStyleColor();
				ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.08f, 0.28f, 0.08f, 1.f));
				ImGui::DragFloat("##sy", &obj.scale.y, 0.05f, 0.01f, 20.f); ImGui::SameLine(0, 1);
				ImGui::PopStyleColor();
				ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.08f, 0.10f, 0.32f, 1.f));
				ImGui::DragFloat("##sz", &obj.scale.z, 0.05f, 0.01f, 20.f);
				ImGui::PopStyleColor();
				ImGui::SameLine(); ImGui::TextDisabled("Scale");

				if (ImGui::IsItemActivated()) dragStartScale = obj.scale;
				if (ImGui::IsItemDeactivatedAfterEdit() && dragStartScale != obj.scale) {
					undoStack.push_back({ selectedObject, obj.position, obj.position, dragStartScale, obj.scale });
					redoStack.clear();
				}

				ImGui::SameLine(0, 8);
				if (ImGui::SmallButton("Rst")) {
					obj.position = { 0, 0.5f, 0 };
					obj.scale = { 1, 1, 1 };
				}

				ImGui::Separator();
				ImGui::Checkbox("Outline", &obj.showOutline);
				if (obj.showOutline) {
					ImGui::ColorEdit3("Outline Color", glm::value_ptr(obj.outlineColor));
					ImGui::SliderFloat("Outline Size", &obj.outlineScale, 0.01f, 0.1f);
				}

				ImGui::Separator();
				ImGui::Text("Folder:");
				ImGui::SameLine();
				if (ImGui::BeginCombo("##folder", obj.folder.empty() ? "(none)" : obj.folder.c_str())) {
					if (ImGui::Selectable("(none)", obj.folder.empty())) obj.folder = "";
					for (auto& f : folders)
						if (ImGui::Selectable(f.name.c_str(), obj.folder == f.name)) obj.folder = f.name;
					ImGui::EndCombo();
				}
			}

			ImGui::Separator();
			if (ImGui::BeginTabBar("MatTabs")) {
				if (ImGui::BeginTabItem("Material")) {
					ImGui::Checkbox("PBR Textures", &obj.usePBR);
					if (!obj.usePBR) {
						ImGui::ColorEdit3("Color", glm::value_ptr(obj.color));
						ImGui::SliderFloat("Specular", &specularStrength, 0.f, 1.f);
						ImGui::SliderInt("Shininess", &shininess, 2, 256);
						ImGui::SliderFloat("Metallic", &obj.metallic, 0.f, 1.f);
						ImGui::SliderFloat("Roughness", &obj.roughness, 0.f, 1.f);

						ImGui::Spacing();
						ImGui::TextDisabled("Quick Presets");
						if (ImGui::SmallButton("Gold")) {
							obj.color = { 1.0f, 0.76f, 0.33f };
							obj.metallic = 1.f; obj.roughness = 0.1f; specularStrength = 0.9f;
						}
						ImGui::SameLine();
						if (ImGui::SmallButton("Chrome")) {
							obj.color = { 0.85f, 0.85f, 0.9f };
							obj.metallic = 1.f; obj.roughness = 0.05f; specularStrength = 1.f;
						}
						ImGui::SameLine();
						if (ImGui::SmallButton("Rubber")) {
							obj.color = { 0.05f, 0.05f, 0.05f };
							obj.metallic = 0.f; obj.roughness = 0.9f; specularStrength = 0.05f;
						}
						ImGui::SameLine();
						if (ImGui::SmallButton("Plastic")) {
							obj.metallic = 0.f; obj.roughness = 0.4f; specularStrength = 0.5f;
						}
					}
					else {
						if (!obj.pbr.hasAlbedo)
							ImGui::ColorEdit3("Albedo Color", glm::value_ptr(obj.pbrColor));
						ImGui::SliderFloat("Metallic (base)", &obj.metallic, 0.f, 1.f);
						ImGui::SliderFloat("Roughness (base)", &obj.roughness, 0.f, 1.f);
						ImGui::Spacing();
						ImGui::TextDisabled("Texture maps");
						ImGui::Separator();

						auto slot = [&](const char* label, unsigned int& tex, bool& has, char* path) {
							drawTextureSlot(label, tex, has, path, 512,
								[&](const char* p) { obj.pbr.loadMap(tex, has, p); },
								[&]() { obj.pbr.clearMap(tex, has, path); });
							};
						slot("Albedo", obj.pbr.albedo, obj.pbr.hasAlbedo, obj.pbr.pathAlbedo);
						slot("Normal", obj.pbr.normal, obj.pbr.hasNormal, obj.pbr.pathNormal);
						slot("Metallic", obj.pbr.metallic, obj.pbr.hasMetallic, obj.pbr.pathMetallic);
						slot("Roughness", obj.pbr.roughness, obj.pbr.hasRoughness, obj.pbr.pathRoughness);
						slot("AO", obj.pbr.ao, obj.pbr.hasAO, obj.pbr.pathAO);
					}
					ImGui::EndTabItem();
				}
				if (ImGui::BeginTabItem("Info")) {
					ImGui::TextDisabled("Name:"); ImGui::SameLine(); ImGui::Text("%s", obj.name.c_str());
					ImGui::TextDisabled("Pos :"); ImGui::SameLine();
					ImGui::Text("%.2f, %.2f, %.2f", obj.position.x, obj.position.y, obj.position.z);
					ImGui::TextDisabled("Scl :"); ImGui::SameLine();
					ImGui::Text("%.2f, %.2f, %.2f", obj.scale.x, obj.scale.y, obj.scale.z);
					ImGui::TextDisabled("Visible:"); ImGui::SameLine(); ImGui::Text("%s", obj.visible ? "Yes" : "No");
					ImGui::TextDisabled("Locked:"); ImGui::SameLine(); ImGui::Text("%s", obj.locked ? "Yes" : "No");
					ImGui::TextDisabled("PBR:"); ImGui::SameLine(); ImGui::Text("%s", obj.usePBR ? "Yes" : "No");
					ImGui::EndTabItem();
				}
				ImGui::EndTabBar();
			}
		}

		ImGui::Separator();
		if (ImGui::BeginTabBar("RenderTabs")) {
			if (ImGui::BeginTabItem("Render")) {
				if (ImGui::Button(useShadows ? "Shadows: ON " : "Shadows: OFF")) useShadows = !useShadows;
				ImGui::Separator();
				ImGui::Text("Directional Light");
				ImGui::DragFloat3("Dir", glm::value_ptr(dirLightDir), 0.01f, -1.f, 1.f);
				ImGui::ColorEdit3("Dir Color", glm::value_ptr(dirLightColor));
				ImGui::SliderFloat("Dir Intensity", &dirLightIntensity, 0.f, 3.f);
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Lights")) {
				ImGui::Text("Point Lights");
				for (int i = 0; i < (int)pointLights.size(); i++) {
					ImGui::PushID(i);
					auto& pl = pointLights[i];
					ImGui::ColorButton("##lc", { pl.color.r, pl.color.g, pl.color.b, 1.f }, 0, { 12, 12 });
					ImGui::SameLine();
					if (ImGui::TreeNode(("Light " + std::to_string(i)).c_str())) {
						ImGui::Checkbox("Enabled", &pl.enabled);
						ImGui::DragFloat3("Position", glm::value_ptr(pl.position), 0.05f);
						ImGui::ColorEdit3("Color", glm::value_ptr(pl.color));
						ImGui::SliderFloat("Intensity", &pl.intensity, 0.f, 5.f);
						ImGui::SliderFloat("Linear", &pl.linear, 0.001f, 1.f);
						ImGui::SliderFloat("Quadratic", &pl.quadratic, 0.001f, 1.f);
						ImGui::TreePop();
					}
					ImGui::PopID();
				}
				ImGui::Separator();
				ImGui::Text("Spot Light");
				ImGui::Checkbox("Enable Spot", &useSpotLight);
				if (useSpotLight) {
					ImGui::ColorEdit3("Spot Color", glm::value_ptr(spotColor));
					ImGui::SliderFloat("Spot Int", &spotIntensity, 0.f, 5.f);
					ImGui::SliderFloat("Cutoff", &spotCutoff, 1.f, 45.f);
					ImGui::SliderFloat("Outer", &spotOuterCutoff, spotCutoff, 60.f);
				}
				ImGui::EndTabItem();
			}
			ImGui::EndTabBar();
		}

		ImGui::End();

		ImGui::SetNextWindowPos({ 0, (float)SCR_HEIGHT - terminalHeight }, ImGuiCond_Always);
		ImGui::SetNextWindowSize({ (float)SCR_WIDTH - 330, terminalHeight }, ImGuiCond_Always);
		ImGui::SetNextWindowBgAlpha(0.97f);
		ImGui::Begin("Terminal", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
		ImGui::BeginChild("TerminalScroll", { 0, terminalHeight - 58 }, false, ImGuiWindowFlags_HorizontalScrollbar);
		for (auto& line : terminalLines)
			ImGui::TextColored(line.color, "%s", line.text.c_str());
		if (terminalScrollToBottom) { ImGui::SetScrollHereY(1.0f); terminalScrollToBottom = false; }
		ImGui::EndChild();

		ImGui::Separator();
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 60);

		if (ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows)) {
			if (ImGui::IsKeyPressed(ImGuiKey_UpArrow) && !cmdHistory.empty()) {
				cmdHistoryIdx = std::min(cmdHistoryIdx + 1, (int)cmdHistory.size() - 1);
				strncpy_s(terminalInput, sizeof(terminalInput), cmdHistory[cmdHistoryIdx].c_str(), sizeof(terminalInput) - 1);
			}
			if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
				cmdHistoryIdx = std::max(cmdHistoryIdx - 1, -1);
				if (cmdHistoryIdx < 0) memset(terminalInput, 0, sizeof(terminalInput));
				else strncpy_s(terminalInput, sizeof(terminalInput), cmdHistory[cmdHistoryIdx].c_str(), sizeof(terminalInput) - 1);
			}
		}

		bool enterPressed = ImGui::InputText("##cmd", terminalInput, sizeof(terminalInput), ImGuiInputTextFlags_EnterReturnsTrue);
		ImGui::SameLine();
		if ((ImGui::Button("Run") || enterPressed) && terminalInput[0] != '\0') {
			std::string raw = terminalInput;
			terminalLines.push_back({ "> " + raw, {0.75f, 0.85f, 1.0f, 1.0f} });
			cmdHistory.insert(cmdHistory.begin(), raw);
			if (cmdHistory.size() > 50) cmdHistory.pop_back();
			cmdHistoryIdx = -1;
			executeCommand(raw, selectedObject, terminalLines, terminalScrollToBottom);
			memset(terminalInput, 0, sizeof(terminalInput));
			terminalScrollToBottom = true;
			ImGui::SetKeyboardFocusHere(-1);
		}
		ImGui::End();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glDeleteProgram(pbrProg);
	glDeleteProgram(shadowProg);
	glDeleteProgram(outlineProg);
	glDeleteProgram(gridProg);
	glfwTerminate();
}