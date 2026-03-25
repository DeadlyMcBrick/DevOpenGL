#define STB_IMAGE_IMPLEMENTATION
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "stb_image.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <string>

constexpr unsigned SCR_WIDTH = 1920, SCR_HEIGHT = 1080;
constexpr unsigned SHADOW_WIDTH = 2048, SHADOW_HEIGHT = 2048;

glm::vec3 cameraPos(0, 3, 8), cameraFront(0, -0.3f, -1), cameraUp(0, 1, 0);
float deltaTime = 0, lastFrame = 0, yaw = -90, pitch = -10, lastX = SCR_WIDTH / 2.f, lastY = SCR_HEIGHT / 2.f;
bool firstMouse = true, uiMode = false;

const char* pbrVertSrc = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
out vec3 FragPos;
out vec3 Normal;
out vec4 FragPosLightSpace;
uniform mat4 model, view, projection, lightSpaceMatrix;
void main(){
    vec4 world = model * vec4(aPos, 1.0);
    FragPos = world.xyz;
    Normal = mat3(transpose(inverse(model))) * aNormal;
    FragPosLightSpace = lightSpaceMatrix * world;
    gl_Position = projection * view * world;
})";

const char* pbrFragSrc = R"(
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec4 FragPosLightSpace;
out vec4 FragColor;

struct DirLight {
    vec3 direction;
    vec3 color;
    float intensity;
};
struct PointLight {
    vec3 position;
    vec3 color;
    float intensity;
    float constant, linear, quadratic;
};
struct SpotLight {
    vec3 position;
    vec3 direction;
    vec3 color;
    float intensity;
    float cutOff, outerCutOff;
    float constant, linear, quadratic;
};

uniform DirLight   dirLight;
uniform PointLight pointLights[4];
uniform int        numPointLights;
uniform SpotLight  spotLight;
uniform bool       useSpotLight;
uniform bool       useShadows;
uniform vec3       objectColor;
uniform vec3       viewPos;
uniform float      specularStrength;
uniform int        shininess;
uniform sampler2D  shadowMap;

float shadowCalc(vec4 fragPosLightSpace, vec3 norm, vec3 lightDir) {
    if (!useShadows) return 0.0;
    vec3 proj = fragPosLightSpace.xyz / fragPosLightSpace.w;
    proj = proj * 0.5 + 0.5;
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

vec3 calcDirLight(DirLight light, vec3 norm, vec3 viewDir) {
    vec3 ld = normalize(-light.direction);
    float diff = max(dot(norm, ld), 0.0);
    vec3 ref = reflect(-ld, norm);
    float spec = pow(max(dot(viewDir, ref), 0.0), shininess);
    float shadow = shadowCalc(FragPosLightSpace, norm, ld);
    vec3 ambient  = 0.15 * light.color * objectColor;
    vec3 diffuse  = diff * light.color * objectColor * light.intensity;
    vec3 specular = specularStrength * spec * light.color;
    return ambient + (1.0 - shadow) * (diffuse + specular);
}

vec3 calcPointLight(PointLight light, vec3 norm, vec3 viewDir) {
    vec3 ld = normalize(light.position - FragPos);
    float diff = max(dot(norm, ld), 0.0);
    vec3 ref = reflect(-ld, norm);
    float spec = pow(max(dot(viewDir, ref), 0.0), shininess);
    float dist = length(light.position - FragPos);
    float att = 1.0 / (light.constant + light.linear * dist + light.quadratic * dist * dist);
    return att * light.intensity * (diff * light.color * objectColor + specularStrength * spec * light.color);
}

vec3 calcSpotLight(SpotLight light, vec3 norm, vec3 viewDir) {
    vec3 ld = normalize(light.position - FragPos);
    float theta = dot(ld, normalize(-light.direction));
    float eps = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / eps, 0.0, 1.0);
    float diff = max(dot(norm, ld), 0.0);
    vec3 ref = reflect(-ld, norm);
    float spec = pow(max(dot(viewDir, ref), 0.0), shininess);
    float dist = length(light.position - FragPos);
    float att = 1.0 / (light.constant + light.linear * dist + light.quadratic * dist * dist);
    return att * intensity * light.intensity * (diff * light.color * objectColor + specularStrength * spec * light.color);
}

void main(){
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 result = calcDirLight(dirLight, norm, viewDir);
    for(int i = 0; i < numPointLights; i++)
        result += calcPointLight(pointLights[i], norm, viewDir);
    if(useSpotLight)
        result += calcSpotLight(spotLight, norm, viewDir);
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

void processInput(GLFWwindow* w) {
	static bool tabPressed = false;
	if (glfwGetKey(w, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(w, true);
	if (glfwGetKey(w, GLFW_KEY_TAB) == GLFW_PRESS && !tabPressed) {
		uiMode = !uiMode;
		glfwSetInputMode(w, GLFW_CURSOR, uiMode ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
		firstMouse = true; tabPressed = true;
	}
	if (glfwGetKey(w, GLFW_KEY_TAB) == GLFW_RELEASE) tabPressed = false;
	if (uiMode) return;
	float s = 5.0f * deltaTime;
	if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) cameraPos += s * cameraFront;
	if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) cameraPos -= s * cameraFront;
	if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * s;
	if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * s;
}

struct Vertex { float x, y, z, nx, ny, nz; };

Vertex cubeVerts[] = {
	{-0.5f,-0.5f,-0.5f, 0,0,-1},{ 0.5f,-0.5f,-0.5f, 0,0,-1},{ 0.5f, 0.5f,-0.5f, 0,0,-1},
	{ 0.5f, 0.5f,-0.5f, 0,0,-1},{-0.5f, 0.5f,-0.5f, 0,0,-1},{-0.5f,-0.5f,-0.5f, 0,0,-1},
	{-0.5f,-0.5f, 0.5f, 0,0, 1},{ 0.5f,-0.5f, 0.5f, 0,0, 1},{ 0.5f, 0.5f, 0.5f, 0,0, 1},
	{ 0.5f, 0.5f, 0.5f, 0,0, 1},{-0.5f, 0.5f, 0.5f, 0,0, 1},{-0.5f,-0.5f, 0.5f, 0,0, 1},
	{-0.5f, 0.5f, 0.5f,-1,0, 0},{-0.5f, 0.5f,-0.5f,-1,0, 0},{-0.5f,-0.5f,-0.5f,-1,0, 0},
	{-0.5f,-0.5f,-0.5f,-1,0, 0},{-0.5f,-0.5f, 0.5f,-1,0, 0},{-0.5f, 0.5f, 0.5f,-1,0, 0},
	{ 0.5f, 0.5f, 0.5f, 1,0, 0},{ 0.5f, 0.5f,-0.5f, 1,0, 0},{ 0.5f,-0.5f,-0.5f, 1,0, 0},
	{ 0.5f,-0.5f,-0.5f, 1,0, 0},{ 0.5f,-0.5f, 0.5f, 1,0, 0},{ 0.5f, 0.5f, 0.5f, 1,0, 0},
	{-0.5f,-0.5f,-0.5f, 0,-1,0},{ 0.5f,-0.5f,-0.5f, 0,-1,0},{ 0.5f,-0.5f, 0.5f, 0,-1,0},
	{ 0.5f,-0.5f, 0.5f, 0,-1,0},{-0.5f,-0.5f, 0.5f, 0,-1,0},{-0.5f,-0.5f,-0.5f, 0,-1,0},
	{-0.5f, 0.5f,-0.5f, 0, 1,0},{ 0.5f, 0.5f,-0.5f, 0, 1,0},{ 0.5f, 0.5f, 0.5f, 0, 1,0},
	{ 0.5f, 0.5f, 0.5f, 0, 1,0},{-0.5f, 0.5f, 0.5f, 0, 1,0},{-0.5f, 0.5f,-0.5f, 0, 1,0},
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
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	return sm;
}

struct ViewportFBO { unsigned int fbo, colorTex, rbo; int width, height; };

ViewportFBO createViewportFBO(int w, int h) {
	ViewportFBO v{ 0, 0, 0, w, h };
	glGenFramebuffers(1, &v.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, v.fbo);
	glGenTextures(1, &v.colorTex);
	glBindTexture(GL_TEXTURE_2D, v.colorTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, v.colorTex, 0);
	glGenRenderbuffers(1, &v.rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, v.rbo);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, v.rbo);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	return v;
}

void resizeViewportFBO(ViewportFBO& v, int w, int h) {
	if (v.width == w && v.height == h) return;
	v.width = w; v.height = h;
	glBindTexture(GL_TEXTURE_2D, v.colorTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
	glBindRenderbuffer(GL_RENDERBUFFER, v.rbo);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
}

struct SceneObject {
	std::string name;
	glm::vec3 position{ 0 }, scale{ 1 };
	glm::vec3 color{ 0.8f, 0.3f, 0.02f };
	bool visible = true;
	bool showOutline = true;
	float outlineScale = 0.04f;
	glm::vec3 outlineColor{ 1.0f, 0.6f, 0.0f };
	std::string folder = "";
};

struct SceneFolder {
	std::string name;
	bool open = true;
};

struct PointLightData {
	glm::vec3 position{ 0, 2, 0 };
	glm::vec3 color{ 1, 1, 1 };
	float intensity = 1.0f;
	float constant = 1.0f, linear = 0.09f, quadratic = 0.032f;
	bool enabled = true;
};

struct TerminalLine {
	std::string text;
	ImVec4 color;
};

unsigned int buildVAO(const void* data, size_t size) {
	unsigned int VAO, VBO;
	glGenVertexArrays(1, &VAO); glGenBuffers(1, &VBO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, x)); glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, nx)); glEnableVertexAttribArray(1);
	return VAO;
}

unsigned int buildGridVAO(std::vector<float>& out) {
	const int G = 20;
	for (int i = -G; i <= G; i++) {
		float f = (float)i;
		out.insert(out.end(), { f, 0.0f, -(float)G, f, 0.0f, (float)G });
		out.insert(out.end(), { -(float)G, 0.0f, f, (float)G, 0.0f, f });
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
	glm::vec3 t0 = (boxMin - ro) * invD;
	glm::vec3 t1 = (boxMax - ro) * invD;
	glm::vec3 tMin = glm::min(t0, t1);
	glm::vec3 tMax = glm::max(t0, t1);
	float tNear = glm::max(glm::max(tMin.x, tMin.y), tMin.z);
	float tFar = glm::min(glm::min(tMax.x, tMax.y), tMax.z);
	if (tFar < 0 || tNear > tFar) return -1.f;
	return tNear;
}

int pickObject(const std::vector<SceneObject>& objects, glm::vec2 mouseVP, int vpW, int vpH,
	glm::mat4 view, glm::mat4 proj) {
	float ndcX = (2.f * mouseVP.x) / vpW - 1.f;
	float ndcY = -(2.f * mouseVP.y) / vpH + 1.f;
	glm::vec4 clip(ndcX, ndcY, -1.f, 1.f);
	glm::vec4 eye = glm::inverse(proj) * clip;
	eye = glm::vec4(eye.x / eye.w, eye.y / eye.w, -1.f, 0.f);
	glm::vec3 worldDir = glm::normalize(glm::vec3(glm::inverse(view) * eye));

	int   best = -1;
	float bestT = 1e9f;
	for (int i = 0; i < (int)objects.size(); i++) {
		if (!objects[i].visible) continue;
		glm::vec3 half = objects[i].scale * 0.5f;
		float t = rayAABB(cameraPos, worldDir,
			objects[i].position - half,
			objects[i].position + half);
		if (t > 0.f && t < bestT) { bestT = t; best = i; }
	}
	return best;
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

	unsigned int cubeVAO = buildVAO(cubeVerts, sizeof(cubeVerts));
	std::vector<float> gridData;
	unsigned int gridVAO = buildGridVAO(gridData);

	ShadowMap   shadowMap = createShadowMap();
	ViewportFBO viewport = createViewportFBO(1280, 720);

	std::vector<SceneObject> objects = {
		{"Cube_0",  {0,  0.5f,  0},  {1,1,1}, {0.8f, 0.3f, 0.02f}},
		{"Cube_1",  {3,  0.5f,  0},  {1,1,1}, {0.2f, 0.5f, 0.9f}},
		{"Cube_2",  {-3, 0.5f,  1},  {1,1,1}, {0.1f, 0.8f, 0.3f}},
	};
	int selectedObject = 0;

	std::vector<SceneFolder> folders;
	bool showNewFolderPopup = false;
	char newFolderName[64] = "";

	std::vector<TerminalLine> terminalLines;
	terminalLines.push_back({ "> Valgut Engine started", {0.4f, 1.0f, 0.4f, 1.0f} });
	char terminalInput[256] = "";
	bool terminalScrollToBottom = false;
	float terminalHeight = 180.f;

	std::vector<PointLightData> pointLights = {
		{{2,  3, 2},  {1.0f, 0.8f, 0.6f}, 2.0f},
		{{-3, 2, -1}, {0.4f, 0.6f, 1.0f}, 1.5f},
	};

	glm::vec3 dirLightDir(-0.5f, -1.0f, -0.5f);
	glm::vec3 dirLightColor(1.0f, 0.98f, 0.9f);
	float     dirLightIntensity = 0.8f;

	bool  useSpotLight = false;
	float spotCutoff = 12.5f;
	float spotOuterCutoff = 17.5f;
	glm::vec3 spotColor(1.0f, 1.0f, 0.8f);
	float     spotIntensity = 2.0f;

	float specularStrength = 0.5f;
	int   shininess = 32;

	bool useShadows = true;

	// viewport window screen-space origin (updated each frame)
	ImVec2 vpScreenPos = { 270, 0 };

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	ImGui::GetStyle().ScaleAllSizes(1.3f);
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init();

	while (!glfwWindowShouldClose(window)) {
		float t = (float)glfwGetTime();
		deltaTime = t - lastFrame; lastFrame = t;
		processInput(window);

		glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
		glm::mat4 proj = glm::perspective(glm::radians(45.f), (float)viewport.width / viewport.height, 0.1f, 200.f);

		glm::vec3 lightOrigin(-5, 10, -5);
		glm::mat4 lightProj = glm::ortho(-12.f, 12.f, -12.f, 12.f, 1.f, 40.f);
		glm::mat4 lightView = glm::lookAt(lightOrigin, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
		glm::mat4 lightSpace = lightProj * lightView;

		glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
		glBindFramebuffer(GL_FRAMEBUFFER, shadowMap.fbo);
		glClear(GL_DEPTH_BUFFER_BIT);
		glUseProgram(shadowProg);
		glUniformMatrix4fv(glGetUniformLocation(shadowProg, "lightSpaceMatrix"), 1, GL_FALSE, glm::value_ptr(lightSpace));
		for (auto& obj : objects) {
			if (!obj.visible) continue;
			glm::mat4 model(1);
			model = glm::translate(model, obj.position);
			model = glm::scale(model, obj.scale);
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
		glUniformMatrix4fv(glGetUniformLocation(pbrProg, "view"), 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(glGetUniformLocation(pbrProg, "projection"), 1, GL_FALSE, glm::value_ptr(proj));
		glUniformMatrix4fv(glGetUniformLocation(pbrProg, "lightSpaceMatrix"), 1, GL_FALSE, glm::value_ptr(lightSpace));
		glUniform3fv(glGetUniformLocation(pbrProg, "viewPos"), 1, glm::value_ptr(cameraPos));
		glUniform1i(glGetUniformLocation(pbrProg, "useShadows"), useShadows);

		glUniform3fv(glGetUniformLocation(pbrProg, "dirLight.direction"), 1, glm::value_ptr(dirLightDir));
		glUniform3fv(glGetUniformLocation(pbrProg, "dirLight.color"), 1, glm::value_ptr(dirLightColor));
		glUniform1f(glGetUniformLocation(pbrProg, "dirLight.intensity"), dirLightIntensity);

		int activePLights = 0;
		for (int i = 0; i < (int)pointLights.size() && i < 4; i++) {
			if (!pointLights[i].enabled) continue;
			std::string b = "pointLights[" + std::to_string(activePLights) + "]";
			glUniform3fv(glGetUniformLocation(pbrProg, (b + ".position").c_str()), 1, glm::value_ptr(pointLights[i].position));
			glUniform3fv(glGetUniformLocation(pbrProg, (b + ".color").c_str()), 1, glm::value_ptr(pointLights[i].color));
			glUniform1f(glGetUniformLocation(pbrProg, (b + ".intensity").c_str()), pointLights[i].intensity);
			glUniform1f(glGetUniformLocation(pbrProg, (b + ".constant").c_str()), pointLights[i].constant);
			glUniform1f(glGetUniformLocation(pbrProg, (b + ".linear").c_str()), pointLights[i].linear);
			glUniform1f(glGetUniformLocation(pbrProg, (b + ".quadratic").c_str()), pointLights[i].quadratic);
			activePLights++;
		}
		glUniform1i(glGetUniformLocation(pbrProg, "numPointLights"), activePLights);

		glUniform1i(glGetUniformLocation(pbrProg, "useSpotLight"), useSpotLight);
		if (useSpotLight) {
			glUniform3fv(glGetUniformLocation(pbrProg, "spotLight.position"), 1, glm::value_ptr(cameraPos));
			glUniform3fv(glGetUniformLocation(pbrProg, "spotLight.direction"), 1, glm::value_ptr(cameraFront));
			glUniform3fv(glGetUniformLocation(pbrProg, "spotLight.color"), 1, glm::value_ptr(spotColor));
			glUniform1f(glGetUniformLocation(pbrProg, "spotLight.intensity"), spotIntensity);
			glUniform1f(glGetUniformLocation(pbrProg, "spotLight.cutOff"), glm::cos(glm::radians(spotCutoff)));
			glUniform1f(glGetUniformLocation(pbrProg, "spotLight.outerCutOff"), glm::cos(glm::radians(spotOuterCutoff)));
			glUniform1f(glGetUniformLocation(pbrProg, "spotLight.constant"), 1.0f);
			glUniform1f(glGetUniformLocation(pbrProg, "spotLight.linear"), 0.09f);
			glUniform1f(glGetUniformLocation(pbrProg, "spotLight.quadratic"), 0.032f);
		}

		glUniform1f(glGetUniformLocation(pbrProg, "specularStrength"), specularStrength);
		glUniform1i(glGetUniformLocation(pbrProg, "shininess"), shininess);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, shadowMap.texture);
		glUniform1i(glGetUniformLocation(pbrProg, "shadowMap"), 0);

		for (int i = 0; i < (int)objects.size(); i++) {
			auto& obj = objects[i];
			if (!obj.visible) continue;
			glm::mat4 model(1);
			model = glm::translate(model, obj.position);
			model = glm::scale(model, obj.scale);

			glStencilFunc(GL_ALWAYS, 1, 0xFF);
			glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
			glStencilMask(0xFF);

			glUseProgram(pbrProg);
			glUniformMatrix4fv(glGetUniformLocation(pbrProg, "model"), 1, GL_FALSE, glm::value_ptr(model));
			glUniform3fv(glGetUniformLocation(pbrProg, "objectColor"), 1, glm::value_ptr(obj.color));
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
			}
		}
		glStencilMask(0xFF);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
		glClearColor(0.08f, 0.08f, 0.08f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// Explorer iMGUI
		ImGui::SetNextWindowPos({ 0, 0 }, ImGuiCond_Always);
		ImGui::SetNextWindowSize({ 270, (float)SCR_HEIGHT - terminalHeight }, ImGuiCond_Always);
		ImGui::SetNextWindowBgAlpha(0.95f);
		ImGui::Begin("Explorer", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
		ImGui::Text("Scene");
		ImGui::SameLine();
		if (ImGui::SmallButton("+ Folder")) {
			showNewFolderPopup = true;
			memset(newFolderName, 0, sizeof(newFolderName));
		}
		ImGui::Separator();
		ImGui::BeginChild("SceneTree", { 0, (float)(SCR_HEIGHT - terminalHeight) * 0.4f }, false);
		if (ImGui::TreeNodeEx("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
			for (auto& folder : folders) {
				bool folderOpen = ImGui::TreeNodeEx(folder.name.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
				if (folderOpen) {
					for (int i = 0; i < (int)objects.size(); i++) {
						if (objects[i].folder != folder.name) continue;
						bool sel = (selectedObject == i);
						if (ImGui::Selectable(objects[i].name.c_str(), sel)) selectedObject = i;
					}
					ImGui::TreePop();
				}
			}
			for (int i = 0; i < (int)objects.size(); i++) {
				if (!objects[i].folder.empty()) continue;
				bool sel = (selectedObject == i);
				if (ImGui::Selectable(objects[i].name.c_str(), sel)) selectedObject = i;
			}
			ImGui::TreePop();
		}
		ImGui::EndChild();
		ImGui::Separator();
		ImGui::Text("Assets");
		ImGui::Separator();
		ImGui::BeginChild("Assets", { 0, 0 }, false);
		const char* assets[] = { "Cube.mesh", "default.mat", "phong.vert", "phong.frag" };
		for (auto& a : assets) ImGui::Selectable(a);
		ImGui::EndChild();
		ImGui::End();

		if (showNewFolderPopup) {
			ImGui::OpenPopup("New Folder");
			showNewFolderPopup = false;
		}
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
		vpScreenPos = ImGui::GetWindowPos();
		ImVec2 vpSize = ImGui::GetContentRegionAvail();
		resizeViewportFBO(viewport, (int)vpSize.x, (int)vpSize.y);
		ImGui::Image((ImTextureID)(intptr_t)viewport.colorTex, vpSize, { 0,1 }, { 1,0 });

		if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
			ImVec2 mp = ImGui::GetMousePos();
			ImVec2 imgMin = ImGui::GetItemRectMin();
			float  rx = mp.x - imgMin.x;
			float  ry = mp.y - imgMin.y;
			int hit = pickObject(objects, { rx, ry }, viewport.width, viewport.height, view, proj);
			if (hit >= 0) selectedObject = hit;
		}

		ImGui::SetCursorPos({ 10, 10 });
		ImGui::TextDisabled("[TAB] Toggle UI  |  WASD Move  |  Mouse Look");
		ImGui::End();
		ImGui::PopStyleVar();

		ImGui::SetNextWindowPos({ (float)SCR_WIDTH - 330, 0 }, ImGuiCond_Always);
		ImGui::SetNextWindowSize({ 330, (float)SCR_HEIGHT }, ImGuiCond_Always);
		ImGui::SetNextWindowBgAlpha(0.95f);
		ImGui::Begin("Inspector", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

		if (selectedObject < (int)objects.size()) {
			auto& obj = objects[selectedObject];
			ImGui::Text("Object: %s", obj.name.c_str());
			ImGui::Separator();
			ImGui::Checkbox("Visible", &obj.visible);
			ImGui::DragFloat3("Position", glm::value_ptr(obj.position), 0.05f);
			ImGui::DragFloat3("Scale", glm::value_ptr(obj.scale), 0.05f, 0.01f, 20.f);
			ImGui::ColorEdit3("Color", glm::value_ptr(obj.color));
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
		ImGui::Text("Material");
		ImGui::SliderFloat("Specular", &specularStrength, 0.f, 1.f);
		ImGui::SliderInt("Shininess", &shininess, 2, 256);

		ImGui::Separator();
		ImGui::Text("Rendering");
		if (ImGui::Button(useShadows ? "Shadows: ON " : "Shadows: OFF"))
			useShadows = !useShadows;

		ImGui::Separator();
		ImGui::Text("Directional Light");
		ImGui::DragFloat3("Dir", glm::value_ptr(dirLightDir), 0.01f, -1.f, 1.f);
		ImGui::ColorEdit3("Dir Color", glm::value_ptr(dirLightColor));
		ImGui::SliderFloat("Dir Intensity", &dirLightIntensity, 0.f, 3.f);

		ImGui::Separator();
		ImGui::Text("Point Lights");
		for (int i = 0; i < (int)pointLights.size(); i++) {
			ImGui::PushID(i);
			auto& pl = pointLights[i];
			std::string label = "Point Light " + std::to_string(i);
			if (ImGui::TreeNode(label.c_str())) {
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
		ImGui::Text("Spot Light (Flashlight)");
		ImGui::Checkbox("Enable Spot", &useSpotLight);
		if (useSpotLight) {
			ImGui::ColorEdit3("Spot Color", glm::value_ptr(spotColor));
			ImGui::SliderFloat("Spot Int", &spotIntensity, 0.f, 5.f);
			ImGui::SliderFloat("Cutoff", &spotCutoff, 1.f, 45.f);
			ImGui::SliderFloat("Outer", &spotOuterCutoff, spotCutoff, 60.f);
		}
		ImGui::End();

		ImGui::SetNextWindowPos({ 0, (float)SCR_HEIGHT - terminalHeight }, ImGuiCond_Always);
		ImGui::SetNextWindowSize({ (float)SCR_WIDTH - 330, terminalHeight }, ImGuiCond_Always);
		ImGui::SetNextWindowBgAlpha(0.95f);
		ImGui::Begin("Terminal", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
		ImGui::BeginChild("TerminalScroll", { 0, terminalHeight - 58 }, false, ImGuiWindowFlags_HorizontalScrollbar);
		for (auto& line : terminalLines)
			ImGui::TextColored(line.color, "%s", line.text.c_str());
		if (terminalScrollToBottom) { ImGui::SetScrollHereY(1.0f); terminalScrollToBottom = false; }
		ImGui::EndChild();
		ImGui::Separator();
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 60);
		bool enterPressed = ImGui::InputText("##cmd", terminalInput, sizeof(terminalInput), ImGuiInputTextFlags_EnterReturnsTrue);
		ImGui::SameLine();
		if ((ImGui::Button("Run") || enterPressed) && terminalInput[0] != '\0') {
			terminalLines.push_back({ std::string("> ") + terminalInput, {1.0f, 1.0f, 1.0f, 1.0f} });
			terminalLines.push_back({ "  Unknown command.", {1.0f, 0.4f, 0.4f, 1.0f} });
			memset(terminalInput, 0, sizeof(terminalInput));
			terminalScrollToBottom = true;
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