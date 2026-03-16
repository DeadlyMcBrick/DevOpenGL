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

const unsigned int SCR_WIDTH = 1920, SCR_HEIGHT = 1080;
glm::vec3 cameraPos(0, 0, 3), cameraFront(0, 0, -1), cameraUp(0, 1, 0);
float deltaTime = 0, lastFrame = 0, yaw = -90, pitch = 0, lastX = 400, lastY = 300;
bool firstMouse = true;
bool uiMode = false;

const char* vertSrc = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
out vec3 FragPos;
out vec3 Normal;
uniform mat4 model, view, projection;
void main(){
    FragPos = vec3(model * vec4(aPos,1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(FragPos,1.0);
})";

const char* fragSrc = R"(
#version 330 core
in vec3 FragPos;
in vec3 Normal;
out vec4 FragColor;
uniform vec3 lightPos, lightColor, objectColor, viewPos;
uniform float specularStrength;
uniform int shininess;
uniform bool useSpecular;
void main(){
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 result = (0.2 + diff) * lightColor * objectColor;
    if(useSpecular){
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
        result += specularStrength * spec * lightColor;
    }
    FragColor = vec4(result, 1.0);
})";

const char* outlineVertSrc = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 model, view, projection;
uniform float outlineScale;
void main(){
    vec3 pos = aPos + aNormal * outlineScale;
    gl_Position = projection * view * model * vec4(pos, 1.0);
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
void main(){
    gl_Position = projection * view * vec4(aPos, 1.0);
})";

const char* gridFragSrc = R"(
#version 330 core
out vec4 FragColor;
uniform vec4 gridColor;
void main(){ FragColor = gridColor; })";

void framebuffer_size_callback(GLFWwindow*, int w, int h) { glViewport(0, 0, w, h); }
void mouse_callback(GLFWwindow*, double xpos, double ypos) {
    if (uiMode) return;
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
    float xo = (xpos - lastX) * 0.1f, yo = (lastY - ypos) * 0.1f;
    lastX = xpos; lastY = ypos;
    yaw += xo; pitch += yo;
    pitch = glm::clamp(pitch, -89.f, 89.f);
    cameraFront = glm::normalize(glm::vec3(cos(glm::radians(yaw)) * cos(glm::radians(pitch)), sin(glm::radians(pitch)), sin(glm::radians(yaw)) * cos(glm::radians(pitch))));
}
void processInput(GLFWwindow* w) {
    if (glfwGetKey(w, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(w, true);
    static bool tabPressed = false;
    float s = 2.5f * deltaTime;
    if (glfwGetKey(w, GLFW_KEY_TAB) == GLFW_PRESS && !tabPressed) {
        uiMode = !uiMode;
        glfwSetInputMode(w, GLFW_CURSOR, uiMode ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
        firstMouse = true; tabPressed = true;
    }
    if (glfwGetKey(w, GLFW_KEY_TAB) == GLFW_RELEASE) tabPressed = false;
    if (uiMode) return;
    if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) cameraPos += s * cameraFront;
    if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) cameraPos -= s * cameraFront;
    if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * s;
    if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * s;
}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Valgut Engine", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_STENCIL_TEST);

    auto compile = [](GLenum type, const char* src) {
        unsigned int s = glCreateShader(type);
        glShaderSource(s, 1, &src, NULL);
        glCompileShader(s);
        return s;
        };

    unsigned int vs = compile(GL_VERTEX_SHADER, vertSrc), fs = compile(GL_FRAGMENT_SHADER, fragSrc);
    unsigned int prog = glCreateProgram();
    glAttachShader(prog, vs); glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs); glDeleteShader(fs);

    unsigned int ovs = compile(GL_VERTEX_SHADER, outlineVertSrc), ofs = compile(GL_FRAGMENT_SHADER, outlineFragSrc);
    unsigned int outlineProg = glCreateProgram();
    glAttachShader(outlineProg, ovs); glAttachShader(outlineProg, ofs);
    glLinkProgram(outlineProg);
    glDeleteShader(ovs); glDeleteShader(ofs);

    unsigned int gvs = compile(GL_VERTEX_SHADER, gridVertSrc), gfs = compile(GL_FRAGMENT_SHADER, gridFragSrc);
    unsigned int gridProg = glCreateProgram();
    glAttachShader(gridProg, gvs); glAttachShader(gridProg, gfs);
    glLinkProgram(gridProg);
    glDeleteShader(gvs); glDeleteShader(gfs);

    float verts[] = {
        -0.5f,-0.5f,-0.5f, 0, 0,-1,  0.5f,-0.5f,-0.5f, 0, 0,-1,  0.5f, 0.5f,-0.5f, 0, 0,-1,
         0.5f, 0.5f,-0.5f, 0, 0,-1, -0.5f, 0.5f,-0.5f, 0, 0,-1, -0.5f,-0.5f,-0.5f, 0, 0,-1,
        -0.5f,-0.5f, 0.5f, 0, 0, 1,  0.5f,-0.5f, 0.5f, 0, 0, 1,  0.5f, 0.5f, 0.5f, 0, 0, 1,
         0.5f, 0.5f, 0.5f, 0, 0, 1, -0.5f, 0.5f, 0.5f, 0, 0, 1, -0.5f,-0.5f, 0.5f, 0, 0, 1,
        -0.5f, 0.5f, 0.5f,-1, 0, 0, -0.5f, 0.5f,-0.5f,-1, 0, 0, -0.5f,-0.5f,-0.5f,-1, 0, 0,
        -0.5f,-0.5f,-0.5f,-1, 0, 0, -0.5f,-0.5f, 0.5f,-1, 0, 0, -0.5f, 0.5f, 0.5f,-1, 0, 0,
         0.5f, 0.5f, 0.5f, 1, 0, 0,  0.5f, 0.5f,-0.5f, 1, 0, 0,  0.5f,-0.5f,-0.5f, 1, 0, 0,
         0.5f,-0.5f,-0.5f, 1, 0, 0,  0.5f,-0.5f, 0.5f, 1, 0, 0,  0.5f, 0.5f, 0.5f, 1, 0, 0,
        -0.5f,-0.5f,-0.5f, 0,-1, 0,  0.5f,-0.5f,-0.5f, 0,-1, 0,  0.5f,-0.5f, 0.5f, 0,-1, 0,
         0.5f,-0.5f, 0.5f, 0,-1, 0, -0.5f,-0.5f, 0.5f, 0,-1, 0, -0.5f,-0.5f,-0.5f, 0,-1, 0,
        -0.5f, 0.5f,-0.5f, 0, 1, 0,  0.5f, 0.5f,-0.5f, 0, 1, 0,  0.5f, 0.5f, 0.5f, 0, 1, 0,
         0.5f, 0.5f, 0.5f, 0, 1, 0, -0.5f, 0.5f, 0.5f, 0, 1, 0, -0.5f, 0.5f,-0.5f, 0, 1, 0,
    };

    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO); glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float))); glEnableVertexAttribArray(1);

    std::vector<float> gridVerts;
    int gridSize = 20;
    float gridStep = 1.0f;
    float gridY = -0.5f;
    for (int i = -gridSize; i <= gridSize; i++) {
        float f = i * gridStep;
        gridVerts.insert(gridVerts.end(), { f, gridY, -(float)gridSize });
        gridVerts.insert(gridVerts.end(), { f, gridY,  (float)gridSize });
        gridVerts.insert(gridVerts.end(), { -(float)gridSize, gridY, f });
        gridVerts.insert(gridVerts.end(), { (float)gridSize, gridY, f });
    }

    unsigned int gridVAO, gridVBO;
    glGenVertexArrays(1, &gridVAO); glGenBuffers(1, &gridVBO);
    glBindVertexArray(gridVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
    glBufferData(GL_ARRAY_BUFFER, gridVerts.size() * sizeof(float), gridVerts.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);

    ////ImGui
    bool Triangle = true;
    float Size = 1.0f;
    float Color[4] = { 0.8f, 0.3f, 0.02f, 1.0f };
    bool useSpecular = true;
    float specularStrength = 0.5f;
    int shininess = 32;
    float lightPos[3] = { 1.2f, 1.0f, 2.0f };
    float Pos[3] = { 0.0f, 0.0f, 0.0f };
    bool showOutline = true;
    float outlineColor[3] = { 1.0f, 0.6f, 0.0f };
    float outlineScale = 0.05f;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetStyle().ScaleAllSizes(1.5f);
    ImGuiIO& io = ImGui::GetIO();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableSetMousePos;

    while (!glfwWindowShouldClose(window)) {
        float t = glfwGetTime(); deltaTime = t - lastFrame; lastFrame = t;
        processInput(window);
        glClearColor(0.12f, 0.12f, 0.12f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(Pos[0], Pos[1], Pos[2]));
        model = glm::rotate(model, t, glm::vec3(0.5f, 1, 0));
        model = glm::scale(model, glm::vec3(Size));
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 proj = glm::perspective(glm::radians(45.f), (float)SCR_WIDTH / SCR_HEIGHT, 0.1f, 100.f);

        glUseProgram(gridProg);
        glUniformMatrix4fv(glGetUniformLocation(gridProg, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(gridProg, "projection"), 1, GL_FALSE, glm::value_ptr(proj));
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUniform4f(glGetUniformLocation(gridProg, "gridColor"), 0.4f, 0.4f, 0.5f, 0.6f);
        glBindVertexArray(gridVAO);
        glDrawArrays(GL_LINES, 0, (int)(gridVerts.size() / 3));
        glDisable(GL_BLEND);

        glUseProgram(prog);
        glUniformMatrix4fv(glGetUniformLocation(prog, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(prog, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(prog, "projection"), 1, GL_FALSE, glm::value_ptr(proj));
        glUniform3fv(glGetUniformLocation(prog, "lightPos"), 1, lightPos);
        glUniform3f(glGetUniformLocation(prog, "lightColor"), 1, 1, 1);
        glUniform3f(glGetUniformLocation(prog, "objectColor"), Color[0], Color[1], Color[2]);
        glUniform3f(glGetUniformLocation(prog, "viewPos"), cameraPos.x, cameraPos.y, cameraPos.z);
        glUniform1f(glGetUniformLocation(prog, "specularStrength"), specularStrength);
        glUniform1i(glGetUniformLocation(prog, "shininess"), shininess);
        glUniform1i(glGetUniformLocation(prog, "useSpecular"), useSpecular);

        glStencilFunc(GL_ALWAYS, 1, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
        glStencilMask(0xFF);
        glBindVertexArray(VAO);
        if (Triangle) glDrawArrays(GL_TRIANGLES, 0, 36);

        if (Triangle && showOutline) {
            glStencilFunc(GL_NOTEQUAL, 1, 0xFF);
            glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
            glStencilMask(0x00);
            glDisable(GL_DEPTH_TEST);
            glUseProgram(outlineProg);
            glUniformMatrix4fv(glGetUniformLocation(outlineProg, "model"), 1, GL_FALSE, glm::value_ptr(model));
            glUniformMatrix4fv(glGetUniformLocation(outlineProg, "view"), 1, GL_FALSE, glm::value_ptr(view));
            glUniformMatrix4fv(glGetUniformLocation(outlineProg, "projection"), 1, GL_FALSE, glm::value_ptr(proj));
            glUniform3f(glGetUniformLocation(outlineProg, "outlineColor"), outlineColor[0], outlineColor[1], outlineColor[2]);
            glUniform1f(glGetUniformLocation(outlineProg, "outlineScale"), outlineScale);
            glBindVertexArray(VAO);
            glDrawArrays(GL_TRIANGLES, 0, 36);
            glStencilMask(0xFF);
            glStencilFunc(GL_ALWAYS, 1, 0xFF);
            glEnable(GL_DEPTH_TEST);
        }
        glStencilMask(0xFF);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(260, SCR_HEIGHT), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.95f);
        ImGui::Begin("Explorer", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
        ImGui::Text("Scene");
        ImGui::Separator();
        float treeHeight = SCR_HEIGHT * 0.45f;
        ImGui::BeginChild("SceneTree", ImVec2(0, treeHeight), false);
        if (ImGui::TreeNodeEx("Assets", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::TreeNode("Scenes")) {
                ImGui::Text("  Main.scene");
                ImGui::TreePop();
            }
            if (ImGui::TreeNode("Meshes")) {
                ImGui::Text("  Cube.obj");
                ImGui::Text("  Plane.obj");
                ImGui::TreePop();
            }
            if (ImGui::TreeNode("Textures")) {
                ImGui::Text("  default.png");
                ImGui::TreePop();
            }
            if (ImGui::TreeNode("Shaders")) {
                ImGui::TreePop();
            }
            ImGui::TreePop();
        }
        ImGui::EndChild();
        ImGui::Separator();
        ImGui::Text("Content");
        ImGui::Separator();
        ImGui::BeginChild("Content", ImVec2(0, 0), false);
        const char* items[] = { "Cube.obj", "Plane.obj", "default.png", "phong.vert", "phong.frag", "Main.scene" };
        for (auto& item : items) {
            ImGui::Selectable(item);
        }
        ImGui::EndChild();
        ImGui::End();

        ImVec2 panelSize(320, SCR_HEIGHT);
        ImGui::SetNextWindowPos(ImVec2(SCR_WIDTH - panelSize.x, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(panelSize, ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.95f);
        ImGui::Begin("Inspector", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
        ImGui::Text("Object");
        ImGui::Checkbox("Visible", &Triangle);
        ImGui::SliderFloat("Size", &Size, 0.5f, 2.0f);
        ImGui::SliderFloat3("Position", Pos, -5.0f, 5.0f);
        ImGui::ColorEdit4("Color", Color);
        ImGui::Separator();
        ImGui::Text("Outline");
        ImGui::Checkbox("Outline", &showOutline);
        if (showOutline) {
            ImGui::ColorEdit3("Outline Color", outlineColor);
            ImGui::SliderFloat("Outline Size", &outlineScale, 0.01f, 0.1f);
        }
        ImGui::Separator();
        ImGui::Text("Lighting");
        ImGui::SliderFloat3("Light Pos", lightPos, -5.0f, 5.0f);
        ImGui::Checkbox("Specular", &useSpecular);
        if (useSpecular) {
            ImGui::SliderFloat("Strength", &specularStrength, 0.0f, 1.0f);
            ImGui::SliderInt("Shininess", &shininess, 2, 256);
        }
        ImGui::End();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window); glfwPollEvents();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glDeleteVertexArrays(1, &VAO); glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &gridVAO); glDeleteBuffers(1, &gridVBO);
    glDeleteProgram(outlineProg);
    glDeleteProgram(gridProg);
    glfwTerminate();
}